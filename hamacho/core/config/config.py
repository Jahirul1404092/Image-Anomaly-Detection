"""Get configurable parameters."""


# TODO: This would require a new design.
# TODO: https://jira.devtools.intel.com/browse/IAAALD-149

import os
import sys

from pathlib import Path
from typing import List, Optional, Union, Iterable
from warnings import warn

import click

from omegaconf import DictConfig, ListConfig, OmegaConf

from hamacho.plug_in.models import torch_model_list
from hamacho.core.data import (
    ensure_folder_format_mask,
    ensure_folder_data_format,
    ensure_mvtec_data_format,
    MVTEC_CATEGORIES,
)

NO_BAD_MODE_NORMALIZATION = "sigma6"


def update_input_size_config(
    config: Union[DictConfig, ListConfig]
) -> Union[DictConfig, ListConfig]:
    """Update config with image size as tuple, effective input size and tiling stride.

    Convert integer image size parameters into tuples, calculate the effective input size based on image size
    and crop size, and set tiling stride if undefined.

    Args:
        config (Union[DictConfig, ListConfig]): Configurable parameters object

    Returns:
        Union[DictConfig, ListConfig]: Configurable parameters with updated values
    """
    # handle image size
    if isinstance(config.dataset.image_size, int):
        config.dataset.image_size = (config.dataset.image_size,) * 2

    config.model.input_size = config.dataset.image_size

    if "tiling" in config.dataset.keys() and config.dataset.tiling.apply:
        if isinstance(config.dataset.tiling.tile_size, int):
            config.dataset.tiling.tile_size = (config.dataset.tiling.tile_size,) * 2
        if config.dataset.tiling.stride is None:
            config.dataset.tiling.stride = config.dataset.tiling.tile_size

    return config


def update_nncf_config(
    config: Union[DictConfig, ListConfig]
) -> Union[DictConfig, ListConfig]:
    """Set the NNCF input size based on the value of the crop_size parameter in the configurable parameters object.

    Args:
        config (Union[DictConfig, ListConfig]): Configurable parameters of the current run.

    Returns:
        Union[DictConfig, ListConfig]: Updated configurable parameters in DictConfig object.
    """
    crop_size = config.dataset.image_size
    sample_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    if "optimization" in config.keys():
        if "nncf" in config.optimization.keys():
            if "input_info" not in config.optimization.nncf.keys():
                config.optimization.nncf["input_info"] = {"sample_size": None}
            config.optimization.nncf.input_info.sample_size = [1, 3, *sample_size]
            if config.optimization.nncf.apply:
                if "update_config" in config.optimization.nncf:
                    return OmegaConf.merge(
                        config, config.optimization.nncf.update_config
                    )
    return config


def update_multi_gpu_training_config(
    config: Union[DictConfig, ListConfig]
) -> Union[DictConfig, ListConfig]:
    """Updates the config to change learning rate based on number of gpus assigned.

    Current behaviour is to ensure only ddp accelerator is used.

    Args:
        config (Union[DictConfig, ListConfig]): Configurable parameters for the current run

    Raises:
        ValueError: If unsupported accelerator is passed

    Returns:
        Union[DictConfig, ListConfig]: Updated config
    """
    # validate accelerator
    if config.trainer.accelerator is not None:
        if config.trainer.accelerator.lower() != "ddp":
            if config.trainer.accelerator.lower() in ("dp", "ddp_spawn", "ddp2"):
                warn(
                    f"Using accelerator {config.trainer.accelerator.lower()} is discouraged. "
                    f"Please use one of [null, ddp]. Setting accelerator to ddp"
                )
                config.trainer.accelerator = "ddp"
            else:
                raise ValueError(
                    f"Unsupported accelerator found: {config.trainer.accelerator}. Should be one of [null, ddp]"
                )
    # Increase learning rate
    # since pytorch averages the gradient over devices, the idea is to
    # increase the learning rate by the number of devices
    if "lr" in config.model:
        # Number of GPUs can either be passed as gpus: 2 or gpus: [0,1]
        n_gpus: Union[int, List] = 1
        if "trainer" in config and "gpus" in config.trainer:
            n_gpus = config.trainer.gpus
        lr_scaler = n_gpus if isinstance(n_gpus, int) else len(n_gpus)
        config.model.lr = config.model.lr * lr_scaler
    return config


def get_configurable_parameters(
    model_name: Optional[str] = None,
    config_path: Optional[Union[Path, str]] = None,
    weight_file: Optional[str] = None,
    config_filename: Optional[str] = "config",
    config_file_extension: Optional[str] = "yaml",
) -> Union[DictConfig, ListConfig]:
    """Get configurable parameters.

    Args:
        model_name: Optional[str]:  (Default value = None)
        config_path: Optional[Union[Path, str]]:  (Default value = None)
        weight_file: Path to the weight file
        config_filename: Optional[str]:  (Default value = "config")
        config_file_extension: Optional[str]:  (Default value = "yaml")

    Returns:
        Union[DictConfig, ListConfig]: Configurable parameters in DictConfig object.
    """
    if model_name is None and config_path is None:
        raise ValueError(
            "Both model_name and model config path cannot be None! "
            "Please provide a model name or path to a config file!"
        )

    if config_path is None:
        config_path = Path(
            f"hamacho/plug_in/models/{model_name}/{config_filename}.{config_file_extension}"
        )

    config = OmegaConf.load(config_path)

    # Dataset Configs
    if "format" not in config.dataset.keys():
        config.dataset.format = "mvtec"

    config = update_input_size_config(config)

    # Project Configs
    data_format = config.dataset.format.lower()
    if data_format in ("folder", "filelist"):
        project_path = (
            config.project.path / Path(config.dataset.category) / config.model.name
        )
    elif data_format == "mvtec":
        project_path = (
            config.project.path
            / Path(data_format)
            / Path(config.model.name)
            / config.dataset.category
        )

    if "test_dir_name" not in config.project.keys():
        config.project.test_dir_name = "test_predictions"

    if "inference_dir_name" not in config.project.keys():
        config.project.inference_dir_name = "inference"

    if "test_dir_name" not in config.project.keys():
        config.project.test_dir_name = "test_predictions"

    if "infer_dir_name" not in config.project.keys():
        config.project.infer_dir_name = "inference"

    (project_path / "weights").mkdir(parents=True, exist_ok=True)
    (project_path / config.project.test_dir_name).mkdir(parents=True, exist_ok=True)
    config.project.path = str(project_path)
    config.project.save_root = str(project_path)
    # loggers should write to project_path folder
    config.trainer.default_root_dir = str(project_path)

    if weight_file:
        config.model.weight_file = weight_file

    config = update_nncf_config(config)

    # thresholding
    if "metrics" in config.keys():
        if "pixel_default" not in config.metrics.threshold.keys():
            config.metrics.threshold.pixel_default = (
                config.metrics.threshold.image_default
            )

        if "image_norm" not in config.metrics.threshold.keys():
            config.metrics.threshold.image_norm = 0.5

        if "pixel_norm" not in config.metrics.threshold.keys():
            config.metrics.threshold.pixel_norm = 0.5

        image_norm = config.metrics.threshold.image_norm
        if image_norm <= 0.01 or image_norm >= 0.99:
            click.secho(
                f"WARNING: 'image_norm' threshold must be within 0.01 and 0.99. \n"
                f"Changing image_norm to 0.5 from {image_norm} as of now. \n"
                f"Please, set it correctly in the config file loacted at {config_path}.",
                fg="yellow",
            )
            config.metrics.threshold.image_norm = 0.5
        pixel_norm = config.metrics.threshold.pixel_norm
        if pixel_norm <= 0.01 or pixel_norm >= 0.99:
            click.secho(
                f"WARNING: 'pixel_norm' threshold must be within 0.01 and 0.99. \n"
                f"Changing pixel_norm to 0.5 from {pixel_norm} as of now. \n"
                f"Please, set it correctly in the config file loacted at {config_path}.",
                fg="yellow",
            )
            config.metrics.threshold.pixel_norm = 0.5

    return config


def validate_model_params(
    ref_config: Union[DictConfig, ListConfig],
    check_config: Union[DictConfig, ListConfig],
    check_config_path: Optional[str] = None,
):
    config_path_err_msg = (
        f"Config located at {check_config_path} does not match other ones"
        if check_config_path is not None
        else ""
    )
    if ref_config["model"] != check_config["model"]:
        ValueError(
            "Please check if all the model parameters of config files "
            f"are the same. {config_path_err_msg}"
        )
    if ref_config["dataset"]["image_size"] != check_config["dataset"]["image_size"]:
        ValueError(
            "Please check if all the image_size parameters of config files "
            f"are the same. {config_path_err_msg}"
        )
    return True


def generate_multi_inferencer_config(
    configs: Iterable[Union[DictConfig, ListConfig]],
    accelerator: str = "auto",
    save_dir: Optional[Union[Path, str]] = None,
    filename: str = "multi-category-config.yaml",
):
    # validate model parameters
    reference_config = configs[0].copy()
    for config in configs:
        validate_model_params(
            ref_config=reference_config,
            check_config=config,
        )

    del reference_config["model"]["weight_file"]
    reference_config = update_input_size_config(reference_config)

    merged_config = {
        "accelerator": accelerator,
        "model": reference_config["model"],
        "categories": {},
    }

    for config in configs:
        category_name = config.dataset.category

        if "seed" in config.project:
            del config.project["seed"]
        if "test" in config.project.save_outputs:
            del config.project.save_outputs["test"]

        config.project.logger = None

        category_config = {
            "dataset": {
                "task": config.dataset.task,
                "category": category_name,
            },
            "model": {
                "weight_file": config.model.weight_file,
            },
            "metrics": {
                "threshold": config.metrics.threshold,
            },
            "project": config.project,
        }

        merged_config["categories"][category_name] = category_config

    final_config = OmegaConf.create(merged_config)

    if save_dir is None:
        return final_config

    save_path = Path(save_dir) / filename
    final_config = OmegaConf.create(merged_config)
    OmegaConf.save(final_config, save_path)

    return final_config


def update_config(
    model: str,
    result_path: str,
    dataset_root: str,
    with_mask_label: bool,
    task_type: str,
    accelerator: str,
    image_size: Optional[int],
    data_format: str,
    category: str,
    batch_size: int,
    split: float,
    seed: int,
    num_workers: int,
    config_path: Optional[str] = None,
    good_file_list: str = "",
    bad_file_list: str = "",
    mask_file_list: Optional[str] = None,
    no_bad_mode: bool = False,
):
    all_models_cfg_path = (
        f"{os.path.dirname(os.path.abspath(__file__))}/../../plug_in/models/"
    )

    if not model.lower() in torch_model_list:
        click.secho("_" * 80, fg="blue")
        click.secho(f"{'.' * 37}ALERT{'.' * 37}", bold=True, fg="yellow")
        click.secho(
            "The Anomaly Detection Model should be "
            f"`patchcore` or `padim`. But got : {model.lower()}\n"
            "Please specify without misspelling. However, the "
            "training will be continued with\n"
            "default model: patchcore.",
            bold=True,
            fg="yellow",
        )
        click.secho("_" * 80, fg="blue")
        model = "patchcore"

    original_config_path = config_path

    # get model's config.
    if config_path is None:
        original_config_path = all_models_cfg_path + f"{model.lower()}/config.yaml"

    # read original config
    dict_obj = OmegaConf.load(original_config_path)

    # update some attributes
    dict_obj["project"]["path"] = f"{result_path}"
    dict_obj["dataset"]["path"] = os.path.join(os.getcwd(), dataset_root)
    dict_obj["dataset"]["seed"] = seed
    dict_obj["dataset"]["split_ratio"] = split
    dict_obj["dataset"]["extensions"] = None
    dict_obj["dataset"]["format"] = data_format
    dict_obj["dataset"]["num_workers"] = num_workers
    dict_obj["trainer"]["accelerator"] = accelerator
    dict_obj["dataset"]["name"] = category
    dict_obj["dataset"]["category"] = category

    if data_format == "folder":
        dict_obj["dataset"]["normal_dir"] = "good"
        if no_bad_mode:
            dict_obj["dataset"]["abnormal_dir"] = None
        else:
            dict_obj["dataset"]["abnormal_dir"] = "bad"
        dict_obj["dataset"]["normal_test_dir"] = None
        if not ensure_folder_data_format(dict_obj):
            sys.exit(1)

    # TODO: Check. https://stackoverflow.com/a/46451650
    if data_format == "filelist":
        if good_file_list and bad_file_list:
            dict_obj["dataset"]["l_normal"] = good_file_list.split(",")
            dict_obj["dataset"]["l_abnormal"] = bad_file_list.split(",")
        else:
            click.secho(
                f"The data-foramt is set as {data_format}.\n"
                "But CUI arguments for `--good-file-list` or `--bad-file-list` is empty.\n"
                "Both of these list are required.\n"
                f"For `--good-file-list`, received: {good_file_list}.\n"
                f"For `--bad-file-list`, received: {bad_file_list}.\n"
            )
            sys.exit(1)

    if mask_file_list:
        with_mask_label = True

    if batch_size is not None:
        dict_obj["dataset"]["train_batch_size"] = batch_size
        dict_obj["dataset"]["test_batch_size"] = batch_size

    if image_size is not None:
        dict_obj["dataset"]["image_size"] = image_size

    if data_format == "mvtec" and category not in MVTEC_CATEGORIES:
        click.secho(
            f"The category {category} is not an MVTec category. \n"
            f"MVTec categories are: {MVTEC_CATEGORIES}",
            bold=True,
            fg="red",
        )
        sys.exit(1)

    if (
        data_format == "mvtec"
        and (Path(dict_obj.dataset.path) / category).exists()
        and not ensure_mvtec_data_format(dict_obj)
    ):
        sys.exit(1)

    if with_mask_label and task_type == "classification":
        click.secho("_" * 60, fg="blue")
        click.secho(f"{'.' * 27}ALERT{'.' * 27}", bold=True, fg="yellow")
        click.secho(
            f"The task type is set {task_type}.\n"
            f"But Training with `--with-mask-label` is also requested {with_mask_label}.\n"
            "If you want to Mask, then set `task_type` as segmentation.\n"
            "Continuing with Classification approach",
            bold=True,
            fg="yellow",
        )
        click.secho("_" * 60, fg="blue")

    if task_type == "classification":
        dict_obj["dataset"]["task"] = task_type
        dict_obj["dataset"]["mask"] = None
        dict_obj["dataset"]["l_abnormal_mask"] = None

    if with_mask_label and task_type == "segmentation":
        dict_obj["dataset"]["task"] = task_type
        dict_obj["dataset"]["mask"] = "mask"

        if data_format == "filelist":
            if mask_file_list:
                dict_obj["dataset"]["l_abnormal_mask"] = mask_file_list.split(",")
            else:
                click.secho(f"{'.' * 37}WARNING{'.' * 37}", bold=True, fg="yellow")
                ans = click.confirm(
                    click.style(
                        "The argument --with-mask-label is given.\n"
                        "But `mask_file_list` is missing in the CUI.\n"
                        "Continue training without mask images?",
                        bold=True,
                        fg="yellow",
                    ),
                    default=True,
                    show_default=True,
                )
                if not ans:
                    sys.exit(0)

                dict_obj["dataset"]["l_abnormal_mask"] = None

        # It will ensure folder structure matching between data/bad and data/good
        if data_format == "folder":
            is_ensured = ensure_folder_format_mask(dict_obj)

            if not is_ensured:
                click.secho(f"{'.' * 37}WARNING{'.' * 37}", bold=True, fg="yellow")
                ans = click.confirm(
                    click.style(
                        "The argument --with-mask-label is given.\n"
                        "But `data/mask` is missing or the mask images does not have the same\n"
                        "corresponding name of the bad images.\n"
                        "Having mask images might give better training results.\n"
                        "Continue training without mask images?",
                        bold=True,
                        fg="yellow",
                    ),
                    default=True,
                    show_default=True,
                )
                if not ans:
                    sys.exit(0)

                dict_obj["dataset"]["mask"] = None

    if not with_mask_label and task_type == "segmentation":
        dict_obj["dataset"]["task"] = task_type
        dict_obj["dataset"]["mask"] = None
        dict_obj["dataset"]["l_abnormal_mask"] = None

    if data_format in ("folder", "filelist"):
        project_path = (
            dict_obj.project.path
            / Path(dict_obj.dataset.category)
            / dict_obj.model.name
        )
    elif data_format == "mvtec":
        project_path = (
            dict_obj.project.path
            / Path(data_format)
            / Path(dict_obj.model.name)
            / dict_obj.dataset.category
        )
    dict_obj.project.save_root = str(project_path.absolute())

    if no_bad_mode:
        normalization_method = NO_BAD_MODE_NORMALIZATION
        click.secho(
            f"Anomalous samples won't be used during validation as no-bad-mode is selected.\n"
            f"Setting normalization method: {normalization_method}.",
            bold=True,
            fg="yellow",
        )
        dict_obj["model"]["normalization_method"] = normalization_method

    # write the update yaml file to the destination folder.
    if data_format.lower() in ("folder", "filelist"):
        config_path = os.path.join(result_path, category, model.lower())
    elif data_format.lower() == "mvtec":
        config_path = os.path.join(
            result_path, data_format.lower(), model.lower(), category
        )
    os.makedirs(config_path, exist_ok=True)

    config_path = f"{config_path}/config.yaml"

    OmegaConf.save(dict_obj, config_path)

    return model, config_path
