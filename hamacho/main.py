import platform
import click
from pathlib import Path
from hamacho.core.utils.warning import hide_warns

hide_warns()  # will reduce all unnecessary p.lightning warnings

from pytorch_lightning import Trainer
from hamacho.core.config import (
    get_configurable_parameters,
    update_config,
)
from hamacho.core.data import get_datamodule
from hamacho.core.utils.callbacks import LoadModelCallback
from hamacho.core.utils.callbacks import get_callbacks
from hamacho.core.utils.loggers import get_experiment_logger
from hamacho.core.utils.click_types import (
    ImageSizeRange,
    BatchSizeType,
    TrainTestSplitType,
    NumWorkersType,
)
from hamacho.core.utils.profilers import HamachoProfiler
from hamacho.plug_in.models import get_model

# Hiding --num-workers for windows OS.
# It causes some issue.
# TODO: Fix.
hide_nw = False
if platform.system().lower() != "linux":
    hide_nw = True


@click.group()
def cli():
    pass

# ----------------- Training + Validation -------------
# fmt: off
# train mode section
@cli.command()
@click.option(
    "--model", 
    type=str, default="patchcore",
    help="Name of the algorithm to train/test"
)
@click.option(
    "--dataset-root", 
    type=click.Path(exists=True, dir_okay=True), default=None,
    help="Path consisting the datasets"
)
@click.option(
    "--result-path", 
    type=str, default="./results",
    help="Path to save the results"
)
@click.option(
    "--task-type", 
    type=click.Choice(("classification", "segmentation")),
    default="segmentation", 
    help="Whether modeling as classification or segmentation approach."
)
@click.option(
    "--with-mask-label", 
    default=False, is_flag=True, 
    help="We can train model with or without ground truth mask."
)
@click.option(
    "--accelerator", 
    type=click.Choice(("cpu", "gpu", "auto")),
    default="auto", 
    help="You can select cpu or gpu or auto for the device."
)
@click.option(
    "--image-size",
    type=ImageSizeRange(), default=None,
    help="Image size of the training images in pixels"
)
@click.option(
    "--batch-size",
    type=BatchSizeType(),
    default=None,
    help="train test batch size"
)
@click.option(
    "--split",
    type=TrainTestSplitType(),
    default=0.2,
    help="train val test split percentage in float"
)
@click.option(
    "--data-format",
    type=click.Choice(("folder", "mvtec", "filelist")),
    default="folder",
    help="format of the dataset in `data/` folder"
)
@click.option(
    "--category",
    type=str,
    required=True,
    help="name of the product in `data/` folder"
)
@click.option(
    "--num-workers",
    type=NumWorkersType(), default=0, hidden=hide_nw,
    help="Number of workers for the dataloader to use"
)
@click.option(
    "--good-file-list",
    type=str,
    help="list of good files"
)
@click.option(
    "--bad-file-list",
    type=str,
    help="list of bad files"
)
@click.option(
    "--mask-file-list",
    type=str,
    default=None,
    help="list of mask files"
)
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "[Expert Only]\n"
        "Edited config path of model config.yaml file"
    )
)
@click.option(
    "--seed",
    type=int, default=420,
    help="Seed value to control dataset random split"
)
@click.option( 
    "--no-bad-mode",
    default=False, is_flag=True,
    help="Select this when only good data is available" 
)
# fmt: on
def train(
    model: str,
    result_path: str,
    dataset_root: str,
    task_type: str,
    with_mask_label: bool,
    accelerator: str,
    batch_size: int,
    data_format: str,
    category: str,
    split: float,
    seed: int,
    image_size: int = None,
    num_workers: int = 0,
    good_file_list: str = "",
    bad_file_list: str = "",
    mask_file_list: str = "",
    config_path: str = None,
    no_bad_mode: bool = False,
):
    if dataset_root is None:
        dataset_root = 'data'

    # update config
    model, config_path = update_config(
        model,
        result_path,
        dataset_root,
        with_mask_label,
        task_type,
        accelerator,
        image_size,
        data_format,
        category,
        batch_size,
        split,
        seed,
        num_workers,
        config_path,
        good_file_list,
        bad_file_list,
        mask_file_list,
        no_bad_mode,
    )

    # ---------------- Training ---------------
    # configuring -> prepare data module -> prepare model -> callbacks -> training
    config = get_configurable_parameters(
        model_name=model,
        config_path=config_path,
    )
    click.secho(f"CPU workers set to : {config.dataset.num_workers}")
    datamodule = get_datamodule(config)
    model = get_model(config)
    experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)
    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
    return trainer


# fmt: off
# inference mode section
@cli.command()
@click.option(
    "--image-path", 
    type=click.Path(exists=True), required=True, 
    help="Path to image(s) to infer."
)
@click.option(
    "--config-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a config file.\n"
         "It is generally results/<product-category>/<model>/config.yaml"
)
@click.option(
    "--save-path",
    type=click.Path(exists=True, dir_okay=True),
    default=None,
    help="Path to save the output image(s)."
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help="Inference batch size"
)
@click.option(
    "--accelerator", 
    type=str, default="auto", 
    help="You can select CPU or GPU or auto for the device."
)
@click.option(
    '--profile',
    type=click.Choice([
        "simple",
        "advanced",
        "expert",
        "custom",
    ]),
    default=None,
    help=(
        "[Only for Developers]\n"
        "simple -> only includes profiling of hamacho functions.\n"
        "advanced -> include profiling of hamacho along with libraries.\n"
        "expert -> include profiling of hamacho along with libraries and built-in functions.\n"
        "custom -> only include the defined functions and modules set in\n"
        "hamacho/core/utils/profilers/custom_profiling/functions.txt and modules.txt"
    )
)
# fmt: on
def inference(image_path, config_path, save_path, batch_size, accelerator, profile):
    if profile is not None:
        profiler = HamachoProfiler(save_path=save_path, type=profile)
        profiler.enable()
    config_path = Path(config_path)
    config = get_configurable_parameters(config_path=config_path)
    config.trainer["accelerator"] = accelerator
    config.dataset["path"] = image_path
    config.dataset["format"] = "inference"
    config.dataset["infer_batch_size"] = batch_size

    if save_path is not None:
        Path(save_path).mkdir(exist_ok=True, parents=True)
        config.project.save_root = save_path
        config.project.inference_dir_name = ""
    save_path = Path(config.project.save_root) / Path(config.project.inference_dir_name)
    
    model = get_model(config)
    callbacks = get_callbacks(config)
    datamodule = get_datamodule(config)
    trainer = Trainer(callbacks=callbacks, **config.trainer)
    trainer.predict(datamodule=datamodule, model=model)
    click.secho(f"Results saved at: {save_path.absolute()}", bold=True)

    if profile is not None:
        profiler.disable()
        profiler.save_path = save_path
        profiler.save_stats(profiler.parse_stats())

    return trainer


if __name__ == "__main__":
    cli()
