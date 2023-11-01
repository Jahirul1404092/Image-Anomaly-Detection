import os
import sys

from pathlib import Path
from typing import Union
import click
from omegaconf import OmegaConf

from hamacho.plug_in.models import torch_model_list
from hamacho.core.data import (
    ensure_folder_format_mask,
    ensure_folder_data_format,
    ensure_mvtec_data_format,
    MVTEC_CATEGORIES
)

# # This script is not using anywhere! 
# def update_cfg(
#     model: str,
#     result_path: str,
#     dataset_root: str,
#     with_mask_label: bool,
#     task_type: str,
#     accelerator: str,
#     image_size: Union[int, None],
#     data_format: str,
#     category: str,
#     batch_size: int,
#     split: float,
#     num_workers: int,
# ):
#     all_models_cfg_path = (
#         f"{os.path.dirname(os.path.abspath(__file__))}/../../plug_in/models/"
#     )

#     if not model.lower() in torch_model_list:
#         click.secho("_" * 80, fg="blue")
#         click.secho(f"{'.' * 37}ALERT{'.' * 37}", bold=True, fg="yellow")
#         click.secho(
#             "The Anomaly Detection Model should be "
#             f"`patchcore` or `padim`. But got : {model.lower()}\n"
#             "Please specify without misspelling. However, the "
#             "training will be continued with\n"
#             "default model: patchcore.",
#             bold=True,
#             fg="yellow",
#         )
#         click.secho("_" * 80, fg="blue")
#         model = "patchcore"

#     # get model's config.
#     original_config_path = all_models_cfg_path + f"{model.lower()}/config.yaml"

#     # read original config
#     dict_obj = OmegaConf.load(original_config_path)
#     # update some attributes
#     dict_obj["project"]["path"] = f"{result_path}/"
#     dict_obj["dataset"]["path"] = os.path.join(os.getcwd(), dataset_root)
#     dict_obj["dataset"]["seed"] = 420
#     dict_obj["dataset"]["split_ratio"] = split
#     dict_obj["dataset"]["extensions"] = None
#     dict_obj["dataset"]["name"] = data_format
#     dict_obj["dataset"]["format"] = data_format
#     dict_obj["dataset"]["category"] = category
#     dict_obj["dataset"]["normal_dir"] = "good"
#     dict_obj["dataset"]["abnormal_dir"] = "bad"
#     dict_obj["dataset"]["normal_test_dir"] = None
#     dict_obj["dataset"]["num_workers"] = num_workers
#     dict_obj["trainer"]["accelerator"] = accelerator

#     if batch_size is not None:
#         dict_obj["dataset"]["train_batch_size"] = batch_size
#         dict_obj["dataset"]["test_batch_size"] = batch_size

#     if image_size is not None:
#         dict_obj["dataset"]["image_size"] = image_size

#     if data_format == "mvtec" and category not in MVTEC_CATEGORIES:
#         click.secho(
#             f"The category {category} is not an MVTec category. \n"
#             f"MVTec categories are: {MVTEC_CATEGORIES}",
#             bold=True,
#             fg="red",
#         )
#         sys.exit(1)

#     if data_format == "mvtec" and not ensure_mvtec_data_format(dict_obj):
#         sys.exit(1)

#     if data_format == "folder" and not ensure_folder_data_format(dict_obj):
#         sys.exit(1)

#     if with_mask_label and task_type == "classification":
#         click.secho("_" * 60, fg="blue")
#         click.secho(f"{'.' * 27}ALERT{'.' * 27}", bold=True, fg="yellow")
#         click.secho(
#             f"The task type is set {task_type}.\n"
#             f"But Training with `--with-mask-label` is also requested {with_mask_label}.\n"
#             "If you want to Mask, then set `task_type` as segmentation.\n"
#             "Continuing with Classification approach",
#             bold=True,
#             fg="yellow",
#         )
#         click.secho("_" * 60, fg="blue")

#     if task_type == "classification":
#         dict_obj["dataset"]["task"] = task_type
#         dict_obj["dataset"]["mask"] = None

#     if with_mask_label and task_type == "segmentation":
#         dict_obj["dataset"]["task"] = task_type
#         dict_obj["dataset"]["mask"] = "mask"

#         # It will ensure folder structure matching between data/bad and data/good
#         if data_format == "folder":
#             is_ensured = ensure_folder_format_mask(dict_obj)

#             if not is_ensured:
#                 click.secho(f"{'.' * 37}WARNING{'.' * 37}", bold=True, fg="yellow")
#                 ans = click.confirm(
#                     click.style(
#                         "The argument --with-mask-label is given.\n"
#                         "But `data/mask` is missing or the mask images does not have the same\n"
#                         "corresponding name of the bad images.\n"
#                         "Having mask images might give better training results.\n"
#                         "Continue training without mask images?",
#                         bold=True,
#                         fg="yellow",
#                     ),
#                     default=True,
#                     show_default=True,
#                 )
#                 if not ans:
#                     sys.exit(0)

#                 dict_obj["dataset"]["mask"] = None

#     if not with_mask_label and task_type == "segmentation":
#         dict_obj["dataset"]["task"] = task_type
#         dict_obj["dataset"]["mask"] = None

#     if data_format == "folder":
#         project_path = (
#             dict_obj.project.path
#             / Path(dict_obj.dataset.category)
#             / dict_obj.model.name
#         )
#     elif data_format in ("btech", "mvtec"):
#         project_path = (
#             dict_obj.project.path
#             / Path(data_format)
#             / Path(dict_obj.model.name)
#             / dict_obj.dataset.category
#         )
#     dict_obj.project.save_root = str(project_path.absolute())

#     # write the update yaml file to the destination folder.
#     if data_format.lower() in ("folder", "filelist"):
#         config_path = os.path.join(result_path, category, model.lower())
#     elif data_format.lower() in ("mvtec", "btech"):
#         config_path = os.path.join(result_path, data_format.lower(),
#                                    model.lower(), category)
#     os.makedirs(config_path, exist_ok=True)
#     config_path = f"{config_path}/config.yaml"
#     OmegaConf.save(dict_obj, config_path)

#     return model, config_path
