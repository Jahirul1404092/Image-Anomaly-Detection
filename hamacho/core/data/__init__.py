"""Hamcho Datasets."""


import warnings
import click
import sys
import json
from typing import Union

from omegaconf import DictConfig
from omegaconf import ListConfig
from pytorch_lightning import LightningDataModule
from torchvision.datasets.folder import IMG_EXTENSIONS


from .folder import FolderDataModule
from .filelist import FilelistDataModule
from .mvtec import MVTec, MVTEC_CATEGORIES
from .inference import InferenceDataModule
from .utils.path import get_directory_tree
from hamacho.core.utils.folder import count_files, list_folders, compare_dir
from hamacho.core.utils.filelist import (
    get_valid_filelist,
    get_valid_paired_filelist,
)


# ref. https://stackoverflow.com/a/2187390
def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + "\n"


warnings.formatwarning = custom_formatwarning

folder_format_standard = {
    "regular": """
data
├── <product category>
│   │
│   ├── good
│   │   ├── [SOME GOOD IMAGE]
│   │   bad
│   │   ├── [SOME BAD IMAGE / SUB-FOLDER]
│   │   mask [optional]
│   │   ├── [SOME MASK LABEL IMAGE W.R.T BAD IMAGE / SUB-FOLDER]
""",
    "no-bad-mode": """
data
├── <product category>
│   │
│   ├── good
│   │   ├── [SOME GOOD IMAGE]
""",
}

mvtec_format_standard = """
data
├── <category>
│   │
│   ├── train
│   │   ├── [SOME GOOD IMAGE / SUB-FOLDER]
│   │   test
│   │   ├── [SOME BAD AND GODD IMAGE / SUB-FOLDER]
│   │   ground_truth [optional]
│   │   ├── [SOME MASK LABEL IMAGE W.R.T BAD IMAGE / SUB-FOLDER]
"""


def ensure_folder_data_format(config):

    data_dir = config["dataset"]["path"]
    category = config["dataset"]["category"]
    dataset_path = f"{data_dir}/{category}"
    data_folders = list_folders(dataset_path)

    # Check whether abnormal_dir is set to None[no-bad-mode selected]
    if config["dataset"]["abnormal_dir"] is None and "good" not in data_folders:
        click.secho(
            "The directory structure is not correct. "
            "The 'good' subdirectory must be present in the "
            "<product category> directory in folder data-format.\n",
            fg="red",
            bold=True,
        )
        click.secho(
            f"Expected Data Tree Structure: \n {folder_format_standard['no-bad-mode']}",
            fg="green",
        )

        # get the tree view of ../data/ directory
        data_tree_view = get_directory_tree(
            data_dir, category=category, limit_to_directories=True
        )

        click.secho(f"But Recevied : \n {data_tree_view}", fg="red")
        click.secho("Exiting. Please ensure that the directory structure is correct.")

        return False

    if config["dataset"]["abnormal_dir"] == "bad" and (
        "good" not in data_folders or "bad" not in data_folders
    ):
        click.secho(
            "The directory structure is not correct. "
            "The 'good' and 'bad' subdirectories must be present in the "
            "<product category> directory in folder data-format.\n",
            fg="red",
            bold=True,
        )
        click.secho(
            f"Expected Data Tree Structure: \n {folder_format_standard['regular']}",
            fg="green",
        )
        # get the tree view of ../data/ directory
        data_tree_view = get_directory_tree(
            data_dir, category=category, limit_to_directories=True
        )

        click.secho(f"But Recevied : \n {data_tree_view}", fg="red")
        click.secho("Exiting. Please ensure that the directory structure is correct.")

        return False

    return True


def ensure_mvtec_data_format(config):

    data_dir = config["dataset"]["path"]
    category = config["dataset"]["category"]
    dataset_path = f"{data_dir}/{category}"
    data_folders = list_folders(dataset_path)

    if "train" not in data_folders or "test" not in data_folders:
        click.secho(
            "The directory structure is not correct. "
            "The 'train', 'test' and 'ground_truth' subdirectories must be present in the "
            "<category> directory in mvtec data-format.\n",
            fg="red",
            bold=True,
        )
        click.secho(
            f"Expected Data Tree Structure: \n {mvtec_format_standard}", fg="green"
        )
        # get the tree view of ../data/ directory
        data_tree_view = get_directory_tree(
            data_dir, category=category, limit_to_directories=True
        )

        click.secho(f"But Recevied : \n {data_tree_view}", fg="red")
        click.secho("Exiting. Please ensure that the directory structure is correct.")

        return False

    return True


def ensure_folder_format_mask(config):

    data_dir = config["dataset"]["path"]
    category = config["dataset"]["category"]
    dataset_path = f"{data_dir}/{category}"
    data_folders = list_folders(dataset_path)

    if "mask" not in data_folders:
        click.secho("The 'mask' directory is missing.\n", fg="yellow")
        click.secho(
            f"Expected Data Tree Structure: \n {folder_format_standard['regular']}",
            fg="green",
        )
        # get the tree view of ../data/ directory
        data_tree_view = get_directory_tree(
            data_dir, category=category, limit_to_directories=True
        )

        click.secho(f"But Recevied : \n {data_tree_view}", fg="red")

        return False

    # compare `data/bad` to `data/mask` and vice versa.
    missing_bad = compare_dir(
        str(dataset_path) + "/mask",
        str(dataset_path) + "/bad",
        extensions=IMG_EXTENSIONS,
    )
    missing_mask = compare_dir(
        str(dataset_path) + "/bad",
        str(dataset_path) + "/mask",
        extensions=IMG_EXTENSIONS,
    )

    if len(missing_mask) == count_files(
        dataset_path + "/bad", extensions=IMG_EXTENSIONS
    ):
        click.secho(
            "WARNING: No mask image found in `data/mask/` folder.\n"
            "Please ensure that the 'bad' and 'mask' image pairs have the same name.",
            bold=True,
            fg="yellow",
        )

        return False

    if missing_bad:
        click.secho("_" * 60, fg="blue")
        click.secho(f"{'.' * 28}INFO{'.' * 28}", bold=True)
        click.secho(
            "There are some extra mask image(s) in `data/mask/` that does\n"
            "not have corresponding bad image(s) in `data/bad/`.\n"
            "Please check all the bad and mask image pairs.",
            bold=True,
        )
        click.secho("_" * 60, fg="blue")
        click.secho(f"The following mask image(s) are extra", bold=True)

        for missing_item, (found_dir, _) in missing_bad.items():
            print(
                f"Filename: {missing_item}\n" f"File Path: {found_dir}/{missing_item}"
            )

        click.secho("_" * 60, fg="blue")

    return True


def get_datamodule(config: Union[DictConfig, ListConfig]) -> LightningDataModule:
    """Get Anomaly Datamodule.

    Args:
        config (Union[DictConfig, ListConfig]): Configuration of the anomaly model.

    Returns:
        PyTorch Lightning DataModule
    """
    datamodule: LightningDataModule

    if config.dataset.format.lower() == "folder":
        datamodule = FolderDataModule(
            root=config.dataset.path,
            normal_dir=config.dataset.normal_dir,
            abnormal_dir=config.dataset.abnormal_dir,
            category=config.dataset.category,
            task=config.dataset.task,
            normal_test_dir=config.dataset.normal_test_dir,
            mask_dir=config.dataset.mask,
            extensions=config.dataset.extensions,
            split_ratio=config.dataset.split_ratio,
            seed=config.dataset.seed,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_val=config.dataset.transform_config.val,
            create_validation_set=config.dataset.create_validation_set,
        )
    elif config.dataset.format.lower() == "filelist":
        l_normal, l_abnormal, l_abnormal_mask, invalid_files = get_valid_filelist(
            config.dataset.l_normal,
            config.dataset.l_abnormal,
            config.dataset.l_abnormal_mask,
            extensions=config.dataset.extensions
        )
   
        if l_abnormal_mask:
            l_abnormal, l_abnormal_mask, invalid_non_paired_files = get_valid_paired_filelist(
                l_abnormal, l_abnormal_mask
            )
            non_abnormal, non_abnormal_mask = invalid_non_paired_files

            if non_abnormal:
                # Report, this items doesn't have pair in abnormal file list.
                click.secho("_" * 60, fg="blue")
                click.secho(f"{'.' * 27}ALERT{'.' * 27}", bold=True, fg="yellow")
                click.secho(
                    "The following abnormal files don't have corresponding\n"
                    "abnormal_mask files. Skipping these files.\n"
                    f"{non_abnormal}",
                    bold=True, 
                    fg="yellow",
                )
            
            if non_abnormal_mask:
                # Report, this items doesn't have pair in normal file list.
                click.secho(
                    "The following abnormal_mask files don't have corresponding\n"
                    "abnormal files. Skipping these files.\n",
                    f"{non_abnormal_mask}",
                    bold=True, 
                    fg="yellow",
                )

        if invalid_files:
            # Report, some invalid fines are found in the good and bad list.
            click.secho("_" * 60, fg="blue")
            click.secho(f"{'.' * 27}ALERT{'.' * 27}", bold=True, fg="yellow")
            click.secho(
                "The following listed files are not valid files.\n"
                "Make sure these files and their locations are valid. \n"
                "Skipping these files to continue the program.",
                bold=True,
                fg="yellow",
            )

            # (Option 1): Better view but some placing issue
            # TODO: decide!
            # click.secho(
            #     json.dumps(invalid_files, indent=4, ensure_ascii=False).encode('utf-8'),
            #     bold=True,
            #     fg="yellow"
            #     )
            # (Option 2)
            click.secho(
                "\n".join("{0} : {1}".format(k, v) for k, v in invalid_files.items()),
                bold=True,
                fg="yellow",
            )
            click.secho("_" * 60, fg="blue")

        if not l_normal and l_abnormal:
            click.secho(f"{'.' * 37}WARNING{'.' * 37}", bold=True, fg="yellow")
            click.secho(
                "After validaiton check of --good_file_list or --bad_file_list \n"
                "becomes empty. That means no valid files are found with these \n"
                "arguments. Please provide some valid image files.\n"
                f"For --good_file_list, became: {l_normal}\n"
                f"For --bad_file_list, became: {l_abnormal}",
                bold=True, 
                fg="yellow",
            )
            sys.exit(1)

        datamodule = FilelistDataModule(
            root=config.dataset.path,
            l_normal=l_normal,
            l_abnormal=l_abnormal,
            l_abnormal_mask=l_abnormal_mask,
            category=config.dataset.category,
            task=config.dataset.task,
            normal_test_dir=config.dataset.normal_test_dir,
            split_ratio=config.dataset.split_ratio,
            seed=config.dataset.seed,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_val=config.dataset.transform_config.val,
            create_validation_set=config.dataset.create_validation_set,
        )
    elif config.dataset.format.lower() == "mvtec":
        datamodule = MVTec(
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
            seed=config.project.seed,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_val=config.dataset.transform_config.val,
            create_validation_set=config.dataset.create_validation_set,
        )
    elif config.dataset.format.lower() == "inference":
        datamodule = InferenceDataModule(
            root=config.dataset.path,
            image_size=tuple(config.dataset.image_size),
            batch_size=config.dataset.infer_batch_size,
            transform_config=config.dataset.transform_config.val,
        )
    else:
        raise ValueError(
            "Unknown dataset! \n"
            "If you use a custom dataset make sure you initialize it in"
            "`get_datamodule` in `hamacho.core.data.__init__.py"
        )

    return datamodule


__all__ = [
    "get_datamodule",
    "FolderDataModule",
    "FilelistDataModule",
    "InferenceDataset",
    "get_directory_tree",
    "MVTEC_CATEGORIES",
]
