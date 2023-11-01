"""Custom Folder Dataset.

This script creates a custom dataset from a folder.
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
import warnings
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import albumentations as A
import pandas as pd
import cv2
import numpy as np
from pandas.core.frame import DataFrame
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from hamacho.core.data.inference import InferenceDataset
from hamacho.core.data.utils import read_image
from hamacho.core.data.utils.split import create_validation_set_from_test_set
from hamacho.core.data.utils.split import split_normal_images_in_train_set
from hamacho.core.pre_processing import PreProcessor

logger = logging.getLogger(__name__)


def _check_and_convert_path(path: Union[str, Path]) -> Path:
    """Check an input path, and convert to Pathlib object.

    Args:
        path (Union[str, Path]): Input path.

    Returns:
        Path: Output path converted to pathlib object.
    """
    if not isinstance(path, Path):
        path = Path(path)
    return path


def make_dataset(
    l_normal: list,
    l_abnormal: list,
    l_abnormal_mask: list,
    category: str,
    normal_test_dir: Optional[Union[str, Path]] = None,
    split: Optional[str] = None,
    split_ratio: float = 0.2,
    seed: int = 0,
    create_validation_set: bool = True,
):
    """Make Folder Dataset.

    Args:
        l_normal (Union[str, Path]): List containing normal images.
        l_abnormal (Union[str, Path]): List containing abnormal images.
        normal_test_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            normal images for the test dataset. Normal test images will be a split of `normal_dir`
            if `None`. Defaults to None.
        mask_dir (Optional[Union[str, Path]], optional): Path to the directory containing
            the mask annotations. Defaults to None.
        split (Optional[str], optional): Dataset split (ie., either train or test). Defaults to None.
        split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.2.
        seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
        create_validation_set (bool, optional):Boolean to create a validation set from the test set.
            Those wanting to create a validation set could set this flag to ``True``.
        extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
            directory.

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """
    filenames = []
    labels = []
    files = {"normal": l_normal, "abnormal": l_abnormal}

    # TODO: We may need to support it later.
    if normal_test_dir:
        files = {**files, **{"normal_test": normal_test_dir}}

    for f_type, paths in files.items():
        filename, label = paths, [f_type] * len(paths)
        filenames += filename
        labels += label


    samples = DataFrame({"image_path": filenames, "label": labels})

    # Create label index for normal (0) and abnormal (1) images.
    samples.loc[
        (samples.label == "normal") | (samples.label == "normal_test"), "label_index"
    ] = 0
    samples.loc[(samples.label == "abnormal"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    # If a path to mask is provided, add it to the sample dataframe.
    if l_abnormal_mask:
        samples["mask_path"] = ""
        normal_df = samples.query("label_index==0")
        abnormal_df = samples.query("label_index==1")
        abnormal_df["mask_path"] = l_abnormal_mask
        samples = pd.concat([normal_df, abnormal_df], axis=0, ignore_index=True)
        samples = samples.astype({"mask_path": "str"})

    # Ensure the pathlib objects are converted to str.
    # This is because torch dataloader doesn't like pathlib.
    samples = samples.astype({"image_path": "str"})

    # Create train/test split.
    # By default, all the normal samples are assigned as train.
    #   and all the abnormal samples are test.
    samples.loc[(samples.label == "normal"), "split"] = "train"
    samples.loc[
        (samples.label == "abnormal") | (samples.label == "normal_test"), "split"
    ] = "test"

    if not normal_test_dir:
        samples = split_normal_images_in_train_set(
            samples=samples, split_ratio=split_ratio, seed=seed, normal_label="normal"
        )

    # If `create_validation_set` is set to True, the test set is split into half.
    if create_validation_set:
        samples = create_validation_set_from_test_set(
            samples, seed=seed, normal_label="normal"
        )

    # Get the data frame for the split.
    if split is not None and split in ["train", "val", "test"]:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class FilelistDataset(Dataset):
    """Filelist Dataset."""

    def __init__(
        self,
        l_normal: list,
        l_abnormal: list,
        l_abnormal_mask: list,
        category: str,
        split: str,
        pre_process: PreProcessor,
        normal_test_dir: Optional[Union[Path, str]] = None,
        split_ratio: float = 0.2,
        task: Optional[str] = None,
        seed: int = 0,
        create_validation_set: bool = False,
    ) -> None:
        """Create FilelistDataset form FoldertDataset.

        Args:
            l_normal (Union[str, Path]): List containing normal images.
            l_abnormal (Union[str, Path]): List containing abnormal images.
            split (Optional[str], optional): Dataset split (ie., either train or test). Defaults to None.
            pre_process (Optional[PreProcessor], optional): Image Pro-processor to apply transform.
                Defaults to None.
            normal_test_dir (Optional[Union[str, Path]], optional): Path to the directory containing
                normal images for the test dataset. Defaults to None.
            split_ratio (float, optional): Ratio to split normal training images and add to the
                test set in case test set doesn't contain any normal images.
                Defaults to 0.2.
            mask_dir (Optional[Union[str, Path]], optional): Path to the directory containing
                the mask annotations. Defaults to None.
            extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
                directory.
            task (Optional[str], optional): Task type. (classification or segmentation) Defaults to None.
            seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
            create_validation_set (bool, optional):Boolean to create a validation set from the test set.
                Those wanting to create a validation set could set this flag to ``True``.

        Raises:
            ValueError: When task is set to classification and `mask_dir` is provided. When `mask_dir` is
                provided, `task` should be set to `segmentation`.

        """
        self.split = split

        if task == "segmentation" and not l_abnormal_mask:
            self.task = "classification"

        if task == "classification" and l_abnormal_mask:
            self.task = "segmentation"

        if task is None or not l_abnormal_mask:
            self.task = "classification"
        else:
            self.task = task
        
        self.category = category
        self.pre_process = pre_process
        self.l_normal = l_normal
        self.l_abnormal = l_abnormal
    
        self.samples = make_dataset(
            l_normal=l_normal,
            l_abnormal=l_abnormal,
            l_abnormal_mask=l_abnormal_mask,
            category=category,
            normal_test_dir=normal_test_dir,
            split=split,
            split_ratio=split_ratio,
            seed=seed,
            create_validation_set=create_validation_set,
        )

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]: Dict of image tensor during training.
                Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """
        item: Dict[str, Union[str, Tensor]] = {}

        image_path = self.samples.image_path[index]
        image = read_image(image_path)

        pre_processed = self.pre_process(image=image)
        item = {"image": pre_processed["image"]}

        if self.split in ["val", "test"]:
            label_index = self.samples.label_index[index]
            data_type = "bad" if label_index == 1 else "good"

            item["image_path"] = image_path
            item["label"] = label_index
            item["data_type"] = data_type
            item["category"] = self.category

            if self.task == "segmentation":
                mask_path = self.samples.mask_path[index]

                # Only Anomalous (1) images has masks in MVTec AD dataset.
                # Therefore, create empty mask for Normal (0) images.
                if label_index == 0:
                    mask = np.zeros(shape=image.shape[:2])
                else:
                    mask = (
                        cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), flags=0)
                        / 255.0
                    )
                pre_processed = self.pre_process(image=image, mask=mask)

                item["mask_path"] = mask_path
                item["image"] = pre_processed["image"]
                item["mask"] = pre_processed["mask"]

        return item


class FilelistDataModule(LightningDataModule):
    """Folder Lightning Data Module."""

    def __init__(
        self,
        root: Path,
        l_normal: list,
        l_abnormal: list,
        l_abnormal_mask: Optional[list] = None,
        task: str = "classification",
        category: str = None,
        normal_test_dir: Optional[Union[Path, str]] = None,
        split_ratio: float = 0.2,
        seed: int = 0,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 8,
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_val: Optional[Union[str, A.Compose]] = None,
        create_validation_set: bool = False,
    ) -> None:
        """FileList Dataset PL Datamodule.

        Args:
            root (Union[str, Path]): Path to the root folder containing normal and abnormal dirs.
            l_normal (str, optional): List containing normal images.
                Defaults to "normal".
            l_abnormal (str, optional): List containing abnormal images.
                Defaults to "abnormal".
            task (str, optional): Task type. Could be either classification or segmentation.
                Defaults to "classification".
            normal_test_dir (Optional[Union[str, Path]], optional): Path to the directory containing
                normal images for the test dataset. Defaults to None.
            mask_dir (Optional[Union[str, Path]], optional): Path to the directory containing
                the mask annotations. Defaults to None.
            extensions (Optional[Tuple[str, ...]], optional): Type of the image extensions to read from the
                directory. Defaults to None.
            split_ratio (float, optional): Ratio to split normal training images and add to the
                test set in case test set doesn't contain any normal images.
                Defaults to 0.2.
            seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
            image_size (Optional[Union[int, Tuple[int, int]]], optional): Size of the input image.
                Defaults to None.
            train_batch_size (int, optional): Training batch size. Defaults to 32.
            test_batch_size (int, optional): Test batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 8.
            transform_config_train (Optional[Union[str, A.Compose]], optional): Config for pre-processing
                during training.
                Defaults to None.
            transform_config_val (Optional[Union[str, A.Compose]], optional): Config for pre-processing
                during validation.
                Defaults to None.
            create_validation_set (bool, optional):Boolean to create a validation set from the test set.
                Those wanting to create a validation set could set this flag to ``True``.

        Examples:
            Assume that we use Folder Dataset for the MVTec/bottle/broken_large category. We would do:
            >>> from anomalib.data import FilelistDataModule
            >>> datamodule = FilelistDataModule(
            ...     l_normal=['1.png', '2.png'], 
            ...     l_abnormal=['2.jpg', '3.jpg'],
            ...     image_size=256
            ... )
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data["image"].shape
            torch.Size([1, 3, 256, 256])

            >>> i, test_data = next(enumerate(datamodule.test_dataloader()))
            >>> test_data.keys()
            dict_keys(['image'])

            We could also create a Folder DataModule for datasets containing mask annotations.
            The dataset expects that mask annotation filenames must be same as the original filename.
            To this end, we modified mask filenames in MVTec AD bottle category.
            Now we could try folder data module using the mvtec bottle broken large category
            >>> datamodule = FilelistDataModule(
            ...     l_normal=['1.png', '2.png'], 
            ...     l_abnormal=['2.jpg', '3.jpg'],
            ...     mask_dir="./datasets/bottle/ground_truth/broken_large",
            ...     image_size=256
            ... )

            >>> i , train_data = next(enumerate(datamodule.train_dataloader()))
            >>> train_data.keys()
            dict_keys(['image'])
            >>> train_data["image"].shape
            torch.Size([16, 3, 256, 256])

            >>> i, test_data = next(enumerate(datamodule.test_dataloader()))
            dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
            >>> print(test_data["image"].shape, test_data["mask"].shape)
            torch.Size([24, 3, 256, 256]) torch.Size([24, 256, 256])

            By default, Folder Data Module does not create a validation set. If a validation set
            is needed it could be set as follows:

            >>> datamodule = FolderDataModule(
            ...     root="./datasets/bottle/test",
            ...     normal="good",
            ...     abnormal="broken_large",
            ...     mask_dir="./datasets/bottle/ground_truth/broken_large",
            ...     image_size=256,
            ...     create_validation_set=True,
            ... )

            >>> i, val_data = next(enumerate(datamodule.val_dataloader()))
            >>> val_data.keys()
            dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
            >>> print(val_data["image"].shape, val_data["mask"].shape)
            torch.Size([12, 3, 256, 256]) torch.Size([12, 256, 256])

            >>> i, test_data = next(enumerate(datamodule.test_dataloader()))
            >>> print(test_data["image"].shape, test_data["mask"].shape)
            torch.Size([12, 3, 256, 256]) torch.Size([12, 256, 256])

        """
        super().__init__()
        self.l_normal = l_normal
        self.l_abnormal = l_abnormal
        self.l_abnormal_mask = l_abnormal_mask
        self.root = _check_and_convert_path(root)
        self.category = category
        self.normal_test = normal_test_dir

        if normal_test_dir:
            self.normal_test = self.root / category / normal_test_dir
            
        self.split_ratio = split_ratio

        self.task = task
        self.transform_config_train = transform_config_train
        self.transform_config_val = transform_config_val
        self.image_size = image_size

        if (
            self.transform_config_train is not None
            and self.transform_config_val is None
        ):
            self.transform_config_val = self.transform_config_train

        self.pre_process_train = PreProcessor(
            config=self.transform_config_train, image_size=self.image_size
        )
        self.pre_process_val = PreProcessor(
            config=self.transform_config_val, image_size=self.image_size
        )

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.create_validation_set = create_validation_set
        self.seed = seed

        self.train_data: Dataset
        self.test_data: Dataset
        if create_validation_set:
            self.val_data: Dataset
        self.inference_data: Dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train, validation and test data.

        Args:
          stage: Optional[str]:  Train/Val/Test stages. (Default value = None)

        """
        logger.info("Setting up train, validation, test and prediction datasets.")
        if stage in (None, "fit"):
            self.train_data = FilelistDataset(
                l_normal=self.l_normal,
                l_abnormal=self.l_abnormal,
                l_abnormal_mask=self.l_abnormal_mask,
                normal_test_dir=self.normal_test,
                category=self.category,
                split="train",
                split_ratio=self.split_ratio,
                pre_process=self.pre_process_train,
                task=self.task,
                seed=self.seed,
                create_validation_set=self.create_validation_set,
            )

        if self.create_validation_set:
            self.val_data = FilelistDataset(
                l_normal=self.l_normal,
                l_abnormal=self.l_abnormal,
                l_abnormal_mask=self.l_abnormal_mask,
                normal_test_dir=self.normal_test,
                category=self.category,
                split="val",
                split_ratio=self.split_ratio,
                pre_process=self.pre_process_val,
                task=self.task,
                seed=self.seed,
                create_validation_set=self.create_validation_set,
            )

        self.test_data = FilelistDataset(
            l_normal=self.l_normal,
            l_abnormal=self.l_abnormal,
            l_abnormal_mask=self.l_abnormal_mask,
            category=self.category,
            split="test",
            normal_test_dir=self.normal_test,
            split_ratio=self.split_ratio,
            pre_process=self.pre_process_val,
            task=self.task,
            seed=self.seed,
            create_validation_set=self.create_validation_set,
        )

        if stage == "predict":
            self.inference_data = InferenceDataset(
                path=self.root,
                image_size=self.image_size,
                transform_config=self.transform_config_val,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        dataset = self.val_data if self.create_validation_set else self.test_data
        return DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(
            self.test_data,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Get predict dataloader."""
        return DataLoader(
            self.inference_data,
            shuffle=False,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )
