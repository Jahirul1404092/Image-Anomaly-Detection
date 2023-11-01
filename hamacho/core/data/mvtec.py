"""MVTec AD Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch
        Lightning DataModule for the MVTec AD dataset.

    If the dataset is not on the file system, the script downloads and
        extracts the dataset and create PyTorch data objects.
"""


import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import VisionDataset

from hamacho.core.data.inference import InferenceDataset
from hamacho.core.data.utils import read_image
from hamacho.core.data.utils.download import (
    get_mvtec_url,
    download_file,
    hash_check_category,
    extract_tar,
)
from hamacho.core.data.utils.split import (
    create_validation_set_from_test_set,
    split_normal_images_in_train_set,
)
from hamacho.core.pre_processing import PreProcessor

logger = logging.getLogger(__name__)

MVTEC_CATEGORIES = {
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
}


def make_mvtec_dataset(
    path: Path,
    split: Optional[str] = None,
    split_ratio: float = 0.1,
    seed: Optional[int] = None,
    create_validation_set: bool = False,
) -> DataFrame:
    """Create MVTec AD samples by parsing the MVTec AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    This function creates a dataframe to store the parsed information based on the following format:
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
    |   | path          | split | label   | image_path    | mask_path                             | label_index |
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
    | 0 | datasets/name |  test |  defect |  filename.png | ground_truth/defect/filename_mask.png | 1           |
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|

    Args:
        path (Path): Path to dataset
        split (str, optional): Dataset split (ie., either train or test). Defaults to None.
        split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.1.
        seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
        create_validation_set (bool, optional): Boolean to create a validation set from the test set.
            MVTec AD dataset does not contain a validation set. Those wanting to create a validation set
            could set this flag to ``True``.

    Examples:
        The following example shows how to get training samples from MVTec AD bottle category:

        >>> root = Path('./MVTec')
        >>> category = 'bottle'
        >>> path = root / category
        >>> path
        PosixPath('MVTec/bottle')

        >>> samples = make_mvtec_dataset(path, split='train', split_ratio=0.1, seed=0)
        >>> samples.head()
           path         split label image_path                           mask_path                   label_index
        0  MVTec/bottle train good MVTec/bottle/train/good/105.png MVTec/bottle/ground_truth/good/105_mask.png 0
        1  MVTec/bottle train good MVTec/bottle/train/good/017.png MVTec/bottle/ground_truth/good/017_mask.png 0
        2  MVTec/bottle train good MVTec/bottle/train/good/137.png MVTec/bottle/ground_truth/good/137_mask.png 0
        3  MVTec/bottle train good MVTec/bottle/train/good/152.png MVTec/bottle/ground_truth/good/152_mask.png 0
        4  MVTec/bottle train good MVTec/bottle/train/good/109.png MVTec/bottle/ground_truth/good/109_mask.png 0

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """
    samples_list = [
        (str(path),) + filename.parts[-3:] for filename in path.glob("**/*.png")
    ]
    if len(samples_list) == 0:
        raise RuntimeError(f"Found 0 images in {path}")

    samples = pd.DataFrame(
        samples_list, columns=["path", "split", "label", "image_path"]
    )
    samples = samples[samples.split != "ground_truth"]

    # Create mask_path column
    samples["mask_path"] = (
        samples.path
        + "/ground_truth/"
        + samples.label
        + "/"
        + samples.image_path.str.rstrip("png").str.rstrip(".")
        + "_mask.png"
    )

    # Modify image_path column by converting to absolute path
    samples["image_path"] = (
        samples.path
        + "/"
        + samples.split
        + "/"
        + samples.label
        + "/"
        + samples.image_path
    )

    # Split the normal images in training set if test set doesn't
    # contain any normal images. This is needed because AUC score
    # cannot be computed based on 1-class
    if sum((samples.split == "test") & (samples.label == "good")) == 0:
        samples = split_normal_images_in_train_set(samples, split_ratio, seed)

    # Good images don't have mask
    samples.loc[(samples.split == "test") & (samples.label == "good"), "mask_path"] = ""

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = 0
    samples.loc[(samples.label != "good"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    if create_validation_set:
        samples = create_validation_set_from_test_set(samples, seed=seed)

    # Get the data frame for the split.
    if split is not None and split in ["train", "val", "test"]:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class MVTecDataset(VisionDataset):
    """MVTec AD PyTorch Dataset."""

    def __init__(
        self,
        root: Union[Path, str],
        category: str,
        pre_process: PreProcessor,
        split: str,
        task: str = "segmentation",
        seed: Optional[int] = None,
        create_validation_set: bool = False,
    ) -> None:
        """Mvtec AD Dataset class.

        Args:
            root: Path to the MVTec AD dataset
            category: Name of the MVTec AD category.
            pre_process: List of pre_processing object containing albumentation compose.
            split: 'train', 'val' or 'test'
            task: ``classification`` or ``segmentation``
            seed: seed used for the random subset splitting
            create_validation_set: Create a validation subset in addition to the train and test subsets

        Examples:
            >>> from hamacho.core.data.mvtec import MVTecDataset
            >>> from hamacho.core.data.transforms import PreProcessor
            >>> pre_process = PreProcessor(image_size=256)
            >>> dataset = MVTecDataset(
            ...     root='./datasets/MVTec',
            ...     category='leather',
            ...     pre_process=pre_process,
            ...     task="classification",
            ...     is_train=True,
            ... )
            >>> dataset[0].keys()
            dict_keys(['image'])

            >>> dataset.split = "test"
            >>> dataset[0].keys()
            dict_keys(['image', 'image_path', 'label'])

            >>> dataset.task = "segmentation"
            >>> dataset.split = "train"
            >>> dataset[0].keys()
            dict_keys(['image'])

            >>> dataset.split = "test"
            >>> dataset[0].keys()
            dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])

            >>> dataset[0]["image"].shape, dataset[0]["mask"].shape
            (torch.Size([3, 256, 256]), torch.Size([256, 256]))
        """
        super().__init__(root)

        if seed is None:
            warnings.warn(
                "seed is None."
                " When seed is not set, images from the normal directory are split between training and test dir."
                " This will lead to inconsistency between runs."
            )

        self.root = Path(root) if isinstance(root, str) else root
        self.category: str = category
        self.split = split
        self.task = task

        self.pre_process = pre_process

        self.samples = make_mvtec_dataset(
            path=self.root / category,
            split=self.split,
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
        item["category"] = self.category

        if self.split in ["val", "test"]:
            label_index = self.samples.label_index[index]
            label = self.samples.label[index]

            item["image_path"] = image_path
            item["label"] = label_index
            item["data_type"] = label

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


@DATAMODULE_REGISTRY
class MVTec(LightningDataModule):
    """MVTec AD Lightning Data Module."""

    def __init__(
        self,
        root: str,
        category: str,
        # TODO: Remove default values. IAAALD-211
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 8,
        task: str = "segmentation",
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_val: Optional[Union[str, A.Compose]] = None,
        seed: Optional[int] = None,
        create_validation_set: bool = False,
    ) -> None:
        """Mvtec AD Lightning Data Module.

        Args:
            root: Path to the MVTec AD dataset
            category: Name of the MVTec AD category.
            image_size: Variable to which image is resized.
            train_batch_size: Training batch size.
            test_batch_size: Testing batch size.
            num_workers: Number of workers.
            task: ``classification`` or ``segmentation``
            transform_config_train: Config for pre-processing during training.
            transform_config_val: Config for pre-processing during validation.
            seed: seed used for the random subset splitting
            create_validation_set: Create a validation subset in addition to the train and test subsets

        Examples:
            >>> from hamacho.core.data import MVTec
            >>> datamodule = MVTec(
            ...     root="./datasets/MVTec",
            ...     category="leather",
            ...     image_size=256,
            ...     train_batch_size=32,
            ...     test_batch_size=32,
            ...     num_workers=8,
            ...     transform_config_train=None,
            ...     transform_config_val=None,
            ... )
            >>> datamodule.setup()

            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image'])
            >>> data["image"].shape
            torch.Size([32, 3, 256, 256])

            >>> i, data = next(enumerate(datamodule.val_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
            >>> data["image"].shape, data["mask"].shape
            (torch.Size([32, 3, 256, 256]), torch.Size([32, 256, 256]))
        """
        super().__init__()

        self.root = root if isinstance(root, Path) else Path(root)
        self.category = category
        self.dataset_path = self.root / self.category
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
        self.task = task
        self.seed = seed

        self.train_data: Dataset
        self.test_data: Dataset
        if create_validation_set:
            self.val_data: Dataset
        self.inference_data: Dataset

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            print(f"Dataset Category not found in {self.root}!!")
            info = f"Downloading the Mvtec AD dataset category: {self.category}."
            # logger.info(info)
            print(info)
            url = get_mvtec_url(self.category)
            dataset_name = f"{self.category}.tar.xz"
            zip_file_path = self.root / dataset_name
            download_file(url, zip_file_path, desc=f"MVTec AD {self.category}")
            logger.info("Checking hash")
            hash_check_category(zip_file_path, self.category)

            logger.info("Extracting the dataset.")
            extract_tar(file_path=zip_file_path, extract_dir=self.root, delete_tar=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train, validation and test data.

        Args:
          stage: Optional[str]:  Train/Val/Test stages. (Default value = None)

        """
        logger.info("Setting up train, validation, test and prediction datasets.")
        if stage in (None, "fit"):
            self.train_data = MVTecDataset(
                root=self.root,
                category=self.category,
                pre_process=self.pre_process_train,
                split="train",
                task=self.task,
                seed=self.seed,
                create_validation_set=self.create_validation_set,
            )

        if self.create_validation_set:
            self.val_data = MVTecDataset(
                root=self.root,
                category=self.category,
                pre_process=self.pre_process_val,
                split="val",
                task=self.task,
                seed=self.seed,
                create_validation_set=self.create_validation_set,
            )

        self.test_data = MVTecDataset(
            root=self.root,
            category=self.category,
            pre_process=self.pre_process_val,
            split="test",
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
