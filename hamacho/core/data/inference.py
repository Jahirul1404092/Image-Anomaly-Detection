"""Inference Dataset."""


from pathlib import Path
from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
)

import albumentations as A
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from hamacho.core.data.utils import get_image_filenames
from hamacho.core.data.utils import read_image
from hamacho.core.pre_processing import PreProcessor


class InferenceDataset(Dataset):
    """Inference Dataset to perform prediction."""

    def __init__(
        self,
        path: Union[str, Path],
        pre_process: Optional[PreProcessor] = None,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        transform_config: Optional[Union[str, A.Compose]] = None,
    ) -> None:
        """Inference Dataset to perform prediction.

        Args:
            path (Union[str, Path]): Path to an image or image-folder.
            pre_process (Optional[PreProcessor], optional): Pre-Processing transforms to
                pre-process the input dataset. Defaults to None.
            image_size (Optional[Union[int, Tuple[int, int]]], optional): Target image size
                to resize the original image. Defaults to None.
            transform_config (Optional[Union[str, A.Compose]], optional): Configuration file
                parse the albumentation transforms. Defaults to None.
        """
        super().__init__()

        self.image_filenames = get_image_filenames(path)

        if pre_process is None:
            self.pre_process = PreProcessor(transform_config, image_size)
        else:
            self.pre_process = pre_process

    def __len__(self) -> int:
        """Get the number of images in the given path."""
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
        """Get dataset item for the index ``index``.
        Args:
            index (int): Index to get the item.
        Returns:
            Dict[str, Union[str, Tensor]]: Dict containing image path, image tensor.
        """
        image_filename = self.image_filenames[index]
        image = read_image(path=image_filename)
        pre_processed = self.pre_process(image=image)

        return {"image": pre_processed["image"], "image_path": image_filename}


class InferenceDataModule(LightningDataModule):
    """Dalamodule for inference"""
    def __init__(
        self,
        root: str,
        image_size: Tuple[int],
        batch_size: int = 1,
        num_workers: int = 1,
        transform_config: Optional[Union[str, A.Compose]] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.batch_szie = batch_size
        self.num_workers = num_workers
        self.transform_config = transform_config

    def setup(self, stage: Optional[str] = None) -> None:
        self.inference_data = InferenceDataset(
            path=self.root,
            image_size=self.image_size,
            transform_config=self.transform_config,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Get predict dataloader."""
        return DataLoader(
            self.inference_data,
            shuffle=False,
            batch_size=self.batch_szie,
            num_workers=self.num_workers,
        )
