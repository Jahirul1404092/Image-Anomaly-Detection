"""Test Dataset."""

import os
from time import sleep

import numpy as np
import pytest
from pathlib import Path

from hamacho.core.utils.filelist import get_valid_filelist, get_valid_paired_filelist
from hamacho.core.config import update_input_size_config
from hamacho.core.data import (
    MVTec,
    FolderDataModule,
    get_datamodule,
    FilelistDataModule,
)
from hamacho.core.pre_processing.transforms import Denormalize, ToNumpy
from tests.helpers.config import get_test_configurable_parameters
from tests.helpers.dataset import TestDataset, get_dataset_path


@pytest.fixture(autouse=True)
def mvtec_data_module():
    datamodule = MVTec(
        root=get_dataset_path(dataset="MVTec"),
        category="toothbrush",
        image_size=(256, 256),
        train_batch_size=1,
        test_batch_size=1,
        num_workers=0,
    )
    datamodule.prepare_data()
    datamodule.setup()

    return datamodule


@pytest.fixture(autouse=True, params=["regular", "no_bad_mode"])
@TestDataset(
    num_train=40,
    num_test=8,
    path="./data/",
    seed=42,
    category_name="shapes",
    data_format="folder",
    remove_dir_on_exit=False,
)
def folder_data_module(
    request,
    path="",
    category="",
):
    """Create Folder Data Module."""
    abnormal_dir = "bad" if request.param == "regular" else None
    mask_dir = "mask" if request.param == "regular" else None
    task = "segmentation" if request.param == "regular" else "classification"

    datamodule = FolderDataModule(
        root=path,
        category=category,
        normal_dir="good",
        abnormal_dir=abnormal_dir,
        mask_dir=mask_dir,
        task=task,
        split_ratio=0.2,
        seed=0,
        image_size=(256, 256),
        train_batch_size=16,
        test_batch_size=4,
        num_workers=0,
        create_validation_set=True,
    )
    datamodule.setup()

    return datamodule


@pytest.fixture(autouse=True)
@TestDataset(
    num_train=1,
    num_test=1,
    path="./data/",
    seed=42,
    category_name="shapes",
    data_format="filelist",
    remove_dir_on_exit=False,
)
def filelist_data_module(path="", category=""):
    """Create Filelist Data Module."""

    good_images_path = os.path.join(path, category, "good")
    bad_images_paths = os.path.join(path, category, "bad")
    mask_images_paths = os.path.join(path, category, "mask")

    good_images = [
        os.path.join(good_images_path, fname) for fname in os.listdir(good_images_path)
    ]
    bad_images = [
        os.path.join(bad_images_paths, fname) for fname in os.listdir(bad_images_paths)
    ]
    mask_images = [
        os.path.join(mask_images_paths, fname)
        for fname in os.listdir(mask_images_paths)
    ]
    l_normal, l_abnormal, l_abnormal_mask, _ = get_valid_filelist(
        good_images, bad_images, mask_images
    )
    l_abnormal, l_abnormal_mask, _ = get_valid_paired_filelist(
        l_abnormal, l_abnormal_mask
    )

    datamodule = FilelistDataModule(
        root=path,
        category=category,
        l_normal=l_normal,
        l_abnormal=l_abnormal,
        l_abnormal_mask=l_abnormal_mask,
        task="segmentation",
        split_ratio=0.2,
        seed=0,
        image_size=(256, 256),
        train_batch_size=16,
        test_batch_size=8,
        num_workers=0,
        create_validation_set=True,
    )
    datamodule.setup()
    return datamodule


@pytest.fixture(autouse=True)
def data_sample(mvtec_data_module):
    _, data = next(enumerate(mvtec_data_module.train_dataloader()))
    return data


class TestMVTecDataModule:
    """Test MVTec AD Data Module."""

    def test_batch_size(self, mvtec_data_module):
        """test_mvtec_datamodule [summary]"""
        _, train_data_sample = next(enumerate(mvtec_data_module.train_dataloader()))
        _, val_data_sample = next(enumerate(mvtec_data_module.val_dataloader()))
        assert train_data_sample["image"].shape[0] == 1
        assert val_data_sample["image"].shape[0] == 1

    def test_val_and_test_dataloaders_has_mask_and_gt(self, mvtec_data_module):
        """Test Validation and Test dataloaders should return filenames, image, mask and label."""
        _, val_data = next(enumerate(mvtec_data_module.val_dataloader()))
        _, test_data = next(enumerate(mvtec_data_module.test_dataloader()))
        assert sorted(
            [
                "category",
                "data_type",
                "image",
                "image_path",
                "label",
                "mask",
                "mask_path",
            ]
        ) == sorted(val_data.keys())
        assert sorted(
            [
                "category",
                "data_type",
                "image",
                "image_path",
                "label",
                "mask",
                "mask_path",
            ]
        ) == sorted(test_data.keys())

    def test_non_overlapping_splits(self, mvtec_data_module):
        """This test ensures that the train and test splits generated are non-overlapping."""
        assert (
            len(
                set(
                    mvtec_data_module.test_data.samples["image_path"].values
                ).intersection(
                    set(mvtec_data_module.train_data.samples["image_path"].values)
                )
            )
            == 0
        ), "Found train and test split contamination"


class TestFolderDataModule:
    """Test Folder Data Module."""

    def test_batch_size(self, folder_data_module):
        """Test batch size."""
        _, train_data_sample = next(enumerate(folder_data_module.train_dataloader()))
        _, test_data_sample = next(enumerate(folder_data_module.test_dataloader()))
        assert train_data_sample["image"].shape[0] == 16
        assert test_data_sample["image"].shape[0] == 4

    def test_val_and_test_dataloaders_has_mask_and_gt(self, folder_data_module):
        """Test Validation and Test dataloaders should return filenames, image, mask and label."""
        _, val_data = next(enumerate(folder_data_module.val_dataloader()))
        _, test_data = next(enumerate(folder_data_module.test_dataloader()))

        if folder_data_module.task == "classification":
            return
        assert sorted(
            [
                "category",
                "data_type",
                "image",
                "image_path",
                "label",
                "mask",
                "mask_path",
            ]
        ) == sorted(val_data.keys())
        assert sorted(
            [
                "category",
                "data_type",
                "image",
                "image_path",
                "label",
                "mask",
                "mask_path",
            ]
        ) == sorted(test_data.keys())


class TestFilelistDataModule:
    """Test Filelist Data Module."""

    def test_batch_size(self, filelist_data_module):
        """Test batch size."""
        _, train_data_sample = next(enumerate(filelist_data_module.train_dataloader()))
        _, test_data_sample = next(enumerate(filelist_data_module.test_dataloader()))
        assert train_data_sample["image"].shape[0] == 16
        assert test_data_sample["image"].shape[0] == 8

    def test_val_and_test_dataloaders(self, filelist_data_module):
        """Test Validation and Test dataloaders should return filenames,
        image, and label."""
        _, val_data = next(enumerate(filelist_data_module.val_dataloader()))
        _, test_data = next(enumerate(filelist_data_module.test_dataloader()))

        assert list(test_data["image"].shape) == [8, 3, 256, 256]
        assert list(val_data["mask"].shape) == [8, 256, 256]
        assert all(x in val_data["label"] for x in [0, 1])
        assert "shapes" in val_data["category"]

        assert sorted(
            [
                "category",
                "mask",
                "mask_path",
                "data_type",
                "image",
                "image_path",
                "label",
            ]
        ) == sorted(val_data.keys())
        assert sorted(
            [
                "category",
                "mask",
                "mask_path",
                "data_type",
                "image",
                "image_path",
                "label",
            ]
        ) == sorted(test_data.keys())


class TestDenormalize:
    """Test Denormalize Util."""

    def test_denormalize_image_pixel_values(self, data_sample):
        """Test Denormalize denormalizes tensor into [0, 256] range."""
        denormalized_sample = Denormalize().__call__(data_sample["image"].squeeze())
        assert denormalized_sample.min() >= 0 and denormalized_sample.max() <= 256

    def test_denormalize_return_numpy(self, data_sample):
        """Denormalize should return a numpy array."""
        denormalized_sample = Denormalize()(data_sample["image"].squeeze())
        assert isinstance(denormalized_sample, np.ndarray)

    def test_denormalize_channel_order(self, data_sample):
        """Denormalize should return a numpy array of order [HxWxC]"""
        denormalized_sample = Denormalize().__call__(data_sample["image"].squeeze())
        assert (
            len(denormalized_sample.shape) == 3 and denormalized_sample.shape[-1] == 3
        )

    def test_representation(self):
        """Test Denormalize representation should return string
        Denormalize()"""
        assert str(Denormalize()) == "Denormalize()"


class TestToNumpy:
    """Test ToNumpy whether it properly converts tensor into numpy array."""

    def test_to_numpy_image_pixel_values(self, data_sample):
        """Test ToNumpy should return an array whose pixels in the range of [0,
        256]"""
        array = ToNumpy()(data_sample["image"])
        assert array.min() >= 0 and array.max() <= 256

    def test_to_numpy_converts_tensor_to_np_array(self, data_sample):
        """ToNumpy returns a numpy array."""
        array = ToNumpy()(data_sample["image"])
        assert isinstance(array, np.ndarray)

    def test_to_numpy_channel_order(self, data_sample):
        """ToNumpy() should return a numpy array of order [HxWxC]"""
        array = ToNumpy()(data_sample["image"])
        assert len(array.shape) == 3 and array.shape[-1] == 3

    def test_one_channel_images(self, data_sample):
        """One channel tensor should be converted to HxW np array."""
        data = data_sample["image"][:, 0, :, :].unsqueeze(0)
        array = ToNumpy()(data)
        assert len(array.shape) == 2

    def test_representation(self):
        """Test ToNumpy() representation should return string `ToNumpy()`"""
        assert str(ToNumpy()) == "ToNumpy()"


class TestConfigToDataModule:
    """Tests that check if the dataset parameters in the config achieve the desired effect."""

    @pytest.mark.parametrize(
        ["input_size", "effective_image_size"],
        [
            (512, (512, 512)),
            ((245, 276), (245, 276)),
            ((263, 134), (263, 134)),
            ((267, 267), (267, 267)),
        ],
    )
    @TestDataset(num_train=20, num_test=10)
    def test_image_size(
        self, input_size, effective_image_size, category="shapes", path=None
    ):
        """Test if the image size parameter works as expected."""
        configurable_parameters = get_test_configurable_parameters(
            dataset_path=path, model_name="padim"
        )
        configurable_parameters.dataset.category = category
        configurable_parameters.dataset.image_size = input_size
        configurable_parameters = update_input_size_config(configurable_parameters)

        data_module = get_datamodule(configurable_parameters)
        data_module.setup()
        assert (
            iter(data_module.train_dataloader()).__next__()["image"].shape[-2:]
            == effective_image_size
        )
