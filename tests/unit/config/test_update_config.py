"""Test update config"""

import os
import tempfile

import pytest
import yaml

from hamacho.core.config import update_config
from hamacho.core.config.config import NO_BAD_MODE_NORMALIZATION
from hamacho.core.data.utils.download import (
    get_mvtec_url,
    download_file,
    hash_check_category,
    extract_tar,
)
from tests.helpers.dataset import TestDataset


class TestUpdateConfig:
    """Test if config is updated correctly"""

    @pytest.mark.parametrize(
        [
            "model_name",
            "is_mask",
            "image_size",
            "data_format",
            "seed",
            "category",
            "batch_size",
            "split",
            "num_workers",
            "task_type",
            "no_bad_mode",
        ],
        [
            (
                "patchcore",
                True,
                None,
                "folder",
                42,
                "shapes",
                1,
                0.2,
                2,
                "classification",
                False,
            ),
            (
                "padim",
                True,
                256,
                "mvtec",
                420,
                "toothbrush",
                2,
                0.1,
                0,
                "segmentation",
                False,
            ),
            (
                "patchcore",
                True,
                None,
                "folder",
                42,
                "shapes",
                1,
                0.2,
                2,
                "classification",
                True,
            ),
        ],
    )

    @TestDataset(num_train=200, num_test=30, path="data", seed=42)
    def test_update(
        self,
        model_name,
        is_mask,
        image_size,
        data_format,
        seed,
        category,
        batch_size,
        split,
        num_workers,
        task_type,
        no_bad_mode,
        path="data",
    ):
        category = "toothbrush" if data_format == "mvtec" else category
        if data_format == "mvtec" and not os.path.exists(os.path.join(path, category)):
            url = get_mvtec_url(category)
            zip_file_path = os.path.join(path, f"{category}.tar.xz")
            download_file(url, file_path=zip_file_path, desc=f"MVTec AD {category}")
            hash_check_category(zip_file_path, category)
            extract_tar(zip_file_path, path, delete_tar=True)

        with tempfile.TemporaryDirectory() as result_path:
            _, config_path = update_config(
                model_name,
                result_path,
                path,
                is_mask,
                task_type,
                "cpu",
                image_size,
                data_format,
                category,
                batch_size,
                split,
                seed,
                num_workers,
                no_bad_mode=no_bad_mode,
            )
            config = yaml.load(open(config_path, "r"), Loader=yaml.SafeLoader)

            mask = None if task_type == "classification" else "mask"
            image_size = 224 if image_size is None else image_size

            assert config["dataset"]["task"] == task_type
            assert config["dataset"]["num_workers"] == num_workers
            assert config["dataset"]["image_size"] == image_size
            assert config["dataset"]["format"] == data_format
            assert config["dataset"]["category"] == category
            assert config["dataset"]["train_batch_size"] == batch_size
            assert config["dataset"]["split_ratio"] == split
            assert config["dataset"]["mask"] == mask
            if no_bad_mode:
                assert config["dataset"]["abnormal_dir"] is None
                assert (
                    config["model"]["normalization_method"] == NO_BAD_MODE_NORMALIZATION
                )
