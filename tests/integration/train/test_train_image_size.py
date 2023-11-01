"""Test trainer for different image size"""

import os
import tempfile

from PIL import Image
from typing import Union

import pytest

from click.testing import CliRunner
from omegaconf import OmegaConf

from hamacho.main import train
from hamacho.plug_in.models import torch_model_list
from tests.helpers.dataset import TestDataset

models_default_img_size = [
    (
        m,
        None,
        OmegaConf.load(
            f"hamacho/plug_in/models/{m}/config.yaml"
        )["dataset"]["image_size"],
        "auto",
    ) \
    for m in torch_model_list
]

class TestTrainImageSize:
    """Test trainer with different image size args"""

    test_dir_name = "test_predictions"
    save_image_dir_name = "images"
    images_parent_dir_name = "predicted_heat_map"

    @pytest.mark.parametrize(
        [
            "model", "in_im_size", "out_im_size", "accelerator"
        ],
        [
            # ("patchcore", None, 224),
            # ("padim", None, 256),
            *models_default_img_size,
            ("patchcore", "32", 64, "auto"),
            ("padim", "90", 96, "auto"),
            ("patchcore", "250", 256, "auto"),
            ("padim", "810", 640, "cpu"),
            ("patchcore", "810", 640, "cpu"),
        ]
    )
    @TestDataset(num_train=16, num_test=4, path='./data/', seed=42,
                 data_format="folder", category_name="shapes")
    def test_im_size(
        self,
        model: str,
        in_im_size: Union[str, None],
        out_im_size: int,
        accelerator: str,
        category="shapes",
        path="./data"
    ):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as result_path:
            result = runner.invoke(train, [
                '--model', model,
                '--image-size', in_im_size,
                '--batch-size', "1",
                '--accelerator', accelerator,
                '--category', category,
                '--result-path', result_path,
            ], standalone_mode=False)
            trainer = result.return_value
            assert result.exit_code == 0
            assert trainer.datamodule.image_size[0] == out_im_size
            assert trainer.datamodule.image_size[1] == out_im_size

            train_sample = next(enumerate(trainer.datamodule.train_dataloader()))[1]
            test_sample = next(enumerate(trainer.datamodule.test_dataloader()))[1]
            assert train_sample["image"].shape[3] == out_im_size
            assert test_sample["image"].shape[3] == out_im_size
            image_filename = os.path.join(
                result_path,
                category,
                model,
                self.test_dir_name,
                self.save_image_dir_name,
                self.images_parent_dir_name,
                "bad",
                "000.png"
            )
            image = Image.open(image_filename)
            assert image.size[0] == out_im_size
            image.close()
