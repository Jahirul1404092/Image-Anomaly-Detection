"""Test inference for different arguments in folder data format"""

import os
import shutil
import tempfile
import pytest

from typing import Union
from pathlib import Path

import torch

from click.testing import CliRunner
from pytorch_lightning import Trainer
from torchvision.datasets.folder import IMG_EXTENSIONS

from hamacho.main import inference, train
from hamacho.core.utils.folder import count_files
from tests.helpers.dataset import TestDataset
from tests.helpers.train import get_pl_accelerator

NUM_TRAIN = 20
NUM_TEST = 6

DATASET_PATH = tempfile.mkdtemp()
RESULT_PATH = tempfile.mkdtemp()
INFER_SAVE_PATH = tempfile.mktemp()
HAS_CUDA = torch.cuda.is_available()


class TestInferenceFolderFormat:
    """Test inferencer with it's args"""
    data_format = "folder"
    model = "padim"
    category = "shapes"
    default_save_path = "inference_results"
    save_image_dir_name = "images"
    images_parent_dir_name = "combined"
    test_images_dir_name = "bad"

    @TestDataset(num_train=NUM_TRAIN, num_test=NUM_TEST,
                 data_format=data_format, seed=42, category_name=category,
                 path=DATASET_PATH, remove_dir_on_exit=False)
    def setup_class(path="", category="", model=model):
        runner = CliRunner()
        runner.invoke(train, [
            '--model', model,
            '--dataset-root', path,
            '--category', category,
            '--image-size', "96",
            '--result-path', RESULT_PATH,
            '--with-mask-label'
        ])

    @pytest.mark.parametrize(
        [
            "in_save_path", "in_accelerator",
            "out_save_path", "single_image"
        ],
        [
            (None, "cpu", default_save_path, True),
            ("infer", "gpu" if HAS_CUDA else "auto", "infer", False),
            (None, "auto", default_save_path, False)
        ]
    )
    def test_args_folder_format(
        self,
        in_save_path: Union[str, None],
        in_accelerator: Union[str, None],
        out_save_path: str,
        single_image: bool
    ):
        runner = CliRunner()
        im_path = os.path.join(
            DATASET_PATH,
            self.category,
            self.test_images_dir_name
        )

        conf_path = os.path.join(
            RESULT_PATH,
            self.category,
            self.model,
            "config.yaml"
        )

        if out_save_path == self.default_save_path:
            out_save_path = os.path.join(
                RESULT_PATH,
                self.category,
                self.model,
                out_save_path,
                self.save_image_dir_name,
                self.images_parent_dir_name,
                self.test_images_dir_name,
            )
        else:
            out_save_path = os.path.join(
                INFER_SAVE_PATH,
                in_save_path,
                self.save_image_dir_name,
                self.images_parent_dir_name,
                self.test_images_dir_name,
            )

        if in_save_path is not None:
            in_save_path = os.path.join(INFER_SAVE_PATH, in_save_path)
            os.makedirs(in_save_path)

        im_path = os.path.join(im_path, "000.png") if single_image else im_path
        out_count = 1 if single_image else NUM_TEST

        result = runner.invoke(inference, [
            '--save-path', in_save_path,
            '--accelerator', in_accelerator,
            '--image-path', im_path,
            '--config-path', conf_path,
        ], standalone_mode=False)

        trainer: Trainer = result.return_value
        print(result.output)
        assert result.exit_code == 0
        assert isinstance(
            trainer.accelerator,
            get_pl_accelerator(in_accelerator)
        )
        assert count_files(
            out_save_path, IMG_EXTENSIONS
        ) == out_count

    def teardown_class(cls):
        # print(DATASET_PATH, RESULT_PATH)
        shutil.rmtree(DATASET_PATH)
        shutil.rmtree(RESULT_PATH)
        shutil.rmtree(INFER_SAVE_PATH)


@pytest.mark.parametrize(
    "model",
    ("padim", "patchcore")
)
@TestDataset(
    num_train=NUM_TRAIN,
    num_test=NUM_TEST,
    path="./data",
    seed=42,
    data_format="folder",
    category_name="shapes",
)
def test_no_bad_mode(model, category="shapes", path=""):
    bad_data_path = Path(f"{path}/{category}/bad")
    os.rename(bad_data_path, bad_data_path.parent / "bad_")
    mask_data_path = Path(f"{path}/{category}/mask")
    shutil.rmtree(str(mask_data_path))
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as result_path, \
        tempfile.TemporaryDirectory() as inference_path:
        runner.invoke(train, [
                "--dataset-root", path,
                "--image-size", "128",
                "--category", category,
                "--model", model,
                "--result-path", result_path,
                "--no-bad-mode",
            ],
            standalone_mode=False,
        )
        im_path = os.path.join(
            path,
            category,
            "bad_"
        )

        conf_path = os.path.join(
            result_path,
            category,
            model,
            "config.yaml"
        )

        in_save_path = os.path.join(inference_path, "inference_results")
        os.makedirs(in_save_path)
        img_out_path = os.path.join(
            in_save_path,
            "images",
            "combined",
            "bad_",
        )
        csv_out_path = os.path.join(
            in_save_path,
            "csv",
            "bad_"
        )

        runner = CliRunner()
        result = runner.invoke(inference, [
            '--save-path', in_save_path,
            '--image-path', im_path,
            '--config-path', conf_path,
        ], standalone_mode=False)
        print(result.stdout)
        assert count_files(img_out_path, IMG_EXTENSIONS) == NUM_TEST
        assert count_files(csv_out_path, ".csv") == NUM_TEST
