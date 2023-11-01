"""Test inference for different arguments in mvtec data format"""

import os
import shutil
import tempfile
import pytest

from typing import Union

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


class TestInferenceMVTecFormat:
    """Test inferencer trained in mvtec format with it's args"""
    data_format = "mvtec"
    model = "patchcore"
    category = "toothbrush"
    default_save_path = "inference_results"
    save_image_dir_name = "images"
    images_parent_dir_name = "combined"

    @TestDataset(num_train=NUM_TRAIN, num_test=NUM_TEST,
                 data_format=data_format, seed=42, category_name=category,
                 path=DATASET_PATH, remove_dir_on_exit=False)
    def setup_class(path="", category="", model=model):
        runner = CliRunner()
        runner.invoke(train, [
            '--model', model,
            '--data-format', "mvtec",
            '--dataset-root', path,
            '--category', category,
            '--image-size', "96",
            '--result-path', RESULT_PATH,
            '--with-mask-label'
        ])

    @pytest.mark.parametrize(
        [
            "in_save_path", "in_accelerator",
            "out_save_path", "single_image", "test_shape",
        ],
        [
            (None, "cpu", default_save_path, True, "star"),
            ("infer", "gpu" if HAS_CUDA else "auto",
             "infer", False, "hexagon"),
            (None, "auto", default_save_path, False, "star")
        ]
    )
    def test_args_mvtec_format(
        self,
        in_save_path: Union[str, None],
        in_accelerator: Union[str, None],
        out_save_path: str,
        single_image: bool,
        test_shape: str
    ):
        runner = CliRunner()
        im_path = os.path.join(
            DATASET_PATH,
            self.category,
            "test",
            test_shape
        )

        conf_path = os.path.join(
            RESULT_PATH,
            "mvtec",
            self.model,
            self.category,
            "config.yaml"
        )

        if out_save_path == self.default_save_path:
            out_save_path = os.path.join(
                RESULT_PATH,
                "mvtec",
                self.model,
                self.category,
                out_save_path,
                self.save_image_dir_name,
                self.images_parent_dir_name,
                test_shape,
            )
        else:
            out_save_path = os.path.join(
                INFER_SAVE_PATH,
                in_save_path,
                self.save_image_dir_name,
                self.images_parent_dir_name,
                test_shape,
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
