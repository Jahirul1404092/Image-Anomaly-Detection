"""Test trainer for different task type"""

import os
import tempfile

import torch
import pytest

from click.testing import CliRunner
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from hamacho.main import train
from hamacho.core.utils.general import get_torch_device
from tests.helpers.dataset import TestDataset
from tests.helpers.train import get_pl_accelerator


class TestTrainAccelerator:
    """Test trainer task type with/without mask label"""

    @pytest.mark.parametrize(
        [
            "model", "accelerator", "key_name"
        ],
        [
            ("padim", "cpu", "model.idx"),
            ("padim", "gpu", "model.idx"),
            ("patchcore", "gpu", "model.memory_bank"),
            ("patchcore", "cpu", "model.memory_bank"),
        ]
    )
    @TestDataset(num_train=20, num_test=6, path='./data/', seed=42,
                 data_format="folder", category_name="shapes")
    def test_acc(self, model, accelerator, key_name,
                 category="shapes", path=""):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as result_path:
            result = runner.invoke(train, [
                '--model', model,
                '--image-size', '64',
                '--accelerator', accelerator,
                '--category', category,
                '--result-path', result_path,
            ], standalone_mode=False)
            trainer = result.return_value
            pl_accelerator = get_pl_accelerator(accelerator)
            # check for error if installed pytorch does not support cuda
            if accelerator == "gpu" and not torch.cuda.is_available():
                assert result.exit_code == 1
                assert isinstance(result.exception, MisconfigurationException)
                return

            assert result.exit_code == 0
            assert isinstance(trainer.accelerator, pl_accelerator)
            weight_path = os.path.join(
                result_path,
                category,
                model,
                "weights",
                "trained_data.hmc"
            )
            device = get_torch_device(accelerator)
            trained_data = torch.load(weight_path)

            tensor = trained_data[key_name]
            assert tensor.device.type == device.type
