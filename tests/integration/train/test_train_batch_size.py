"""Test trainer for different batch size"""

import tempfile

import yaml
import pytest

from typing import Union
from click.testing import CliRunner

from hamacho.main import train
from hamacho.plug_in.models import torch_model_list
from tests.helpers.dataset import TestDataset


models_default_bs = []
for m in torch_model_list:
    with open(f"hamacho/plug_in/models/{m}/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    trbs = config["dataset"]["train_batch_size"]
    tsbs = config["dataset"]["test_batch_size"]
    models_default_bs.append((m, None, trbs, tsbs))


class TestTrainBatchSize:
    """Test trainer with different batch size"""

    @pytest.mark.parametrize(
        [
            "model", "input_bs", "train_bs", "test_bs"
        ],
        [
            *models_default_bs,
            ("patchcore", "16", 16, 16),
            ("padim", "12", 12, 12),
        ]
    )
    @TestDataset(num_train=40, num_test=24, path='./data/', seed=42,
                 data_format="folder", category_name="shapes")
    def test_bs(self,
                model: str,
                input_bs: Union[str, None],
                train_bs: int,
                test_bs: int,
                category="shapes",
                path="./data"):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as result_path:
            result = runner.invoke(train, [
                '--model', model,
                '--image-size', "64",
                '--batch-size', input_bs,
                '--category', category,
                '--result-path', result_path,
            ], standalone_mode=False)
            trainer = result.return_value
            assert result.exit_code == 0
            assert trainer.datamodule.train_batch_size == train_bs
            assert trainer.datamodule.test_batch_size == test_bs

            train_sample = next(enumerate(trainer.datamodule.train_dataloader()))[1]
            test_sample = next(enumerate(trainer.datamodule.test_dataloader()))[1]
            assert train_sample["image"].shape[0] == train_bs
            assert test_sample["image"].shape[0] == test_bs
