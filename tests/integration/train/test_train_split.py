"""Test trainer for different batch size"""

import tempfile

import pytest
from click.testing import CliRunner

from hamacho.main import train
from tests.helpers.dataset import TestDataset

NUM_TRAIN = 20
NUM_TEST = 6

class TestTrainSplit:
    """Test trainer with different batch size"""

    @pytest.mark.parametrize(
        [
            "input_split", "out_split"
        ],
        [
            ("0.1", 0.1),
            ("0.8", 0.8),
            ("-0.1", 0.2),
            ("1.1", 0.2),
        ]
    )
    @TestDataset(num_train=NUM_TRAIN, num_test=NUM_TEST, path='./data/', seed=42,
                 data_format="folder", category_name="shapes")
    def test_split(
        self,
        input_split: str,
        out_split: float,
        category="shapes",
        path="./data"
    ):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as result_path:
            result = runner.invoke(train, [
                '--image-size', "128",
                '--split', input_split,
                '--batch-size', "1",
                '--category', category,
                '--result-path', result_path,
            ], standalone_mode=False)
            trainer = result.return_value
            dm = trainer.datamodule
            assert result.exit_code == 0
            assert dm.split_ratio == out_split
            batch_len = (NUM_TRAIN * out_split + NUM_TEST)
            assert trainer.num_test_batches[0] == batch_len
            assert len(list(dm.test_dataloader())) == batch_len
