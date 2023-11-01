"""Test trainer for different task type"""

import tempfile

import pytest

from click.testing import CliRunner

from hamacho.main import train
from tests.helpers.dataset import TestDataset


class TestTrainTaskType:
    """Test trainer task type with/without mask label"""

    @pytest.mark.parametrize(
        [
            "task", "with_mask"
        ],
        [
            ("classification", True),
            ("segmentation", True),
            ("classification", False),
            ("segmentation", False),
        ]
    )
    @TestDataset(num_train=20, num_test=6, path='./data/', seed=42,
                 data_format="folder", category_name="shapes")
    def test_task(self, task, with_mask, category="shapes", path="./data"):
        runner = CliRunner()
        with_mask_arg = ('--with-mask-label',) if with_mask else tuple()
        with tempfile.TemporaryDirectory() as result_path:
            result = runner.invoke(train, [
                '--task-type', task,
                *with_mask_arg,
                '--category', category,
                '--result-path', result_path,
            ], standalone_mode=False)
            trainer = result.return_value
            assert result.exit_code == 0
            assert trainer.datamodule.task == task
            if task == "classification" and with_mask:
                assert "But Training with `--with-mask-label` is also requested" in result.output
