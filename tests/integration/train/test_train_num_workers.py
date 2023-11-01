"""Test trainer for different batch size"""

import platform
import tempfile

import pytest

from typing import Union
from click.testing import CliRunner

from hamacho.main import train
from hamacho.core.utils.general import get_cpu_count
from tests.helpers.dataset import TestDataset


class TestTrainBatchSize:
    """Test trainer with different batch size"""

    cpu_count = get_cpu_count()
    headroom = 2
    current_system = platform.system().lower()
    supported_system = "linux"

    @pytest.mark.parametrize(
        [
            "model", "input_nw", "out_nw"
        ],
        [
            ("patchcore", "0", 0),
            ("padim", "-1", 0),
            ("patchcore", str(cpu_count), cpu_count - headroom),
            ("padim", str(cpu_count + 2), cpu_count - headroom),
        ]
    )
    @TestDataset(num_train=20, num_test=6, path=None, seed=42,
                 data_format="folder", category_name="shapes")
    def test_bs(self,
                model: str,
                input_nw: str,
                out_nw: str,
                category="shapes",
                path=""):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as result_path:
            result = runner.invoke(train, [
                '--model', model,
                '--dataset-root', path,
                '--image-size', "64",
                '--num-workers', input_nw,
                '--category', category,
                '--result-path', result_path,
            ], standalone_mode=False)
            trainer = result.return_value

            if self.current_system == self.supported_system:
                dm = trainer.datamodule
                assert result.exit_code == 0
                assert dm.num_workers == out_nw
                assert dm.train_dataloader().num_workers == out_nw
            else:
                assert result.exit_code == 1
