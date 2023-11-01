"""Test trainer for with tiler enabled"""

import tempfile

import yaml
import pytest
from pathlib import Path

from click.testing import CliRunner

from hamacho.core.pre_processing import Tiler
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


class TestTrainTiler:
    """Test trainer with tiling enabled"""

    @pytest.mark.parametrize(
        [
            "model", "enable_tiling",
        ],
        [
            ("patchcore", True),
            ("padim", True),
            ("padim", False),
        ]
    )
    @TestDataset(num_train=20, num_test=6, path='./data/', seed=42,
                 data_format="folder", category_name="shapes")
    def test_tiler(self, model, enable_tiling, category="shapes", path="./data"):
        with open(f"hamacho/plug_in/models/{model}/config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        config["dataset"]["tiling"]["apply"] = enable_tiling
        config["dataset"]["tiling"]["tile_size"] = 128
        config["dataset"]["tiling"]["stride"] = 128
        config["dataset"]["tiling"]["remove_border_count"] = 0
        config["dataset"]["tiling"]["use_random_tiling"] = False
        config["dataset"]["tiling"]["random_tile_count"] = 16

        temp_config_path = f"temp_{model}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as result_path:
            result = runner.invoke(train, [
                '--task-type', "segmentation",
                '--category', category,
                '--result-path', result_path,
                '--config-path', temp_config_path,
            ], standalone_mode=False)

        trainer = result.return_value
        assert result.exit_code == 0
        tiler = trainer.model.model.tiler
        if enable_tiling:
            assert isinstance(tiler, Tiler)
            assert tiler.tile_size_h == tiler.tile_size_w == 128
            assert tiler.stride_h == tiler.stride_w == 128
            assert tiler.remove_border_count == 0
            assert tiler.tile_count == 16
        else:
            assert tiler is None
        Path(temp_config_path).unlink(missing_ok=True)
        