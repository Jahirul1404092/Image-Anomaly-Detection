"""Test trainer for every model"""

import os
import tempfile

import torch
import pytest

from importlib import import_module

from click.testing import CliRunner
from omegaconf import OmegaConf

from hamacho.main import train
from hamacho.plug_in.models import torch_model_list, get_model
from tests.helpers.dataset import TestDataset


class TestTrainedModel:
    """Test trainer has loaded the correct model"""

    @pytest.mark.parametrize(
        [
            "model", "category"
        ],
        [
            (m, "shapes") for m in torch_model_list
        ]
    )
    @TestDataset(
        num_train=20, 
        num_test=6, 
        path='./data/', 
        seed=42,
        data_format="filelist", 
        category_name="shapes")
    def test_model(self, model, category, path='./data/'):
        lm = import_module(f"hamacho.plug_in.models.{model}.lightning_model")
        pl_model = getattr(lm, f"{model.capitalize()}Lightning")
        tm = import_module(f"hamacho.plug_in.models.{model}.torch_model")
        torch_model = getattr(tm, f"{model.capitalize()}Model")
        runner = CliRunner()
        input_size = 192
        with tempfile.TemporaryDirectory() as result_path:
            result = runner.invoke(train, [
                '--model', model,
                '--image-size', str(input_size),
                '--accelerator', "cpu",
                '--category', category,
                '--result-path', result_path,
            ], standalone_mode=False)
            trainer = result.return_value
            assert result.exit_code == 0
            assert isinstance(trainer.model, pl_model)
            assert isinstance(trainer.model.model, torch_model)

            weight_path = os.path.join(
                result_path,
                category,
                model,
                "weights",
                "trained_data.hmc"
            )
            config_path = os.path.join(
                result_path,
                category,
                model,
                "config.yaml"
            )
            trained_data = torch.load(weight_path)
            config = OmegaConf.load(config_path)
            config.model.input_size = (input_size, input_size)
            model = get_model(config)
            model.load_trained_data(trained_data)
            model.eval()
            assert isinstance(model, pl_model)
            assert isinstance(model.model, torch_model)
