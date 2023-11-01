"""Quick sanity check on models."""



import tempfile

import pytest
import torch

from hamacho.plug_in.models import torch_model_list
from tests.helpers.dataset import TestDataset
from tests.helpers.model import model_load_test, setup_model_train


class TestModel:
    """Do a sanity check on the models."""

    @pytest.mark.parametrize(
        ["model_name", "nncf"],
        [(m, False) for m in torch_model_list],
    )
    @TestDataset(num_train=10, num_test=5)
    def test_model(self, model_name, nncf, category="shapes", path="./data"):
        """Test the models on only 1 epoch as a sanity check before merge."""
        with tempfile.TemporaryDirectory() as project_path:
            # Train test
            device = [0]
            if not torch.cuda.is_available():
                device = 1

            config, datamodule, model, trainer = setup_model_train(
                model_name,
                dataset_path=path,
                project_path=project_path,
                nncf=nncf,
                category=category,
                weight_file="trained_data.hmc",
                device=device,
            )
            results = trainer.test(model=model, datamodule=datamodule)[0]

            # Test model load
            model_load_test(config, datamodule, results)
