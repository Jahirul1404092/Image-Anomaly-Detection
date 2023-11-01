"""Test Config Getter."""

import pytest
from hamacho.core.config import get_configurable_parameters


MODEL_NAME = "patchcore"


class TestConfig:
    """Test Config Getter."""

    def test_get_configurable_parameters_return_correct_model_name(self):
        """Configurable parameter should return the correct model name."""
        configurable_parameters = get_configurable_parameters(
            MODEL_NAME, config_path=f"./hamacho/plug_in/models/{MODEL_NAME}/config.yaml"
        )
        assert configurable_parameters.model.name == MODEL_NAME

    def test_get_configurable_parameter_fails_with_none_arguments(self):
        """Configurable parameter should raise an error with none arguments."""
        with pytest.raises(ValueError):
            get_configurable_parameters()
