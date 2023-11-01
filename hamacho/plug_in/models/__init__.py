"""Load Anomaly Model."""



import os
from importlib import import_module
from typing import List
from typing import Union

from omegaconf import DictConfig
from omegaconf import ListConfig
from torch import load

from hamacho.plug_in.models.components import AnomalyModule

torch_model_list: List[str] = [
    "padim",
    "patchcore",
]


def get_model(config: Union[DictConfig, ListConfig]) -> AnomalyModule:
    """Load model from the configuration file.

    Works only when the convention for model naming is followed.

    The convention for writing model classes is
    `hamacho.plug_in.models.<model_name>.model.<Model_name>Lightning`
    `hamacho.plug_in.models.stfpm.model.StfpmLightning`

    Args:
        config (Union[DictConfig, ListConfig]): Config.yaml loaded using OmegaConf

    Raises:
        ValueError: If unsupported model is passed

    Returns:
        AnomalyModule: Anomaly Model
    """
    model_list: List[str] = [
        "cflow",
        "dfkde",
        "dfm",
        "ganomaly",
        "padim",
        "patchcore",
        "stfpm",
    ]
    model: AnomalyModule

    if config.model.name in model_list:
        module = import_module(f"hamacho.plug_in.models.{config.model.name}")
        model = getattr(module, f"{config.model.name.capitalize()}Lightning")(config)

    else:
        raise ValueError(f"Unknown model {config.model.name}!")

    if "init_weights" in config.keys() and config.init_weights:
        model.load_state_dict(
            load(os.path.join(config.project.path, config.init_weights))["state_dict"],
            strict=False,
        )

    return model
