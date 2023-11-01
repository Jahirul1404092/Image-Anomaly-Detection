"""Callbacks for Hamacho models."""


import os
from importlib import import_module
from typing import List
from typing import Union

import yaml
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback

from .cdf_normalization import CdfNormalizationCallback
from .metrics_configuration import MetricsConfigurationCallback
from .min_max_normalization import MinMaxNormalizationCallback
from .sigma6_normalization import SigmaSixNormalizationCallback
from .model_loader import LoadModelCallback
from .model_saver import SaveModelCallback
from .timer import TimerCallback
from .visualizer_callback import VisualizerCallback
from .csv import CSVMetricsLoggerCallback
from .plot import PlotMetrics
from .tiler_configuration import TilerConfigurationCallback

__all__ = [
    "CdfNormalizationCallback",
    "MetricsConfigurationCallback",
    "MinMaxNormalizationCallback",
    SigmaSixNormalizationCallback,
    "LoadModelCallback",
    "TilerConfigurationCallback",
    "TimerCallback",
    "VisualizerCallback",
]


def get_callbacks(config: Union[ListConfig, DictConfig]) -> List[Callback]:
    """Return base callbacks for all the lightning models.

    Args:
        config (DictConfig): Model config

    Return:
        (List[Callback]): List of callbacks.
    """
    callbacks: List[Callback] = []

    callbacks.append(TimerCallback())

    # Add metric configuration to the model via MetricsConfigurationCallback
    image_metric_names = OmegaConf.select(
        config, "metrics.image", default=None
    )
    pixel_metric_names = OmegaConf.select(
        config, "metrics.pixel", default=None
    )
    image_threshold = OmegaConf.select(
        config, "metrics.threshold.image_default", default=None
    )
    pixel_threshold = OmegaConf.select(
        config, "metrics.threshold.pixel_default", default=None
    )
    norm_image_threshold = OmegaConf.select(
        config, "metrics.threshold.image_norm", default=None
    )
    norm_pixel_threshold = OmegaConf.select(
        config, "metrics.threshold.pixel_norm", default=None
    )
    metrics_callback = MetricsConfigurationCallback(
        adaptive_threshold=config.metrics.threshold.adaptive,
        default_image_threshold=image_threshold,
        default_pixel_threshold=pixel_threshold,
        image_metric_names=image_metric_names,
        pixel_metric_names=pixel_metric_names,
    )
    callbacks.append(metrics_callback)

    if "weight_file" in config.model.keys():
        weights_path = os.path.join(
            config.project.path, config.model.weight_file
        )
    else:
        weights_path = os.path.join(config.project.path, "weights")
        os.makedirs(weights_path, exist_ok=True)
        weights_path = os.path.join(weights_path, "trained_data.hmc")

    save_model = SaveModelCallback(weights_path)
    load_model = LoadModelCallback(weights_path)
    callbacks.extend((save_model, load_model))

    if (
        "normalization_method" in config.model.keys()
        and not config.model.normalization_method == "none"
    ):
        if config.model.normalization_method == "cdf":
            if config.model.name in ["padim", "stfpm"]:
                if "nncf" in config.optimization and config.optimization.nncf.apply:
                    raise NotImplementedError(
                        "CDF Score Normalization is currently not compatible with NNCF."
                    )
                callbacks.append(CdfNormalizationCallback())
            else:
                raise NotImplementedError(
                    "Score Normalization is currently supported for PADIM and STFPM only."
                )
        elif config.model.normalization_method == "min_max":
            callbacks.append(
                MinMaxNormalizationCallback(
                    norm_image_threshold=norm_image_threshold,
                    norm_pixel_threshold=norm_pixel_threshold,
                )
            )
        elif config.model.normalization_method == "sigma6":
            callbacks.append(
                SigmaSixNormalizationCallback(
                    norm_image_threshold=norm_image_threshold,
                    norm_pixel_threshold=norm_pixel_threshold,
                )
            )
        else:
            raise ValueError(
                f"Normalization method not recognized: {config.model.normalization_method}"
            )
    
    # Add tiler configuration callback to the model via TilerConfigurationCallback
    if "tiling" in config.dataset.keys():
        enable = OmegaConf.select(config, "dataset.tiling.apply", default=False)
        tile_size = OmegaConf.select(config, "dataset.tiling.tile_size", default=None)
        stride = OmegaConf.select(config, "dataset.tiling.stride", default=None)
        remove_border_count = OmegaConf.select(config, "dataset.tiling.remove_border_count", default=0)
        use_random_tiling = OmegaConf.select(config, "dataset.tiling.use_random_tiling", default=False)
        random_tile_count = OmegaConf.select(config, "dataset.tiling.random_tile_count", default=16)
        tiler_callback = TilerConfigurationCallback(
            enable=enable,
            tile_size=tile_size,
            stride=stride,
            remove_border_count=remove_border_count,
            tile_count=random_tile_count,
            )
        callbacks.append(tiler_callback)

    test_save_outputs = OmegaConf.select(
        config,
        f"project.save_outputs.test.image[{config.dataset.task}]",
        default=None
    )
    pred_save_outputs = OmegaConf.select(
        config,
        f"project.save_outputs.inference.image[{config.dataset.task}]",
        default=None
    )
    add_label_on_image = OmegaConf.select(
        config,
        "project.save_outputs.add_label_on_image",
        default=None
    )
    save_combined_result_image = OmegaConf.select(
        config,
        "project.save_outputs.save_combined_result_as_image",
        default=False
    )

    if not config.project.log_images_to == []:
        callbacks.append(
            VisualizerCallback(
                task=config.dataset.task,
                save_root=config.project.save_root,
                test_dir_name=config.project.test_dir_name,
                inference_dir_name=config.project.inference_dir_name,
                test_save_outputs=test_save_outputs,
                pred_save_outputs=pred_save_outputs,
                add_label_on_image=add_label_on_image,
                save_combined_result_image=save_combined_result_image,
                log_images_to=config.project.log_images_to,
                inputs_are_normalized=not config.model.normalization_method == "none",
            )
        )

    csv_test_save_outputs = OmegaConf.select(
        config,
        "project.save_outputs.test.csv",
        default=[]
    )
    csv_pred_save_outputs = OmegaConf.select(
        config,
        "project.save_outputs.inference.csv",
        default=[]
    )
    # avoid None value if key not found by OmegaConf.select
    csv_test_save_outputs = csv_test_save_outputs or []
    csv_pred_save_outputs = csv_pred_save_outputs or []
    callbacks.append(
        CSVMetricsLoggerCallback(
            save_root=config.project.save_root,
            test_dir_name=config.project.test_dir_name,
            inference_dir_name=config.project.inference_dir_name,
            test_save_outputs=csv_test_save_outputs,
            pred_save_outputs=csv_pred_save_outputs,
        )
    )
    callbacks.append(
        PlotMetrics(
            save_root=config.project.save_root,
            test_dir_name=config.project.test_dir_name,
        )
    )

    if "optimization" in config.keys():
        if "nncf" in config.optimization and config.optimization.nncf.apply:
            # NNCF wraps torch's jit which conflicts with kornia's jit calls.
            # Hence, nncf is imported only when required
            nncf_module = import_module("hamacho.core.utils.callbacks.nncf.callback")
            nncf_callback = getattr(nncf_module, "NNCFCallback")
            nncf_config = yaml.safe_load(OmegaConf.to_yaml(config.optimization.nncf))
            callbacks.append(
                nncf_callback(
                    config=nncf_config,
                    export_dir=os.path.join(config.project.path, "compressed"),
                )
            )
        if "openvino" in config.optimization and config.optimization.openvino.apply:
            from .openvino import (
                OpenVINOCallback,  # pylint: disable=import-outside-toplevel
            )

            callbacks.append(
                OpenVINOCallback(
                    input_size=config.model.input_size,
                    dirpath=os.path.join(config.project.path, "openvino"),
                    filename="openvino_model",
                )
            )

    return callbacks
