"""This module contains Torch inference implementations."""


from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union, Literal

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from hamacho.core.config import generate_multi_inferencer_config
from hamacho.core.data.utils import read_image
from hamacho.core.deploy.optimize import get_model_metadata
from hamacho.core.pre_processing import PreProcessor
from hamacho.core.utils.callbacks import (
    CSVMetricsLoggerCallback,
    MetricsConfigurationCallback,
    VisualizerCallback,
    get_callbacks
)
from hamacho.core.utils.general import DummyTrainer, get_torch_device
from hamacho.plug_in.models import get_model
from hamacho.plug_in.models.components import AnomalyModule

from hamacho.core.deploy.inferencers.base import Inferencer


class TorchInferencer(Inferencer):
    """PyTorch implementation for the inference.

    Args:
        config (DictConfig): Configurable parameters that are used
            during the training stage.
        model_source (Union[str, Path, AnomalyModule]): Path to the model ckpt file or the Anomaly model.
        meta_data_path (Union[str, Path], optional): Path to metadata file. If none, it tries to load the params
                from the model state_dict. Defaults to None.
    """

    def __init__(
        self,
        config: Union[DictConfig, ListConfig],
        model_source: Union[str, Path, AnomalyModule],
        device: Optional[torch.device] = None,
        meta_data_path: Union[str, Path] = None,
        visualizer_callback: Optional[VisualizerCallback] = None,
        metrics_callback: Optional[MetricsConfigurationCallback] = None,
        csv_callback: Optional[CSVMetricsLoggerCallback] = None,
    ):
        self.config = config
        self.device = (
            get_torch_device(config.trainer.accelerator)
            if device is None else device
        )
        if isinstance(model_source, AnomalyModule):
            self.model = model_source.to(self.device)
        else:
            self.model = self.load_model(model_source)

        self.meta_data = self._load_meta_data(meta_data_path)

        self.input_size = self.config.dataset.image_size
        self._init_pre_processor()
        self._set_callbacks(
            visualizer_callback=visualizer_callback,
            metrics_callback=metrics_callback,
            csv_callback=csv_callback,
        )

    def _set_callbacks(
        self,
        visualizer_callback: Optional[VisualizerCallback] = None,
        metrics_callback: Optional[MetricsConfigurationCallback] = None,
        csv_callback: Optional[CSVMetricsLoggerCallback] = None,
    ):
        if metrics_callback is not None:
            metrics_callback.setup(DummyTrainer, self.model)
            self._set_thresholds(config=self.config)

        self.metrics_callback = metrics_callback
        self.visualizer_callback = visualizer_callback
        self.csv_callback = csv_callback

        if self.csv_callback is not None:
            self.csv_callback.on_predict_start(
                DummyTrainer, self.model
            )

    def _init_pre_processor(
        self,
    ):
        transform_config = (
            self.config.transform
            if "transform" in self.config.keys() else None
        )
        self.pre_processor = PreProcessor(
            transform_config,
            tuple(self.input_size)
        )

    def _set_thresholds(self, config):
        if self.model.image_metrics is not None \
            and "image_norm" in config.metrics.threshold.keys():
            self.model.image_metrics.set_threshold(
                config.metrics.threshold.image_norm
            )
        elif self.model.image_metrics is not None:
            self.model.image_metrics.set_threshold(0.5)

        if self.model.pixel_metrics is not None \
            and "pixel_norm" in config.metrics.threshold.keys():
            self.model.pixel_metrics.set_threshold(
                config.metrics.threshold.pixel_norm
            )
        elif self.model.pixel_metrics is not None:
            self.model.pixel_metrics.set_threshold(0.5)

    def _load_meta_data(
        self, path: Optional[Union[str, Path]] = None
    ) -> Union[Dict, DictConfig]:
        """Load metadata from file or from model state dict.

        Args:
            path (Optional[Union[str, Path]], optional): Path to metadata file. If none, it tries to load the params
                from the model state_dict. Defaults to None.

        Returns:
            Dict: Dictionary containing the meta_data.
        """
        meta_data: Union[DictConfig, Dict[str, Union[float, Tensor, np.ndarray]]]
        if path is None:
            meta_data = get_model_metadata(self.model)
        else:
            meta_data = super()._load_meta_data(path)
        return meta_data

    def warmup(
        self,
    ):
        """
        Allocate gpu or cpu memory to make the first prediction faster
        than usual. This should be called right after creating the instance
        """
        img = torch.empty(
            (1, 3, *self.input_size),
            dtype=torch.float32,
            device=self.device
        )
        self.forward({"image": img})

    def load_model(self, path: Union[str, Path]) -> AnomalyModule:
        """Load the PyTorch model.

        Args:
            path (Union[str, Path]): Path to model ckpt file.

        Returns:
            (AnomalyModule): PyTorch Lightning model.
        """
        model = get_model(self.config)
        model.load_trained_data(
            torch.load(path, map_location=self.device)
        )
        model.eval()
        model = model.to(self.device)
        return model

    def pre_process(self, image: np.ndarray) -> Tensor:
        """Pre process the input image by applying transformations.

        Args:
            image (np.ndarray): Input image

        Returns:
            Tensor: pre-processed image.
        """
        pre_processed = self.pre_processor(image=image)

        if len(pre_processed["image"]) == 3:
            pre_processed["image"] = pre_processed["image"].unsqueeze(0)

        item = {"image": pre_processed["image"].to(self.device)}

        return item

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward-Pass input tensor to the model.

        Args:
            image (Tensor): Input tensor.

        Returns:
            Tensor: Output predictions.
        """
        return self.model.validation_step(batch, 0)

    def post_process(
        self,
        predictions: Dict[str, Tensor],
        meta_data: Optional[Union[Dict, DictConfig]] = None
    ) -> Tuple[np.ndarray, float]:
        """Post process the output predictions.

        Args:
            predictions (Tensor): Raw output predicted by the model.
            meta_data (Dict, optional): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
                Defaults to None.

        Returns:
            np.ndarray: Post processed predictions that are ready to be visualized.
        """
        if meta_data is None:
            meta_data = self.meta_data
            predictions["image_threshold"] = self.model.image_threshold.value
        else:
            predictions["image_threshold"] = meta_data["image_threshold"]

        self.model._post_process(predictions, keep_denormalized=True)
        anomaly_map, pred_score = predictions["anomaly_maps"], predictions["pred_scores"]

        anomaly_map, pred_score = self._normalize(anomaly_map, pred_score, meta_data)

        predictions["anomaly_maps"] = anomaly_map
        predictions["pred_scores"] = pred_score

        if self.metrics_callback is not None:
            predictions["image_threshold_norm"] = self.model.image_metrics.threshold

        if "image_path" not in predictions:
            return predictions

        if self.visualizer_callback is not None:
            self.visualizer_callback.on_predict_batch_end(
                DummyTrainer, self.model, predictions,
                predictions, 0, None
            )

        if self.csv_callback is not None:
            self.csv_callback.on_predict_batch_end(
                DummyTrainer, self.model, predictions,
                predictions, 0, None
            )
            self.csv_callback.on_predict_end(
                DummyTrainer, self.model
            )

        return predictions


class MultiCategoryTorchInferencer(TorchInferencer):
    """
    Do inference for multiple products
    
    Args:
        config_paths: model config file paths
        inference_save_path: custom directory to save the results
    """

    def __init__(
        self,
        configs: Iterable[Union[DictConfig, ListConfig]],
        device: Literal["cuda", "cpu"] = "cuda",
        inference_save_path: Optional[Union[str, Path]] = None,
    ):
        self.initialized = False
        self.meta_datas = {}
        self.trained_datas = {}
        self.metrics_callbacks: Dict[str, MetricsConfigurationCallback] = {}
        self.visualizer_callbacks: Dict[str, VisualizerCallback] = {}
        self.csv_callbacks: Dict[str, CSVMetricsLoggerCallback] = {}
        self.inference_save_path = inference_save_path
        self.device = torch.device(device)

        self.main_configs_list = list(configs)
        # if no config_path given, don't initialize
        if self.main_configs_list:
            self._initialize()

    def _initialize(self):
        """
        call all functions that loads needed data
        """
        self.config = generate_multi_inferencer_config(
            configs=self.main_configs_list
        )
        self.init_model()

        self.input_size = self.config.model.input_size
        self._init_pre_processor()

        self._load_categories()
        self.initialized = True

    def init_model(self):
        """
        Initialize the model
        """
        model_name = self.config.model.name
        if model_name != "patchcore":
            ValueError(
                "Multi Category Inference is only supported for "
                f"patchcore model. But config for {model_name} was given"
            )
        self.model = get_model(self.config)
        self.model.eval()
        self.model = self.model.to(self.device)

    def add_category_config(
        self,
        config: Union[DictConfig, ListConfig],
    ):
        """
        add a new category and load it's data
        """
        category = config.dataset.category

        configs_list = self.main_configs_list.copy()
        configs_list.append(config)
        self.config = generate_multi_inferencer_config(
            configs=configs_list
        )
        self.main_configs_list = configs_list

        # initialize if this is the first config
        if not self.initialized:
            self._initialize()

        self._load_category_datas(
            category=category,
            category_config=self.config.categories[category]
        )

    def remove_category(self, category: str):
        """
        remove a category from instance
        """
        del self.config.categories[category]
        del self.meta_datas[category]
        del self.trained_datas[category]
        del self.metrics_callbacks[category]
        del self.visualizer_callbacks[category]
        del self.csv_callbacks[category]

    def _load_categories(self) -> None:
        """
        load all categories at start from config
        """
        for category, category_config in self.config.categories.items():
            self._load_category_datas(
                category=category,
                category_config=category_config,
            )

    def _load_category_datas(
        self,
        category: str,
        category_config: Union[DictConfig, ListConfig],
    ):
        """
        load all the data of a single category
        """
        self._load_category_trained_data(
            category=category,
            category_config=category_config,
        )
        if self.inference_save_path is not None:
            category_config.project.save_root = self.inference_save_path
            category_config.project.inference_dir_name = category

        self._set_category_callbacks(
            category=category,
            category_config=category_config,
        )

    def _load_category_trained_data(
        self,
        category: str,
        category_config: Union[DictConfig, ListConfig],
    ):
        """
        load category meta datas and trained_data.hmc
        """
        trained_data_keys = self.model.get_trained_data_keys()
        save_root = category_config.project.save_root
        weight_file = category_config.model.weight_file
        trained_data_path = Path(save_root) / Path(weight_file)
        trained_data = torch.load(
            trained_data_path,
            map_location=self.device
        )

        cached_trained_data = {}
        for data_key in trained_data_keys:
            cached_trained_data[data_key] = trained_data[data_key]

        self._load_category_meta_data(trained_data, category)
        self.trained_datas[category] = cached_trained_data

    def _set_category_callbacks(
        self,
        category: str,
        category_config: Union[DictConfig, ListConfig],
    ):
        """
        add callbacks of a category using category config
        """
        category_config.model.normalization_method = (
            self.config.model.normalization_method
        )
        callbacks = get_callbacks(config=category_config)
        for callback in callbacks:
            if isinstance(callback, MetricsConfigurationCallback):
                callback.setup(DummyTrainer, self.model)
                self.metrics_callbacks[category] = callback
            if isinstance(callback, VisualizerCallback):
                self.visualizer_callbacks[category] = callback
            if isinstance(callback, CSVMetricsLoggerCallback):
                callback.on_predict_start(DummyTrainer, self.model)
                self.csv_callbacks[category] = callback

    def _load_category_meta_data(
        self,
        data: Dict[str, Tensor],
        category: str,
    ):
        """
        load meta data for each category
        """
        cached_meta_data = {
            "min": data["min_max.min"],
            "max": data["min_max.max"],
            "image_threshold": data["image_threshold.value"],
            "pixel_threshold": data["pixel_threshold.value"],
            "image_mean": data["training_distribution.image_mean"],
            "image_std": data["training_distribution.image_std"],
            "pixel_mean": data["training_distribution.pixel_mean"],
            "pixel_std": data["training_distribution.pixel_std"],
        }
        self.meta_datas[category] = get_model_metadata(
            self.model,
            cached_meta_data
        )

    def warmup(self):
        """
        Allocate gpu or cpu memory to make the first prediction faster
        than usual. This should be called right after creating the instance
        """
        if not self.initialized:
            return

        img = torch.empty(
            (1, 3, *self.input_size),
            dtype=torch.float32,
            device=self.device
        )
        self.forward(
            {"image": img},
            self.trained_datas[list(self.trained_datas.keys())[0]]
        )

    def forward(
        self,
        batch: Dict[str, Tensor],
        trained_data: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        return self.model.validation_step(batch, 0, trained_data)

    def post_process(
        self,
        predictions: Dict[str, Tensor],
        meta_data: Optional[Union[Dict, DictConfig]] = None
    ) -> Tuple[np.ndarray, float]:
        return super().post_process(predictions, meta_data)

    def predict(
        self,
        image: Union[str, np.ndarray, Path],
        category: str,
    ) -> Tuple[np.ndarray, float]:

        item = {}
        meta_data = self.meta_datas[category]
        self.metrics_callback = self.metrics_callbacks[category]
        self.visualizer_callback = self.visualizer_callbacks[category]
        self.csv_callback = self.csv_callbacks[category]
        self._set_thresholds(self.config.categories[category])

        if isinstance(image, (str, Path)):
            item["image_path"] = [image]
            image_arr: np.ndarray = read_image(image)
        else:  # image is already a numpy array. Kept for mypy compatibility.
            image_arr = image

        pre_processed = self.pre_process(image_arr)
        item.update(pre_processed)
        predictions = self.forward(item, self.trained_datas[category])
        predictions = self.post_process(predictions, meta_data=meta_data)

        return predictions
