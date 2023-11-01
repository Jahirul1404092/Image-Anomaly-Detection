import os
from typing import Dict, List
from pathlib import Path

from torch import Tensor
import torch

from omegaconf import OmegaConf

from ts.context import Context

from hamacho.core.deploy import TorchInferencer

from hamacho.core.deploy.model_serving.torchserve_handler import TorchServeHandler
from hamacho.core.utils.general import DummyTrainer
from hamacho.core.utils.callbacks import (
    get_callbacks,
    VisualizerCallback,
    MetricsConfigurationCallback,
    CSVMetricsLoggerCallback
)

class TsHandler(TorchServeHandler):
    serve_folder = ''

    def initialize(self, context: Context) -> None:
        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        print(properties)

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        trained_data_path = os.path.join(model_dir, serialized_file)
        model_config_path = os.path.join(model_dir, "config.yaml")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() else "cpu"
        )

        self.config = OmegaConf.load(model_config_path)
        ### get_configurable_parameters manipulates output path, 
        ### which eventually outputs unexpected folders 
        # self.config = get_configurable_parameters(
        #     config_path=model_config_path
        # )

        callbacks = get_callbacks(config=self.config)

        for callback in callbacks:
            if isinstance(callback, MetricsConfigurationCallback):
                metrics_callback = callback
            if isinstance(callback, VisualizerCallback):
                visualizer_callback = callback
            if isinstance(callback, CSVMetricsLoggerCallback):
                csv_callback = callback

        self.inferencer = TorchInferencer(
            config=self.config,
            model_source=trained_data_path,
            device=self.device,
            metrics_callback=metrics_callback,
            visualizer_callback=visualizer_callback,
            csv_callback=csv_callback
        )
        self.inferencer.warmup()

        self.initialized = True

    def postprocess(self, predictions: Dict[str, Tensor]) -> List[Dict[str, float]]:
        if "image_path" not in predictions:
            predictions['image_path'] = [f'{self.serve_folder}tmp_name.png']

        if self.inferencer.visualizer_callback is not None:
            self.inferencer.visualizer_callback.on_predict_batch_end(
                DummyTrainer, self.inferencer.model, predictions,
                predictions, 0, None
            )

        predictions = super().postprocess(predictions)
        return predictions
