
import os
import base64

import cv2
import torch
import numpy as np

from typing import Dict, List, Union

from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler
from torch import Tensor

from hamacho.core.deploy import TorchInferencer
from hamacho.core.config import get_configurable_parameters
from hamacho.core.utils.callbacks import (
    get_callbacks,
    MetricsConfigurationCallback
)


# https://github.com/pytorch/serve/blob/master/docs/custom_service.md#custom-handler-with-class-level-entry-point
class TorchServeHandler(BaseHandler):
    """
    A custom torchserve handler class
    """

    def __init__(self) -> None:
        self._context = None
        self.initialized = False
        self.inferencer = None
        self.device = None

    def initialize(self, context: Context) -> None:
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """
        print(context.__dict__)
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

        self.config = get_configurable_parameters(
            config_path=model_config_path
        )

        callbacks = get_callbacks(config=self.config)

        for callback in callbacks:
            if isinstance(callback, MetricsConfigurationCallback):
                metrics_callback = callback

        self.inferencer = TorchInferencer(
            config=self.config,
            model_source=trained_data_path,
            device=self.device,
            metrics_callback=metrics_callback,
        )
        self.inferencer.warmup()

        self.initialized = True

    def preprocess(
        self,
        data: List[Dict[str, Union[str, bytes, bytearray]]],
    ) -> Dict[str, Tensor]:
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytes or bytesarray
            if isinstance(image, bytes):
                image = np.fromstring(image, dtype=np.uint8)
            elif isinstance(image, bytearray):
                image = np.frombuffer(image, dtype=np.uint8)

            # do pre processing
            if isinstance(image, np.ndarray):
                image = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
                image = self.inferencer.pre_process(image)["image"]
                image = image.squeeze()
            elif isinstance(image, list):
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        print("BATCH-SIZE", len(images))

        return {"image": torch.stack(images).to(self.device)}

    def handle(
        self,
        data: List[Dict[str, Union[str, bytes, bytearray]]],
        _context: Context
    ) -> List[Dict[str, float]]:
        """
        Invoked by TorchServe for prediction request.
        Does pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param _context: Initial context contains model server system properties.
        :return: prediction output
        """
        images = self.preprocess(data)
        pred = self.inferencer.forward(images)
        return self.postprocess(pred)

    def postprocess(
        self,
        predictions: Dict[str, Tensor]
    ) -> List[Dict[str, float]]:
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        pred = self.inferencer.post_process(predictions)
        results = []
        for i, (pred_score, pred_score_denormalized) in enumerate(
            zip(
                pred["pred_scores"],
                pred["pred_scores_denormalized"],
            )
        ):
            result = {
                "pred_score": float(pred_score_denormalized),
                "image_threshold": float(pred["image_threshold"]),
            }
            if "image_threshold_norm" in pred:
                result["pred_score_norm"] = float(pred_score)
                result["image_threshold_norm"] = float(
                    pred["image_threshold_norm"]
                )

            results.append(result)

        return results
