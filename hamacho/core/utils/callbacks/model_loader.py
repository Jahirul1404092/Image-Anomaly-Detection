"""Callback that loads model weights from the state dict."""


import logging

import torch
from pytorch_lightning import Callback

from hamacho.plug_in.models.components import AnomalyModule

logger = logging.getLogger(__name__)


class LoadModelCallback(Callback):
    """Callback that loads the model weights from the state dict."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def on_test_start(
        self, _trainer, pl_module: AnomalyModule
    ) -> None:
        """Call when inference begins.
        Loads the model weights from ``weights_path`` into the PyTorch module.
        """
        self._load_model(pl_module)

    def on_predict_start(
        self, _trainer, pl_module: AnomalyModule
    ) -> None:
        """Call when inference begins.
        Loads the model weights from ``weights_path`` into the PyTorch module.
        """
        self._load_model(pl_module)

    def _load_model(
        self,
        pl_module: AnomalyModule,
    ) -> None:
        """
        Loads the trained_data loacated at weight_path
        """
        logger.info("Loading the model from %s", self.weights_path)
        state_dict = torch.load(self.weights_path, map_location=pl_module.device)
        pl_module.load_trained_data(state_dict)
