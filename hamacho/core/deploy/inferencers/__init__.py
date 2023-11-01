"""Inferencers for Torch and OpenVINO."""


from .base import Inferencer
from .openvino import OpenVINOInferencer
from .torch import TorchInferencer, MultiCategoryTorchInferencer

__all__ = [
    "Inferencer",
    "TorchInferencer",
    "OpenVINOInferencer",
    "MultiCategoryTorchInferencer",
]
