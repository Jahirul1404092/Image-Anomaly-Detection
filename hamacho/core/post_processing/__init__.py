"""Methods to help post-process raw model outputs."""


from typing import Dict
from torch import Tensor

from .post_process import add_anomalous_label
from .post_process import add_normal_label
from .post_process import anomaly_map_to_color_map
from .post_process import compute_mask
from .post_process import superimpose_anomaly_map
from .post_process import anomaly_map_to_grayscale
from .visualizer import Visualizer

__all__ = [
    "add_anomalous_label",
    "add_normal_label",
    "anomaly_map_to_color_map",
    "anomaly_map_to_grayscale",
    "superimpose_anomaly_map",
    "compute_mask",
    "Visualizer",
]


def parse_single_result(
    pred: Dict[str, Tensor],
    return_anomaly_map=False,
    return_normalized=False,
):
    result = {
        "pred_score": float(pred["pred_scores_denormalized"]),
        "image_threshold": float(pred["image_threshold"]),
    }
    if return_normalized and "image_threshold_norm" in pred:
        result["pred_score_norm"] = float(pred["pred_scores"])
        result["image_threshold_norm"] = float(pred["image_threshold_norm"])

    if return_anomaly_map:
        result["anomaly_map"] = pred["anomaly_maps"].cpu().numpy()

    return result
