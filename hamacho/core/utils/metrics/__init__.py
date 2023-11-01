"""Custom anomaly evaluation metrics."""
import importlib
import warnings
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torchmetrics
from omegaconf import DictConfig
from omegaconf import ListConfig

from .adaptive_threshold import AdaptiveThreshold
from .anomaly_score_distribution import AnomalyScoreDistribution
from .auroc import AUROC
from .collection import MetricCollection
from .min_max import MinMax
from .optimal_f1 import OptimalF1

from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryConfusionMatrix
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryPrecision

__all__ = [
    "AUROC",
    "OptimalF1",
    "AdaptiveThreshold",
    "AnomalyScoreDistribution",
    "MinMax",
    "BinaryAccuracy",
    "BinaryPrecision",
    "BinaryRecall",
    "BinaryConfusionMatrix",
]


def get_metrics(
    config: Union[ListConfig, DictConfig]
) -> Tuple[MetricCollection, MetricCollection]:
    """Create metric collections based on the config.

    Args:
        config (Union[DictConfig, ListConfig]): Config.yaml loaded using OmegaConf

    Returns:
        MetricCollection: Image-level metric collection
        MetricCollection: Pixel-level metric collection
    """
    image_metric_names = (
        config.metrics.image if "image" in config.metrics.keys() else []
    )
    pixel_metric_names = (
        config.metrics.pixel if "pixel" in config.metrics.keys() else []
    )
    image_metrics = metric_collection_from_names(image_metric_names, "Image Level ")
    pixel_metrics = metric_collection_from_names(pixel_metric_names, "Pixel Level ")
    return image_metrics, pixel_metrics


def metric_collection_from_names(
    metric_names: List[str], prefix: Optional[str]
) -> MetricCollection:
    """Create a metric collection from a list of metric names.

    The function will first try to retrieve the metric from the metrics defined in the metrics module,
    then in TorchMetrics package.

    Args:
        metric_names (List[str]): List of metric names to be included in the collection.
        prefix (Optional[str]): prefix to assign to the metrics in the collection.

    Returns:
        MetricCollection: Collection of metrics.
    """
    metrics_module = importlib.import_module("hamacho.core.utils.metrics")
    metrics = MetricCollection([], prefix=prefix)
    for metric_name in metric_names:
        if hasattr(metrics_module, metric_name):
            metric_cls = getattr(metrics_module, metric_name)
            metrics.add_metrics(metric_cls())
        elif hasattr(torchmetrics, metric_name):
            try:
                metric_cls = getattr(torchmetrics, metric_name)
                metrics.add_metrics(metric_cls())
            except TypeError:
                warnings.warn(
                    f"Incorrect constructor arguments for {metric_name} metric from TorchMetrics package."
                )
        else:
            warnings.warn(
                f"No metric with name {metric_name} found in hamacho metrics or TorchMetrics."
            )
    return metrics
