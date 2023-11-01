"""Base Inferencer for Torch and OpenVINO."""


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from hamacho.core.data.utils import read_image
from hamacho.core.post_processing.normalization.cdf import \
    normalize as normalize_cdf
from hamacho.core.post_processing.normalization.cdf import standardize
from hamacho.core.post_processing.normalization.min_max import \
    normalize as normalize_min_max


class Inferencer(ABC):
    """Abstract class for the inference.

    This is used by both Torch and OpenVINO inference.
    """

    @abstractmethod
    def load_model(self, path: Union[str, Path]):
        """Load Model."""
        raise NotImplementedError

    @abstractmethod
    def pre_process(self, image: np.ndarray) -> Union[np.ndarray, Tensor]:
        """Pre-process."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, image: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        """Forward-Pass input to model."""
        raise NotImplementedError

    @abstractmethod
    def post_process(
        self, predictions: Union[np.ndarray, Tensor], meta_data: Optional[Dict]
    ) -> Tuple[np.ndarray, float]:
        """Post-Process."""
        raise NotImplementedError

    def predict(
        self,
        image: Union[str, np.ndarray, Path],
        meta_data: Optional[dict] = None,
    ) -> Tuple[np.ndarray, float]:
        """Perform a prediction for a given input image.

        The main workflow is (i) pre-processing, (ii) forward-pass, (iii) post-process.

        Args:
            image (Union[str, np.ndarray]): Input image whose output is to be predicted.
                It could be either a path to image or numpy array itself.

            superimpose (bool): If this is set to True, output predictions
                will be superimposed onto the original image. If false, `predict`
                method will return the raw heatmap.

            overlay_mask (bool): If this is set to True, output segmentation mask on top of image.

        Returns:
            np.ndarray: Output predictions to be visualized.
        """

        item = {}
        if meta_data is None:
            if hasattr(self, "meta_data"):
                meta_data = getattr(self, "meta_data")
            else:
                meta_data = {}
        if isinstance(image, (str, Path)):
            item["image_path"] = [image]
            image_arr: np.ndarray = read_image(image)
        else:  # image is already a numpy array. Kept for mypy compatibility.
            image_arr = image
        meta_data["image_shape"] = image_arr.shape[:2]

        pre_processed = self.pre_process(image_arr)
        item.update(pre_processed)
        predictions = self.forward(item)
        predictions = self.post_process(predictions, meta_data=meta_data)

        return predictions

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Call predict on the Image.

        Args:
            image (np.ndarray): Input Image

        Returns:
            np.ndarray: Output predictions to be visualized
        """
        return self.predict(image)

    def _normalize(
        self,
        anomaly_maps: Union[Tensor, np.ndarray],
        pred_scores: Union[Tensor, np.float32],
        meta_data: Union[Dict, DictConfig],
    ) -> Tuple[Union[np.ndarray, Tensor], float]:
        """Applies normalization and resizes the image.

        Args:
            anomaly_maps (Union[Tensor, np.ndarray]): Predicted raw anomaly map.
            pred_scores (Union[Tensor, np.float32]): Predicted anomaly score
            meta_data (Dict): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.

        Returns:
            Tuple[Union[np.ndarray, Tensor], float]: Post processed predictions that are ready to be visualized and
                predicted scores.


        """

        # min max normalization
        if "min" in meta_data and "max" in meta_data:
            anomaly_maps = normalize_min_max(
                anomaly_maps,
                meta_data["pixel_threshold"],
                meta_data["min"],
                meta_data["max"],
            )
            pred_scores = normalize_min_max(
                pred_scores,
                meta_data["image_threshold"],
                meta_data["min"],
                meta_data["max"],
            )

        # standardize pixel scores
        if "pixel_mean" in meta_data.keys() and "pixel_std" in meta_data.keys():
            anomaly_maps = standardize(
                anomaly_maps,
                meta_data["pixel_mean"],
                meta_data["pixel_std"],
                center_at=meta_data["image_mean"],
            )
            anomaly_maps = normalize_cdf(anomaly_maps, meta_data["pixel_threshold"])

        # standardize image scores
        if "image_mean" in meta_data.keys() and "image_std" in meta_data.keys():
            pred_scores = standardize(
                pred_scores, meta_data["image_mean"], meta_data["image_std"]
            )
            pred_scores = normalize_cdf(pred_scores, meta_data["image_threshold"])

        return anomaly_maps, pred_scores

    def _load_meta_data(
        self, path: Optional[Union[str, Path]] = None
    ) -> Union[DictConfig, Dict[str, Union[float, np.ndarray, Tensor]]]:
        """Loads the meta data from the given path.

        Args:
            path (Optional[Union[str, Path]], optional): Path to JSON file containing the metadata.
                If no path is provided, it returns an empty dict. Defaults to None.

        Returns:
            Union[DictConfig, Dict]: Dictionary containing the metadata.
        """
        meta_data: Union[DictConfig, Dict[str, Union[float, np.ndarray, Tensor]]] = {}
        if path is not None:
            config = OmegaConf.load(path)
            meta_data = cast(DictConfig, config)
        return meta_data
