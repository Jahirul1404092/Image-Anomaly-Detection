"""Utilities for optimization and OpenVINO conversion."""


import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
from torch import Tensor

from hamacho.plug_in.models.components import AnomalyModule


def get_model_metadata(
    model: AnomalyModule,
    cached_meta_data: Optional[Dict[str, Tensor]] = None
) -> Dict[str, Tensor]:
    """Get meta data related to normalization from model.

    Args:
        model (AnomalyModule): Anomaly model which contains metadata related to normalization.

    Returns:
        Dict[str, Tensor]: metadata
    """
    meta_data = {}
    if cached_meta_data is None:
        cached_meta_data = {
            "image_threshold": model.image_threshold.cpu().value,
            "pixel_threshold": model.pixel_threshold.cpu().value,
            "pixel_mean": model.training_distribution.pixel_mean.cpu(),
            "image_mean": model.training_distribution.image_mean.cpu(),
            "pixel_std": model.training_distribution.pixel_std.cpu(),
            "image_std": model.training_distribution.image_std.cpu(),
            "min": model.min_max.min.cpu(),
            "max": model.min_max.max.cpu(),
        }
    # Remove undefined values by copying in a new dict
    for key, val in cached_meta_data.items():
        if isinstance(val, Tensor) and not torch.isinf(val).all():
            meta_data[key] = val
        elif isinstance(val, np.ndarray) and not np.isinf(val).all():
            meta_data[key] = val

    del cached_meta_data
    return meta_data


def export_convert(
    model: AnomalyModule,
    input_size: Union[List[int], Tuple[int, int]],
    onnx_path: Union[str, Path],
    export_path: Union[str, Path],
):
    """Export the model to onnx format and convert to OpenVINO IR.

    Args:
        model (AnomalyModule): Model to convert.
        input_size (Union[List[int], Tuple[int, int]]): Image size used as the input for onnx converter.
        onnx_path (Union[str, Path]): Path to output onnx model.
        export_path (Union[str, Path]): Path to exported OpenVINO IR.
    """
    height, width = input_size
    torch.onnx.export(
        model.model,
        torch.zeros((1, 3, height, width)).to(model.device),
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )
    optimize_command = (
        "mo --input_model " + str(onnx_path) + " --output_dir " + str(export_path)
    )
    os.system(optimize_command)
    with open(
        Path(export_path) / "meta_data.json", "w", encoding="utf-8"
    ) as metadata_file:
        meta_data = get_model_metadata(model)
        # Convert metadata from torch
        for key, value in meta_data.items():
            if isinstance(value, Tensor):
                meta_data[key] = value.numpy().tolist()
        json.dump(meta_data, metadata_file, ensure_ascii=False, indent=4)
