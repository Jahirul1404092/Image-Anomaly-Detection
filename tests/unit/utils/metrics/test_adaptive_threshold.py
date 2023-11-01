"""Tests for the adaptive threshold metric."""


import shutil
import random

import pytest
import torch

from hamacho.core.config import get_configurable_parameters
from hamacho.core.utils.metrics import AdaptiveThreshold
from tests.helpers.dataset import TestDataset
from tests.helpers.dirs import create_dir
from tests.helpers.train import run_train_test

DATASET_PATH = "./datasets"
MVTEC_PATH = f"{DATASET_PATH}/MVTec/"


@pytest.fixture(autouse=True)
def setup():
    dataset_created = create_dir(DATASET_PATH)
    mvtec_created = create_dir(MVTEC_PATH)
    yield
    if mvtec_created:
        shutil.rmtree(MVTEC_PATH)
    if dataset_created:
        shutil.rmtree(DATASET_PATH)


@pytest.mark.parametrize(
    ["labels", "preds", "target_threshold"],
    [
        (
            torch.Tensor([0, 0, 0, 1, 1]),
            torch.Tensor([2.3, 1.6, 2.6, 7.9, 3.3]),
            3.3,
        ),  # standard case
        (
            torch.Tensor([1, 0, 0, 0]),
            torch.Tensor([4, 3, 2, 1]),
            4,
        ),  # 100% recall for all thresholds
    ],
)
def test_adaptive_threshold(labels, preds, target_threshold):
    """Test if the adaptive threshold computation returns the desired value."""

    adaptive_threshold = AdaptiveThreshold(default_value=0.5)
    adaptive_threshold.update(preds, labels)
    threshold_value = adaptive_threshold.compute()

    assert threshold_value == target_threshold


@TestDataset(num_train=200, num_test=30, path=DATASET_PATH, seed=42)
def test_non_adaptive_threshold(path=DATASET_PATH, category="shapes"):
    config = get_configurable_parameters(
        config_path="hamacho/plug_in/models/padim/config.yaml"
    )
    config.dataset.path = path
    config.dataset.category = category
    config.model.normalization_method = "none"
    config.metrics.threshold.adaptive = False
    config.trainer.fast_dev_run = True
    config.project.log_images_to = []
    config.metrics.image = ["F1Score"]
    config.metrics.pixel = ["F1Score"]

    image_threshold = random.random()
    pixel_threshold = random.random()
    config.metrics.threshold.image_default = image_threshold
    config.metrics.threshold.pixel_default = pixel_threshold

    (trainer,) = run_train_test(config)
    assert trainer.model.image_metrics.F1Score.threshold == image_threshold
    assert trainer.model.pixel_metrics.F1Score.threshold == pixel_threshold
