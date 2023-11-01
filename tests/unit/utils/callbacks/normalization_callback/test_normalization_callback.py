import shutil

import pytest
import tempfile

from pytorch_lightning import seed_everything

from hamacho.core.config import get_configurable_parameters, update_config
from tests.helpers.dataset import TestDataset
from tests.helpers.dirs import create_dir
from tests.helpers.train import run_train_test
from tests.helpers.inference import get_meta_data


DATASET_PATH = "./datasets"
MVTEC_PATH = f"{DATASET_PATH}/MVTec/"


@pytest.fixture(autouse=True)
def setup_normalizer():
    dataset_created = create_dir(DATASET_PATH)
    mvtec_created = create_dir(MVTEC_PATH)
    yield
    if mvtec_created:
        shutil.rmtree(MVTEC_PATH)
    if dataset_created:
        shutil.rmtree(DATASET_PATH)


@pytest.mark.parametrize(
    ["norm_method", "compare_with_norm_method"],
    [
        ("none", "cdf"),
        ("none", "min_max"),
    ],
)
@TestDataset(num_train=200, num_test=30, path=DATASET_PATH, seed=42)
def test_normalizer(
    norm_method, compare_with_norm_method, path=DATASET_PATH, category="shapes"
):
    config = get_configurable_parameters(
        config_path="hamacho/plug_in/models/padim/config.yaml"
    )
    config.dataset.path = path
    config.dataset.category = category
    config.metrics.threshold.adaptive = True
    config.project.log_images_to = []
    config.metrics.image = ["F1Score", "AUROC"]

    # first method
    config.model.normalization_method = norm_method
    seed_everything(42)
    _, results_first_method = run_train_test(config, run_test=True)

    # second method
    config.model.normalization_method = compare_with_norm_method
    seed_everything(42)
    _, results_second_method = run_train_test(config, run_test=True)

    # performance should be the same
    for metric in ["Image Level AUROC", "Image Level F1Score"]:
        assert round(results_first_method[0][metric], 3) == round(
            results_second_method[0][metric], 3
        )


@TestDataset(
    num_train=80,
    num_test=30,
    path=DATASET_PATH,
    seed=42,
    data_format="folder",
    remove_dir_on_exit=True,
)
def test_sigma6_normalizer(path=DATASET_PATH, model="padim", category="shapes"):
    """Check whether F1-optimized adaptive threshold is override by validation
    data distribution according to sigma6 rule.
    """
    config = get_configurable_parameters(
        config_path="hamacho/plug_in/models/padim/config.yaml"
    )
    with tempfile.TemporaryDirectory() as result_path:
        _, config_path = update_config(
            model,
            result_path,
            dataset_root=path,
            with_mask_label=False,
            task_type="classification",
            image_size=None,
            accelerator="auto",
            data_format="folder",
            category=category,
            batch_size=32,
            split=0.2,
            seed=42,
            num_workers=0,
            no_bad_mode=True,
        )
        config = get_configurable_parameters(
            model_name=model,
            config_path=config_path,
        )

        trainer, _ = run_train_test(config, run_test=True)

        meta_data = get_meta_data(trainer.model, trainer.datamodule.image_size)
        image_threshold = meta_data["image_threshold"].item()
        mean = meta_data["stats"]["image_mean"].item()
        std = meta_data["stats"]["image_std"].item()
        assert round(image_threshold, 2) == round(mean + 3 * std, 2)
