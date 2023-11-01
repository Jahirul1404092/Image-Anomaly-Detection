"""Test Multi Category Torch Inferencer"""

import os
import shutil

from tempfile import TemporaryDirectory
from typing import Tuple

import cv2
import pytest
import torch

from omegaconf import DictConfig, OmegaConf

from hamacho.core.deploy import MultiCategoryTorchInferencer
from tests.helpers.dataset import TestDataset
from tests.helpers.model import setup_model_train


def setup_category(
    category: str,
    config: DictConfig,
    project_path: str,
    model: str,
    weights_file: str = "trained_data.hmc"
) -> Tuple[str, str]:
    category_path = os.path.join(
        project_path,
        "results",
        category,
        model,
    )
    os.makedirs(category_path)
    config = config.copy()
    config.dataset.category = category
    config.project.path = category_path
    config.project.save_root = category_path
    config.project.log_images_to = ["local"]
    config_path = os.path.join(
        category_path,
        "config.yaml"
    )
    OmegaConf.save(
        config,
        config_path,
    )
    weights_path = os.path.join(
        category_path,
        weights_file,
    )
    shutil.copy2(
        os.path.join(
            project_path,
            weights_file,
        ),
        weights_path
    )

    return config, config_path, category_path


class TestMultiCatTorchInferencer:

    default_save_path = "inference_results"
    save_image_dir_name = "images"
    images_parent_dir_name = "combined"
    image_type = "good"

    @pytest.mark.parametrize(
        (
            "model_name", "device", "inference_dir", "category_count"
        ),
        (
            ("patchcore", "cpu", None, 3),
            ("patchcore", "cuda", "infer_res", 2),
        )
    )
    @TestDataset(num_train=10, num_test=3)
    def test_inference(
        self,
        model_name: str,
        device: str,
        inference_dir: str,
        category_count: int,
        category="shapes",
        path: str = "./data",
        weight_file: str = "trained_data.hmc",
    ):
        """
        Test the `MultiCategoryTorchInferencer`
        Trains two categories with the Patchcore model
        and tests if inference is working as expected
        """
        pl_device = [0] if device == "cuda" else 1
        # fallback to cpu if cuda not found
        if not torch.cuda.is_available():
            pl_device = 1
            device = "cpu"

        with TemporaryDirectory() as project_path:
            config, *_ = setup_model_train(
                model_name,
                dataset_path=path,
                project_path=project_path,
                nncf=False,
                category=category,
                fast_run=False,
                weight_file=weight_file,
                accelerator="gpu" if device == "cuda" else "cpu",
                device=pl_device,
            )
            # if custom inference directory given
            if inference_dir is not None:
                self.default_save_path = ""
                inference_dir = os.path.join(
                    project_path, inference_dir
                )

            configs = []
            image_paths = []
            category_paths = []
            categories = []
            # prepare the categories
            for i in range(category_count):
                category_name = f"{category}_{i}"
                config, _, category_path = setup_category(
                    category_name, config, project_path, model_name
                )
                categories.append(category_name)
                configs.append(config)
                if inference_dir is not None:
                    category_path = os.path.join(
                        project_path, inference_dir, category_name
                    )
                category_paths.append(category_path)
                image_path = os.path.join(
                    path,
                    category,
                    "test",
                    self.image_type,
                    f"00{i}.png",
                )
                image_paths.append(image_path)

            inferencer = MultiCategoryTorchInferencer(
                configs=configs,
                device=device,
                inference_save_path=inference_dir,
            )

            for category_name, image_path, category_path in zip(
                categories, image_paths, category_paths
            ):
                pred = inferencer.predict(
                    image=image_path,
                    category=category_name,
                )
                out_save_path = os.path.join(
                    category_path,
                    self.default_save_path,
                    self.save_image_dir_name,
                    self.images_parent_dir_name,
                    self.image_type,
                    os.path.basename(image_path),
                )
                assert os.path.exists(out_save_path)
                assert "pred_scores" in pred
                assert "anomaly_maps" in pred

            # test add_category_config function
            another_category_name = f"{category}_a"
            another_config, _, _ = setup_category(
                another_category_name, config, project_path, model_name
            )
            inferencer.add_category_config(another_config)
            image = cv2.imread(image_paths[0])
            pred = inferencer.predict(
                image=image,
                category=another_category_name,
            )
            assert "pred_scores" in pred
            assert "anomaly_maps" in pred

            # test removal of category
            inferencer.remove_category(another_category_name)
            with pytest.raises(KeyError):
                inferencer.predict(
                    image=image,
                    category=another_category_name,
                )
