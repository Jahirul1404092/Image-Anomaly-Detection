"""Tests for Torch and OpenVINO inferencers."""



from tempfile import TemporaryDirectory

import pytest
import torch

from pytorch_lightning import Trainer

from hamacho.core.data import get_datamodule
from hamacho.core.deploy import TorchInferencer
from hamacho.core.utils.callbacks import get_callbacks
from hamacho.plug_in.models import get_model
from tests.helpers.dataset import TestDataset
from tests.helpers.inference import MockImageLoader, get_meta_data
from tests.helpers.config import get_model_config


class TestInferencers:
    @pytest.mark.parametrize(
        [
            "model_name", "accelerator",
        ],
        [
            ("padim", "auto"),
            ("patchcore", "auto"),
            ("padim", "cpu"),
            ("patchcore", "cpu"),
        ],
    )
    @TestDataset(num_train=10, num_test=1, use_mvtec=False)
    def test_torch_inference_model_accelerator(
        self,
        model_name: str,
        accelerator: str,
        category: str = "shapes",
        path: str = "./datasets/MVTec",
    ):
        """Tests Torch inference.
        Model is not trained as this checks that the inferencers are working.
        Args:
            model_name (str): Name of the model
        """
        with TemporaryDirectory() as project_path:
            model_config = get_model_config(
                model_name=model_name,
                dataset_path=path,
                category=category,
                project_path=project_path,
                accelerator=accelerator,
            )
            model_config.model.weight_file = "trained_data.hmc"

            model = get_model(model_config)
            datamodule = get_datamodule(model_config)
            callbacks = get_callbacks(model_config)
            trainer = Trainer(**model_config.trainer, logger=False, callbacks=callbacks)

            trainer.fit(model=model, datamodule=datamodule)
            model.eval()

            # Test torch inferencer
            torch_inferencer = TorchInferencer(model_config, model)
            torch_dataloader = MockImageLoader(
                model_config.dataset.image_size, total_count=1
            )
            meta_data = get_meta_data(model, model_config.dataset.image_size)

            for image in torch_dataloader():
                prediction = torch_inferencer.predict(
                    image, meta_data=meta_data
                )
                assert "pred_scores" in prediction
                assert "anomaly_maps" in prediction

    @pytest.mark.parametrize(
        [
            "model", "accelerator",
        ],
        [
            ("padim", "auto"),
            ("patchcore", "auto"),
        ],
    )
    @TestDataset(num_train=10, num_test=1, use_mvtec=False)
    def test_torch_inference_no_bad_mode(
        self,
        model: str,
        accelerator: str,
        category: str = "shapes",
        path: str = "./datasets/MVTec",
    ):
        """Tests Torch inference.
        Model is not trained as this checks that the inferencers are working.
        """
        with TemporaryDirectory() as project_path:
            model_config = get_model_config(
                model_name=model,
                dataset_path=path,
                category=category,
                project_path=project_path,
                accelerator=accelerator,
            )
            model_config.model.weight_file = "trained_data.hmc"
            model_config.dataset.abnormal_dir = None
            model_config.model.normalization_method = "sigma6"

            model = get_model(model_config)
            datamodule = get_datamodule(model_config)
            callbacks = get_callbacks(model_config)
            trainer = Trainer(**model_config.trainer, logger=False, callbacks=callbacks)

            trainer.fit(model=model, datamodule=datamodule)
            model.eval()

            # Test torch inferencer
            torch_inferencer = TorchInferencer(model_config, model)
            assert "min" in torch_inferencer.meta_data
            assert "max" in torch_inferencer.meta_data
            assert "image_mean" not in torch_inferencer.meta_data
            assert "image_std" not in torch_inferencer.meta_data
            assert "pixel_mean" not in torch_inferencer.meta_data
            assert "pixel_std" not in torch_inferencer.meta_data
            torch_dataloader = MockImageLoader(
                model_config.dataset.image_size, total_count=1
            )
            meta_data = get_meta_data(model, model_config.dataset.image_size)

            for image in torch_dataloader():
                prediction = torch_inferencer.predict(
                    image, meta_data=meta_data
                )
                assert "pred_scores" in prediction
                assert "anomaly_maps" in prediction
