
import os

import pytest

from tempfile import TemporaryDirectory

from ts.context import Context
from pytorch_lightning import Trainer
from omegaconf import OmegaConf

from hamacho.core.deploy import TorchServeHandler
from hamacho.core.data import get_datamodule
from hamacho.core.utils.callbacks import get_callbacks
from hamacho.plug_in.models import get_model
from tests.helpers.dataset import TestDataset
from tests.helpers.config import get_model_config


class TestHandler:
    @pytest.mark.parametrize(
        "model_name",
        ["padim", "patchcore"],
    )
    @TestDataset(num_train=10, num_test=1, use_mvtec=False)
    def test_handler(
        self,
        model_name: str,
        category: str = "shapes",
        path: str = "./datasets/MVTec",
    ):
        """Tests TorchServe handler.
        Args:
            model_name (str): Name of the model
        """
        dataset_path = path
        with TemporaryDirectory() as project_path:
            model_config = get_model_config(
                model_name=model_name,
                dataset_path=dataset_path,
                category=category,
                project_path=project_path,
                accelerator="auto",
            )
            model_config.model.weight_file = "trained_data.hmc"
            model_config.project.pop("save_outputs")
            OmegaConf.save(
                model_config,
                os.path.join(project_path, "config.yaml")
            )

            model = get_model(model_config)
            datamodule = get_datamodule(model_config)
            callbacks = get_callbacks(model_config)
            trainer = Trainer(
                **model_config.trainer,
                logger=False,
                callbacks=callbacks
            )

            trainer.fit(model=model, datamodule=datamodule)
            model.eval()

            model_dir = project_path
            dummy_manifest = {
                "createdOn": "DD/MM/YYYY HH:MM:SS", # not needed
                "runtime": "python",
                "model": {
                    "modelName": category,
                    "serializedFile": "trained_data.hmc",
                    "handler": "torchserve_handler.py",
                    "modelVersion": "0.1"
                },
                "archiverVersion": "0.7.0"
            }
            handler = TorchServeHandler()
            context = Context(
                model_name=model_name,
                model_dir=model_dir,
                manifest=dummy_manifest,
                batch_size=1,
                mms_version="",
                gpu=0,
            )
            handler.initialize(context)
            image_path = os.path.join(
                dataset_path,
                category,
                "test",
                "good",
                "000.png"
            )
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            # 2 image in batch
            datas = [
                {"data": image_bytes},
                {"data": image_bytes},
            ]
            preds = handler.handle(datas, context)

            assert len(preds) == 2
            assert "pred_score" in preds[0]
