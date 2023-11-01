import os
import subprocess
import logging

logger = logging.getLogger(name=__name__)
import re
from pathlib import Path
import shutil
from threading import Lock
import tempfile

from flask import Blueprint, request
from flask_restful import Resource, Api, reqparse
from werkzeug.exceptions import BadRequest
from jsonschema import validate, ValidationError
from pytorch_lightning import Trainer
from omegaconf import OmegaConf

from api.model.schemas import Schemas
from api.util.dbtool import DbTool
from api.util.const import Const, win2posix
from api.util.verifier import proxy_license_verifier

from hamacho.core.data import get_datamodule
from hamacho.core.config import get_configurable_parameters

from hamacho.core.utils.callbacks import (
    get_callbacks,
)
from hamacho.plug_in.models import get_model
from hamacho.core.utils.loggers import get_experiment_logger

app = Blueprint("train", __name__)
api = Api(app)

parser = reqparse.RequestParser()


def train(model_id, good, bad, mask, param=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        conf = OmegaConf.load(Const.config_path)
        conf.dataset.category = str(model_id)
        tmppath = Path(tmpdir) / "config.yaml"
        OmegaConf.save(conf, tmppath)
        config = get_configurable_parameters(
            model_name="patchcore",
            config_path=tmppath,
        )
    config.dataset.format = "filelist"
    config.dataset.l_normal = good
    config.dataset.l_abnormal = bad
    config.dataset.l_abnormal_mask = None if len(mask) == 0 else mask
    config.dataset.mask = None if len(mask) == 0 else "mask"

    if param:
        config = OmegaConf.merge(config, param)

    # if tiling is enabled.
    if config.dataset.tiling.apply:
        config.dataset.tiling.tile_size = 128
        config.dataset.tiling.stride = 128
        config.dataset.tiling.remove_border_count = 0
        config.dataset.tiling.use_random_tiling = False
        config.dataset.tiling.random_tile_count = 16

    (Path(config.project.path) / "weights").mkdir(parents=True, exist_ok=True)
    datamodule = get_datamodule(config)
    model = get_model(config)
    experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
    return config


class TrainManager(Resource):
    method_decorators = [proxy_license_verifier]

    lock = Lock()
    # /train api
    def post(self):
        args = parser.parse_args()
        try:
            # parse request
            validate(request.json, Schemas.train_schema)
            # get image info
            with DbTool() as db:
                if "model_tag" in request.json:
                    prevmodel = db.getModels(request.json["model_tag"])
                    if len(prevmodel) > 0:
                        raise BadRequest("duplicate model tag")
                if "image_tag" in request.json:
                    ds, l_image = db.getDataset(request.json["image_tag"])
                else:
                    l_image = request.json["image_id"]
                    ds, _ = db.getDataset(request.json["image_id"])
            good = [win2posix(x) for x in ds["good"]]
            bad = [win2posix(x) for x in ds["bad"]]
            mask = [win2posix(x) for x in ds["mask"]]
            with DbTool() as db:
                last_id = db.getLastModelId()
                if last_id is None:
                    raise Exception("failed to get last model id")
                current_id = last_id + 1
            with TrainManager.lock:
                # train model
                config = train(
                    current_id, good, bad, mask, request.json.get("parameters", None)
                )
                project_path = Path(config.project.path)
                if (project_path / "weights" / "trained_data.hmc").exists():
                    # copy weights
                    savedir = Path(config.project.save_root)
                    del config.dataset.l_normal
                    del config.dataset.l_abnormal
                    # save config
                    OmegaConf.save(config, savedir / "config.yaml")
                    with DbTool() as db:
                        # insert model path
                        i_id = db.addModel(
                            str(savedir),
                            l_image=l_image,
                            model_type=config.model.name,  # no check, assume patchcore
                            tag=request.json.get("model_tag", None),
                            d_param=request.json.get("parameters", None),
                            version=Const.model_version,
                        )
                        if i_id != current_id:
                            raise AssertionError("assetion error model id")
                    return i_id
                else:
                    logger.error("failed to save weight file")
                    return "failed to save weight file", 500

        except ValidationError as e:
            logger.error(str(e), stack_info=True)
            return "invalid input", 400
        except BadRequest as e:
            logger.error(str(e), exc_info=True)
            return "invalid input", e.code
        except Exception as e:
            logger.error(str(e), stack_info=True)
            return "application error", 500


api.add_resource(TrainManager, "/train")
