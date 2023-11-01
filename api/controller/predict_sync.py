import shutil
import logging
logger = logging.getLogger(name=__name__)
import json
from pathlib import Path
from collections import defaultdict
from threading import Lock
from typing import Union, Tuple, Dict, Optional

import torch
import numpy as np
import pandas as pd

from flask import request
from flask_restful import Resource
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from omegaconf import OmegaConf, DictConfig
from jsonschema import validate, ValidationError
from torch import Tensor

from api.util.dbtool import DbTool
from api.model.schemas import Schemas
from api.util.const import win2posix
from api.util.verifier import proxy_license_verifier
from hamacho.core.deploy.inferencers import MultiCategoryTorchInferencer
from hamacho.core.post_processing import parse_single_result
from hamacho.core.data.utils import read_image

class SyncInferencer(MultiCategoryTorchInferencer):
    serve_id = ''
    serve_folder = ''

    def set_serve_id(self, sid: int):
        self.serve_id = sid
        self.serve_folder = f"serve_id_{self.serve_id}/"
        return self.serve_folder

    def post_process(self, predictions: Dict[str, Tensor], meta_data: Optional[Union[Dict, DictConfig]] = None) -> Tuple[np.ndarray, float]:
        # no need to append parent directory. no good to too dependent on local directory name
        if 'image_path' in predictions:
            for i in range(len(predictions['image_path'])):
                predictions['image_path'][i] = f"{self.serve_folder}{Path(predictions['image_path'][i]).name}"
        return super().post_process(predictions, meta_data)

class SyncPredictManager(Resource):
    method_decorators = [proxy_license_verifier]

    d_predictor: Dict[str, SyncInferencer] = dict()
    d_config = defaultdict(dict)
    lock = Lock()

    def do_csv_metrics_backup(self, results_path: Path):
        self.do_csv_merge(results_path)
        main_path = results_path / "metrics" / "pred_outputs.csv"
        backup_path = results_path / "metrics" / "pred_outputs.bak.csv"
        if main_path.exists() and not backup_path.exists():
            shutil.copyfile(main_path, backup_path)

    def do_csv_merge(self, results_path: Path):
        main_path = results_path / "metrics" / "pred_outputs.csv"
        backup_path = results_path / "metrics" / "pred_outputs.bak.csv"
        if not (main_path.exists() and backup_path.exists()):
            return

        m_df = pd.read_csv(main_path, index_col=[0])
        b_df = pd.read_csv(backup_path, index_col=[0])

        f_df = pd.concat((b_df, m_df), ignore_index=True)
        f_df.to_csv(main_path, encoding='utf-8_sig')

        backup_path.unlink()

    def post(self):
        if request.path.endswith('/servemodel'):
            try:
                # parse request
                model_id = request.json.get('model_id', None)
                tag = request.json.get('tag', None)
                mode = request.json.get('mode', 'batch')
                if (model_id and tag) or not (model_id or tag):
                    return 'Invalid Input', 400
                # get model info
                with DbTool() as db:
                    model = db.getModels(model_id or tag)
                if len(model) == 0:
                    return 'Model not found', 404
                model_id = model[0]['model_id']
                model_path = Path(model[0]['model_path'])
                # get config
                config = OmegaConf.load(model_path / 'config.yaml')
                if 'parameters' in request.json:
                    config = OmegaConf.merge(config, request.json['parameters'])
                model_type = config.model.name
                mode_folder = ''
                if mode == 'batch':
                    mode_folder = 'batch_mode'
                elif mode == 'online':
                    mode_folder = 'online_mode'
                out_path = (
                    Path(config.project.path) / 
                    config.project.inference_dir_name /
                    mode_folder
                )
                config.project.inference_dir_name = f"{config.project.inference_dir_name}/{mode_folder}"
                SyncPredictManager.d_config[model_type][model_id] = config
                # serve inferencer
                with SyncPredictManager.lock:
                    if model_type in SyncPredictManager.d_predictor:
                        SyncPredictManager.d_predictor[model_type].add_category_config(config)
                    else:
                        accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
                        SyncPredictManager.d_predictor[model_type] = SyncInferencer(
                            [config],
                            torch.device(accelerator)
                        )
                        SyncPredictManager.d_predictor[model_type].warmup()

                self.do_csv_metrics_backup(out_path)

                with DbTool() as db:
                    db.updateInferencePath(model_id, str(out_path))
                    serve_id = db.addServing(model_id, request.json.get('parameters', None), mode)
                serve_folder = SyncPredictManager.d_predictor[model_type].set_serve_id(serve_id)
                config.project.serve_id = serve_id
                config.project.serve_folder = serve_folder
                SyncPredictManager.d_config[model_type][model_id] = config
                logger.debug(f'model {model_id} served')
                return 'OK', 200
            except BadRequest as e:
                logger.error(str(e), exc_info=True)
                return 'invalid input', e.code
            except Exception as e:
                logger.error(str(e), exc_info=True)
                return 'application error', 500
        elif request.path.endswith('/predict'):
            as_file = request.args.get('as_file', default=False, type=lambda a: a.lower() == 'true')
            try:
                if not as_file:
                    # parse request
                    json_data = request.json
                    l_paths = request.json['image_paths']
                    validate(json_data, Schemas.predict_schema)
                else:
                    json_data: dict = json.loads(request.form['json'])
                    l_paths = []
                    validate(json_data, Schemas.predict_as_file_schema)

                save_mode = json_data.get('save', 'none')
                model_id_or_tag = json_data.get('model_id', None) or json_data.get('tag', None)
                # get model info
                with DbTool() as db:
                    model = db.getModels(model_id_or_tag)
                if len(model) == 0:
                    return 'Model not found', 404
                model_id = model[0]['model_id']
                model_type = model[0]['model_type']
                inference_path = model[0]['inference_path']
                if as_file:
                    serve_folder = SyncPredictManager.d_config[model_type][model_id].project.serve_folder
                    input_save_dir = Path(inference_path) / "images" / "input_image" / serve_folder
                    input_save_dir.mkdir(parents=True, exist_ok=True)
                    for img in request.files.getlist('images'):
                        img_dir = f'{input_save_dir}/{secure_filename(Path(img.filename).name)}'
                        img.stream.seek(0)
                        img.save(img_dir)
                        l_paths.append(img_dir)
                # check inferencer and config
                if model_type not in SyncPredictManager.d_predictor:
                    logger.error(f'Model {model_type} not served yet')
                    return f'Model {model_type} not served yet', 404
                if model_id not in SyncPredictManager.d_config.get(model_type, None):
                    logger.error(f'Model id {model_id} not served yet')
                    return f'Model id {model_id} not served yet', 404
                l_result = []
                # infer
                with SyncPredictManager.lock:
                    for path in l_paths:
                        localpath = win2posix(path)
                        pred = SyncPredictManager.d_predictor[model_type].predict(localpath, str(model_id))
                        result_path = Path(inference_path) / "images" / "prediction" / pred['image_path'][0]
                        input_path = Path(inference_path) / "images" / "input_image" / pred['image_path'][0]
                        csv_path = (Path(inference_path) / "csv" / pred['image_path'][0]).with_suffix('.csv')
                        result = parse_single_result(pred=pred, return_normalized=True)
                        result['image_path'] = str(input_path)
                        result['result_path'] = str(result_path)
                        l_result.append(result)

                        save_to_db = False
                        if save_mode == "all":
                            save_to_db = True
                        elif save_mode == "ok_only" and result['pred_score_norm'] <= result['image_threshold_norm']:
                            save_to_db = True
                        elif save_mode == "ng_only" and result['pred_score_norm'] >= result['image_threshold_norm']:
                            save_to_db = True

                        if save_to_db:
                            with DbTool() as db:
                                db.addInference(model_id, str(input_path), result, str(result_path), str(csv_path))

                        result['saved'] = save_to_db

                return l_result, 200
            except ValidationError as e:
                logger.error(str(e), exc_info=True)
                return 'Invalid Input', 400
            except BadRequest as e:
                logger.error(str(e), exc_info=True)
                return 'invalid input', e.code
            except Exception as e:
                logger.error(str(e), exc_info=True)
                return 'application error', 500
        else:
            return 'method not allowed', 405

    def delete(self):
        if not request.path.endswith('/unservemodel'):
            return 'method not allowed', 405
        try:
            # parse request
            model_id = request.args.get('model_id', None, type=int)
            tag = request.args.get('tag', None, type=str)
            if (model_id and tag) or not (model_id or tag):
                return 'Invalid Input', 400
            # get model info
            with DbTool() as db:
                model = db.getModels(model_id or tag)
            if len(model) == 0:
                logger.error('Model not found')
                return 'Model not found', 404
            model_id = model[0]['model_id']
            model_type = model[0]['model_type']
            # check and unserve inferencer
            if model_id not in SyncPredictManager.d_config[model[0]['model_type']]:
                logger.error('Model not served yet')
                return 'Model not found', 404

            config = SyncPredictManager.d_config[model_type][model_id]
            self.do_csv_merge(Path(config.project.path) / config.project.inference_dir_name)
            del SyncPredictManager.d_config[model_type][model_id]

            if len(SyncPredictManager.d_config[model_type]) > 0:
                with SyncPredictManager.lock:
                    SyncPredictManager.d_predictor[model_type].remove_category(str(model_id))
            else:
                del SyncPredictManager.d_predictor[model_type]
            logger.debug(f'model {model_id} unserved')
            return 'OK', 200
        except BadRequest as e:
            logger.error(str(e), exc_info=True)
            return 'invalid input', e.code
        except Exception as e:
            logger.error(str(e), exc_info=True)
            return 'application error', 500
