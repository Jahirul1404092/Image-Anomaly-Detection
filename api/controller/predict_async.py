import os
import shutil
import subprocess
import logging
logger = logging.getLogger(name=__name__)
import json
import tempfile

from collections import defaultdict
from pathlib import Path
from threading import Lock
from distutils.dir_util import copy_tree, remove_tree
# depricated. Will be removed on python 3.12
# replacement will be setuptool, under development

import requests
import pandas as pd

from flask import request
from flask_restful import Resource
from werkzeug.exceptions import BadRequest
from omegaconf import OmegaConf
from jsonschema import validate, ValidationError

from api.util.dbtool import DbTool
from api.util.const import Const, win2posix
from api.util.verifier import proxy_license_verifier
from api.model.schemas import Schemas
# from api.grpc.torchserve_grpc_client import (
#     infer,
#     register,
#     unregister
# )

class AsyncPredictManager(Resource):
    method_decorators = [proxy_license_verifier]

    proc = None
    d_config = defaultdict(dict)
    lock = Lock()
    @classmethod
    def start_server(cls):
        Path('.torchserve').mkdir(exist_ok=True)
        env = os.environ
        env['PYTHONPATH'] = os.path.abspath(os.path.curdir)
        Path(f'{Const.save_dir}/model_store').mkdir(parents=True, exist_ok=True)
        cls.proc = subprocess.Popen([
            'torchserve',
            '--model-store',
            f'{Const.save_dir}/model_store',
            '--start',
            '--ncs'
        ], stdout=open(".torchserve/stdout.log", 'w'), 
            stderr=open(".torchserve/stderr.log", 'w'),
            env=env)

    @classmethod
    def stop_server(cls):
        if cls.proc:
            subprocess.run(['torchserve', '--stop'])
            cls.proc = None

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
            s_url = "http://localhost:8081"
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
                savedir = Path(model[0]['model_path'])
                mardir = Path(f'{Const.save_dir}/model_store')
                mardir.mkdir(parents=True, exist_ok=True)
                model_path = Path(model[0]['model_path'])
                # get config
                config = OmegaConf.load(model_path / 'config.yaml')
                if 'parameters' in request.json:
                    config = OmegaConf.merge(config, request.json['parameters'])
                model_type = config.model.name
                # need absolute
                config.project.relative_path = config.project.path
                config.project.path = str(Path(config.project.path).absolute())
                config.project.save_root = str(Path(config.project.save_root).absolute())
                mode_folder = ''
                if mode == 'batch':
                    mode_folder = 'batch_modee'
                elif mode == 'online':
                    mode_folder = 'online_modee'
                out_path = (
                    Path(config.project.path) / 
                    config.project.inference_dir_name /
                    mode_folder
                )
                config.project.inference_dir_name = f"{config.project.inference_dir_name}/{mode_folder}"
                AsyncPredictManager.d_config[model_type][model_id] = config
                with DbTool() as db:
                    out_path = Path(config.project.relative_path) / config.project.inference_dir_name
                    db.updateInferencePath(model_id, str(out_path))
                    last_serve_id = db.getLastServeId()

                serve_id = last_serve_id + 1
                config.project.serve_id = serve_id
                serve_folder = f"serve_id_{serve_id}/"
                config.project.serve_folder = serve_folder
                AsyncPredictManager.d_config[model_type][model_id] = config
                # create mar file
                with tempfile.TemporaryDirectory() as tempdir:
                    s_handler = ("from api.util.ts_handler import TsHandler\n"
                    "class TsHandlerImpl(TsHandler):\n"
                    f"    serve_folder = '{serve_folder}'\n"
                    )
                    handler_path = str(Path(tempdir) / 'ts_handler.py')
                    with open(handler_path, 'w') as fp:
                        fp.write(s_handler)
                    OmegaConf.save(config, str(Path(tempdir) / 'config.yaml'))
                    mar_name = f"{config.model.name}_{model_id}"
                    subprocess.run(['torch-model-archiver', '--model-name', mar_name,
                            '--serialized-file', savedir / 'weights/trained_data.hmc',
                            # '--model-file', f'hamacho/plug_in/models/{config.model.name}/lightning_model.py', 
                            '--handler', handler_path,
                            '--extra-files', str(Path(tempdir) / 'config.yaml'), 
                            '--export-path', mardir, '--version', 
                            f'v{Const.model_version}', '--force'])
                # register mar
                # res = register(mar_name, mar_name + '.mar')
                res = requests.post(s_url + f'/models?url={mar_name}.mar', timeout=(10, 60))
                if res.status_code != 200:
                    logger.error(f'model registration error {res.status_code}')
                    return 'model registration error', res.status_code
                self.do_csv_metrics_backup(out_path)
                # scale mar
                res = requests.put(s_url + f'/models/{mar_name}?synchronous=true', 
                                   timeout=(10, 120)) # takes some seconds loading backbone
                if res.status_code != 200:
                    logger.error(f'model scaling error {res.status_code}')
                    return 'model scaling error', res.status_code

                with DbTool() as db:
                    db.addServing(model_id, request.json.get('parameters', None), mode)
                return 'OK', 200
            except BadRequest as e:
                logger.error(str(e), exc_info=True)
                return 'invalid input', e.code
            except Exception as e:
                logger.error(str(e), exc_info=True)
                return 'application error', 500
        elif request.path.endswith('/predict'):
            s_url = "http://localhost:8080"
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

                if len(model) == 0 or model[0].get('inference_path', None) is None:
                    return 'Model not found', 404
                model_id = model[0]['model_id']
                model_type = model[0]['model_type']
                inference_path = Path(model[0]['inference_path'])
                inference_path.mkdir(exist_ok=True, parents=True)
                mar_name = f"{model_type}_{str(model_id)}"
                results = []
                lasterror = None
                serve_folder = AsyncPredictManager.d_config[model_type][model_id].project.serve_folder
                if as_file:
                    input_save_dir = Path(inference_path) / "images" / "input_image" / serve_folder
                    input_save_dir.mkdir(parents=True, exist_ok=True)
                    for img in request.files.getlist('images'):
                        img_dir = f'{input_save_dir}/{img.filename}'
                        img.stream.seek(0)
                        l_paths.append((img_dir, img.stream.read()))
                # https://github.com/pytorch/serve/blob/master/docs/inference_api.md#curl-example didn't work. strange syntax.
                # localfiles={'data': open(re.sub(f"^{host_dir}", base_dir, path), 'rb') for path in request.json['image_paths']}
                # res =requests.post(f"{s_url}/predictions/{mar_name}", files=localfiles)
                for path in l_paths:
                    if as_file:
                        im_data = path[1]
                        path = path[0]
                        localpath = Path(win2posix(path))
                    else:
                        localpath = Path(win2posix(path))
                        with open(localpath, 'rb') as fp:
                            im_data = fp.read()

                    with AsyncPredictManager.lock:
                        # infer
                        # res = infer(mar_name, fp)
                        res = requests.put(f"{s_url}/predictions/{mar_name}", data=im_data, timeout=(10, 60))
                        if res.status_code == 200:
                            input_path = Path(inference_path) / "images" / "input_image" / serve_folder / localpath.name
                            # with open(input_path, 'wb') as f:
                            #     f.write(im_data)
                            result_path = Path(inference_path) / "images" / "prediction" / serve_folder / localpath.name
                            csv_path = (Path(inference_path) / "csv" / serve_folder / localpath.name).with_suffix('.csv')
                            old_csv = csv_path.parent / 'tmp_name.csv'
                            old_csv.rename(csv_path)
                            result = res.json()
                            # rename output file
                            for f in inference_path.glob('**/tmp_name.png'):
                                f.rename(f.parent / (localpath.stem + localpath.suffix))
                            result['image_path'] = str(input_path)
                            result['result_path'] = str(result_path)
                            results.append(result)

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
                        else:
                            logger.error(f'prediction error {res.status_code}')
                            lasterror = res.status_code
                if len(results) == 0:
                    logger.error('no image predicted successfully')
                    return 'prediction error', lasterror
                else:
                    if lasterror:
                        n_failed = len(request.json["image_paths"]) - len(results)
                        logger.warning(f'{n_failed} files failed')
                    else:
                        logger.debug(f'{len(results)} images predicted')
                    return results, 200
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
            s_url = "http://localhost:8081"
            # parse request
            model_id = request.args.get('model_id', None, type=int)
            tag = request.args.get('tag', None, type=str)
            if (model_id and tag) or not (model_id or tag):
                logger.error('Invalid Input')
                return 'Invalid Input', 400
            # get model info
            with DbTool() as db:
                model = db.getModels(model_id or tag)
            if len(model) == 0:
                logger.error('Model not found')
                return 'Model not found', 404
            model_id = model[0]['model_id']
            model_type = model[0]['model_type']
            mar_name = f"{model_type}_{str(model_id)}"
            # unregister
            # res = unregister(mar_name)
            res = requests.delete(s_url + f'/models/{mar_name}')
            if res.status_code != 200:
                logger.error(f'model unregistration error {res.status_code}')
                return 'model unregistration error', res.status_code

            if model_id in AsyncPredictManager.d_config[model_type]:
                config = AsyncPredictManager.d_config[model_type][model_id]
                self.do_csv_merge(Path(config.project.path) / config.project.inference_dir_name)
                del AsyncPredictManager.d_config[model_type][model_id]

            with DbTool() as db:
                model = db.updateInferencePath(model_id, None)
            logger.debug(f'model {mar_name} unregistered')
            return 'OK', 200
        except BadRequest as e:
            logger.error(str(e), exc_info=True)
            return 'invalid input', e.code
        except Exception as e:
            logger.error(str(e), exc_info=True)
            return 'application error', 500
