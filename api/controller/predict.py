import os
import logging
logger = logging.getLogger(name=__name__)
from pathlib import Path

from flask import Blueprint, request
from flask_restful import Api, Resource

from api.controller.predict_sync import SyncPredictManager
from api.controller.predict_async import AsyncPredictManager
from api.util.const import Const
from api.util.dbtool import DbTool
from api.util.verifier import proxy_license_verifier


app = Blueprint('predict', __name__)
api = Api(app)

if os.environ.get('PREDICTION_MODE', 'SYNC') == 'SYNC':
    api.add_resource(SyncPredictManager, '/predict', '/servemodel', '/unservemodel')
else:
    (Path(Const.save_dir) / 'model_store').mkdir(exist_ok=True, parents=True)
    AsyncPredictManager.start_server()
    api.add_resource(AsyncPredictManager, '/predict', '/servemodel', '/unservemodel')

class Inference(Resource):
    method_decorators = [proxy_license_verifier]

    def get(self):
        if request.path.endswith('/listresults'):
            try:
                model_id = request.args.get('model_id', None)
                mode = request.args.get('inference_mode', None)
                with DbTool() as db:
                    if model_id is None:
                        res = db.listInferences()
                        if len(res) == 0:
                            return 'inference not found', 404
                    else:
                        res = db.listModelInferences(model_id, mode)
                return res, 200
            except Exception as e:
                logger.error(str(e))
                return 'application error', 500

        elif request.path.endswith('/getresult'):
            try:
                inf_id = request.args.get('inference_id', 0, type=int)
                with DbTool() as db:
                    res = db.getInference(inf_id)
                    if res == {}:
                        return 'inference_id not found', 404
                return res, 200
            except Exception as e:
                logger.error(str(e))
                return 'application error', 500
        else:
            return 'method not allowed', 405

    def delete(self):
        if request.path.endswith('/delresult'):
            try:
                inf_id = request.args.get('inference_id', 0, type=int)
                with DbTool() as db:
                    res = db.delInference(inf_id)
                    if res:
                        return 'OK', 200
                    else:
                        return 'inference_id not found', 404
            except Exception as e:
                logger.error(str(e))
                return 'application error', 500
        else:
            return 'method not allowed', 405

api.add_resource(Inference, '/listresults', '/getresult', '/delresult')
