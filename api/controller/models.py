import os
import subprocess
import logging
logger = logging.getLogger(name=__name__)
import re
from pathlib import Path
import shutil

from flask import Blueprint, request
from flask_restful import Resource, Api, reqparse
from werkzeug.exceptions import BadRequest

from api.util.dbtool import DbTool
from api.util.const import Const
from api.util.verifier import proxy_license_verifier

app = Blueprint('models', __name__)
api = Api(app)

parser = reqparse.RequestParser()

class ModelManager(Resource):
    method_decorators = [proxy_license_verifier]

    def get(self):
        if request.path.endswith('/listmodels'):
            try:
                with DbTool() as db:
                    models = db.getModels()
                if len(models) == 0:
                    return "No Models Found", 404
                res = []
                for m in models:
                    res.append({
                        "model_id": m['model_id'],
                        "tag": m.get('tag', None),
                        "created": m['created'].strftime("%Y/%m/%d %H:%M:%S")
                    })
                logger.debug(f'{len(models)} models selected')
                return res, 200
            except Exception as e:
                logger.error(str(e), exc_info=True)
                return 'application error', 500
        elif request.path.endswith('/modeldetails'):
            try:
                model_id = request.args.get('model_id', 0, type=int)
                tag = request.args.get('tag', "", type=str)
                if (model_id and tag) or not (model_id or tag):
                    return 'Invalid Request', 400
                with DbTool() as db:
                    models = db.getModels(model_id or tag)
                if len(models) == 0:
                    logger.error('Model not found')
                    return 'Model Not Found', 404
                model = models[0]
                ret = {
                    "image_ids": model['image_ids'],
                    "tag": model['tag'],
                    "parameters": model['parameters'],
                    "created": model['created'].strftime("%Y/%m/%d %H:%M:%S")
                }
                logger.debug(f'model {model["model_id"]} selected')
                return ret, 200
            except BadRequest as e:
                logger.error(str(e), exc_info=True)
                return 'invalid input', e.code
            except Exception as e:
                logger.error(str(e), exc_info=True)
                return 'application error', 500
        else:
            return 'method not allowed', 405

    def delete(self):
        # /delmodel api
        if not request.path.endswith('/delmodel'):
            return 'method not allowwd', 405
        try:
            obj = request.json
            if (obj is None or (not isinstance(obj, list) and 
                not isinstance(obj, str) and not isinstance(obj, int))):
                logger.error('Invalid Request')
                return 'Invalid Request', 400
            with DbTool() as db:
                models = db.getModels(obj)
                for mod in models:
                    shutil.rmtree(mod['model_path'])
                # delete real file object
                res = db.delModel(obj)
            if len(res) == 0:
                logger.error('Model Not Found')
                return 'Model Not Found', 404
            logger.debug(f'model(s) {obj} deleted')
            return res, 200
        except BadRequest as e:
            logger.error(str(e), exc_info=True)
            return 'invalid input', e.code
        except Exception as e:
            logger.error(str(e), exc_info=True)
            return 'application error', 500

api.add_resource(ModelManager, '/listmodels', '/modeldetails', '/delmodel')
