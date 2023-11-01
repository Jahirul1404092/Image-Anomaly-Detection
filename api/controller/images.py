import logging
logger = logging.getLogger(name=__name__)
from flask import Blueprint, request
from flask_restful import Resource, Api, reqparse
from jsonschema import validate, ValidationError
from werkzeug.exceptions import BadRequest

from api.util.dbtool import DbTool
from api.model.schemas import Schemas
from api.util.const import Const
from api.util.verifier import proxy_license_verifier

app = Blueprint('images', __name__)
api = Api(app)

class ImageManager(Resource):
    method_decorators = [proxy_license_verifier]

    def get(self):
        if request.path.endswith('/listimages'):
            try:
                with DbTool() as db:
                    im = db.getTrainingImages()
                    images = [{'image_id':x['image_id'], 
                            'image_path':x['image_path'],
                            'image_tag':x['tag']} for x in im]
                if len(im) == 0:
                    return 'No Images', 404
                return images, 200
            except Exception as e:
                logger.error(str(e), exc_info=True)
                return 'application error', 500
        elif request.path.endswith('/imagedetails'):
            image_id = request.args.get('image_id', 0, type=int)
            image_tag = request.args.get('image_tag', None, type=str)
            with DbTool() as db:
                try:
                    if image_id:
                        if image_tag:
                            return 'cannot specify both id and tag', 400
                        images = db.getTrainingImages(image_id)
                    elif image_tag:
                        images = db.getTrainingImages(image_tag)
                    else:
                        return 'neither id nor tag is specified', 400
                    for i in range(len(images)):
                        del images[i]['created']
                    if len(images) > 0:
                        return images, 200
                    else:
                        return 'Not Found', 404
                except Exception as e:
                    logger.error(str(e), exc_info=True)
                    return 'application error', 500
        else:
            return 'method not allowed', 405

    def post(self):
        # /addimage api
        if not request.path.endswith('/addimage'):
            return 'method not allowed', 405
        try:
            validate(request.json, Schemas.addimage_schema)
            group = 'good' if 'group' not in request.json else request.json['group']
            tag = request.json.get('image_tag', None)
            with DbTool() as db:
                res = db.addTrainingImages(request.json['image_path'], tag, group)
            if len(res) > 0:
                return res
            else:
                logger.warning('unknown error')
                return 'unknown error', 500
        except ValidationError as e:
            logger.error(str(e), exc_info=True)
            return 'invalid input', 400
        except BadRequest as e:
            logger.error(str(e), exc_info=True)
            return 'invalid input', e.code
        except Exception as e:
            logger.error(str(e), exc_info=True)
            return 'application error', 500

    def delete(self):
        # /delimage api
        if not request.path.endswith('/delimage'):
            return 'method not allowed', 405
        with DbTool() as db:
            try:
                if 'image_id' in request.json:
                    if 'image_tag' in request.json:
                        return 'cannot specify both id and tag', 400
                    if isinstance(request.json['image_id'], list):
                        res = db.delTrainingImages(request.json['image_id'])
                    else:
                        return 'image_id should be list', 400
                elif 'image_tag' in request.json:
                    res = db.delTrainingImages(request.json['image_tag'])
                else:
                    return 'neither id nor tag is specified', 400
                if res > 0:
                    return 'OK', 200
                elif res == 0:
                    return 'Not Found', 404
                else:
                    raise ValueError('unknown error')
            except BadRequest as e:
                logger.error(str(e), exc_info=True)
                return 'invalid input', e.code
            except Exception as e:
                logger.error(str(e), exc_info=True)
                return 'application error', 500

api.add_resource(ImageManager, '/listimages', '/imagedetails', '/delimage', '/addimage')
