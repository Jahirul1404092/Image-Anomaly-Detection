import unittest
from unittest.mock import patch, Mock
import os
import logging
import io
import json

FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
from api.tests.helper import *

from PIL import Image
from torch import Tensor
from werkzeug.exceptions import BadRequest

# Ignore license verification.
from api.tests.helper import verifier_mock
patch("api.util.verifier.proxy_license_verifier", verifier_mock).start()

os.environ['PREDICTION_MODE'] = 'SYNC'
from api.api import app
from api.util.dbtool import DbTool
from api.util.const import Const
from api.controller.predict_sync import SyncPredictManager

class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        os.environ['DBPATH'] = '/tmp/test_db.sqlite'
        SyncPredictManager.d_predictor.clear()
        SyncPredictManager.d_config.clear()
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists(os.environ['DBPATH']):
            os.remove(os.environ['DBPATH'])
        return super().tearDown()

    def testServe(self):
        with app.test_client() as client:
            # no model
            with patch('api.controller.predict_sync.SyncInferencer'):
                ret = client.post(Const.version_prefix + '/servemodel', json={'model_id':1})
            self.assertEqual(ret.status_code, 404)
            with DbTool() as db:
                db.addModel('hamacho/plug_in/models/patchcore', [1, 2, 3], 'patchcore', 
                            {}, 'bottle')
                db.addModel('hamacho/plug_in/models/patchcore', [1, 2, 3], 'patchcore', 
                            {}, 'cup')
            # normal
            with patch('api.controller.predict_sync.SyncInferencer') as mockpred:
                instance = mockpred.return_value
                instance.set_serve_id.return_value = 'serve_id_0'
                ret = client.post(Const.version_prefix + '/servemodel', 
                                  json={'model_id':1, 'parameters':{'dataset':{'seed':4200}}})
                self.assertEqual(ret.status_code, 200)
                # 2nd serve
                ret = client.post(Const.version_prefix + '/servemodel', 
                                  json={'model_id':2, 'parameters':{'dataset':{'seed':4300}}})
                self.assertEqual(ret.status_code, 200)
            # argrment error
            ret = client.post(Const.version_prefix + '/servemodel', json={})
            self.assertEqual(ret.status_code, 400)
            ret = client.post(Const.version_prefix + '/servemodel', json={'model_id':1, 'tag':'bottle'})
            self.assertEqual(ret.status_code, 400)
            # exception
            with patch('api.controller.predict_sync.OmegaConf.load', 
                       side_effect=Exception('Expected mock error')):
                ret = client.post(Const.version_prefix + '/servemodel', json={'tag':'bottle'})
            self.assertEqual(ret.status_code, 500)
            with patch('api.controller.predict_sync.request') as MockRequest:
                MockPath = Mock()
                MockPath.endswith.return_value = False
                MockRequest.path = MockPath
                res = client.post(Const.version_prefix + '/servemodel')
                self.assertEqual(res.status_code, 405)
            # broken json
            res = client.post(Const.version_prefix + '/servemodel', 
                                data='{"abc"123}'.encode(),
                                headers={'content-type':'application/json'})
            self.assertEqual(res.status_code, 400)

    def testUnserve(self):
        with app.test_client() as client:
            # no model
            with patch('api.controller.predict_sync.SyncInferencer'):
                ret = client.delete(Const.version_prefix + '/unservemodel?model_id=1')
            self.assertEqual(ret.status_code, 404)
            with DbTool() as db:
                db.addModel('hamacho/plug_in/models/patchcore', [1, 2, 3], 'patchcore', 
                            {}, 'bottle')
                db.addModel('hamacho/plug_in/models/patchcore', [1, 2, 3], 'patchcore', 
                            {}, 'cup')
            # no model served yet
            with patch('api.controller.predict_sync.SyncInferencer'):
                ret = client.delete(Const.version_prefix + '/unservemodel?model_id=1')
            self.assertEqual(ret.status_code, 404)
            with patch('api.controller.predict_sync.SyncInferencer') as mockpred:
                instance = mockpred.return_value
                instance.set_serve_id.return_value = 'serve_id_0'
                # serve model 1
                ret = client.post(Const.version_prefix + '/servemodel', json={'model_id':1})
                self.assertEqual(ret.status_code, 200)
                # serve model 2
                ret = client.post(Const.version_prefix + '/servemodel', json={'model_id':2})
                self.assertEqual(ret.status_code, 200)
                # invalid input none
                ret = client.delete(Const.version_prefix + '/unservemodel')
                self.assertEqual(ret.status_code, 400)
                # invalid input both
                ret = client.delete(Const.version_prefix + '/unservemodel?model_id=2&tag=bottle')
                self.assertEqual(ret.status_code, 400)
                # normal unserve 1
                ret = client.delete(Const.version_prefix + '/unservemodel?model_id=1')
                self.assertEqual(ret.status_code, 200)
                # normal unserve 2
                ret = client.delete(Const.version_prefix + '/unservemodel?tag=cup')
                self.assertEqual(ret.status_code, 200)
            # exception
            with patch('api.controller.predict_sync.DbTool') as MockDb:
                dbmock = Mock()
                dbmock.getModels.side_effect = Exception('Expected mock error')
                instance = MockDb.return_value
                instance.__enter__.return_value = dbmock
                res = client.delete(Const.version_prefix + '/unservemodel?tag=cup')
                self.assertEqual(res.status_code, 500)
                dbmock.getModels.side_effect = BadRequest('Expected mock bad request')
                res = client.delete(Const.version_prefix + '/unservemodel?tag=cup')
                self.assertEqual(res.status_code, 400)
                
    
    def testPredict(self):
        with app.test_client() as client:
            data1 = {"model_id":1, "image_paths":["/test/imgs/037.png", "/test/imgs/049.png"]}
            data2 = {"tag":"cup", "image_paths":["/test/imgs/028.png", "/test/imgs/040.png"]}
            data_win = {"model_id":1, "image_paths":["C:\\test\\imgs\\028.png", "C:\\test\\imgs\\040.png"]}
            invaliddata = {"model_id":1, "tag":"cup", 
                         "image_paths":["/test/imgs/028.png", "/test/imgs/040.png"]}
            with patch('api.controller.predict_sync.SyncInferencer'):
                # not found
                ret = client.post(Const.version_prefix + '/predict', json=data1)
                self.assertEqual(ret.status_code, 404)
            with DbTool() as db:
                db.addModel('hamacho/plug_in/models/patchcore', [1, 2, 3], 'patchcore', 
                            {}, 'bottle')
                db.addModel('hamacho/plug_in/models/patchcore', [1, 2, 3], 'patchcore', 
                            {}, 'cup')
            with patch('api.controller.predict_sync.SyncInferencer') as mockpred:
                inst = mockpred.return_value
                inst.set_serve_id.return_value = 'serve_id_0'
                inst.predict.return_value = {
                    "pred_scores_denormalized":0.1,
                    "image_threshold":0.5,
                    "pred_scores":0.4,
                    "image_threshold_norm":0.5,
                    "anomaly_maps": Tensor([0.7, 0.5, 0.3]),
                    "image_path": ['abc/efg.jpg']
                }
                expected = {'pred_score': 0.1, 'image_threshold': 0.5, 'pred_score_norm': 0.4, 'image_threshold_norm': 0.5}
                # not served yet
                ret = client.post(Const.version_prefix + '/predict', json=data1)
                self.assertEqual(ret.status_code, 404)
                # serve model 1
                ret = client.post(Const.version_prefix + '/servemodel', json={'model_id':1})
                self.assertEqual(ret.status_code, 200)
                # normal
                ret = client.post(Const.version_prefix + '/predict', json=data1)
                self.assertEqual(ret.status_code, 200)
                self.assertEqual(ret.json[0], {
                    'image_path': 'results/inference_results/batch_mode/images/input_image/abc/efg.jpg',
                    'pred_score': 0.1, 'image_threshold': 0.5, 'pred_score_norm': 0.4, 'image_threshold_norm': 0.5,
                    'result_path': 'results/inference_results/batch_mode/images/prediction/abc/efg.jpg',
                    'saved': False
                })
                # normal win path
                os.environ['HOST_DATA_DIR'] = "C:\\\\test"
                os.environ['BASE_DATA_DIR'] = "/tmp"
                ret = client.post(Const.version_prefix + '/predict', json=data_win)
                self.assertEqual(ret.status_code, 200)
                self.assertEqual(inst.predict.call_args_list[-1][0][0], '/tmp/imgs/040.png')
                self.assertEqual(inst.predict.call_args_list[-2][0][0], '/tmp/imgs/028.png')
                del os.environ['HOST_DATA_DIR']
                del os.environ['BASE_DATA_DIR']
                # model 2 not served yet
                ret = client.post(Const.version_prefix + '/predict', json=data2)
                self.assertEqual(ret.status_code, 404)
                # serve model 2
                ret = client.post(Const.version_prefix + '/servemodel', json={'model_id':2})
                self.assertEqual(ret.status_code, 200)
                # normal
                ret = client.post(Const.version_prefix + '/predict', json=data2)
                self.assertEqual(ret.status_code, 200)
                # normal as_file=true
                img = Image.new('RGB', size=(224, 224), color=(0,0,0))
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='jpeg')
                files = {
                    'json': json.dumps({'model_id': 1}),
                    'images': [(img_bytes.getvalue(), 'test_file.jpg')],
                }
                ret = client.post(Const.version_prefix + '/predict?as_file=true', data=files, content_type='multipart/form-data')
                self.assertEqual(ret.status_code, 200)
                # validation error
                ret = client.post(Const.version_prefix + '/predict', json=invaliddata)
                self.assertEqual(ret.status_code, 400)
                # exception
                with patch('api.controller.predict_sync.DbTool') as MockDb:
                    dbmock = Mock()
                    dbmock.getModels.side_effect = Exception('Expected mock error')
                    instance = MockDb.return_value
                    instance.__enter__.return_value = dbmock
                    ret = client.post(Const.version_prefix + '/predict', json=data2)
                    self.assertEqual(ret.status_code, 500)
            # broken json
            res = client.post(Const.version_prefix + '/predict', 
                                data='{"abc"123}'.encode(),
                                headers={'content-type':'application/json'})
            self.assertEqual(res.status_code, 400)
