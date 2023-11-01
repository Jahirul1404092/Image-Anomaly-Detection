import unittest
from unittest.mock import patch, Mock
import os
import json
import io
import logging
import requests
import time
from pathlib import Path
from PIL import Image

FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
from api.tests.helper import *
from werkzeug.exceptions import BadRequest

# Ignore license verification.
from api.tests.helper import verifier_mock
patch("api.util.verifier.proxy_license_verifier", verifier_mock).start()

os.environ['PREDICTION_MODE'] = 'ASYNC'
from api.api import app
from api.util.dbtool import DbTool
from api.util.const import Const
from api.controller.predict_async import AsyncPredictManager

class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # wait for server up
        time.sleep(5)
        if Path('api/api.py').exists():
            cls.cython = False
        else:
            cls.cython = True
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        AsyncPredictManager.stop_server()
        return super().tearDownClass()
    
    def setUp(self) -> None:
        os.environ['DBPATH'] = '/tmp/test_db.sqlite'
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists(os.environ['DBPATH']):
            os.remove(os.environ['DBPATH'])
        return super().tearDown()

    def testServe(self):
        with app.test_client() as client:
            # no model
            ret = client.post(Const.version_prefix + '/servemodel', json={'model_id':1})
            self.assertEqual(ret.status_code, 404)
            with DbTool() as db:
                db.addModel('api', [1, 2, 3], 'patchcore', 
                            {}, 'bottle')
                # db.addModel('hamacho/plug_in/models/patchcore', [1, 2, 3], 'patchcore', 
                #             {}, 'cup')
            # normal
            with patch('api.controller.predict_async.requests') as mockreq:
                r200 = requests.Response()
                r200.status_code = 200
                mockreq.post.return_value = r200
                mockreq.put.return_value = r200
                ret = client.post(Const.version_prefix + '/servemodel', 
                                    json={'model_id':1, 'parameters':{'dataset':{'seed':4200}}})
                self.assertEqual(ret.status_code, 200)
            # 2nd serve no model data
            ret = client.post(Const.version_prefix + '/servemodel', 
                                json={'model_id':2, 'parameters':{'dataset':{'seed':4300}}})
            self.assertEqual(ret.status_code, 404)
            with DbTool() as db:
                db.addModel('api', [1, 2, 3], 'patchcore', 
                            {}, 'cup')
            with patch('api.controller.predict_async.requests') as mockreq:
                r200 = requests.Response()
                r200.status_code = 200
                r400 = requests.Response()
                r400.status_code = 400
                # registration error
                mockreq.post.return_value = r400
                ret = client.post(Const.version_prefix + '/servemodel', json={'model_id':2})
                self.assertEqual(ret.status_code, 400)
                # scaling error
                mockreq.post.return_value = r200
                mockreq.put.return_value = r400
                ret = client.post(Const.version_prefix + '/servemodel', json={'model_id':2})
                self.assertEqual(ret.status_code, 400)
            # argument error
            ret = client.post(Const.version_prefix + '/servemodel', json={})
            self.assertEqual(ret.status_code, 400)
            ret = client.post(Const.version_prefix + '/servemodel', json={'model_id':1, 'tag':'bottle'})
            self.assertEqual(ret.status_code, 400)
            # exception
            with patch('api.controller.predict_async.DbTool', 
                       side_effect=Exception('Expected mock error')):
                ret = client.post(Const.version_prefix + '/servemodel', json={'tag':'bottle'})
            self.assertEqual(ret.status_code, 500)
            with patch('api.controller.predict_async.request') as MockRequest:
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
            ret = client.delete(Const.version_prefix + '/unservemodel?model_id=1')
            self.assertEqual(ret.status_code, 404)
            with DbTool() as db:
                db.addModel('hamacho/plug_in/models/patchcore', [1, 2, 3], 'patchcore', 
                            {}, 'bottle')
                db.addModel('hamacho/plug_in/models/patchcore', [1, 2, 3], 'patchcore', 
                            {}, 'cup')
            # no model served yet
            with patch('api.controller.predict_async.requests') as mockreq:
                r404 = requests.Response()
                r404.status_code = 404
                mockreq.delete.return_value = r404
                ret = client.delete(Const.version_prefix + '/unservemodel?model_id=1')
                self.assertEqual(ret.status_code, 404)
            with patch('api.controller.predict_async.requests') as mockreq:
                r200 = requests.Response()
                r200.status_code = 200
                mockreq.delete.return_value = r200
                # serve model 1
                # invalid input none
                ret = client.delete(Const.version_prefix + '/unservemodel')
                self.assertEqual(ret.status_code, 400)
                # invalid input both
                ret = client.delete(Const.version_prefix + '/unservemodel?model_id=2&tag=bottle')
                self.assertEqual(ret.status_code, 400)
                # normal unserve
                ret = client.delete(Const.version_prefix + '/unservemodel?model_id=1')
                self.assertEqual(ret.status_code, 200)
                # normal unserve 2
                ret = client.delete(Const.version_prefix + '/unservemodel?tag=cup')
                self.assertEqual(ret.status_code, 200)
            # exception
            with patch('api.controller.predict_async.DbTool') as MockDb:
                dbmock = Mock()
                dbmock.getModels.side_effect = Exception('Expected mock error')
                instance = MockDb.return_value
                instance.__enter__.return_value = dbmock
                res = client.delete(Const.version_prefix + '/unservemodel?tag=cup')
                self.assertEqual(res.status_code, 500)
                # bad request
                dbmock.getModels.side_effect = BadRequest('Expected mock bac request')
                res = client.delete(Const.version_prefix + '/unservemodel?tag=cup')
                self.assertEqual(res.status_code, 400)

    def testPredict(self):
        with app.test_client() as client:
            data1 = {"model_id":1, "image_paths":["/test/imgs/037.png", "/test/imgs/049.png"]}
            data2 = {"tag":"cup", "image_paths":["/test/imgs/028.png", "/test/imgs/040.png"]}
            data_win = {"tag":"cup", "image_paths":["C:\\test\\imgs\\028.png", "C:\\test\\imgs\\040.png"]}
            invaliddata = {"model_id":1, "tag":"cup", 
                         "image_paths":["/test/imgs/028.png", "/test/imgs/040.png"]}
            # not found
            ret = client.post(Const.version_prefix + '/predict', json=data1)
            self.assertEqual(ret.status_code, 404)
            with DbTool() as db:
                m_id = db.addModel('api', [1, 2, 3], 'patchcore', 
                            {}, 'bottle')
                db.updateInferencePath(m_id, 'results')
                m_id = db.addModel('api', [1, 2, 3], 'patchcore', 
                            {}, 'cup')
                db.updateInferencePath(m_id, 'results')
            # cannot perform in cython mode. open mock does not work
            if not self.cython:
                with patch('api.controller.predict_async.open') as mockopen:
                    with patch('api.controller.predict_async.requests') as mockreq:
                        r200 = requests.Response()
                        r200.status_code = 200
                        r400 = requests.Response()
                        r400.status_code = 400
                        expected = {
                            "pred_score": 4.309920310974121,
                            "image_threshold": 3.438771963119507,
                            "pred_score_norm": 0.8594635725021362,
                            "image_threshold_norm": 0.5
                            }
                        r200.json = Mock(return_value=expected)
                        
                        mockreq.put.return_value = r400
                        # post method error
                        ret = client.post(Const.version_prefix + '/predict', json=data1)
                        self.assertEqual(ret.status_code, 400)
                        
                        # no result
                        mockreq.put.side_effect = [r400, r200] 
                        ret = client.post(Const.version_prefix + '/predict', json=data1)
                        self.assertEqual(ret.status_code, 200)
                        # normal
                        mockreq.put.side_effect = None
                        mockreq.put.return_value = r200
                        ret = client.post(Const.version_prefix + '/predict', json=data1)
                        self.assertEqual(ret.status_code, 200)
                        self.assertEqual(ret.json[0], expected)
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
                        # win path
                        os.environ['HOST_DATA_DIR'] = 'C:\\\\test'
                        os.environ['BASE_DATA_DIR'] = '/tmp'
                        ret = client.post(Const.version_prefix + '/predict', json=data_win)
                        self.assertEqual(ret.status_code, 200)
                        self.assertEqual(str(mockopen.call_args_list[-1][0][0]), '/tmp/imgs/040.png')
                        self.assertEqual(str(mockopen.call_args_list[-2][0][0]), '/tmp/imgs/028.png')
                        del os.environ['HOST_DATA_DIR']
                        del os.environ['BASE_DATA_DIR']
                        # # no results
                        # with patch('api.controller.predict_async.request'):
                        #     ret = client.post(Const.version_prefix + '/predict', json=data2)
                        #     self.assertEqual(ret.status_code, 200)
                        # validation error
                        ret = client.post(Const.version_prefix + '/predict', json=invaliddata)
                        self.assertEqual(ret.status_code, 400)
                        # exception
                        with patch('api.controller.predict_async.DbTool') as MockDb:
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
