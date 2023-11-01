import os
import unittest
import logging
FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
from unittest.mock import Mock, patch

# Ignore license verification.
from api.tests.helper import verifier_mock
patch("api.util.verifier.proxy_license_verifier", verifier_mock).start()

from api.api import app
from api.util.dbtool import DbTool
from api.util.const import Const

os.environ['PREDICTION_MODE'] = 'SYNC'

class TestCase(unittest.TestCase):
    def setUp(self):
        os.environ['DBPATH'] = '/tmp/test_db.sqlite'

    def tearDown(self):
        if os.path.exists(os.environ['DBPATH']):
            os.remove(os.environ['DBPATH'])

    def testListResults(self):
        with app.test_client() as client:
            # empty
            res = client.get(Const.version_prefix + '/listresults')
            self.assertEqual(res.status_code, 404)
            expected = {}
            with DbTool() as db:
                res = db.addServing(1, {'param1':1,'param2':3}, mode='batch')
                self.assertEqual(res, 1)
                res = db.addServing(2, None, mode='batch')
                self.assertEqual(res, 2)
                inf_id = db.addInference(1, 'img1.png', {'score': 100}, '/tmp/result', '/csv/path')
                expected[str(inf_id)] = 'img1.png'
                inf_id = db.addInference(2, 'img2.png', {'score': 110}, '/tmp/result2', '/csv/path')
                expected[str(inf_id)] = 'img2.png'
                # normal
                res = client.get(Const.version_prefix + '/listresults')
                self.assertEqual(res.json, expected)
                # application error
                with patch('api.controller.predict.DbTool') as MockDb:
                    dbmock = Mock()
                    dbmock.listInferences.side_effect = ValueError('Expected mock error')
                    instance = MockDb.return_value
                    instance.__enter__.return_value = dbmock
                    res = client.get(Const.version_prefix + '/listresults')
                self.assertEqual(res.status_code, 500)

    def testGetResult(self):
        l_inf_id = []
        with DbTool() as db:
            res = db.addModel('/abc/def.ckp', [1, 2, 3, 4, 5, 6], 'patchcore',
                              tag='bottle')
            self.assertEqual(res, 1)
            res = db.addModel('/xyz/pqr.ckp', [11, 12, 13, 14, 15], 'patchcore', 
                              tag='cup')
            self.assertEqual(res, 2)
            res = db.addServing(1, {'param1':1,'param2':3}, mode='batch')
            self.assertEqual(res, 1)
            res = db.addServing(2, None, mode='batch')
            self.assertEqual(res, 2)
            inf_id = db.addInference(1, 'img1.png', {'score': 100}, '/tmp/result', '/csv/path')
            l_inf_id.append(inf_id)
            inf_id = db.addInference(2, 'img2.png', {'score': 110}, '/tmp/result2', '/csv/path')
            l_inf_id.append(inf_id)
            with app.test_client() as client:
                # inference id not found
                res = client.get(Const.version_prefix + '/getresult?inference_id=100')
                self.assertEqual(res.status_code, 404)
                # normal
                res = client.get(Const.version_prefix + f'/getresult?inference_id={l_inf_id[0]}')
                self.assertEqual(res.json['inference_id'], l_inf_id[0])
                self.assertEqual(res.json['image_path'], 'img1.png')
                self.assertEqual(res.json['result_json'], {'score': 100})
                self.assertEqual(res.json['result_path'], '/tmp/result')
                self.assertEqual(res.json['model_id'], 1)
                self.assertEqual(res.json['parameter'], {'param1': 1, 'param2': 3})
                res = client.get(Const.version_prefix + f'/getresult?inference_id={l_inf_id[1]}')
                self.assertEqual(res.json['inference_id'], l_inf_id[1])
                self.assertEqual(res.json['image_path'], 'img2.png')
                self.assertEqual(res.json['result_json'], {'score': 110})
                self.assertEqual(res.json['result_path'], '/tmp/result2')
                self.assertEqual(res.json['model_id'], 2)
                self.assertEqual(res.json['parameter'], {})
                # application error
                with patch('api.controller.predict.DbTool') as MockDb:
                    dbmock = Mock()
                    dbmock.getInference.side_effect = ValueError('Expected mock error')
                    instance = MockDb.return_value
                    instance.__enter__.return_value = dbmock
                    res = client.get(Const.version_prefix + f'/getresult?inference_id={l_inf_id[0]}')
                    self.assertEqual(res.status_code, 500)

    def testDelResult(self):
        l_inf_id = []
        with DbTool() as db:
            res = db.addServing(1, {'param1':1,'param2':3}, mode='batch')
            self.assertEqual(res, 1)
            res = db.addServing(2, None, mode='batch')
            self.assertEqual(res, 2)
            inf_id = db.addInference(1, 'img1.png', {'score': 100}, '/tmp/result', '/csv/path')
            l_inf_id.append(inf_id)
            inf_id = db.addInference(2, 'img2.png', {'score': 110}, '/tmp/result2', '/csv/path')
            l_inf_id.append(inf_id)
            with app.test_client() as client:
                res = client.get(Const.version_prefix + '/listresults')
                self.assertEqual(list(res.json.keys()), ["1", "2"])
                # inference id not found
                res = client.delete(Const.version_prefix + '/delresult?inference_id=100')
                self.assertEqual(res.status_code, 404)
                res = client.get(Const.version_prefix + '/listresults')
                self.assertEqual(list(res.json.keys()), ["1", "2"])
                # normal
                res = client.delete(Const.version_prefix + f'/delresult?inference_id={l_inf_id[0]}')
                self.assertEqual(res.status_code, 200)
                res = client.get(Const.version_prefix + '/listresults')
                self.assertEqual(list(res.json.keys()), ["2"])
                # application error
            with patch('api.controller.predict.DbTool') as MockDb:
                dbmock = Mock()
                dbmock.delInference.side_effect = ValueError('Expected mock error')
                instance = MockDb.return_value
                instance.__enter__.return_value = dbmock
                res = client.delete(Const.version_prefix + f'/delresult?inference_id={l_inf_id[1]}')
                self.assertEqual(res.status_code, 500)
