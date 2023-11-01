import unittest
from unittest.mock import patch, Mock
import os
import logging
from pathlib import Path
import shutil
import re

FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
from api.tests.helper import *
from werkzeug.exceptions import BadRequest

# Ignore license verification.
from api.tests.helper import verifier_mock
patch("api.util.verifier.proxy_license_verifier", verifier_mock).start()

from api.api import app
from api.util.dbtool import DbTool
from api.util.const import Const

class TestCase(unittest.TestCase):
    def setUp(self):
        os.environ['DBPATH'] = '/tmp/test_db.sqlite'

    def tearDown(self):
        if os.path.exists(os.environ['DBPATH']):
            os.remove(os.environ['DBPATH'])

    def testListModels(self):
        with app.test_client() as client:
            res = client.get(Const.version_prefix + '/listmodels')
            self.assertEqual(res.status_code, 404)
            with DbTool() as db:
                db.addModel('/abc/bottle.png', [1, 2, 3], 'patchcore', 
                            tag='bottle')
                db.addModel('/def/cup.png', [5, 6, 7, 8], 'patchcore')
            res = client.get(Const.version_prefix + '/listmodels')
            self.assertEqual(len(res.json), 2)
            self.assertEqual(res.json[0]['model_id'], 1)
            self.assertEqual(res.json[0]['tag'], 'bottle')
            self.assertEqual(res.json[1]['model_id'], 2)
            self.assertIsNone(res.json[1]['tag'])
            self.assertIsNotNone(re.match(r'[0-9]{4}/[0-9]{2}/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}', 
                                          res.json[0]['created']))
            # error case
            with patch('api.controller.models.DbTool') as MockDb:
                dbmock = Mock()
                dbmock.getModels.side_effect = Exception('Expected mock error')
                instance = MockDb.return_value
                instance.__enter__.return_value = dbmock
                res = client.get(Const.version_prefix + '/listmodels')
                self.assertEqual(res.status_code, 500)
            with patch('api.controller.models.request') as MockRequest:
                MockPath = Mock()
                MockPath.endswith.return_value = False
                MockRequest.path = MockPath
                res = client.get(Const.version_prefix + '/listmodels')
                self.assertEqual(res.status_code, 405)
                
    def testModelDetails(self):
        with app.test_client() as client:
            res = client.get(Const.version_prefix + '/modeldetails?model_id=1')
            self.assertEqual(res.status_code, 404)
            res = client.get(Const.version_prefix + '/modeldetails?tag=bottle')
            self.assertEqual(res.status_code, 404)
            with DbTool() as db:
                db.addModel('/abc/bottle/1', [1, 2, 3], 'patchcore', 
                            tag='bottle')
                db.addModel('/def/cup/1', [5, 6, 7, 8], 'patchcore', 
                            d_param={'dataset':{'seed':1}})
            res = client.get(Const.version_prefix + '/modeldetails?tag=bottle')
            self.assertEqual(res.json['image_ids'], [1, 2, 3])
            self.assertEqual(res.json['tag'], 'bottle')
            self.assertIsNotNone(re.match(r'[0-9]{4}/[0-9]{2}/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}', 
                                          res.json['created']))
            res = client.get(Const.version_prefix + '/modeldetails?model_id=2')
            self.assertEqual(res.json['image_ids'], [5, 6, 7, 8])
            self.assertIsNone(res.json['tag'])
            self.assertEqual(res.json['parameters'], {'dataset':{'seed':1}})
            self.assertIsNotNone(re.match(r'[0-9]{4}/[0-9]{2}/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}', 
                                          res.json['created']))
            res = client.get(Const.version_prefix + '/modeldetails?tag=bottle&model_id=1')
            self.assertEqual(res.status_code, 400)
            # error case
            with patch('api.controller.models.DbTool') as MockDb:
                dbmock = Mock()
                dbmock.getModels.side_effect = Exception('Expected mock error')
                instance = MockDb.return_value
                instance.__enter__.return_value = dbmock
                res = client.get(Const.version_prefix + '/modeldetails?tag=bottle')
                self.assertEqual(res.status_code, 500)
                dbmock.getModels.side_effect = BadRequest('Expected mock bad request')
                res = client.get(Const.version_prefix + '/modeldetails?tag=bottle')
                self.assertEqual(res.status_code, 400)

    def testDelModel(self):
        def insertModels():
            with DbTool() as db:
                db.addModel('/abc/bottle/1', [1, 2, 3], 'patchcore', tag='bottle')
                db.addModel('/def/cup/1', [5, 6, 7, 8], 'patchcore', 
                            d_param={'dataset':{'seed':1}})
                db.addModel('/abc/cup/2', [101, 102, 103], 'patchcore', tag='cup')
                db.addModel('/def/phone/1', [45, 46, 47, 48], 'patchcore', 
                            d_param={'dataset':{'seed':2}})
                db.addModel('/abc/bottle/2', [10, 11, 12, 13, 14], 'patchcore', tag='dog')
                db.addModel('/def/cup/3', [75, 76, 77, 78], 'patchcore', 
                            d_param={'dataset':{'seed':3}})
            
        with app.test_client() as client:
            with patch('api.controller.models.shutil.rmtree'):
                res = client.delete(Const.version_prefix + '/delmodel', json=[1, 2])
                self.assertEqual(res.status_code, 404)
                res = client.delete(Const.version_prefix + '/delmodel', json=[])
                self.assertEqual(res.status_code, 404)
                res = client.delete(Const.version_prefix + '/delmodel', json='')
                self.assertEqual(res.status_code, 404)
                res = client.delete(Const.version_prefix + '/delmodel', json={'a':1})
                self.assertEqual(res.status_code, 400)
                res = client.delete(Const.version_prefix + '/delmodel', json=1.1)
                self.assertEqual(res.status_code, 400)
                insertModels()
                res = client.delete(Const.version_prefix + '/delmodel', json=1)
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.json, [1])
                res = client.delete(Const.version_prefix + '/delmodel', json='dog')
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.json, [5])
                res = client.delete(Const.version_prefix + '/delmodel', json=[2, 'cup', 4])
                self.assertEqual(res.status_code, 200)
                self.assertEqual(sorted(res.json), [2, 3, 4])
                with DbTool() as db:
                    res = db.getModels()
                self.assertEqual(len(res), 1)
                self.assertEqual(res[0]['model_id'], 6)
            with patch('api.controller.models.shutil.rmtree') as mocksh:
                mocksh.side_effect = Exception('Expected Mock Error')
                res = client.delete(Const.version_prefix + '/delmodel', json=6)
                self.assertEqual(res.status_code, 500)
                mocksh.side_effect = BadRequest('Expected Mock bad request')
                res = client.delete(Const.version_prefix + '/delmodel', json=6)
                self.assertEqual(res.status_code, 400)
            with patch('api.controller.models.shutil.rmtree'):
                res = client.delete(Const.version_prefix + '/delmodel', json=6)
                self.assertEqual(res.status_code, 200)
