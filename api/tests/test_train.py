import unittest
from unittest.mock import patch, Mock
import os
import logging
from pathlib import Path

FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
from api.tests.helper import *

# Ignore license verification.
patch("api.util.verifier.proxy_license_verifier", verifier_mock).start()

from api.api import app
from api.util.const import Const

class TestCase(unittest.TestCase):
    def setUp(self):
        os.environ['DBPATH'] = '/tmp/test_db.sqlite'

    def tearDown(self):
        if os.path.exists(os.environ['DBPATH']):
            os.remove(os.environ['DBPATH'])

    @sub_test([
        dict(req = {
            'image_tag':'bottle',
            'model_tag':'bottle'
        }, expect = 200),
        dict(req = {
            'image_id': [1, 2, 3, 4],
            'parameters': {'dataset':{'seed': 43}}
        }, expect = 200),
        dict(req = {
            'image_tag':'bottle',
            'image_id': [1, 2, 3, 4]
        }, expect = 400),
        dict(req = {
            'image_id': [1, 2, 3, 4],
            'parameters': {
                'dataset': {'tiling': {'apply': True}}
            }
        }, expect=200),
    ])
    def testTrainMan(self, req, expect):
        Path('/app/results/2/patchcore/weights').mkdir(parents=True, exist_ok=True)
        weightpath = Path('/app/results/2/patchcore/weights/trained_data.hmc')
        if weightpath.exists():
            weightpath.unlink()
        with app.test_client() as client:
            dbmock = Mock()
            data = {'good':['/abc/def.png'], 
                     'bad':['/efg/hij.png'], 
                     'mask':[]}
            dbmock.getDataset.return_value = (data, [1, 2, 3])
            dbmock.getLastModelId.return_value = 1
            dbmock.addModel.return_value = 2
            with patch('api.controller.train.DbTool') as MockDb:
                instance = MockDb.return_value
                instance.__enter__.return_value = dbmock
                dbmock.getModels.return_value=[]
                with patch('api.controller.train.train', return_value=Mock()):
                    # error case failed to save weight file
                    if expect != 400:
                        res = client.post(Const.version_prefix + '/train', json=req)
                        self.assertEqual(res.status_code, 500)
                    weightpath.touch()
                    pathpatch = patch('api.controller.train.Path')
                    pathpatch.start()
                    shutilpatch = patch('api.controller.train.shutil')
                    shutilpatch.start()
                    confpatch = patch('api.controller.train.OmegaConf.save')
                    confpatch.start()
                    mergepatch = patch('api.controller.train.OmegaConf.merge')
                    mergepatch.start()
                    getconfpatch = patch('api.controller.train.get_configurable_parameters')
                    getconfpatch.start()
                    getdmpatch = patch('api.controller.train.get_datamodule')
                    getdmpatch.start()
                    getmodelpatch = patch('api.controller.train.get_model')
                    getmodelpatch.start()
                    getcbpatch = patch('api.controller.train.get_callbacks')
                    getcbpatch.start()
                    subppatch = patch('api.controller.train.subprocess')
                    subppatch.start()
                    # normal case
                    res = client.post(Const.version_prefix + '/train', json=req)
                    self.assertEqual(res.status_code, expect)

                    # broken json
                    res = client.post(Const.version_prefix + '/train', 
                                      data='{"abc"123}'.encode(),
                                      headers={'content-type':'application/json'})
                    self.assertEqual(res.status_code, 400)

                    # error case model id assetion error
                    if expect != 400:
                        dbmock.addModel.return_value = 100
                        res = client.post(Const.version_prefix + '/train', json=req)
                        self.assertEqual(res.status_code, 500)
                    

                    # error case getlastModelId
                    if expect != 400:
                        dbmock.getLastModelId.return_value = None
                        res = client.post(Const.version_prefix + '/train', json=req)
                        self.assertEqual(res.status_code, 500)

                    if expect == 200:
                        dbmock.addModel.return_value = 2
                        dbmock.getLastModelId.return_value = 1
                        with patch('api.controller.train.train') as trmock:
                            os.environ['HOST_DATA_DIR'] = 'C:\\\\abc'
                            os.environ['BASE_DATA_DIR'] = '/tmp'
                            dataw = {'good':['C:\\abc\\def.png'], 
                                    'bad':['C:\\abc\\hij.png'], 
                                    'mask':['C:\\abc\\xyz.png']}
                            dbmock.getDataset.return_value = (dataw, [1, 2, 3])
                            res = client.post(Const.version_prefix + '/train', json=req)
                            self.assertEqual(res.status_code, expect)
                            self.assertEqual(trmock.call_args[0], (2, ['/tmp/def.png'],
                                                                   ['/tmp/hij.png'],
                                                                   ['/tmp/xyz.png'],
                                                                   req.get('parameters', None)))
                        
                    pathpatch.stop()
                    shutilpatch.stop()
                    confpatch.stop()
                    mergepatch.stop()
                    subppatch.stop()
                    getconfpatch.stop()
                    getdmpatch.stop()
                    getmodelpatch.stop()
                    getcbpatch.stop()
                    subppatch.stop()
                    if expect != 400:
                        with patch.object(dbmock, "getDataset", 
                                          side_effect=ValueError('Expected Mock Error')):
                            res = client.post(Const.version_prefix + '/train', json=req)
                            self.assertEqual(res.status_code, 500)
                    # duplicate model tag
                    if 'model_tag' in req:
                        with patch.object(dbmock, "getModels", 
                                            return_value=[1,2]):
                            res = client.post(Const.version_prefix + '/train', json=req)
                            self.assertEqual(res.status_code, 400)
