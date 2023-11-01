import unittest
from unittest.mock import patch
import os
from api.tests.base_integration import IntegTestCase
import time
import shutil
from pathlib import Path
import logging
FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
import psutil

# Ignore license verification.
from api.tests.helper import verifier_mock
patch("api.util.verifier.proxy_license_verifier", verifier_mock).start()

from api.tests.helper import sub_test, generateRandomImage, createLetters
from api.tests.helper import createShapes, merge_dicts, isDicInDic, verifier_mock

os.environ['PREDICTION_MODE'] = 'ASYNC'
from api.api import app
from api.util.dbtool import DbTool
from api.util.const import Const
from api.controller.predict_async import AsyncPredictManager

class TestCase(IntegTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        time.sleep(3)   # wait for server up
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        AsyncPredictManager.stop_server()
        return super().tearDownClass()
    
    def setUp(self):
        self.app = app
        if Path('results').exists():
            shutil.rmtree('results', ignore_errors=True)
        if Path('/tmp/integration_db.sqlite').exists():
            Path('/tmp/integration_db.sqlite').unlink()
        os.environ['DBPATH'] = '/tmp/integration_db.sqlite'
        AsyncPredictManager.start_server()
        return super().setUp()

    def tearDown(self):
        AsyncPredictManager.stop_server()
        return super().tearDown()

    @sub_test([
        dict(data={'dataset':{'tiling':{"apply": True}}}),
        dict(data={'dataset':{'split_ratio':0.1}}),
        dict(data={'model':{'coreset_sampling_ratio':0.2}})
    ])
    def testServe(self, data):
        self.setUp()
        return super().doTestServe(data)

    def testPredict(self):
        return super().doTestPredict()

    def testThread(self):
        return super().doTestThread()

    @sub_test([
        dict(data=[1, {'project':{'save_outputs':{'inference':{'image':{'segmentation':[
            'input_image', 'histogram', 'ground_truth_mask', 'predicted_heat_map', 'predicted_mask', 'segmentation_result'
        ]}, 'csv':['anomaly_map', 'metrics']}, 'save_combined_result_as_image': True}}
                   ,'dataset':{'task':'segmentation'}}]),
        dict(data=[2, {'project':{'save_outputs':{'inference':{'image':{'classification':[
            'input_image', 'histogram', 'prediction'
        ]}, 'csv':['anomaly_map', 'metrics']}, 'save_combined_result_as_image': True}}
                   ,'dataset':{'task':'classification'}}]),
    ])
    def testPredictOutputResults(self, data):
        with DbTool(os.environ['DBPATH']) as db:
            db.dropAll()
        # sqlite3 python3.8 delays this transaction, try python 3.9
        time.sleep(1)
        shutil.rmtree('./results', ignore_errors=True)
        AsyncPredictManager.start_server()
        
        ret = super().doTestPredictOutputResults(data[1])
        if data[0] == 1:
            self.assertEqual(ret.status_code, 200)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/images/input_image").glob('**/*.png'))), 12)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/images/predicted_mask").glob('**/*.png'))), 12)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/images/combined").glob('**/*.png'))), 12)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/images/predicted_heat_map").glob('**/*.png'))), 12)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/csv").glob('**/*.csv'))), 12)
            self.assertTrue(Path("./results/1/patchcore/inference_results/metrics/pred_outputs.csv").exists())
            with self.app.test_client() as client:
                l_inf = client.get(Const.version_prefix + '/listresults')
                self.assertEqual(len(l_inf.json), 12)
                res = client.get(Const.version_prefix + '/getresult?inference_id=1')
                self.assertEqual(res.json['inference_id'], 1)
                self.assertEqual(res.json['model_id'], 1)
                self.assertEqual(res.json['result_path'], str(Path('results/1/patchcore/inference_results').absolute()))
                with DbTool() as db:
                    db.dropAll()
        if data[0] == 2:
            self.assertEqual(ret.status_code, 200)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/images/input_image").glob('**/*.png'))), 12)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/images/combined").glob('**/*.png'))), 12)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/images/prediction").glob('**/*.png'))), 12)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/csv").glob('**/*.csv'))), 12)
            self.assertTrue(Path("./results/1/patchcore/inference_results/metrics/pred_outputs.csv").exists())
            with self.app.test_client() as client:
                l_inf = client.get(Const.version_prefix + '/listresults')
                self.assertEqual(len(l_inf.json), 12)
                res = client.get(Const.version_prefix + '/getresult?inference_id=1')
                self.assertEqual(res.json['inference_id'], 1)
                self.assertEqual(res.json['model_id'], 1)
                self.assertEqual(res.json['result_path'], str(Path('results/1/patchcore/inference_results').absolute()))

        AsyncPredictManager.stop_server()
