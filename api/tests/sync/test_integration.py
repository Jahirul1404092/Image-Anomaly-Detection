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

os.environ['PREDICTION_MODE'] = 'SYNC'
from api.api import app
from api.util.dbtool import DbTool
from api.util.const import Const
from api.controller.predict_sync import SyncPredictManager

class TestCase(IntegTestCase):
    def setUp(self):
        self.app = app
        SyncPredictManager.d_predictor.clear()
        SyncPredictManager.d_config.clear()
        if Path('results').exists():
            shutil.rmtree('results', ignore_errors=True)
        if Path('/tmp/integration_db.sqlite').exists():
            Path('/tmp/integration_db.sqlite').unlink()
        os.environ['DBPATH'] = '/tmp/integration_db.sqlite'
        return super().setUp()

    def tearDown(self):
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
        ]}, 'csv':['anomaly_map', 'metrics']}, 'save_combined_result_as_image': True}}}])
    ])
    def testPredictOutputResults(self, data):
        shutil.rmtree('./results', ignore_errors=True)
        ret = super().doTestPredictOutputResults(data[1])
        self.assertEqual(ret.status_code, 200)
        if data[0] == 1:
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/batch_mode/images/input_image").glob('**/*.png'))), 12)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/batch_mode/images/predicted_mask").glob('**/*.png'))), 12)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/batch_mode/images/combined").glob('**/*.png'))), 12)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/batch_mode/images/predicted_heat_map").glob('**/*.png'))), 12)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/batch_mode/csv").glob('**/*.csv'))), 12)
            self.assertTrue(Path("./results/1/patchcore/inference_results/batch_mode/metrics/pred_outputs.csv").exists())
            with self.app.test_client() as client:
                l_inf = client.get(Const.version_prefix + '/listresults')
                self.assertEqual(len(l_inf.json), 12)
                res = client.get(Const.version_prefix + '/getresult?inference_id=1')
                self.assertEqual(res.json['inference_id'], 1)
                self.assertEqual(res.json['model_id'], 1)
                self.assertEqual(res.json['result_path'], 'results/1/patchcore/inference_results/batch_mode/images/prediction/serve_id_1/009.png')
                with DbTool() as db:
                    db.dropAll()
        if data[0] == 2:
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/batch_mode/images/input_image").glob('**/*.png'))), 12)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/batch_mode/images/combined").glob('**/*.png'))), 12)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/batch_mode/images/prediction").glob('**/*.png'))), 12)
            self.assertEqual(len(list(Path("./results/1/patchcore/inference_results/batch_mode/csv").glob('**/*.csv'))), 12)
            self.assertTrue(Path("./results/1/patchcore/inference_results/batch_mode/metrics/pred_outputs.csv").exists())
            with self.app.test_client() as client:
                l_inf = client.get(Const.version_prefix + '/listresults')
                self.assertEqual(len(l_inf.json), 12)
                res = client.get(Const.version_prefix + '/getresult?inference_id=1')
                self.assertEqual(res.json['inference_id'], 1)
                self.assertEqual(res.json['model_id'], 1)
                self.assertEqual(res.json['result_path'], 'results/1/patchcore/inference_results/batch_mode/images/prediction/serve_id_1/009.png')
        # sqlite3 python3.8 delays this transaction, try python 3.9
        time.sleep(1)
