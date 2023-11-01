import unittest
from unittest.mock import patch
import os
import logging
from pathlib import Path
import shutil
import re
import sys
import tempfile
import time
FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)

from omegaconf import OmegaConf

from api.api import app
from api.tests.helper import sub_test, generateRandomImage, createLetters
from api.tests.helper import createShapes, merge_dicts, isDicInDic, verifier_mock
from api.util.dbtool import DbTool
from api.util.const import Const

class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 事前準備ライセンスファイルのチェック
        paid_lic_path = 'api/tests/licenses/paid/valid.lic'
        if not os.path.isfile(paid_lic_path):
            print('実行前に有償版ライセンスファイルを準備する: {}'.format(paid_lic_path))
        paid_lic_path = 'api/tests/licenses/paid/expired.lic'
        if not os.path.isfile(paid_lic_path):
            print('実行前に有償版ライセンスファイルを準備する: {}'.format(paid_lic_path))
            sys.exit()

        time.sleep(3)   # wait for server up
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        return super().tearDownClass()
    
    def setUp(self):
        if Path('results').exists():
            shutil.rmtree('results', ignore_errors=True)
        if Path('/tmp/integration_db.sqlite').exists():
            Path('/tmp/integration_db.sqlite').unlink()
        os.environ['DBPATH'] = '/tmp/integration_db.sqlite'
        os.environ.pop("OBFUSCATED_VERIFIER_DUMMY_DIR", None)
        os.environ.pop("PYARMOR_LICENSE", None)
        return super().setUp()

    def tearDown(self):
        os.environ.pop("OBFUSCATED_VERIFIER_DUMMY_DIR", None)
        os.environ.pop("PYARMOR_LICENSE", None)
        return super().tearDown()
        
    def doTrain(self, tasks=[0, 1], with_bad=False, with_mask=False, params={}):
        l_res = []
        for i in tasks:
            generator = [createLetters, createShapes][i]
            with tempfile.TemporaryDirectory() as imgpath:
                trainsize = 50
                l_good, l_bad, l_mask = generator(imgpath, trainsize)
                with app.test_client() as client:
                    image_ids = []
                    category =  ['letters', 'shapes'][i]
                    res = client.post(Const.version_prefix + '/addimage', 
                                    json={'image_path': l_good,
                                    'image_tag':category})
                    image_ids.extend(res.json)
                    if with_bad:
                        if with_mask:
                            res = client.post(Const.version_prefix + '/addimage', 
                                            json={'image_path': l_mask,
                                            'group': 'mask',
                                            'image_tag':category})
                            image_ids.extend(res.json)
                        res = client.post(Const.version_prefix + '/addimage', 
                                        json={'image_path': l_bad,
                                        'group': 'bad',
                                        'image_tag':category})
                        image_ids.extend(res.json)
                        params = merge_dicts(params,{
                            'model':{'normalization_method':'min_max'}})
                    if i == 0:
                        res = client.post(Const.version_prefix + '/train', 
                                        json={'image_tag': category,
                                                'model_tag': category,
                                                'parameters': params})
                    else:
                        res = client.get(Const.version_prefix + '/listimages')
                        imgs = [x['image_id'] for x in res.json if x.get('image_tag', None)==category]
                        res = client.post(Const.version_prefix + '/train',
                                          json={'image_id': imgs,
                                                'parameters': params})
                    l_res.append(res)
                    if res.status_code == 200:
                        # self.assertEqual(res.json, i + 1)
                        with DbTool(os.environ['DBPATH']) as db:
                            m = db.getModels(res.json)[0]
                            # conf = OmegaConf.load(Path(m['model_path']) / 'config.yaml')
                            self.assertEqual(m['model_id'], res.json)
                            # self.assertEqual(m['model_path'], conf.project.path)
                            self.assertEqual(m['tag'], category if i == 0 else None)
                            self.assertEqual(m['parameters'], params)
        return l_res

    def testTrainWithDuplicatedTags(self):
        os.environ['HOST_DATA_DIR'] = '/test'
        os.environ['BASE_DATA_DIR'] = '/tmp'
        os.environ["OBFUSCATED_VERIFIER_DUMMY_DIR"] = "obfuscated"
        os.environ["PYARMOR_LICENSE"] = "api/tests/licenses/paid/valid.lic"
        savedir = Path('./results')
        l_res = self.doTrain()
        self.assertTrue(all([res.status_code==200 for res in l_res]))
        self.assertTrue((savedir / '1').exists())
        self.assertTrue(Path(os.environ['DBPATH']).exists())
        l_res = self.doTrain()
        self.assertTrue(any([res.status_code==400 for res in l_res]))

    @sub_test([
        dict(data=[True, False, {'dataset':{'tiling':{"apply": True}}}]),
        dict(data=[True, False, {'model':{'backbone':'resnet18'}}]),
        dict(data=[True, True, {'model':{'coreset_sampling_ratio':0.2}}]),
        dict(data=[False, False, {'dataset':{'split_ratio':0.1}}]),
        dict(data=[False, False, {'trainer':{'accelerator':'cpu'}}]),
        dict(data=[False, False, {'dataset':{'num_workers':4}}]),
        dict(data=[True, True, {'dataset':{'task':'segmentation'}}]),
        dict(data=[True, False, {'model':{'normalization_method':'min_max'}}]),
        dict(data=[True, False, {'dataset':{'train_batch_size':15}}]),
        dict(data=[False, False, {'metrics':{'threshold':{'image_norm':0.3}}}]),
        dict(data=[False, False, {'metrics':{'threshold':{'pixcel_norm':0.3}}}]),
        dict(data=[False, False, {'project':{'save_outputs':{'add_label_on_image': True,
                                                             'test':{'image':{'classification':['prediction']}}}}}])
    ])
    def testTrainWithOptions(self, data):
        os.environ['HOST_DATA_DIR'] = '/test'
        os.environ['BASE_DATA_DIR'] = '/tmp'
        os.environ["OBFUSCATED_VERIFIER_DUMMY_DIR"] = "obfuscated"
        os.environ["PYARMOR_LICENSE"] = "api/tests/licenses/paid/valid.lic"
        l_res = self.doTrain(with_bad=data[0], with_mask=data[1], params=data[2])
        self.assertTrue(all([x.status_code == 200 for x in l_res]))
        for res in l_res:
            config = OmegaConf.load(f'results/{res.json}/patchcore/config.yaml')
            self.assertTrue(isDicInDic(data[2], config))
        with DbTool(os.environ['DBPATH']) as db:
            db.dropAll()
        # sqlite3 python3.8 delays this transaction, try python 3.9
        time.sleep(1)

    def testImage(self):
        os.environ["OBFUSCATED_VERIFIER_DUMMY_DIR"] = "obfuscated"
        os.environ["PYARMOR_LICENSE"] = "api/tests/licenses/paid/valid.lic"
        with tempfile.TemporaryDirectory() as imgpath:
            l_good, l_bad, l_mask = createLetters(imgpath, 50)
            with app.test_client() as client:
                # error case no image
                res = client.post(Const.version_prefix + '/addimage', 
                                json={'image_tag':'letters'})
                self.assertEqual(res.status_code, 400)
                # error case invalid group
                res = client.post(Const.version_prefix + '/addimage', 
                                json={'image_path': l_good,
                                'image_tag':'letters',
                                'group': 'better'})
                self.assertEqual(res.status_code, 400)
                # method not allowed
                res = client.delete(Const.version_prefix + '/addimage', 
                                json={'image_path': l_good,
                                'image_tag':'letters'})
                self.assertEqual(res.status_code, 405)
                # get error case no image
                res = client.get(Const.version_prefix + '/listimages')
                self.assertEqual(res.status_code, 404)
                res = client.get(Const.version_prefix + '/imagedetails?image_id=1')
                self.assertEqual(res.status_code, 404)
                # delete error case no image
                res = client.delete(Const.version_prefix + '/delimage',
                                    json={'image_id':[1,2]})
                self.assertEqual(res.status_code, 404)
                # normal
                res = client.post(Const.version_prefix + '/addimage', 
                                json={'image_path': l_good,
                                'image_tag':'letters'})
                self.assertEqual(len(res.json), 50)
                # method not allowed
                res = client.post(Const.version_prefix + '/listimages')
                self.assertEqual(res.status_code, 405)
                res = client.post(Const.version_prefix + '/imagedetails?image_id=1')
                self.assertEqual(res.status_code, 405)
                # no image
                res = client.get(Const.version_prefix + '/imagedetails?image_id=51')
                self.assertEqual(res.status_code, 404)
                res = client.get(Const.version_prefix + '/imagedetails?image_tag=shapes')
                self.assertEqual(res.status_code, 404)
                # bad param
                res = client.delete(Const.version_prefix + '/delimage?image_id=1', json={})
                self.assertEqual(res.status_code, 400)
                res = client.get(Const.version_prefix + '/imagedetails',
                                 json={'image_id':1})
                self.assertEqual(res.status_code, 400)
                # no image
                res = client.delete(Const.version_prefix + '/delimage',
                                    json={'image_id': [51, 52]})
                self.assertEqual(res.status_code, 404)
                res = client.delete(Const.version_prefix + '/delimage',
                                    json={'image_tag': 'shapes'})
                self.assertEqual(res.status_code, 404)
                # normal
                res = client.get(Const.version_prefix + '/listimages')
                self.assertEqual(len(res.json), 50)
                self.assertTrue(all([x['image_tag']=='letters' for x in res.json]))
                res = client.get(Const.version_prefix + '/imagedetails?image_id=50')
                self.assertEqual(len(res.json), 1)
                self.assertEqual(res.json[0]['group'], 'good')
                # delete by id
                res = client.delete(Const.version_prefix + '/delimage',
                                    json={'image_id': [45, 50]})
                self.assertEqual(res.status_code, 200)
                res = client.get(Const.version_prefix + '/imagedetails?image_id=45')
                self.assertEqual(res.status_code, 404)
                # delete by tag
                res = client.delete(Const.version_prefix + '/delimage',
                                    json={'image_tag': 'letters'})
                self.assertEqual(res.status_code, 200)
                res = client.get(Const.version_prefix + '/imagedetails?image_tag=letters')
                self.assertEqual(res.status_code, 404)

    def testTestOutputs(self):
        os.environ['HOST_DATA_DIR'] = '/test'
        os.environ['BASE_DATA_DIR'] = '/tmp'
        os.environ["OBFUSCATED_VERIFIER_DUMMY_DIR"] = "obfuscated"
        os.environ["PYARMOR_LICENSE"] = "api/tests/licenses/paid/valid.lic"
        l_res = self.doTrain(with_bad=True, with_mask=True, params={'dataset':{'task':'segmentation'},
                                     'project':{
                                         'save_outputs':{
                                             'test':{
                                                 'image':{'segmentation':
                                                     ['input_image', 
                                                      'histogram',
                                                      'ground_truth_mask',
                                                      'predicted_heat_map',
                                                      'predicted_mask',
                                                      'segmentation_result']},
                                                 'csv':['anomaly_map, metrics']},
                                             'save_combined_result_as_image':True}}})
        self.assertTrue(all([x.status_code==200 for x in l_res]))
        for i in [1, 2]:
            self.assertEqual(len(list(Path(f'results/{i}/patchcore/test_predictions/images/combined').glob('**/*.png'))), 20)
            self.assertEqual(len(list(Path(f'results/{i}/patchcore/test_predictions/images/input_image').glob('**/*.png'))), 20)
            self.assertEqual(len(list(Path(f'results/{i}/patchcore/test_predictions/images/predicted_heat_map').glob('**/*.png'))), 20)
            self.assertEqual(len(list(Path(f'results/{i}/patchcore/test_predictions/images/predicted_mask').glob('**/*.png'))), 20)
            self.assertTrue(Path(f'results/{i}/patchcore/test_predictions/metrics/confusion_matrix.png').exists())
            self.assertTrue(Path(f'results/{i}/patchcore/test_predictions/metrics/image-level-roc.png').exists())
            self.assertTrue(Path(f'results/{i}/patchcore/test_predictions/metrics/image-level-scores-histogram.png').exists())
            self.assertTrue(Path(f'results/{i}/patchcore/test_predictions/metrics/pixel_level_test_outputs.csv').exists())
            self.assertTrue(Path(f'results/{i}/patchcore/test_predictions/metrics/pixel-level-roc.png').exists())
        with DbTool(os.environ['DBPATH']) as db:
            db.dropAll()
        shutil.rmtree('results', ignore_errors=True)
        l_res = self.doTrain(with_bad=True, with_mask=True, params={'dataset':{'task':'classification'},
                                     'project':{
                                         'save_outputs':{
                                             'test':{
                                                 'image':{'classification':
                                                     ['input_image', 
                                                      'histogram',
                                                      'prediction']},
                                                 'csv':['anomaly_map, metrics']},
                                             'save_combined_result_as_image':True}}})
        self.assertTrue(all([x.status_code==200 for x in l_res]))
        for i in [1, 2]:
            self.assertEqual(len(list(Path(f'results/{i}/patchcore/test_predictions/images/combined').glob('**/*.png'))), 20)
            self.assertEqual(len(list(Path(f'results/{i}/patchcore/test_predictions/images/input_image').glob('**/*.png'))), 20)
            self.assertEqual(len(list(Path(f'results/{i}/patchcore/test_predictions/images/prediction').glob('**/*.png'))), 20)
            self.assertTrue(Path(f'results/{i}/patchcore/test_predictions/metrics/confusion_matrix.png').exists())
            self.assertTrue(Path(f'results/{i}/patchcore/test_predictions/metrics/image-level-roc.png').exists())
            self.assertTrue(Path(f'results/{i}/patchcore/test_predictions/metrics/image-level-scores-histogram.png').exists())

    def testModelOutputFolder(self):
        os.environ['HOST_DATA_DIR'] = '/test'
        os.environ['BASE_DATA_DIR'] = '/tmp'
        os.environ["OBFUSCATED_VERIFIER_DUMMY_DIR"] = "obfuscated"
        os.environ["PYARMOR_LICENSE"] = "api/tests/licenses/paid/valid.lic"
        model_dir = '/tmp/model_dir'
        test_dir = '/tmp/test_dir'
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(test_dir, ignore_errors=True)
        Path(test_dir).mkdir(exist_ok=True, parents=True)
        self.doTrain(tasks=[0], params={'project':{
            'path':model_dir, 'save_root':test_dir, 'save_outputs':{
                'test':{
                    'image':{
                        'classification':['prediction']},
                    'csv':['metrics']}}}})
        self.assertTrue(Path(f'{model_dir}/weights/trained_data.hmc').exists())
        self.assertEqual(len(list(Path(f'{test_dir}/test_predictions/images/prediction').glob('**/*.png'))), 10)
        self.assertTrue(Path(f'{test_dir}/test_predictions/metrics/confusion_matrix.png').exists())

    def testInvalidLicense(self):
        def _call_all_api(expect_status_code: int):
            # 学習画像 登録
            res = client.post(Const.version_prefix + "/addimage", json={"image_path": ["path"]})
            self.assertEqual(res.status_code, expect_status_code)

            # 学習画像 一覧取得
            res = client.get(Const.version_prefix + "/listimages")
            self.assertEqual(res.status_code, expect_status_code)

            # 学習画像 詳細取得
            res = client.get(Const.version_prefix + "/imagedetails?image_id=1")
            self.assertEqual(res.status_code, expect_status_code)

            # 学習画像 削除
            res = client.delete(Const.version_prefix + "/delimage", json={'image_id': [1, 2]})
            self.assertEqual(res.status_code, expect_status_code)

            # 学習
            res = client.post(Const.version_prefix + "/train", json={'image_id': [1, 2]})
            self.assertEqual(res.status_code, expect_status_code)

            # 学習済みモデル 一覧取得
            res = client.get(Const.version_prefix + "/listmodels")
            self.assertEqual(res.status_code, expect_status_code)

            # 学習済みモデル 詳細取得
            res = client.get(Const.version_prefix + "/modeldetails?model_id=1")
            self.assertEqual(res.status_code, expect_status_code)

            # 学習済みモデル 削除
            res = client.delete(Const.version_prefix + "/delmodel", json=1)
            self.assertEqual(res.status_code, expect_status_code)

            # モデルサービング
            res = client.post(Const.version_prefix + "/servemodel", json={'tag': 'bottle'})
            self.assertEqual(res.status_code, expect_status_code)

            # モデル登録解除
            res = client.delete(Const.version_prefix + "/unservemodel?tag=letters")
            self.assertEqual(res.status_code, expect_status_code)

            # 推論
            res = client.post(Const.version_prefix + "/predict", json={'tag': 'bottle', 'image_paths': ''})
            self.assertEqual(res.status_code, expect_status_code)

            # 推論結果 一覧取得
            res = client.get(Const.version_prefix + "/listresults")
            self.assertEqual(res.status_code, expect_status_code)

            # 推論結果 詳細取得
            res = client.get(Const.version_prefix + "/getresult?inference_id=1")
            self.assertEqual(res.status_code, expect_status_code)

            # 推論結果 削除
            res = client.delete(Const.version_prefix + "/delresult?inference_id=1")
            self.assertEqual(res.status_code, expect_status_code)

        with app.test_client() as client:
            # OBFUSCATED_VERIFIER_DUMMY_DIR not found
            _call_all_api(500)

            # Verifier file not found
            os.environ["OBFUSCATED_VERIFIER_DUMMY_DIR"] = "invalid/path"
            _call_all_api(500)

            # PYARMOR_LICENSE not found
            os.environ["OBFUSCATED_VERIFIER_DUMMY_DIR"] = "obfuscated"
            _call_all_api(500)

            # Invalid license
            os.environ["PYARMOR_LICENSE"] = "invalid/path"
            _call_all_api(401)

            # Expired
            os.environ["PYARMOR_LICENSE"] = "api/tests/licenses/paid/expired.lic"
            _call_all_api(401)

            # Trial license
            os.environ["PYARMOR_LICENSE"] = "api/tests/licenses/trial/expired.lic"
            _call_all_api(500)
