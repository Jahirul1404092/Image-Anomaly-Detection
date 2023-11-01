import unittest
import os
import logging
from pathlib import Path
import shutil
import re
import tempfile
import time
FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
import psutil
import threading

from omegaconf import OmegaConf

from api.tests.helper import *
from api.util.dbtool import DbTool
from api.util.const import Const

class IntegTestCase(unittest.TestCase):
    def doTrain(self, idx, with_bad=False, with_mask=False, params={}):
        generator = [createLetters, createShapes][idx]
        with tempfile.TemporaryDirectory() as imgpath:
            trainsize = 50
            l_good, l_bad, l_mask = generator(imgpath, trainsize)
            with self.app.test_client() as client:
                image_ids = []
                category =  ['letters', 'shapes'][idx]
                res = client.post(Const.version_prefix + '/addimage', 
                                json={'image_path': l_good,
                                'image_tag':category})
                image_ids.extend(res.json)
                if with_bad:
                    res = client.post(Const.version_prefix + '/addimage', 
                                    json={'image_path': l_bad,
                                    'group': 'bad',
                                    'image_tag':category})
                    image_ids.extend(res.json)
                    params = merge_dicts(params,{
                        'model':{'normalization_method':'min_max'},
                        'dataset':{'seed':43}
                        })
                else:
                    params = merge_dicts(params, {
                        'dataset':{'seed':43}
                        })
                res = client.post(Const.version_prefix + '/train', 
                                json={'image_tag': category,
                                        'model_tag': category,
                                        'parameters': params})
                # self.assertEqual(res.json, i + 1)
                with DbTool(os.environ['DBPATH']) as db:
                    m = db.getModels(res.json)[0]
                    conf = OmegaConf.load(str(Path(m['model_path']) / 'config.yaml'))
                    self.assertEqual(m['model_id'], res.json)
                    self.assertEqual(m['model_path'], conf.project.path)
                    self.assertEqual(m['tag'], category)
                    self.assertEqual(m['parameters'], params)
        return res

    def doTestPredict(self):
        shutil.rmtree('./results', ignore_errors=True)
        os.environ['HOST_DATA_DIR'] = '/test'
        os.environ['BASE_DATA_DIR'] = '/tmp'
        self.doTrain(0)
        self.doTrain(1)
        with tempfile.TemporaryDirectory() as testdir:
            p_test = Path(testdir)
            (p_test / 'letters').mkdir()
            (p_test / 'shapes').mkdir()
            createLetters(str(p_test / 'letters'), 10)
            createShapes(str(p_test / 'shapes'), 8, '日本語')
            hostdir = os.environ.get('HOST_DATA_DIR', '/test')
            basedir = os.environ.get('BASE_DATA_DIR', '/tmp')
            l_letters = [re.sub(f"^{basedir}", hostdir, str(p)) 
                         for p in (p_test / 'letters').glob('*.png')]
            l_shapes = [re.sub(f"^{basedir}", hostdir, str(p)) 
                         for p in (p_test / 'shapes').glob('*.png')]
            with self.app.test_client() as client:
                # predict no model served
                ret = client.post(Const.version_prefix + '/predict',
                                  json={"model_id":1, "image_paths":l_letters})
                self.assertEqual(ret.status_code, 404)
                mem = psutil.virtual_memory() 
                print("#### before serve", mem.percent, mem.used, mem.available)
                # serve 1 by model id
                ret = client.post(Const.version_prefix + '/servemodel',
                                json={'model_id': 1, 'parameters':{'dataset':{'seed':4200}}})
                self.assertEqual(ret.status_code, 200)
                mem = psutil.virtual_memory() 
                print("#### after serve", mem.percent, mem.used, mem.available)

                # predict 1 error case both model id and tag
                ret = client.post(Const.version_prefix + '/predict',
                                  json={"model_id":1, "tag":"letters", "image_paths":l_letters})
                self.assertEqual(ret.status_code, 400)
                # predict 1 error case niether both model id nor tag
                ret = client.post(Const.version_prefix + '/predict',
                                  json={"image_paths":l_letters})
                self.assertEqual(ret.status_code, 400)
                # predict 1 normal
                ret = client.post(Const.version_prefix + '/predict',
                                  json={"model_id":1, "image_paths":l_letters})
                self.assertEqual(ret.status_code, 200)
                self.assertEqual(len(ret.json), len(l_letters))
                imgs = list(Path('results/1/patchcore/inference_results/batch_mode/images/prediction/serve_id_1').glob('*.png'))
                self.assertEqual(len(imgs), 12)

                # serve 2 by tag
                mem = psutil.virtual_memory() 
                print("#### before serve 2", mem.percent, mem.used, mem.available)
                ret = client.post(Const.version_prefix + '/servemodel',
                                json={'tag': 'shapes', 'parameters':{'dataset':{'seed':42}}})
                self.assertEqual(ret.status_code, 200)
                mem = psutil.virtual_memory() 
                print("#### after serve 2", mem.percent, mem.used, mem.available)
                # predict 2 normal
                ret = client.post(Const.version_prefix + '/predict',
                                  json={"tag":'shapes', "image_paths":l_shapes})
                imgs = list(Path('results/2/patchcore/inference_results/batch_mode/images/prediction/serve_id_2').glob('*.png'))
                self.assertEqual(len(imgs), 9)
                self.assertTrue(all(['日本語' in str(x) for x in imgs]))
                # same images again. this should fail on windows.
                ret = client.post(Const.version_prefix + '/predict',
                                  json={"tag":'shapes', "image_paths":l_shapes})
                imgs = list(Path('results/2/patchcore/inference_results/batch_mode/images/prediction/serve_id_2').glob('*.png'))
                self.assertEqual(len(imgs), 9)
                self.assertTrue(all(['日本語' in str(x) for x in imgs]))

                self.assertEqual(ret.status_code, 200)
                self.assertEqual(len(ret.json), len(l_shapes))
                # unserver 1 and predict
                ret = client.delete(Const.version_prefix + '/unservemodel?tag=letters')
                self.assertEqual(ret.status_code, 200)
                ret = client.post(Const.version_prefix + '/predict',
                                  json={"tag":"letters", "image_paths":l_letters})
                self.assertEqual(ret.status_code, 404)
                # unserver 2
                ret = client.delete(Const.version_prefix + '/unservemodel?tag=shapes')

    # @sub_test([
    #     dict(data={'dataset':{'split_ratio':0.1}}),
    #     dict(data={'model':{'coreset_sampling_ratio':0.2}})
    # ])
    def doTestServe(self, data):
        shutil.rmtree('./results', ignore_errors=True)
        os.environ['HOST_DATA_DIR'] = '/test'
        os.environ['BASE_DATA_DIR'] = '/tmp'
        self.doTrain(0, params=data)
        self.doTrain(1, params=data)
        with tempfile.TemporaryDirectory() as testdir:
            p_test = Path(testdir)
            (p_test / 'letters').mkdir()
            (p_test / 'shapes').mkdir()
            createLetters(str(p_test / 'letters'), 10)
            createShapes(str(p_test / 'shapes'), 8, '日本語')
            hostdir = os.environ.get('HOST_DATA_DIR', '/test')
            basedir = os.environ.get('BASE_DATA_DIR', '/tmp')
            l_letters = [re.sub(f"^{basedir}", hostdir, str(p)) 
                         for p in (p_test / 'letters').glob('*.png')]
            l_shapes = [re.sub(f"^{basedir}", hostdir, str(p)) 
                         for p in (p_test / 'shapes').glob('*.png')]
            with self.app.test_client() as client:
                # error case both model id and tag
                ret = client.post(Const.version_prefix + '/servemodel',
                                json={'model_id': 1, 'tag': 'letters', 'parameters':{'dataset':{'seed':4200}}})
                self.assertEqual(ret.status_code, 400)
                # error case neither model id nor tag
                ret = client.post(Const.version_prefix + '/servemodel',
                                json={'parameters':{'dataset':{'seed':4200}}})
                self.assertEqual(ret.status_code, 400)
                # error case wrong method
                ret = client.put(Const.version_prefix + '/servemodel',
                                json={'model_id': 1, 'parameters':{'dataset':{'seed':4200}}})
                self.assertEqual(ret.status_code, 405)
                ret = client.delete(Const.version_prefix + '/servemodel',
                                json={'model_id': 1, 'parameters':{'dataset':{'seed':4200}}})
                self.assertEqual(ret.status_code, 405)
                # serve 1 by model id
                ret = client.post(Const.version_prefix + '/servemodel',
                                json={'model_id': 1, 'parameters':{'dataset':{'seed':4200}}})
                self.assertEqual(ret.status_code, 200)
                mem = psutil.virtual_memory() 
                print("#### after serve 1", mem.percent, mem.used, mem.available)
                # serve 2 by tag
                ret = client.post(Const.version_prefix + '/servemodel',
                                json={'tag': 'shapes', 'parameters':{'dataset':{'seed':42}}})
                self.assertEqual(ret.status_code, 200)
                mem = psutil.virtual_memory() 
                print("#### after serve 1", mem.percent, mem.used, mem.available)
                # unserve error method not allowed
                ret = client.post(Const.version_prefix + '/unservemodel?model_id=1')
                self.assertEqual(ret.status_code, 405)
                # unserve error case both model id and tag
                ret = client.delete(Const.version_prefix + '/unservemodel?model_id=1&tag=letters')
                self.assertEqual(ret.status_code, 400)
                # unserve error case niether model id nor tag
                ret = client.delete(Const.version_prefix + '/unservemodel')
                self.assertEqual(ret.status_code, 400)
                # unserve by id
                ret = client.delete(Const.version_prefix + '/unservemodel?model_id=2')
                self.assertEqual(ret.status_code, 200)
                # unserve by tag
                ret = client.delete(Const.version_prefix + '/unservemodel?tag=letters')
                self.assertEqual(ret.status_code, 200)

    def doTestThread(self):
        shutil.rmtree('./results', ignore_errors=True)
        def doOnThread(idx):
            os.environ['HOST_DATA_DIR'] = '/test'
            os.environ['BASE_DATA_DIR'] = '/tmp'
            self.doTrain(idx)
            params = [('letters', createLetters), ('shapes', createShapes)]
            with tempfile.TemporaryDirectory() as testdir:
                hostdir = os.environ.get('HOST_DATA_DIR', '/test')
                basedir = os.environ.get('BASE_DATA_DIR', '/tmp')
                p_test = Path(testdir)
                (p_test / params[idx][0]).mkdir()
                params[idx][1](str(p_test / params[idx][0]), 8, '日本語')
                l_data = [re.sub(f"^{basedir}", hostdir, str(p)) 
                            for p in (p_test / params[idx][0]).glob('*.png')]
                with self.app.test_client() as client:
                    ret = client.post(Const.version_prefix + '/servemodel',
                                json={'tag': params[idx][0]})
                    self.assertEqual(ret.status_code, 200)
                    ret = client.post(Const.version_prefix + '/predict',
                                      json={'tag': params[idx][0], 'image_paths': l_data})
                    self.assertEqual(ret.status_code, 200)
                    self.assertEqual(len(ret.json), len(l_data))
                    ret = client.delete(Const.version_prefix + f'/unservemodel?tag={params[idx][0]}')
        th0 = threading.Thread(target=doOnThread, args=[0])
        th1 = threading.Thread(target=doOnThread, args=[1])
        th0.start()
        th1.start()
        th0.join()
        th1.join()

    def doTestPredictOutputResults(self, data):
        os.environ['HOST_DATA_DIR'] = '/test'
        os.environ['BASE_DATA_DIR'] = '/tmp'
        self.doTrain(0, params=data)
        # self.doTrain(1, params=data)
        with tempfile.TemporaryDirectory() as testdir:
            p_test = Path(testdir)
            (p_test / 'letters').mkdir()
            (p_test / 'shapes').mkdir()
            createLetters(str(p_test / 'letters'), 10)
            createShapes(str(p_test / 'shapes'), 8, '日本語')
            hostdir = os.environ.get('HOST_DATA_DIR', '/test')
            basedir = os.environ.get('BASE_DATA_DIR', '/tmp')
            l_letters = [re.sub(f"^{basedir}", hostdir, str(p)) 
                         for p in (p_test / 'letters').glob('*.png')]
            l_shapes = [re.sub(f"^{basedir}", hostdir, str(p)) 
                         for p in (p_test / 'shapes').glob('*.png')]
            with self.app.test_client() as client:
                ret = client.post(Const.version_prefix + '/servemodel',
                                json={'tag': 'letters', 'parameters': data})
                ret = client.post(Const.version_prefix + '/predict',
                                json={"tag":"letters", "image_paths":l_letters, "save": 'all'})
                client.delete(Const.version_prefix + '/unservemodel?tag=letters')
        return ret
