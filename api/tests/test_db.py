import unittest
from unittest.mock import patch
import sqlite3
import os
import logging
FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
import datetime

from api.tests.helper import sub_test
from api.util.dbtool import *
from api.util.const import Const

class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get('DBPATH', None):
            del os.environ['DBPATH']
        cls.dbpath = '/tmp/test_db.sqlite'
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.dbpath)
        return super().tearDownClass()
    
    def setUp(self):
        return super().setUp()
    
    def tearDown(self):
        with DbTool(path=self.dbpath) as db:
            db.dropAll()
        return super().tearDown()
    
    def testCreate(self):
        db = DbTool(path=self.dbpath)
        con = sqlite3.connect(self.dbpath)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        l_tabels = [t[0] for t in cur.fetchall()]
        con.close()
        self.assertEqual(sorted(l_tabels), sorted(['serving',
                                                   'inference',
                                                   'sqlite_sequence',
                                                   'training_images',
                                                   'models']))

    def testTrainingImages(self):
        with DbTool(path=self.dbpath) as db:
            res = db.addTrainingImages(['/abc/def.png',
                                        '/abc/こんにちは.png'], 'bottle')
            self.assertEqual(res, [1, 2])
            res = db.addTrainingImages(['/mn/xyz.png'], 'cup')
            self.assertEqual(res, [3])
            res = db.addTrainingImages(['/abc/mnk.png'], group='bad')
            self.assertEqual(res, [4])
            l_img = db.getTrainingImages()
            self.assertEqual(len(l_img), 4)
            self.assertEqual(l_img[0]['image_path'],'/abc/def.png')
            self.assertEqual(l_img[0]['group'], 'good')
            self.assertEqual(l_img[1]['image_path'],'/abc/こんにちは.png')
            self.assertEqual(l_img[2]['tag'],'cup')
            self.assertIsNone(l_img[3]['tag'])
            self.assertEqual(l_img[3]['group'], 'bad')
            l_img = db.getTrainingImages(3)
            self.assertEqual(len(l_img), 1)
            self.assertEqual(l_img[0]['image_path'],'/mn/xyz.png')
            self.assertEqual(l_img[0]['tag'],'cup')
            l_img = db.getTrainingImages('bottle')
            self.assertEqual(len(l_img), 2)
            self.assertEqual(l_img[0]['image_path'],'/abc/def.png')
            self.assertEqual(l_img[1]['image_path'],'/abc/こんにちは.png')
            res = db.delTrainingImages('cup')
            self.assertGreater(res, 0)
            self.assertEqual(len(db.getTrainingImages('cup')), 0)
            self.assertEqual(len(db.getTrainingImages()), 3)
            res = db.delTrainingImages([1, 4])
            self.assertGreater(res, 0)
            res = db.getTrainingImages()
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0]['image_id'], 2)
            res = db.addTrainingImages(['/abc/mnk.png'], group='bad')
            self.assertEqual(res, [5])
            with patch.object(db.sess, 'bulk_save_objects', 
                              side_effect=ValueError('Expected Mock Error')):
                res = db.addTrainingImages(['/abc/000.png'])
                self.assertEqual(len(res), 0)
            with patch.object(db.sess, 'query', side_effect=ValueError('Expected Mock Error')):
                res = db.getTrainingImages()
                self.assertEqual(len(res), 0)
                res = db.delTrainingImages()
                self.assertEqual(res, -1)

    def testModels(self):
        with DbTool(path=self.dbpath) as db:
            params = {'dataset':{'seed': 43, 'category':'cup'}}
            def insertModel():
                cur_id = db.getLastModelId() + 1
                res = db.addModel('/abc/def.ckp', [1, 2, 3, 4, 5], 'patchcore', 
                                  params, 'bottle')
                self.assertEqual(res, cur_id)
                res = db.addModel('/xyz/pqr.ckp', [5, 6, 7, 8], 'patchcore', 
                                  d_param=None, tag='cup')
                self.assertEqual(res, cur_id + 1)
                res = db.addModel('/abc/nmk.ckp', [100, 101, 102, 103, 104, 105], 'patchcore', 
                                  d_param=None, tag='glass', version=Const.model_version + 1)
                self.assertEqual(res, cur_id + 2)
                res = db.addModel('/abc/xyz.ckp', [50, 51, 52, 53, 54, 55], 'patchcore')
                self.assertEqual(res, cur_id + 3)
            insertModel()
            l_model = db.getModels()
            self.assertEqual(len(l_model), 3)
            self.assertEqual(l_model[0]['model_id'], 1)
            self.assertEqual(l_model[1]['model_path'], '/xyz/pqr.ckp')
            self.assertIsNone(l_model[2]['tag'])
            l_model = db.getModels(version=Const.model_version + 1)
            self.assertEqual(len(l_model), 1)
            l_model = db.getModels('bottle')
            self.assertEqual(len(l_model), 1)
            l_model = db.getModels(2)
            self.assertEqual(len(l_model), 1)
            res = db.delModel('cup')
            # 2 will be deleted
            self.assertEqual(res, [2])
            self.assertEqual(len(db.getModels()), 2)
            res = db.delModel(version=Const.model_version + 1)
            # 3 will be deleted
            self.assertEqual(res, [3])
            self.assertEqual(len(db.getModels()), 2)
            self.assertEqual(len(db.getModels(version=Const.model_version + 1)), 0)
            res = db.delModel(1, version=Const.model_version + 1)
            self.assertEqual(res, [])
            # nothing deleted
            res = db.delModel([1, 4])
            self.assertEqual(res, [1, 4])
            # delete everything
            self.assertEqual(len(db.getModels()), 0) 

            insertModel()
            l_mid = [x['model_id'] for x in db.getModels()]
            res = db.delModel(l_mid[1:], version=Const.model_version)
            # 2, 4 will be deleted
            self.assertEqual(res, l_mid[1:])
            l_model = db.getModels()
            self.assertEqual(len(l_model), 1)
            self.assertEqual(l_model[0]['model_id'], l_mid[0])
            m_res = db.getModels('glass', version=Const.model_version + 1)
            res = db.delModel('glass', version=Const.model_version + 1)
            # 3 will be deleted
            self.assertEqual(res, [m['model_id'] for m in m_res])
            l_model = db.getModels(version=Const.model_version + 1)
            self.assertEqual(len(l_model), 0)
            l_mid = [x['model_id'] for x in db.getModels()]
            res = db.delModel()
            self.assertEqual(res, l_mid)
            # nothing
            self.assertEqual(len(db.getModels()), 0)
            insertModel()
            last_id = db.getLastModelId()
            res = db.getModels(['cup', 'bottle', last_id])
            self.assertEqual(len(res), 3)
            res = db.delModel(['cup', 'bottle', last_id])
            self.assertEqual(len(res), 3)
            with patch.object(db.sess, 'add', 
                              side_effect=ValueError('Expected Mock Error')):
                res = db.addModel('/zzz/aaa.png', [5, 6, 7, 8], 'patchcore')
                self.assertEqual(res, 0)
            with patch.object(db.sess, 'query', 
                              side_effect=ValueError('Expected Mock Error')):
                res = db.getModels('cup')
                self.assertEqual(len(res), 0)
            with patch.object(db.sess, 'query', 
                              side_effect=ValueError('Expected Mock Error')):
                res = db.delModel()
                self.assertFalse(res)

    @sub_test([
        dict(data=[
            ('/abc/def.png', 'bottle', 'good'),
            ('/abc/bcd.png', 'cup', 'good'),
            ('/abd/efg.png', 'bottle', 'bad'),
            ('/abd/qrz.png', 'cup', 'bad'),
            ('/ace/efg.png', 'bottle', 'mask'),
            ('/ace/qrz.png', 'cup', 'mask')
        ]),
        dict(data=[
            ('/abc/def.png', 'bottle', 'good'),
            ('/abc/bcd.png', 'cup', 'good'),
            ('/abd/efg.png', 'bottle', 'bad'),
            ('/abd/qrz.png', 'cup', 'bad')
        ]),
    ])
    def testDataset(self, data):
        with DbTool(path=self.dbpath) as db:
            # get by list
            a = {'good':[], 'bad':[], 'mask':[]}
            for path, tag, group in data:
                db.addTrainingImages([path], tag, group)
                a[group].append(path)
            l_id = db.getTrainingImages()
            q, l_img = db.getDataset([x['image_id'] for x in l_id])
            self.assertEqual(q, a)
            db.delTrainingImages()
            # get by tag
            a = {'good':[], 'bad':[], 'mask':[]}
            for path, tag, group in data:
                db.addTrainingImages([path], tag, group)
                if tag == 'cup':
                    a[group].append(path)
            q, l_img = db.getDataset('cup')
            self.assertEqual(q, a)
            db.delTrainingImages()
            self.assertIsNone(db.getDataset('bottle')[0])

    def testLastModelId(self):
        with DbTool(path=self.dbpath) as db:
            res = db.getLastModelId()
            self.assertEqual(res, 0)
            res = db.addModel('/abc/def.ckp', [1, 2, 3, 4, 5, 6], 'patchcore',
                              tag='bottle')
            self.assertEqual(res, 1)
            res = db.addModel('/xyz/pqr.ckp', [11, 12, 13, 14, 15], 'patchcore', 
                              tag='cup')
            self.assertEqual(res, 2)
            res = db.delModel(2)
            self.assertEqual(res, [2])
            res = db.getModels()
            self.assertEqual(len(res), 1)
            res = db.getLastModelId()
            self.assertEqual(res, 2)
            with patch.object(db.sess, 'execute', 
                                side_effect=ValueError('Expected Mock Error')):
                res = db.getLastModelId()
                self.assertIsNone(res)

    def testUpdateInferencePath(self):
        with DbTool(path=self.dbpath) as db:
            params = {'dataset':{'seed': 43, 'category':'cup'}}
            cur_id = db.getLastModelId() + 1
            res = db.addModel('/abc/def.ckp', [1, 2, 3, 4, 5], 'patchcore', 
                                params, 'bottle')
            self.assertEqual(res, cur_id)
            res = db.addModel('/xyz/pqr.ckp', [5, 6, 7, 8], 'patchcore', 
                                d_param=None, tag='cup')
            self.assertEqual(res, cur_id + 1)
            res = db.getModels()
            self.assertIsNone(res[0]['inference_path'])
            res = db.updateInferencePath(res[0]['model_id'], '/abc/def')
            res = db.getModels()
            self.assertEqual(res[0]['inference_path'], '/abc/def')
            res = db.updateInferencePath(res[0]['model_id'], None)
            res = db.getModels()
            self.assertIsNone(res[0]['inference_path'])

    def testInference(self):
        with DbTool(path=self.dbpath) as db:
            res = db.addModel('/abc/def.ckp', [1, 2, 3, 4, 5, 6], 'patchcore',
                              tag='bottle')
            self.assertEqual(res, 1)
            res = db.addModel('/xyz/pqr.ckp', [11, 12, 13, 14, 15], 'patchcore', 
                              tag='cup')
            self.assertEqual(res, 2)
            # error add serving
            res = db.addServing(None, {'param1':1,'param2':3}, mode='batch')
            self.assertEqual(res, 0)
            # normal add serving
            res = db.addServing(1, {'param1':1,'param2':3}, mode='batch')
            self.assertEqual(res, 1)
            res = db.addServing(2, None, 'online')
            self.assertEqual(res, 2)
            res = db.getLastServeId()
            self.assertEqual(res, 2)
            # empty inference
            res = db.listInferences()
            self.assertEqual(res, {})
            # error add inference
            res = db.addInference(None, 'img1.png', {'result1':1, 'result2':3}, '/home/ubuntu', '/csv/path')
            self.assertEqual(res, 0)
            # normal add inference
            res = db.addInference(1, 'img2.png', {'result1':1, 'result2':3}, '/home/ubuntu', '/csv/path')
            self.assertEqual(res, 1)
            res = db.addInference(2, 'img3.png', {'result1':9, 'result2':10}, '/home/debian', '/csv/path')
            self.assertEqual(res, 2)
            # error list inferences
            with patch.object(db.sess, 'execute', 
                              side_effect=ValueError('Expected Mock Error')):
                res = db.listInferences()
                self.assertEqual(res, {})
            # normal list inferences
            res = db.listInferences()
            self.assertEqual(res, {1: 'img2.png',  2: 'img3.png'})
            res = db.getInference(1)
            self.assertEqual(res['inference_id'], 1)
            self.assertEqual(res['image_path'], 'img2.png')
            self.assertEqual(res['result_json'], {"result1": 1, "result2": 3})
            self.assertEqual(res['result_path'], "/home/ubuntu")
            self.assertEqual(res['model_id'], 1)
            self.assertEqual(res['parameter'], {'param1': 1, 'param2': 3})
            self.assertAlmostEqual(
                datetime.datetime.fromisoformat(res['infered']).timestamp(),
                datetime.datetime.now().timestamp(), -1)
            res = db.getInference(2)
            self.assertEqual(res['inference_id'], 2)
            self.assertEqual(res['image_path'], 'img3.png')
            self.assertEqual(res['result_json'], {"result1": 9, "result2": 10})
            self.assertEqual(res['result_path'], "/home/debian")
            self.assertEqual(res['model_id'], 2)
            self.assertEqual(res['parameter'], {})
            self.assertAlmostEqual(
                datetime.datetime.fromisoformat(res['infered']).timestamp(),
                datetime.datetime.now().timestamp(), -1)
            res = db.getInference(5)
            self.assertEqual(res, {})
            # delete empty inference
            res = db.delInference(5)
            self.assertFalse(res)
            # error delete inference
            with patch.object(db.sess, 'execute', 
                              side_effect=ValueError('Expected Mock Error')):
                res = db.delInference(1)
                self.assertFalse(res)
            # normal delete inference
            res = db.delInference(1)
            self.assertTrue(res)
