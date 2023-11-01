import os
import json
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
class TestCase(unittest.TestCase):
    def setUp(self):
        os.environ['DBPATH'] = '/tmp/test_db.sqlite'

    def tearDown(self):
        if os.path.exists(os.environ['DBPATH']):
            os.remove(os.environ['DBPATH'])

    def testListImage(self):
        with app.test_client() as client:
            res = client.get(Const.version_prefix + '/listimages')
            self.assertEqual(res.status_code, 404)
        with DbTool() as db:
            res = db.addTrainingImages(['/abc/def.png',
                                        '/abc/ghq.png'], 'bottle')
            res = db.addTrainingImages(['/mn/xyz.png'], 'cup')
            res = db.addTrainingImages(['/abc/mnk.png'])
        with app.test_client() as client:
            res = client.get(Const.version_prefix + '/listimages')
            self.assertEqual(res.status_code, 200)
            self.assertEqual(json.loads(res.data),
                             json.loads('[{"image_id": 1, "image_path": "/abc/def.png", "image_tag": "bottle"}, {"image_id": 2, "image_path": "/abc/ghq.png", "image_tag": "bottle"}, {"image_id": 3, "image_path": "/mn/xyz.png", "image_tag": "cup"}, {"image_id": 4, "image_path": "/abc/mnk.png", "image_tag": null}]'))
            with patch('api.controller.images.DbTool.getTrainingImages', side_effect=ValueError('expected mock error')):
                res = client.get(Const.version_prefix + '/listimages')
                self.assertEqual(res.status_code, 500)
            with patch('api.controller.images.request') as MockRequest:
                MockPath = Mock()
                MockPath.endswith.return_value = False
                MockRequest.path = MockPath
                res = client.get(Const.version_prefix + '/listimages')
                self.assertEqual(res.status_code, 405)

    def testImageDetails(self):
        with app.test_client() as client:
            res = client.get(Const.version_prefix + '/imagedetails?image_id=1')
            self.assertEqual(res.status_code, 404)
        with DbTool() as db:
            db.addTrainingImages(['/abc/def.png',
                                        '/abc/ghq.png'], 'bottle')
            db.addTrainingImages(['/mn/xyz.png'], 'cup')
            db.addTrainingImages(['/abc/mnk.png'])
        with app.test_client() as client:
            res = client.get(Const.version_prefix + '/imagedetails?image_id=1')
            self.assertEqual(res.json, [{'image_id': 1, 'image_path': '/abc/def.png', 'tag': 'bottle', 'group': 'good'}])
            res = client.get(Const.version_prefix + '/imagedetails?image_tag=bottle')
            self.assertEqual(res.json, [{'image_id': 1, 'image_path': '/abc/def.png', 'tag': 'bottle', 'group': 'good'},{'image_id': 2, 'image_path': '/abc/ghq.png', 'tag': 'bottle', 'group': 'good'}])
            res = client.get(Const.version_prefix + '/imagedetails?image_tag=bottle&image_id=1')
            self.assertEqual(res.status_code, 400)
            res = client.get(Const.version_prefix + '/imagedetails')
            self.assertEqual(res.status_code, 400)
            with patch('api.controller.images.DbTool.getTrainingImages', side_effect=ValueError('Expected mock error')):
                res = client.get(Const.version_prefix + '/imagedetails?image_id=1')
                self.assertEqual(res.status_code, 500)

    def testAddImage(self):
        with app.test_client() as client:
            res = client.get(Const.version_prefix + '/listimages')
            self.assertEqual(res.status_code, 404)
            q = {
                "image_path": [
                    "/abc/def.png",
                    "/xyz/qrs.jpg",
                    "/nmk/efg.png"
                ],
                "group": "good",
                "image_tag": "bottle"
            }
            res = client.post(Const.version_prefix + '/addimage', json=q)
            self.assertEqual(res.json, [1, 2, 3])
            with DbTool() as db:
                res = db.getTrainingImages()
                self.assertEqual(len(res), 3)
            res = client.post(Const.version_prefix + '/addimage', json={"image_path":[]})
            self.assertEqual(res.status_code, 400)
            del q['group']
            del q['image_tag']
            res = client.post(Const.version_prefix + '/addimage', json=q)
            self.assertEqual(res.json, [4, 5, 6])
            # 日本語
            jp = {
                "image_path": [
                    "/フォルダ1/日本語.png",
                    "/フォルダ1/英語.png"
                ],
                "group": "bad",
                "image_tag": "タグ1"
            }
            res = client.post(Const.version_prefix + '/addimage', data=json.dumps(jp),
                              headers={'content-type':'application/json'})
            with DbTool() as db:
                imgs = db.getTrainingImages("タグ1")
                self.assertEqual(len(imgs), 2)
                self.assertEqual(sorted([x['image_path'] for x in imgs]), 
                                 sorted(["/フォルダ1/日本語.png", "/フォルダ1/英語.png"]))
            # 日本語cp932 fail
            cp932 = json.dumps(jp).encode('cp932').decode('unicode_escape').encode('cp932')
            res = client.post(Const.version_prefix + '/addimage', data=cp932, 
                              headers={'content-type':'application/json'})
            self.assertEqual(res.status_code, 400)
            # 日本語cp932 ok なぜか fail, charset=cp932 が効いてない。受信側では utf-8 となるバグ？
            # res = client.post(Const.version_prefix + '/addimage', data=cp932, 
            #                   content_type='application/json', charset='cp932')
            # self.assertEqual(res.status_code, 200)
            with patch('api.controller.images.DbTool.addTrainingImages', return_value=[]):
                res = client.post(Const.version_prefix + '/addimage', json=q)
            self.assertEqual(res.status_code, 500)
            with patch('api.controller.images.DbTool.addTrainingImages', side_effect=ValueError):
                res = client.post(Const.version_prefix + '/addimage', json=q)
            self.assertEqual(res.status_code, 500)

    def testDelImage(self):
        with app.test_client() as client:
            res = client.delete(Const.version_prefix + '/delimage', 
                                json={'image_id':[1,3]})
            self.assertEqual(res.status_code, 404)
        with DbTool() as db:
            db.addTrainingImages(['/abc/def.png',
                                        '/abc/ghq.png'], 'bottle')
            db.addTrainingImages(['/mn/xyz.png'], 'cup')
            db.addTrainingImages(['/abc/mnk.png'])
        with app.test_client() as client:
            res = client.delete(Const.version_prefix + '/delimage', 
                                json={'image_id':[1,3]})
            self.assertEqual(res.status_code, 200)
        with DbTool() as db:
            res = db.getTrainingImages()
            self.assertEqual(len(res), 2)
        with app.test_client() as client:
            res = client.delete(Const.version_prefix + '/delimage', 
                                json={'image_id':[2],'image_tag':'cup'})
            self.assertEqual(res.status_code, 400)
            res = client.delete(Const.version_prefix + '/delimage', 
                                json={'image_id':2})
            self.assertEqual(res.status_code, 400)
            res = client.delete(Const.version_prefix + '/delimage', 
                                json={'img_id':[2]})
            self.assertEqual(res.status_code, 400)
            with patch('api.controller.images.DbTool.delTrainingImages', side_effect=ValueError('Expected mock error')):
                res = client.delete(Const.version_prefix + '/delimage', 
                                    json={'image_id':[2]})
                self.assertEqual(res.status_code, 500)
            res = client.delete(Const.version_prefix + '/delimage', 
                                json={'image_tag':'bottle'})
            self.assertEqual(res.status_code, 200)
            with DbTool() as db:
                res = db.getTrainingImages()
                self.assertEqual(len(res), 1)
            with patch('api.controller.images.DbTool.delTrainingImages', return_value=-1):
                res = client.delete(Const.version_prefix + '/delimage', 
                                    json={'image_id':[2]})
                self.assertEqual(res.status_code, 500)
