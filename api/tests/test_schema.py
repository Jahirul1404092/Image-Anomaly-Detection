import unittest
from jsonschema import validate, ValidationError
from api.model.schemas import Schemas

class TestCase(unittest.TestCase):
    def testAddImageSchema(self):
        q = {
                "image_path": [
                    "/abc/def.png",
                    "/def/xyz.jpg"
                ],
                "group": "good",
                "image_tag": "cup"
            }
        validate(q, Schemas.addimage_schema)
        del q['image_tag']
        validate(q, Schemas.addimage_schema)
        # error cases
        q['group'] = 'marginal'
        with self.assertRaises(ValidationError):
            validate(q, Schemas.addimage_schema)
        q['group'] = 'bad'
        validate(q, Schemas.addimage_schema)
        q['image_path'] = []
        with self.assertRaises(ValidationError):
            validate(q, Schemas.addimage_schema)

    def testTrainSchema(self):
        q = {
                "image_id": [1, 2, 3, 8, 9],
                "model_tag": 'bottle',
                "parameters":{
                    "image_size": "244",
                    "model:backbone": "resnet18"
                }
            }
        validate(q, Schemas.train_schema)
        q['image_id'].append("10")
        with self.assertRaises(ValidationError):
            validate(q, Schemas.train_schema)
        q['image_id'].pop()
        q["image_tag"]="cup"
        with self.assertRaises(ValidationError):
            validate(q, Schemas.train_schema)
        del q["image_id"]
        validate(q, Schemas.train_schema)
        q["parameters"]['image_size']=244
        with self.assertRaises(ValidationError):
            validate(q, Schemas.train_schema)
        del q["parameters"]
        validate(q, Schemas.train_schema)
        q["parameters"] = {
            "dataset": {
                "seed": "42"
            }
        }
        validate(q, Schemas.train_schema)
        del q['model_tag']
        q['image_id'] = []
        del q['image_tag']
        with self.assertRaises(ValidationError):
            validate(q, Schemas.train_schema)

    def testPredictSchema(self):
        q = {
                "model_id": 11,
                "image_paths": [
                    "/abc/def.png",
                    "/xyz/pqr.jpg"
                    ],
                "parameters":{
                    "image_size": "244",
                    "train:lr": "0.01"
                }
            }
        validate(q, Schemas.predict_schema)
        q['save'] = 'all'
        validate(q, Schemas.predict_schema)
        q['save'] = 'none'
        validate(q, Schemas.predict_schema)
        q['save'] = 'ok_only'
        validate(q, Schemas.predict_schema)
        q['save'] = 'ng_only'
        validate(q, Schemas.predict_schema)
        q['tag']='bottle'
        with self.assertRaises(ValidationError):
            validate(q, Schemas.predict_schema)
        del q['model_id']
        validate(q, Schemas.predict_schema)
        q['parameters']['image_size']=244
        with self.assertRaises(ValidationError):
            validate(q, Schemas.predict_schema)
        q['parameters']['image_size']="244"
        del q['tag']
        with self.assertRaises(ValidationError):
            validate(q, Schemas.predict_schema)
        q['model_id'] = '11'
        with self.assertRaises(ValidationError):
            validate(q, Schemas.predict_schema)
        q['model_id'] = 11
        q['image_paths'] = []
        with self.assertRaises(ValidationError):
            validate(q, Schemas.predict_schema)
