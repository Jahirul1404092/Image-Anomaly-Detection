import datetime
import enum

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, BigInteger, Enum, JSON

"""
database entity classes
"""

Base = declarative_base()

def initialize(dbpath:str):
    engine = sqlalchemy.create_engine(f'sqlite:///{dbpath}', echo=False)
    Base.metadata.create_all(bind=engine)

class TrainingImages(Base):
    image_id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    image_path = Column(String(255))
    tag = Column(String(128))
    group = Column(String(4))
    created = Column(DateTime, default=datetime.datetime.now())
    __tablename__ = 'training_images'
    __table_args__=({"sqlite_autoincrement": True})
    def as_dict(self):
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "tag": self.tag,
            "group": self.group,
            "created": self.created
            }

class Model(Base):
    model_id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    model_path = Column(String(255))
    model_type = Column(String(20))
    inference_path = Column(String(255))
    tag = Column(String(128), unique=True)
    version = Column(Integer)
    image_ids = Column(JSON)
    parameters = Column(JSON)
    created = Column(DateTime, default=datetime.datetime.now())
    __tablename__ = 'models'
    __table_args__=({"sqlite_autoincrement": True})
    def as_dict(self):
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "inference_path": self.inference_path,
            "tag": self.tag,
            "image_ids": self.image_ids,
            "parameters": self.parameters,
            "version": self.version,
            "created": self.created
            }


class InferenceMode(enum.Enum):
    batch = 'batch'
    online = 'online'


class Serving(Base):
    serve_id = Column(Integer, primary_key=True, nullable=False)
    model_id = Column(Integer, nullable=False)
    parameter = Column(JSON, nullable=True)
    mode = Column(Enum(InferenceMode), nullable=True)
    served = Column(DateTime, default=datetime.datetime.now())
    __tablename__ = 'serving'
    __table_args__ = ({"sqlite_autoincrement": True})
    def as_dict(self):
        return {
            "serve_id": self.serve_id,
            "model_id": self.model_id,
            "parameter": self.parameter,
            "mode": self.mode,
            "served": self.served
            }

class Inference(Base):
    inference_id = Column(Integer, primary_key=True, nullable=False)
    serve_id = Column(Integer, nullable=False)
    image_path = Column(String(255), nullable=False)
    result_json = Column(JSON, nullable=True)
    result_path = Column(String(255))
    csv_path = Column(String(255), nullable=True)
    infered = Column(DateTime, default=datetime.datetime.now())
    __tablename__ = 'inference'
    __table_args__ = ({"sqlite_autoincrement": True})
    def as_dict(self):
        return {
            "inference_id": self.inference_id,
            "image_path": self.image_path,
            "serve_id": self.serve_id,
            "result_json": self.result_json,
            "result_path": self.result_path,
            "csv_path": self.csv_path,
            "infered": self.infered
            }
