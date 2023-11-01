import os
from typing import Union, Literal
import logging
logger = logging.getLogger(name=__file__)
import json

import sqlalchemy
from sqlalchemy import or_
from sqlalchemy.sql import text as sqltext, func
from sqlalchemy.orm import sessionmaker, aliased
from api.util.const import Const
from api.model.entity import TrainingImages, Model, initialize, Inference, Serving
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
# when fk constraints are necessary
# def _fk_pragma_on_connect(dbapi_con, con_record):
#     dbapi_con.execute('pragma foreign_keys=ON')

class DbTool():
    """
    db tool
    use context manager
    """
    def __init__(self, path:str=None):
        dbpath = os.environ.get('DBPATH', path)
        self.engine = sqlalchemy.create_engine(f'sqlite:///{dbpath}')
        # sqlalchemy.event.listen(self.engine, 'connect', _fk_pragma_on_connect)
        initialize(dbpath)
    
    def __enter__(self):
        self.sess = sessionmaker(bind=self.engine)()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def addTrainingImages(self, l_path:list, tag:str=None, group:str='good'):
        """
        add trainig images
        in:
            l_path: list of path to the image file
            tag: user convenience
            group: good or bad
        out:
            list of image_id
            empty list on error
        """
        try:
            res = self.sess.execute("SELECT * FROM sqlite_sequence WHERE name = 'training_images'").one_or_none() or [0, 0]
            n_id = res[1]
            # n_id = self.sess.execute(sql).one()[1]
            ret = self.sess.bulk_save_objects(
                [TrainingImages(image_path=path,
                                tag=tag,
                                group=group)
                    for path in l_path], return_defaults=True)
            self.sess.commit()
            return list(range(n_id + 1, n_id + 1 + len(l_path)))
        except Exception as e:
            logger.error(str(e))
            self.sess.rollback()
            return []
    
    def getTrainingImages(self, u_id:Union[str, int]=None):
        """
        get trainig images
        in:
            i_id: image_id or tag
                    or None for everything
        out:
            list of dict
            empty list on error
        """
        try:
            l_row = []
            if u_id is None:
                rows = self.sess.query(TrainingImages).all()
            elif isinstance(u_id, str):
                #i_id is tag
                rows = self.sess.query(TrainingImages).filter(TrainingImages.tag==u_id).all()
            else:
                rows = self.sess.query(TrainingImages).filter(TrainingImages.image_id==u_id).all()
            for row in rows:
                l_row.append(row.as_dict())
            return l_row
        except Exception as e:
            logger.error(str(e))
            return []
    
    def getDataset(self, u_id:Union[list, str]):
        """
        get dataset
        in:
            list of image_id or tag
        out:
            dataset dict
            or None on failure
        """
        try:
            if isinstance(u_id, str):
                rows = self.sess.query(TrainingImages).filter(TrainingImages.tag==u_id).all()
                l_id = [r.image_id for r in rows]
            else:
                l_id = u_id
                rows = self.sess.query(TrainingImages).filter(TrainingImages.image_id.in_(u_id)).all()
            d_path = {'good':[], 'bad':[], 'mask':[]}
            if len(rows) == 0:
                raise(ValueError('No Data'))
            for row in rows:
                d_path[row.group].append(row.image_path)
            return d_path, l_id
        except Exception as e:
            logger.error(str(e))
            return None, None
    
    def delTrainingImages(self, u_id:Union[list, str]=None):
        """
        delete trainig images
        in:
            l_id: list of image_id or tag 
                    or None for everything
        out:
            connt: count of deleted images or -1 on error
        """
        try:
            if u_id is None:
                rows = self.sess.query(TrainingImages)
            elif isinstance(u_id, str):
                rows = self.sess.query(TrainingImages).filter(TrainingImages.tag==u_id)
            else:
                rows = self.sess.query(TrainingImages).filter(TrainingImages.image_id.in_(u_id))
            count = rows.count()
            rows.delete()
            self.sess.commit()
            return count
        except Exception as e:
            logger.error(str(e))
            self.sess.rollback()
            return -1

    def addModel(self, path:str, l_image:list, model_type:str, d_param:dict=None, 
                 tag:str=None, version:int=None):
        """
        add a model weight path
        in:
            path: path to model weight file
            l_image: list of image ids
            model_type: model name
            d_param: dict of parameters
            tag: user convenience
            version: model version
        out:
            model_id
            0 on failure
        """
        try:
            modelversion = Const.model_version if version is None else version
            m = Model(
                model_path = path,
                image_ids = l_image,
                model_type = model_type,
                parameters = d_param,
                version = modelversion,
                tag = tag
            )
            self.sess.add(m)
            self.sess.commit()
            sql = sqltext("select max(model_id) from models")
            n_id = self.sess.execute(sql).one()[0]
            return n_id
        except Exception as e:
            self.sess.rollback()
            logger.error(str(e))
            return 0

    def getModels(self, u_id:Union[int, str, list]=None, version:int=None):
        """
        get models
        in: 
            m_id: model id or tag or list of them
                    or None for everything
            version: model version
        out:
            dict of model
        """
        try:
            models = []
            mv = Const.model_version if version is None else version
            if u_id is None:
                rows = self.sess.query(Model).filter(Model.version==mv).all()
            elif isinstance(u_id, str):
                rows = self.sess.query(Model).filter(Model.tag==u_id, Model.version==mv).all()
            elif isinstance(u_id, int):
                rows = self.sess.query(Model).filter(Model.model_id==u_id, Model.version==mv).all()
            else:
                tags = [x for x in u_id if isinstance(x, str)]
                imgs = [x for x in u_id if isinstance(x, int)]
                rows = self.sess.query(Model).filter(
                    or_(Model.model_id.in_(imgs), Model.tag.in_(tags)), Model.version==mv).all()
                
            for row in rows:
                models.append(row.as_dict())
            return models
        except Exception as e:
            logger.error(str(e))
            return []
        
    def delModel(self, u_id:Union[int, str, list]=None, version=None):
        """
        delete model weight path
        in:
            u_id: list of model id or tag
                    or None for everything
            version: model version
        out:
            True: success
            False: failure
        """
        try:
            if u_id is None:
                if version is None:
                    rows = self.sess.query(Model)
                else:
                    rows = self.sess.query(Model).filter(Model.version==version)
            elif isinstance(u_id, str):
                if version is None:
                    rows = self.sess.query(Model).filter(Model.tag==u_id)
                else:
                    rows = self.sess.query(Model).filter(Model.tag==u_id, Model.version==version)
            elif isinstance(u_id, int):
                if version is None:
                    rows = self.sess.query(Model).filter(Model.model_id==u_id)
                else:
                    rows = self.sess.query(Model).filter(Model.model_id==u_id, Model.version==version)
            else:
                tags = [x for x in u_id if isinstance(x, str)]
                imgs = [x for x in u_id if isinstance(x, int)]
                if version is None:
                    rows = self.sess.query(Model).filter(
                        or_(Model.model_id.in_(imgs), Model.tag.in_(tags)))
                else:
                    rows = self.sess.query(Model).filter(
                        or_(Model.model_id.in_(imgs), Model.tag.in_(tags)), Model.version==version)
            ids = [row.model_id for row in rows.all()]
            rows.delete()
            self.sess.commit()
            return ids
                
        except Exception as e:
            logger.error(str(e))
            self.sess.rollback()
            return []

    def getLastModelId(self):
        try:
            # res = self.sess.query(func.max(Model.model_id).label('maxid')).one_or_none()
            res = self.sess.execute("SELECT * FROM sqlite_sequence WHERE name = 'models'").one_or_none() or [0, 0]
            return res[1]
        except Exception as e:
            logger.error(str(e), exc_info=True)
            return None

    def dropAll(self):
        """
        for test purpose
        """
        res = self.sess.execute("SELECT name FROM sqlite_schema WHERE type ='table' AND name NOT LIKE 'sqlite_%'")
        for t in res.all():
            self.sess.execute(f"drop table {t[0]}")
        self.sess.commit()

    def updateInferencePath(self, model_id, inference_path, version=None):
        try:
            modelversion = Const.model_version if version is None else version
            res = self.sess.query(Model).filter(Model.model_id==model_id, 
                                                Model.version==modelversion).first()
            res.inference_path = inference_path
            # vvv should be faster
            # res = self.sess.execute(f'update models set inference_path="{inference_path}" where model_id={model_id} and version={modelversion}')
            self.sess.commit()
        except Exception as e:
            logger.error(str(e), exc_info=True)
            self.sess.rollback()
            return False
        return True
    
    def addServing(self, model_id, d_param, mode: Literal['batch', 'online']):
        """
        add serving
        in: model_id, d_param
        out: serve_id
        """
        try:
            s = Serving(
                model_id=model_id,
                mode=mode,
                parameter=d_param,
            )
            self.sess.add(s)
            self.sess.commit()
            n_id = self.sess.query(func.max(Serving.serve_id)).one()
            return n_id[0]
        except Exception as e:
            logger.error(str(e))
            self.sess.rollback()
            return 0

    def getLastServeId(self):
        """
        Returns last served id
        """
        try:
            res = self.sess.query(func.max(Serving.serve_id)).one()
            return res[0]
        except Exception as e:
            logger.error(str(e))
            return None

    def addInference(self, model_id, image_path, result_json, result_path, csv_path):
        """
        add inference
        in: model_id, result_json, resut_path
        out: inference_id
        """
        try:
            srv_id = self.sess.query(func.max(Serving.serve_id)).filter(Serving.model_id==model_id).one()[0]
            inf = Inference(
                serve_id=srv_id,
                image_path=image_path,
                result_json=result_json,
                result_path=result_path,
                csv_path=csv_path,
            )
            self.sess.add(inf)
            self.sess.commit()
            inf_id = self.sess.query(func.max(Inference.inference_id)).one()
            return inf_id[0]
        except Exception as e:
            logger.error(str(e))
            self.sess.rollback()
            return 0

    def listInferences(self):
        """
        list inferences
        out: dict of inference_id:image_path
        """
        try:
            iq = self.sess.query(
               Inference.inference_id,
               Inference.image_path
            ).all()
            return {r[0]:r[1] for r in iq}
        except Exception as e:
            logger.error(str(e))
            return {}

    def listModelInferences(self, model_id: int, mode: Literal['batch', 'online'] = None):
        """
        list inferences by model id
        """
        try:
            sql = "select a.inference_id, a.image_path, a.result_json, a.result_path, a.csv_path, a.infered, b.model_id, b.mode " \
                  f"from inference a join serving b on a.serve_id = b.serve_id where b.model_id = {model_id}"
            if mode is not None:
                sql += f" and b.mode = '{mode}'"
            results = self.sess.execute(sql).all()
            return [{
                'inference_id': res[0],
                'image_path': res[1],
                'result_json': json.loads(res[2]),
                'result_path': res[3],
                'csv_path': res[4],
                'infered': res[5],
                'model_id': res[6],
                'mode': res[7]
                } for res in results]
        except Exception as e:
            logger.error(str(e))
            return {}

    def getInference(self, inf_id):
        """
        get inference
        in: inference id
        out: inference_id, imaga_path, result_json, result_path, model_id, parameter
        """
        try:
            sql = f"select a.inference_id, a.image_path, a.result_json, a.result_path, a.infered, b.model_id, b.parameter from inference a join serving b on a.serve_id = b.serve_id where a.inference_id = {inf_id}"
            res = self.sess.execute(sql).one()
            return {'inference_id': res[0], 'image_path': res[1], 'result_json': json.loads(res[2]), 'result_path': res[3], 'infered': res[4], 'model_id': res[5], 'parameter': json.loads(res[6]) or {}}
        except sqlalchemy.exc.NoResultFound as e:
            logger.error(str(e))
            return {}

    def delInference(self, inf_id):
        """
        delete inference
        in: inference id
        out: true or false
        """
        try:
            sql = f"delete from inference where inference_id={inf_id}"
            res = self.sess.execute(sql)
            if res.rowcount:
                self.sess.commit()
                return True
            else:
                return False
        except Exception as e:
            logger.error(str(e))
            self.sess.rollback()
            return False
