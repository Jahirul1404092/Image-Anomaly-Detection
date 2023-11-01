import logging
logger = logging.getLogger(name=__name__)

import grpc
from api.grpc import (
    inference_pb2,
    inference_pb2_grpc,
    management_pb2,
    management_pb2_grpc
)

class GRPCResponse(object):
    def __init__(self, code=200, jobj=None):
        status_code = code
    def json(self):
        return self.jobj

def get_inference_stub():
    channel = grpc.insecure_channel('localhost:7070')
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub

def get_management_stub():
    channel = grpc.insecure_channel('localhost:7071')
    stub = management_pb2_grpc.ManagementAPIsServiceStub(channel)
    return stub

def infer(model_name, fp):
    stub = get_inference_stub()
    # with open(model_input, 'rb') as f:
    #     data = f.read()

    input_data = {'data': fp.read()}
    response = stub.Predictions(
        inference_pb2.PredictionsRequest(model_name=model_name,
                                         input=input_data))

    try:
        prediction = response.prediction.decode('utf-8')
        return GRPCResponse(jobj=prediction)
    except grpc.RpcError as e:
        logger.error(f"Failed inference.")
        logger.error(str(e.details()))
        return GRPCResponse(500)

def register(model_name):
    stub = get_management_stub()
    marfile = f"{model_name}.mar"
    params = {
        'url': marfile,
        'initial_workers': 1,
        # 'synchronous': True,
        'model_name': model_name
    }
    try:
        stub.RegisterModel(
            management_pb2.RegisterModelRequest(**params))
        logger.debut(f"Model {model_name} registered successfully")
        return GRPCResponse()
    except grpc.RpcError as e:
        logger.error(f"Failed to register model {model_name}.")
        logger.error(str(e.details()))
        return GRPCResponse(500)

def unregister(model_name):
    stub = get_management_stub()
    try:
        stub.UnregisterModel(
            management_pb2.UnregisterModelRequest(model_name=model_name))
        logger.debug(f"Model {model_name} unregistered successfully")
        return GRPCResponse()
    except grpc.RpcError as e:
        logger.error(f"Failed to unregister model {model_name}.")
        logger.error(str(e.details()))
        return GRPCResponse(500)
