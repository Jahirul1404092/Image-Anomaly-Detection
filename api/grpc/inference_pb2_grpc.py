# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import api.grpc.inference_pb2 as inference__pb2


class InferenceAPIsServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Ping = channel.unary_unary(
                '/org.pytorch.serve.grpc.inference.InferenceAPIsService/Ping',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=inference__pb2.TorchServeHealthResponse.FromString,
                )
        self.Predictions = channel.unary_unary(
                '/org.pytorch.serve.grpc.inference.InferenceAPIsService/Predictions',
                request_serializer=inference__pb2.PredictionsRequest.SerializeToString,
                response_deserializer=inference__pb2.PredictionResponse.FromString,
                )


class InferenceAPIsServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Ping(self, request, context):
        """Check health status of the TorchServe server.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Predictions(self, request, context):
        """Predictions entry point to get inference using default model version.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InferenceAPIsServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Ping': grpc.unary_unary_rpc_method_handler(
                    servicer.Ping,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=inference__pb2.TorchServeHealthResponse.SerializeToString,
            ),
            'Predictions': grpc.unary_unary_rpc_method_handler(
                    servicer.Predictions,
                    request_deserializer=inference__pb2.PredictionsRequest.FromString,
                    response_serializer=inference__pb2.PredictionResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'org.pytorch.serve.grpc.inference.InferenceAPIsService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class InferenceAPIsService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Ping(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/org.pytorch.serve.grpc.inference.InferenceAPIsService/Ping',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            inference__pb2.TorchServeHealthResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Predictions(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/org.pytorch.serve.grpc.inference.InferenceAPIsService/Predictions',
            inference__pb2.PredictionsRequest.SerializeToString,
            inference__pb2.PredictionResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
