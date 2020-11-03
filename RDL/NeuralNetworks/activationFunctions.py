from enum import Enum
import numpy
from math import exp


class _Linear:
    @staticmethod
    def forward(input_data):
        return input_data

    @staticmethod
    def backward(input_data):
        return numpy.ones(shape=input_data.shape)


class _Relu:
    @staticmethod
    def forward(input_data):
        computed = list()
        for value in input_data:
            computed.append(max(0, value))
        computed = numpy.asarray(computed)
        computed.reshape(input_data.shape)
        return computed

    @staticmethod
    def backward(input_data):
        computed = list()
        for value in input_data:
            if value > 0:
                computed.append(1)
            else:
                computed.append(0)
        computed = numpy.asarray(computed)
        computed.reshape(input_data.shape)
        return computed


class _Sigmoid:
    @staticmethod
    def forward(input_data):
        computed = list()
        for value in input_data:
            computed.append(1/(1+exp(-value)))
        computed = numpy.asarray(computed)
        computed.reshape(input_data.shape)
        return computed

    @staticmethod
    def backward(input_data):
        pass


class _Tanh:
    @staticmethod
    def forward(input_data):
        computed = list()
        for value in input_data:
            computed.append(2 * (1 / (1 + exp(-value))) - 1)
        computed = numpy.asarray(computed)
        computed.reshape(input_data.shape)
        return computed

    @staticmethod
    def backward(input_data):
        pass


class _Swish:
    @staticmethod
    def forward(input_data):
        pass

    @staticmethod
    def backward(input_data):
        pass


class _Softmax:
    @staticmethod
    def forward(input_data):
        pass

    @staticmethod
    def backward(input_data):
        pass


class ActivationFunctions(Enum):
    LINEAR = _Linear
    RELU = _Relu
    SIGMOID = _Sigmoid
    TANH = _Tanh
    SWISH = _Swish
    SOFTMAX = _Softmax
