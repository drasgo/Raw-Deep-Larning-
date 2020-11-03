from enum import Enum
from math import exp
import numpy


class _Ops:
    @staticmethod
    def forward(input_data: numpy.Array):
        pass

    @staticmethod
    def derivative(input_data: numpy.Array):
        pass


class _Linear(_Ops):
    @staticmethod
    def forward(input_data: numpy.Array):
        return input_data

    @staticmethod
    def derivative(input_data: numpy.Array):
        return numpy.ones(shape=input_data.shape)


class _Relu(_Ops):
    @staticmethod
    def forward(input_data: numpy.Array):
        computed = list()
        for value in input_data:
            computed.append(max(0, value))
        computed = numpy.asarray(computed)
        computed.reshape(input_data.shape)
        return computed

    @staticmethod
    def derivative(input_data: numpy.Array):
        computed = list()
        for value in input_data:
            if value > 0:
                computed.append(1)
            else:
                computed.append(0)
        computed = numpy.asarray(computed)
        computed.reshape(input_data.shape)
        return computed


class _Sigmoid(_Ops):
    @staticmethod
    def forward(input_data: numpy.Array):
        computed = list()
        for value in input_data:
            computed.append(1/(1+exp(-value)))
        computed = numpy.asarray(computed)
        computed.reshape(input_data.shape)
        return computed

    @staticmethod
    def derivative(input_data: numpy.Array):
        pass


class _Tanh(_Ops):
    @staticmethod
    def forward(input_data: numpy.Array):
        computed = list()
        for value in input_data:
            computed.append(2 * (1 / (1 + exp(-value))) - 1)
        computed = numpy.asarray(computed)
        computed.reshape(input_data.shape)
        return computed

    @staticmethod
    def derivative(input_data: numpy.Array):
        pass


class _Swish(_Ops):
    @staticmethod
    def forward(input_data: numpy.Array):
        pass

    @staticmethod
    def derivative(input_data: numpy.Array):
        pass


class _Softmax(_Ops):
    @staticmethod
    def forward(input_data: numpy.Array):
        pass

    @staticmethod
    def derivative(input_data: numpy.Array):
        pass


class ActivationFunctions(Enum):
    LINEAR = _Linear
    RELU = _Relu
    SIGMOID = _Sigmoid
    TANH = _Tanh
    SWISH = _Swish
    SOFTMAX = _Softmax
