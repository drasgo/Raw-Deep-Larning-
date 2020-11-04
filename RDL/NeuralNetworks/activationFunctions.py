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
        for rows in range(input_data):
            for columns in range(input_data[rows]):
                input_data[rows][columns] = (max(0, input_data[rows][columns]))
        return input_data

    @staticmethod
    def derivative(input_data: numpy.Array):
        for rows in range(input_data):
            for columns in range(input_data[rows]):
                if input_data[rows][columns] > 0:
                    input_data[rows][columns] = 1
                else:
                    input_data[rows][columns] = 0
        return input_data


class _LeakyRelu(_Ops):
    @staticmethod
    def forward(input_data: numpy.Array):
        for rows in range(input_data):
            for columns in range(input_data[rows]):
                input_data[rows][columns] = (max(0.01, input_data[rows][columns]))
        return input_data

    @staticmethod
    def derivative(input_data: numpy.Array):
        for rows in range(input_data):
            for columns in range(input_data[rows]):
                if input_data[rows][columns] > 0:
                    input_data[rows][columns] = 1
                else:
                    input_data[rows][columns] = 0.01
        return input_data


class _Sigmoid(_Ops):
    @staticmethod
    def forward(input_data: numpy.Array):
        for rows in range(input_data):
            for columns in range(input_data[rows]):
                input_data[rows][columns] = 1 / (1 + exp(-input_data[rows][columns]))
        return input_data

    @staticmethod
    def derivative(input_data: numpy.Array):
        """f'(x) = f(x) * (1 - f(x))"""
        return input_data * (1 - input_data)


class _Tanh(_Ops):
    @staticmethod
    def forward(input_data: numpy.Array):
        for rows in range(input_data):
            for columns in range(input_data[rows]):
                input_data[rows][columns] = 2 * (1 / (1 + exp(-input_data[rows][columns]))) - 1
        return input_data

    @staticmethod
    def derivative(input_data: numpy.Array):
        """f'(x) = 1 - f(x)^2"""
        return 1 - input_data**2


class _Swish(_Ops):
    """
    Swish implemented following https://arxiv.org/abs/1710.05941
    f(x) = x * sigmoid(bx) where b can be a constant or a trainable parameter.
    In this implementation, b is costant 1
    """
    @staticmethod
    def forward(input_data: numpy.Array):
        for rows in range(input_data):
            for columns in range(input_data[rows]):
                input_data[rows][columns] = input_data[rows][columns] * (1 / (1 + exp(-input_data[rows][columns])))
        return input_data

    @staticmethod
    def derivative(input_data: numpy.Array):
        """f'(x) = f(x) + (sigmoid(x) * (1 - f(x)))"""
        return input_data + (_Sigmoid.forward(input_data) * (1 - input_data))


class _Softmax(_Ops):
    @staticmethod
    def forward(input_data: numpy.Array):
        return numpy.exp(input_data) / numpy.sum(input_data)

    @staticmethod
    def derivative(input_data: numpy.Array):
        for rows in range(input_data):
            for columns in range(input_data[rows]):
                if rows == columns:
                    delta = 1
                else:
                    delta = 0
                input_data[rows][columns] = input_data[rows][columns] * (delta - input_data[rows][columns])



class ActivationFunctions(Enum):
    LINEAR = _Linear
    RELU = _Relu
    LRELU = _LeakyRelu
    SIGMOID = _Sigmoid
    TANH = _Tanh
    SWISH = _Swish
    SOFTMAX = _Softmax
