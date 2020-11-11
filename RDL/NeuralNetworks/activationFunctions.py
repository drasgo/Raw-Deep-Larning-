from enum import Enum
from math import exp
import numpy


class _Ops:
    """ """
    @staticmethod
    def compute(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        pass

    @staticmethod
    def derivative(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        pass


class _Linear(_Ops):
    """ """
    name = "Linear"
    @staticmethod
    def compute(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        return input_data

    @staticmethod
    def derivative(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        return numpy.ones(shape=input_data.shape)


class _Relu(_Ops):
    """ """
    name = "Relu"
    @staticmethod
    def compute(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        for rows in range(input_data.shape[0]):
            for columns in range(input_data.shape[1]):
                input_data[rows][columns] = max(0, input_data[rows][columns])
        return input_data

    @staticmethod
    def derivative(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        for rows in range(input_data.shape[0]):
            for columns in range(input_data.shape[1]):
                if input_data[rows][columns] > 0:
                    input_data[rows][columns] = 1
                else:
                    input_data[rows][columns] = 0
        return input_data


class _LeakyRelu(_Ops):
    """ """
    name = "Leaky Relu"
    @staticmethod
    def compute(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        for rows in range(input_data.shape[0]):
            for columns in range(input_data.shape[1]):
                input_data[rows][columns] = max(0.01, input_data[rows][columns])
        return input_data

    @staticmethod
    def derivative(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        for rows in range(input_data.shape[0]):
            for columns in range(input_data.shape[1]):
                if input_data[rows][columns] > 0:
                    input_data[rows][columns] = 1
                else:
                    input_data[rows][columns] = 0.01
        return input_data


class _Sigmoid(_Ops):
    """ """
    name = "Sigmoid"
    @staticmethod
    def compute(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        for rows in range(input_data.shape[0]):
            for columns in range(input_data.shape[1]):
                input_data[rows][columns] = 1 / (1 + exp(-input_data[rows][columns]))
        return input_data

    @staticmethod
    def derivative(input_data: numpy.array) -> numpy.array:
        """f'(x) = f(x) * (1 - f(x))

        :param input_data: numpy.array: 

        """
        return input_data * (numpy.ones(input_data.shape) - input_data)


class _Tanh(_Ops):
    """ """
    name = "Tanh"
    @staticmethod
    def compute(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        for rows in range(input_data.shape[0]):
            for columns in range(input_data.shape[1]):
                pos_e = exp(input_data[rows][columns])
                neg_e = exp(-input_data[rows][columns])
                input_data[rows][columns] = (pos_e - neg_e) / (pos_e + neg_e)
        return input_data

    @staticmethod
    def derivative(input_data: numpy.array) -> numpy.array:
        """f'(x) = 1 - f(x)^2

        :param input_data: numpy.array: 

        """
        return numpy.ones(input_data.shape) - input_data ** 2


class _Swish(_Ops):
    """Swish implemented following https://arxiv.org/abs/1710.05941
    f(x) = x * sigmoid(bx) where b can be a constant or a trainable parameter.
    In this implementation, b is costant 1


    """
    name = "Swish"
    @staticmethod
    def compute(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        for rows in range(input_data.shape[0]):
            for columns in range(input_data.shape[1]):
                input_data[rows][columns] = input_data[rows][columns] * (
                    1 / (1 + exp(-input_data[rows][columns]))
                )
        return input_data

    @staticmethod
    def derivative(input_data: numpy.array) -> numpy.array:
        """f'(x) = f(x) + (sigmoid(x) * (1 - f(x)))

        :param input_data: numpy.array: 

        """
        return input_data + (_Sigmoid.compute(input_data) * (numpy.ones(input_data.shape) - input_data))


class _Softmax(_Ops):
    """ """
    name = "Softmax"
    @staticmethod
    def compute(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        pos_e = numpy.exp(input_data)
        return pos_e / numpy.sum(pos_e)

    @staticmethod
    def derivative(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        for rows in range(input_data.shape[0]):
            for columns in range(input_data.shape[1]):
                if rows == columns:
                    delta = 1
                else:
                    delta = 0
                input_data[rows][columns] = input_data[rows][columns] * (
                    delta - input_data[rows][columns]
                )
        return input_data


class ActivationFunctions(Enum):
    """ """
    LINEAR = _Linear
    RELU = _Relu
    LRELU = _LeakyRelu
    SIGMOID = _Sigmoid
    TANH = _Tanh
    SWISH = _Swish
    SOFTMAX = _Softmax
