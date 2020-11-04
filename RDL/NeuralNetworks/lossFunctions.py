from enum import Enum
from math import log
import numpy


class _MSE:
    @staticmethod
    def forward(output, target):
        loss = output - target
        return numpy.sum(numpy.transpose(loss) * loss)

    @staticmethod
    def derivative(output, target):
        return numpy.sum(output - target)


class _CrossEntropy:
    @staticmethod
    def forward(output, target):
        loss = target * log(numpy.transpose(output))
        return - numpy.sum(loss)

    @staticmethod
    def derivative(output, target):
        return numpy.sum(output - target)


class LossFunctions(Enum):
    MSE = _MSE
    CROSS_ENTROPY = _CrossEntropy
