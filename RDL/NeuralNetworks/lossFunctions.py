from enum import Enum

class _MSE:
    @staticmethod
    def forward(output, target):
        pass

    @staticmethod
    def derivative(output, target):
        pass


class _CrossEntropy:
    @staticmethod
    def forward(output, target):
        pass

    @staticmethod
    def derivative(output, target):
        pass


class LossFunctions(Enum):
    MSE = _MSE
    CROSS_ENTROPY = _CrossEntropy
