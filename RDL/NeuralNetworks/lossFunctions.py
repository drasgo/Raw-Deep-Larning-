from enum import Enum

class _LossFunctionImplementation:
    @staticmethod
    def mse(output, target):
        pass


class LossFunctions(Enum):
    MSE = _LossFunctionImplementation.mse