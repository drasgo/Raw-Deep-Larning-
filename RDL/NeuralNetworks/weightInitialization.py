from enum import Enum
import numpy
from RDL.NeuralNetworks.activationFunctions import ActivationFunctions


class _InitializationTechniques:
    @staticmethod
    def weight_initialization(shape: tuple, activation_function: ActivationFunctions) -> numpy.ndarray:
        pass


class _Zero(_InitializationTechniques):
    @staticmethod
    def weight_initialization(shape: tuple, activation_function: ActivationFunctions) -> numpy.ndarray:
        pass


class _Ones(_InitializationTechniques):
    @staticmethod
    def weight_initialization(shape: tuple, activation_function: ActivationFunctions) -> numpy.ndarray:
        pass


class _Random(_InitializationTechniques):
    @staticmethod
    def weight_initialization(shape: tuple, activation_function: ActivationFunctions) -> numpy.ndarray:
        pass


class _Suggested(_InitializationTechniques):
    @staticmethod
    def weight_initialization(shape: tuple, activation_function: ActivationFunctions) -> numpy.ndarray:
        pass


class _He(_InitializationTechniques):
    @staticmethod
    def weight_initialization(shape: tuple, activation_function: ActivationFunctions) -> numpy.ndarray:
        pass


class _Xavier(_InitializationTechniques):
    @staticmethod
    def weight_initialization(shape: tuple, activation_function: ActivationFunctions) -> numpy.ndarray:
        pass


class WeightInitializations(Enum):
    ZERO = _Zero
    ONES = _Ones
    RANDOM = _Random
    SUGGESTED = _Suggested
    HE = _He
    XAVIER = _Xavier
