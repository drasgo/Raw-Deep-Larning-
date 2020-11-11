from enum import Enum
import numpy
from functools import reduce
from RDL.NeuralNetworks.activationFunctions import ActivationFunctions


class _InitializationTechniques:
    """ """
    @staticmethod
    def weight_initialization(
        shape: tuple, activation_function: ActivationFunctions
    ) -> numpy.ndarray:
        """

        :param shape: tuple: 
        :param activation_function: ActivationFunctions: 

        """
        pass


class _Zero(_InitializationTechniques):
    """ """
    @staticmethod
    def weight_initialization(
        shape: tuple, activation_function: ActivationFunctions
    ) -> numpy.ndarray:
        """
        :param shape: tuple: 
        :param activation_function: ActivationFunctions:
        """
        return numpy.zeros(shape)


class _Ones(_InitializationTechniques):
    """ """
    @staticmethod
    def weight_initialization(
        shape: tuple, activation_function: ActivationFunctions
    ) -> numpy.ndarray:
        """
        :param shape: tuple: 
        :param activation_function: ActivationFunctions:
        """
        return numpy.ones(shape)


class _Random(_InitializationTechniques):
    """ """
    @staticmethod
    def weight_initialization(
        shape: tuple, activation_function: ActivationFunctions
    ) -> numpy.ndarray:
        """
        :param shape: tuple: 
        :param activation_function: ActivationFunctions:
        """
        return numpy.random.rand(shape[0], shape[1])


class _Costum:
    """ """
    @staticmethod
    def weight_initialization(shape: tuple, matrix: numpy.ndarray) -> numpy.ndarray:
        """
        :param shape: tuple: 
        :param matrix: numpy.ndarray:
        """
        size = 1
        for elem in shape:
            size *= elem
        if size == matrix.size:
            matrix = matrix.reshape(shape)
        return matrix


class _Suggested(_InitializationTechniques):
    """ """
    @staticmethod
    def weight_initialization(
        shape: tuple, activation_function: ActivationFunctions
    ) -> numpy.ndarray:
        """
        # TODO
        :param shape: tuple: 
        :param activation_function: ActivationFunctions:
        """
        pass


class _He(_InitializationTechniques):
    """ """
    @staticmethod
    def weight_initialization(
        shape: tuple, activation_function: ActivationFunctions
    ) -> numpy.ndarray:
        """
        # TODO
        :param shape: tuple: 
        :param activation_function: ActivationFunctions:
        """
        pass


class _Xavier(_InitializationTechniques):
    """ """
    @staticmethod
    def weight_initialization(
        shape: tuple, activation_function: ActivationFunctions
    ) -> numpy.ndarray:
        """
        # TODO
        :param shape: tuple: 
        :param activation_function: ActivationFunctions:
        """
        pass


class WeightInitializations(Enum):
    """ """
    ZERO = _Zero
    ONES = _Ones
    RANDOM = _Random
    SUGGESTED = _Suggested
    HE = _He
    XAVIER = _Xavier
    COSTUM = _Costum
