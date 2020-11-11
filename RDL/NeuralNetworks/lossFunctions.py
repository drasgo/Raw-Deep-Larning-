from enum import Enum
import numpy


class _MSE:
    """ """
    @staticmethod
    def compute(output, target):
        """

        :param output: 
        :param target: 

        """
        loss = output - target
        return numpy.sum(numpy.dot(numpy.transpose(loss), loss))

    @staticmethod
    def derivative(output, target):
        """

        :param output: 
        :param target: 

        """
        return numpy.sum(output - target)


class _CrossEntropy:
    """ """
    @staticmethod
    def compute(output, target):
        """

        :param output: 
        :param target: 

        """
        output[output==0] = 0.00000001
        output[output==1] = 0.99999999
        return -numpy.sum(target * numpy.log(output) + (1-target)*numpy.log(1-output))

    @staticmethod
    def derivative(output, target):
        """

        :param output: 
        :param target: 

        """
        # output[output == 0] = 0.00000001
        # output[output == 1] = 0.99999999
        # der_loss = ((-target)/output) + ((1 - target)/(1 - output))
        # print(der_loss)
        # input()
        return target - output



class LossFunctions(Enum):
    """ """
    MSE = _MSE
    CROSS_ENTROPY = _CrossEntropy
