from RDL.NeuralNetworks.baseNeuralNetwork import BaseNeuralNetwork
from RDL.configs.debug import Verbosity, logger
import numpy


class MLP(BaseNeuralNetwork):
    """ """
    def __init__(self, verbose: Verbosity=Verbosity.RELEASE) -> numpy.array:
        super().__init__(verbose)

    def forward(self, input_data):
        """

        :param input_data: 

        """
        logger("STARTING FORWARD PASS FOR INPUT DATA: " + str(input_data), self.verbose)
        input_data = self.prepare_data(input_data)
        # Add check for input data. E.g. correct size, all numeric values, etc.
        output = None
        for input_layer in self.input_layers:
            # TODO: implement multiple input layers
            prev_layer = input_layer
            prev_layer["data"] = input_data

            while True:
                curr_layer = self.structure[prev_layer["output_layer"]]
                output_data = (
                    numpy.dot(curr_layer["weight"], prev_layer["data"])
                    + curr_layer["bias"]
                )
                logger("Forward pass for layer " + curr_layer["name"] +
                       "\nInput data: " + str(prev_layer["data"]) +
                       "\nLayer weights: " + str(curr_layer["weight"]) +
                       "\nLayer biases: " + str(curr_layer["bias"]) +
                       "\nLayer activation function: " + str(curr_layer["activation"].value.name) +
                       "\n(Curr matrix * Prev data) + Curr biases = " + str(output_data), self.verbose)
                output_data = curr_layer["activation"].value.forward(output_data)
                curr_layer["data"] = output_data
                prev_layer = curr_layer
                logger(curr_layer["activation"].value.name + "(Curr data) = " + str(output_data), self.verbose)

                if curr_layer["type"] == "output":
                    output = curr_layer["data"]
                    break
        logger("FINISHED FORWARD PASS", self.verbose)
        return output

    def backward(self, output_data, target_data):
        """Compute sensitivities from m to 1 layer
        Then new_W(i) = old_W(i) - decay * S(i) (sensitivity of layer i) * (input to layer i)T (meaning transpose of input layer i)
        sensitivity of i ( S(i) ) = derivative of function Fi * old_W(i+1)T * S(i+1)
        except for layer m, where sensitivity of m ( S(m) ) = derivative of function Fi (usually linear) * derivative of loss (integer)

        :param output_data: 
        :param target_data: 

        """
        logger("STARTING BACKWARD PASS", self.verbose)
        loss = self.loss_function.value.forward(output_data, target_data)
        loss_derivative = self.loss_function.value.derivative(output_data, target_data)
        logger("Loss: " + str(loss) + "\nLoss derivative: " + str(loss_derivative), self.verbose)

        for output_layer in self.output_layers:
            curr_layer = output_layer

            while True:
                if curr_layer["type"] == "input":
                    break

                function_derivative = curr_layer["activation"].value.derivative(curr_layer["data"])
                if curr_layer["type"] == "output":
                    # S(output) = f'(x) * loss'
                    sensitivity = function_derivative * loss_derivative
                    logger("Computed sensitivity (function derivative * loss derivative) for layer " +
                           curr_layer["name"] + ": " + str(sensitivity), self.verbose)
                else:
                    # S(i) = f'(x) * W(i+1)T * S(i+1)
                    diag_function_derivative = numpy.diagflat(function_derivative)
                    transposed_weight_matrix = numpy.transpose(prev_layer["weight"])

                    logger("Computing sensitivity for layer " + curr_layer["name"] +
                           "\n -data computed in forward step: " + str(curr_layer["data"]) +
                           "\n -function derivative: " + str(function_derivative) +
                           "\n -diagonal of function derivative: " + str(diag_function_derivative) +
                           "\n -prev layer (" + prev_layer["name"] + ") weight matrix: " + str(prev_layer["weight"]) +
                           "\n -prev layer transposed weight matrix: " + str(transposed_weight_matrix) +
                           "\n -prev layer sensitivity: " + str(prev_layer["sensitivity"]) +
                           "\n -diagonal function derivative shape: " + str(diag_function_derivative.shape) +
                           "\n -transposed prev weights shape: " + str(transposed_weight_matrix.shape), self.verbose)

                    sensitivity = numpy.dot(diag_function_derivative, transposed_weight_matrix)
                    logger("Function derivative * weight matrix transposed: " + str(sensitivity), self.verbose)
                    sensitivity = numpy.dot(sensitivity, prev_layer["sensitivity"])
                    logger("Computed sensitivity (function derivative * weight matrix transposed * prev layer sensitivity) "
                           "for layer " + curr_layer["name"] + ": " + str(sensitivity), self.verbose)

                curr_layer["sensitivity"] = sensitivity
                updates = sensitivity * numpy.transpose(
                    self.structure[curr_layer["input_layer"]]["data"]
                )
                curr_layer["weight_update"] += updates
                prev_layer = curr_layer
                curr_layer = self.structure[prev_layer["input_layer"]]

        logger("FINISHED BACKWARD PASS", self.verbose)
        return loss
