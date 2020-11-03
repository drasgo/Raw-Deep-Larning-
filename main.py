from RDL.NeuralNetworks.MLP import MLP
import RDL.configs.terminal_commands
import RDL.configs.mnist_load
from RDL.NeuralNetworks.activationFunctions import ActivationFunctions
import numpy


if __name__ == "__main__":
    # grab values from terminal
    net = MLP()
    net.add_input_layer("input", 2)
    net.add_layer("h1", "input", 3, ActivationFunctions.TANH)
    net.add_layer("h2", "h1", 5, ActivationFunctions.SIGMOID)
    net.add_output_layer("output", "h2", 1)
    net.add_loss_function("loss")
    net.commit_structure()
    import pprint
    pprint.pprint(net.structure)
    input_data = numpy.reshape(numpy.array([1,2]), (2, 1))
    net.forward(input_data)

    # get data

    # create new mlp

    # train-test or run nn with new data

    # print stats or result

    pass
