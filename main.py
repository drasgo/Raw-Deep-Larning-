from RDL.NeuralNetworks.MLP import MLP
import RDL.configs.terminal_commands
import RDL.configs.mnist_load
from RDL.NeuralNetworks.activationFunctions import ActivationFunctions
from RDL.NeuralNetworks.lossFunctions import LossFunctions
import numpy
import pprint


if __name__ == "__main__":
    # grab values from terminal
    # optional: load model
    # get data

    # create new mlp
    net = MLP()
    net.add_input_layer("input", 2)
    net.add_layer("h1", "input", 3, ActivationFunctions.TANH)
    net.add_layer("h2", "h1", 5, ActivationFunctions.SIGMOID)
    net.add_output_layer("output", "h2", 1)
    net.add_loss_function(LossFunctions.MSE)
    net.commit_structure()
    pprint.pprint(net.structure)

    # train-test or run nn with new data
    input_data = numpy.reshape(numpy.array([1, 2]), (2, 1))
    output_data = numpy.array([3])
    validation_input = numpy.reshape(numpy.array([4, 5]), (2, 1))
    validation_output = numpy.array([6])
    net.train(
        input_data,
        output_data,
        validation_input,
        validation_output,
        epochs=1,
        batch_size=1,
    )

    # print stats or result

    pass
