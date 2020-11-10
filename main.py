from RDL.NeuralNetworks.MLP import MLP
# import RDL.configs.terminal_commands
import RDL.configs.mnist_load
from RDL.NeuralNetworks.activationFunctions import ActivationFunctions
from RDL.NeuralNetworks.lossFunctions import LossFunctions
from RDL.configs.debug import Verbosity
import numpy
import pprint


if __name__ == "__main__":
    # grab values from terminal
    # optional: load model
    # get data

    # create new mlp
    net = MLP(verbose=Verbosity.DEBUG)
    net.add_input_layer("input", 2)
    net.add_layer("h1", "input", 3, ActivationFunctions.TANH)
    net.add_layer("h2", "h1", 5, ActivationFunctions.SIGMOID)
    net.add_output_layer("output", "h2", 1)
    net.add_loss_function(LossFunctions.MSE)
    net.commit_structure()
    pprint.pprint(net.structure)

    # train-test or run nn with new data
    input_data = numpy.array([[1, 2], [3,4]])
    output_data = numpy.array([[3], [5]])
    validation_input = numpy.array([[4, 5], [5,6]])
    validation_output = numpy.array([[6], [7]])
    net.train(
        input_data=input_data,
        target_data=output_data,
        train_pairs=2,
        validation_input=validation_input,
        validation_target=validation_output,
        epochs=3,
        batch_size=1,
    )
    test_data = numpy.array([[6,7]])
    test_target = numpy.array([8])
    net.test(
        test_input=test_data,
        test_target=test_target,
        test_pairs=1)

    # print stats or result

    pass
