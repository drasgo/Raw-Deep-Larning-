from RDL.NeuralNetworks.MLP import MLP
# import RDL.configs.terminal_commands
from RDL.configs.mnist_load import load_synth, load_mnist
from RDL.NeuralNetworks.activationFunctions import ActivationFunctions
from RDL.NeuralNetworks.lossFunctions import LossFunctions
from RDL.configs.debug import Verbosity
import numpy
import pprint


if __name__ == "__main__":
    # grab values from terminal
    # optional: load model
    # get data
    train, test, _ = load_synth(num_train=100, num_val=30)
    input_train, target_train = train
    input_validation = input_train[-70:]
    input_train = input_train[:-30]
    target_validation = target_train[-70:]
    target_train = target_train[:-30]
    input_test, target_test = test
    print("Input train shape: " + str(input_train.shape))
    print("Target train shape: " + str(target_train.shape))
    pprint.pprint(target_train)
    print("Input validation shape: " + str(target_validation.shape))
    print("Target validation shape: " + str(input_validation.shape))
    print("Input test shape: " + str(input_test.shape))
    print("Target test shape: " + str(target_test.shape))
    # input()
    # _____________________________
    # train, test, _ = load_mnist()
    # input_train, target_train = train
    # input_validation = input_train[-5000:]
    # input_train = input_train[:-5000]
    # target_validation = target_train[-5000:]
    # target_train = target_train[:-5000]
    # input_test, target_test = test
    # print("Input train shape: " + str(input_train.shape))
    # print("Target train shape: " + str(target_train.shape))
    # pprint.pprint(target_train)
    # print("Input validation shape: " + str(target_validation.shape))
    # print("Target validation shape: " + str(input_validation.shape))
    # print("Input test shape: " + str(input_test.shape))
    # print("Target test shape: " + str(target_test.shape))
    # input()
    # ____________________________
    # create new mlp
    net = MLP(verbose=Verbosity.RELEASE)
    # net = MLP(verbose=Verbosity.DEBUG)
    net.add_input_layer("input", 2)
    net.add_layer("h1", "input", 6, ActivationFunctions.TANH)
    # net.add_layer("h2", "h1", 5, ActivationFunctions.SIGMOID)
    net.add_output_layer("output", "h1", 1)
    net.add_loss_function(LossFunctions.MSE)
    net.commit_structure()
    pprint.pprint(net.structure)

    # train-test or run nn with new data
    # input_data = numpy.array([[1, 2], [3,4]])
    # output_data = numpy.array([[3], [5]])
    # validation_input = numpy.array([[4, 5], [5,6]])
    # validation_output = numpy.array([[6], [7]])
    net.train(
        input_data=input_train,
        target_data=target_train,
        input_size=2,
        validation_input=input_validation,
        validation_target=target_validation,
        epochs=10,
        batch_size=5,
    )
    # test_data = numpy.array([[6,7]])
    # test_target = numpy.array([8])
    net.test(
        test_input=input_test,
        test_target=target_test,
        input_size=2)

    # print stats or result

    pass
