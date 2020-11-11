from RDL.NeuralNetworks.MLP import MLP
# import RDL.configs.terminal_commands
from RDL.configs.mnist_load import load_synth, load_mnist
from RDL.NeuralNetworks.activationFunctions import ActivationFunctions
from RDL.NeuralNetworks.lossFunctions import LossFunctions
from RDL.NeuralNetworks.weightInitialization import WeightInitializations
from RDL.configs.debug import Verbosity
import numpy
import pprint

def create_one_hot_vector(set: numpy.ndarray) -> numpy.ndarray:
    vectors = []
    # print(set)
    # max_val = 10
    for elem in set:
        temp = [0 for element in range(10)]
        temp[elem] = 1
        vectors.append(temp)
    vectors = numpy.array(vectors)
    return vectors


if __name__ == "__main__":
    # grab values from terminal
    # optional: load model
    # get data
    # train, test, _ = load_synth(num_train=100, num_val=30)
    # input_train, target_train = train
    # input_validation = input_train[-70:]
    # input_train = input_train[:-30]
    # target_validation = target_train[-70:]
    # target_train = target_train[:-30]
    # input_test, target_test = test
    # print("Input train shape: " + str(input_train.shape))
    # print("Target train shape: " + str(target_train.shape))
    # pprint.pprint(target_train)
    # print("Input validation shape: " + str(target_validation.shape))
    # print("Target validation shape: " + str(input_validation.shape))
    # print("Input test shape: " + str(input_test.shape))
    # print("Target test shape: " + str(target_test.shape))
    # input()
    # _____________________________
    train, test, _ = load_mnist()
    input_train, target_train = train
    input_validation = input_train[-20:]
    # input_validation = input_train[:-50]
    input_train = input_train[:40]
    # input_train = input_train[500:]
    target_validation = target_train[-20:]
    target_validation = create_one_hot_vector(target_validation)
    # target_validation = target_train[:-50]
    target_train = target_train[:40]
    target_train = create_one_hot_vector(target_train)
    # print(target_train.shape)
    # print(target_validation.shape)
    # input()
    # target_train = target_train[500:]
    input_test, target_test = test
    input_test = input_test[:10]
    target_test = target_test[:10]
    target_test = create_one_hot_vector(target_test)

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
    # net = MLP(verbose=Verbosity.RELEASE)
    net = MLP(verbose=Verbosity.DEBUG)
    net.add_input_layer("input", 784)
    net.add_layer("h1", "input", 50, ActivationFunctions.SIGMOID, weight_initialization=WeightInitializations.ONES)
    # net.add_layer("h1", "input", 300, ActivationFunctions.SIGMOID, weight_initialization=WeightInitializations.ONES)
    # net.add_layer("h2", "h1", 5, ActivationFunctions.SIGMOID)
    net.add_output_layer("output", "h1", 10, activation_function=ActivationFunctions.SOFTMAX)
    net.add_loss_function(LossFunctions.CROSS_ENTROPY)
    net.add_learning_rate(0.1)
    net.commit_structure()
    # pprint.pprint(net.structure)

    # train-test or run nn with new data
    # input_data = numpy.array([[1, 2], [3,4]])
    # output_data = numpy.array([[3], [5]])
    # validation_input = numpy.array([[4, 5], [5,6]])
    # validation_output = numpy.array([[6], [7]])
    net.train(
        input_data=input_train,
        target_data=target_train,
        validation_input=input_validation,
        validation_target=target_validation,
        epochs=10,
        batch_size=2,
    )
    # test_data = numpy.array([[6,7]])
    # test_target = numpy.array([8])
    net.test(
        test_input=input_test,
        test_target=target_test)

    # print stats or result

    pass
