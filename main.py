from RDL.NeuralNetworks.MLP import MLP

# import RDL.configs.terminal_commands
from RDL.configs.mnist_load import load_synth, load_mnist
from RDL.NeuralNetworks.activationFunctions import ActivationFunctions
from RDL.NeuralNetworks.lossFunctions import LossFunctions
from RDL.NeuralNetworks.weightInitialization import WeightInitializations
from RDL.configs.debug import Verbosity
import numpy
import pprint


def create_one_hot_vector(dataset: numpy.ndarray) -> numpy.ndarray:
    """

    :param dataset: numpy.ndarray: 

    """
    vectors = []
    # print(set)
    # max_val = 10
    for elem in dataset:
        temp = [0 for _ in range(10)]
        temp[elem] = 1
        vectors.append(temp)
    vectors = numpy.array(vectors)
    return vectors


def mnist(m_train, m_test):
    """

    :param m_train: 
    :param m_test: 

    """
    i_train, t_train = m_train
    v_input = i_train[-100:]
    # input_validation = input_train[:-50]
    i_train = i_train[:200]
    # input_train = input_train[500:]
    v_target = t_train[-100:]
    v_target = create_one_hot_vector(v_target)
    # target_validation = target_train[:-50]
    t_train = t_train[:200]
    t_train = create_one_hot_vector(t_train)
    # target_train = target_train[500:]
    t_input, t_target = m_test
    t_input = t_input[:100]
    t_target = t_target[:100]
    t_target = create_one_hot_vector(t_target)
    return i_train, t_train, v_input, v_target, t_input, t_target


def synth(s_train, s_test):
    """

    :param s_train: 
    :param s_test: 

    """
    # get data
    i_train, t_train = s_train
    v_input = i_train[-70:]
    i_train = i_train
    v_target = t_train[-70:]
    v_target = create_one_hot_vector(v_target)
    t_train = t_train
    t_train = create_one_hot_vector(t_train)
    t_input, t_target = s_test
    t_target = create_one_hot_vector(t_target)
    return i_train, t_train, v_input, v_target, t_input, t_target


def network_1(net: MLP):
    """

    :param net: 

    """
    net.add_normalization(False)
    net.add_standardization(False)
    net.add_input_layer(
        layer_name="input",
        input_nodes=2,
    )
    net.add_layer(
        layer_name="h1",
        input_layer="input",
        nodes=3,
        activation_function=ActivationFunctions.SIGMOID,
        weight_initialization=WeightInitializations.COSTUM,
        custum_weigths=numpy.array([[1, -1], [1, -1], [1, -1]]),
    )

    net.add_output_layer(
        layer_name="output",
        input_layer="h1",
        nodes=2,
        activation_function=ActivationFunctions.SOFTMAX,
        weight_initialization=WeightInitializations.COSTUM,
        custum_weigths=numpy.array([[1, -1, -1], [1, -1, -1]]),
    )
    net.add_loss_function(LossFunctions.CROSS_ENTROPY)
    net.add_learning_rate(0.001)
    return net


def network_2(net: MLP):
    """

    :param net: 

    """
    # net.add_normalization(False)
    # net.add_standardization(False)
    net.add_input_layer("input", 2)
    net.add_layer(
        layer_name="h1",
        input_layer="input",
        nodes=3,
        activation_function=ActivationFunctions.SIGMOID,
        # weight_initialization=WeightInitializations.COSTUM,
        # custum_weigths=numpy.array([[1, -1], [1, -1], [1, -1]])
    )

    net.add_output_layer(
        layer_name="output",
        input_layer="h1",
        nodes=10,
        activation_function=ActivationFunctions.SOFTMAX,
        # weight_initialization=WeightInitializations.COSTUM,
        # custum_weigths=numpy.array([[1, -1, -1], [1, -1, -1]])
    )
    net.add_loss_function(LossFunctions.CROSS_ENTROPY)
    net.add_learning_rate(0.001)
    return net


if __name__ == "__main__":
    # grab values from terminal
    # optional: load model
    dataset = "synth"
    if dataset == "mnist":
        train, test, _ = load_mnist()
        (
            input_train,
            target_train,
            input_validation,
            target_validation,
            input_test,
            target_test,
        ) = mnist(train, test)
    else:
        train, test, _ = load_synth(1000, 100)
        (
            input_train,
            target_train,
            input_validation,
            target_validation,
            input_test,
            target_test,
        ) = synth(train, test)

    # create new mlp
    net = MLP(verbose=Verbosity.RELEASE)
    net = network_2(net)
    # net = MLP(verbose=Verbosity.DEBUG)

    net.commit_structure()
    # input_train = numpy.array([[1, -1]])
    # target_train = numpy.array([1,0])
    net.train(
        input_data=input_train,
        target_data=target_train,
        # validation_input=input_validation,
        # validation_target=target_validation,
        epochs=1,
        batch_size=1,
    )
    net.test(
        test_input=input,
        test_target=target)
