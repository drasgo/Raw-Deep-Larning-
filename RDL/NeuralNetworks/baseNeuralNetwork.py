import numpy
import multiprocessing
from RDL.NeuralNetworks.activationFunctions import ActivationFunctions
from RDL.NeuralNetworks.weightInitialization import WeightInitializations
from RDL.NeuralNetworks.lossFunctions import LossFunctions
from RDL.configs.debug import logger, Verbosity
from typing import Tuple


class BaseNeuralNetwork:
    """ """

    def __init__(self, verbose: Verbosity = Verbosity.RELEASE):
        self.context = {}
        self.structure = {}
        self.loss_function = None
        self.loss = None
        self.input_layers = []
        self.output_layers = []
        self.learning_rate = 0.1
        self.normalization = True
        self.standardization = False
        self.parallel_structure = multiprocessing.Manager().dict()
        self.validation_threshold = 0.001
        self.verbose = verbose

    def add_input_layer(self, layer_name: str, input_nodes: int):
        """
        Adds a new input layer
        :param layer_name: str: 
        :param input_nodes: int:
        """
        self.structure[layer_name] = {
            "name": layer_name,
            "type": "input",
            "nodes": input_nodes,
        }
        logger("Added input layer: ", self.verbose)
        logger(self.structure[layer_name], self.verbose)

    def add_layer(
        self,
        layer_name: str,
        input_layer: str,
        nodes: int,
        activation_function: ActivationFunctions,
        weight_initialization: WeightInitializations = WeightInitializations.ONES,
        custum_weigths: numpy.ndarray = numpy.array([]),
        bias_initialization: WeightInitializations = WeightInitializations.ZERO,
    ):
        """
        Adds a new hidden layer. The weight initialization technique can be specified. More information are available in
        the file weightInizialization.py
        :param layer_name: str: 
        :param input_layer: str: 
        :param nodes: int: 
        :param activation_function: ActivationFunctions: 
        :param weight_initialization: WeightInitializations:  (Default value = WeightInitializations.ONES)
        :param custum_weigths: numpy.ndarray:  (Default value = numpy.array([]))
        :param bias_initialization: WeightInitializations:  (Default value = WeightInitializations.ZERO)
        """
        self.structure[layer_name] = {
            "name": layer_name,
            "type": "hidden",
            "nodes": nodes,
            "activation": activation_function,
            "input_layer": input_layer,
            "weight_initialization": weight_initialization,
            "costum_weights": custum_weigths,
            "bias_initialization": bias_initialization,
        }
        logger("Added hidden layer: ", self.verbose)
        logger(self.structure[layer_name], self.verbose)

    def add_output_layer(
        self,
        layer_name: str,
        input_layer: str,
        nodes: int,
        activation_function: ActivationFunctions = ActivationFunctions.LINEAR,
        weight_initialization: WeightInitializations = WeightInitializations.ONES,
        custum_weigths: numpy.ndarray = numpy.array([]),
        bias_initialization: WeightInitializations = WeightInitializations.ZERO,
    ):
        """
        Adds a new output layer. The weight initialization technique can be specified. More information are available in
        the file weightInizialization.py
        :param layer_name: str: 
        :param input_layer: str: 
        :param nodes: int: 
        :param activation_function: ActivationFunctions:  (Default value = ActivationFunctions.LINEAR)
        :param weight_initialization: WeightInitializations:  (Default value = WeightInitializations.ONES)
        :param custum_weigths: numpy.ndarray:  (Default value = numpy.array([]))
        :param bias_initialization: WeightInitializations:  (Default value = WeightInitializations.ZERO)
        """
        self.structure[layer_name] = {
            "name": layer_name,
            "type": "output",
            "activation": activation_function,
            "nodes": nodes,
            "input_layer": input_layer,
            "weight_initialization": weight_initialization,
            "costum_weights": custum_weigths,
            "bias_initialization": bias_initialization,
        }
        logger("Added output layer: ", self.verbose)
        logger(self.structure[layer_name], self.verbose)

    def add_standardization(self, standardize: bool = True):
        """
        :param standardize: bool:  (Default value = True)
        """
        self.standardization = standardize
        logger("Added standardization for inputs ", self.verbose)

    def add_normalization(self, normalize: bool = True):
        """
        :param normalize: bool:  (Default value = True)
        """
        self.normalization = normalize
        logger("Added normalization for inputs ", self.verbose)

    def add_operation(self, operation):
        """
        :param operation:
        """
        # TODO: Not supported yet
        pass

    def add_loss_function(self, loss_function: LossFunctions):
        """
        The loss objects are stored in lossFunctions.py
        :param loss_function: LossFunctions:
        """
        self.loss_function = loss_function
        logger("Added loss function: " + str(loss_function), self.verbose)

    def add_learning_rate(self, learning_rate: float):
        """
        :param learning_rate: float:
        """
        self.learning_rate = learning_rate
        logger("Added learning rate: " + str(learning_rate), self.verbose)

    def sanity_check(self):
        """This method performs a sanity control of the structure. Orderly, it checks that:
        - a loss function is provided
        - there is only one Input layer
        - there is only one Output layer
        - every layer has an input layer (data comes from a past layer), except Input layer
        - every layer has an ouput layer (data goes to another layer), except Output layer
        - if Cross Entropy is the chosen loss function (for the moment) the last activation function has to be Softmax

        """
        if self.loss_function is None:
            print("Error: loss function missing or not recognized.")
            exit()

        if not (
            0
            < len(
                [
                    layer
                    for layer in self.structure
                    if self.structure[layer]["type"] == "input"
                ]
            )
            < 2
        ):
            print("Error: supported only one input layer.")
            exit()

        if not (
            0
            < len(
                [
                    layer
                    for layer in self.structure
                    if self.structure[layer]["type"] == "output"
                ]
            )
            < 2
        ):
            print("Error: supported only one output layer.")
            exit()

        for layer in self.structure:
            if (
                self.structure[layer]["type"] == "output"
                and self.structure[layer]["activation"] != ActivationFunctions.SOFTMAX
                and self.loss_function == LossFunctions.CROSS_ENTROPY
            ):
                print("Error: cross-entropy supported only with softmax.")
                exit()
            if (
                self.structure[layer]["type"] != "input"
                and self.structure[layer]["input_layer"] not in self.structure
            ):
                print(
                    "Error: in layer "
                    + layer
                    + " input layer specified is "
                    + self.structure[layer]["input_layer"]
                    + " but no layer with this name exists."
                )
                exit()

            if self.structure[layer]["type"] != "output" and not any(
                "input_layer" in self.structure[out_layer]
                and self.structure[out_layer]["input_layer"] == layer
                for out_layer in self.structure
            ):
                print(
                    "Error: for layer "
                    + layer
                    + " no output layer found. Possible error designing structure."
                )
                exit()

    def commit_structure(self):
        """
        Performs a sanity check on the layers given and generates the final network's structure, which is going to be
        saved in self.structure. Each layer - except the input layer - has a weight matrix and a bias matrix,
        initialized here.
        The every layer in self.structure has the following structure:
        {
                "name": str,
                "type": str, (right now it an be only "input", "hidden", and "output")
                "nodes": int,
                "weight": numpy.ndarray,
                "weight_update": numpy.ndarray, (used for storing the weight updates when using batches backprop)
                "bias": numpy.ndarray,
                "bias_update": bias_update, (as "weight_update")
                "data": numpy.ndarray, (used for temporary storing the output of this layer for the backward pass)
                "sensitivity": numpy.ndarray, (used for the backward pass)
                "activation": ActivationFunctions, (Has the reference of the activation function object associated with
                this layer. For more informations, the activation functions are stored in activationFunctions.py)
                "input_layer": str, (specify the layer before this one)
                "output_layer": str, (specify the layer after this one)
            }

        """
        self.sanity_check()
        new_structure = dict()
        for layer in self.structure:
            input_layer = ""
            output_layer = ""
            activation_function = ""
            weights = None
            weights_update = None
            bias_update = None
            biases = None

            if self.structure[layer]["type"] != "output":
                output_layer = [
                    out_layer
                    for out_layer in self.structure
                    if "input_layer" in self.structure[out_layer]
                    and self.structure[out_layer]["input_layer"] == layer
                ]

                output_layer = output_layer[0]

            if self.structure[layer]["type"] != "input":
                activation_function = self.structure[layer]["activation"]
                input_layer = self.structure[layer]["input_layer"]
                # TODO: add weight initialization.
                if (
                    self.structure[layer]["weight_initialization"]
                    == WeightInitializations.COSTUM
                ):
                    weights = self.structure[layer][
                        "weight_initialization"
                    ].value.weight_initialization(
                        (
                            self.structure[layer]["nodes"],
                            self.structure[input_layer]["nodes"],
                        ),
                        self.structure[layer]["costum_weights"],
                    )
                else:
                    weights = self.structure[layer][
                        "weight_initialization"
                    ].value.weight_initialization(
                        (
                            self.structure[layer]["nodes"],
                            self.structure[input_layer]["nodes"],
                        ),
                        activation_function,
                    )
                weights_update = numpy.zeros(weights.shape)
                # TODO: add bias initialization
                biases = numpy.zeros((self.structure[layer]["nodes"], 1))
                bias_update = numpy.zeros(biases.shape)

            new_structure[layer] = {
                "name": layer,
                "type": self.structure[layer]["type"],
                "nodes": self.structure[layer]["nodes"],
                "weight": weights,
                "weight_update": weights_update,
                "bias": biases,
                "bias_update": bias_update,
                "data": None,
                "sensitivity": None,
                "activation": activation_function,
                "input_layer": input_layer,
                "output_layer": output_layer,
            }

            if new_structure[layer]["type"] == "input":
                self.input_layers.append(new_structure[layer])
            if new_structure[layer]["type"] == "output":
                self.output_layers.append(new_structure[layer])
        self.structure = new_structure
        logger("Final structure committed: ", self.verbose)
        logger(self.structure, self.verbose)

    def prepare_data(self, input_data: numpy.ndarray) -> numpy.ndarray:
        """
        If normalization and/or standardization are selected, performs normalization and/or standardization on the input
        data.
        :param input_data: numpy.ndarray:
        :return numpy.ndarray
        """
        if self.standardization is True:
            logger("Data standardization: \nBefore: " + str(input_data), self.verbose)
            input_data = self.standardize_data(input_data)
            logger("After: " + str(input_data), self.verbose)

        if self.normalization is True:
            logger("Data normalization: \nBefore: " + str(input_data), self.verbose)
            input_data = self.normalize_data(input_data)
            logger("After: " + str(input_data), self.verbose)

        return input_data

    @staticmethod
    def normalize_data(input_data: numpy.ndarray) -> numpy.ndarray:
        """
        Normalize input data
        # TODO: export in another module and add other standardization techniques
        :param input_data: numpy.ndarray:
        :return numpy.ndarray
        """
        return (input_data - numpy.min(input_data)) / (
            numpy.max(input_data) - numpy.min(input_data)
        )

    @staticmethod
    def standardize_data(input_data: numpy.ndarray) -> numpy.ndarray:
        """
        Standardize input data
        # TODO: export in another module and add other standardization techniques
        :param input_data: numpy.ndarray:
        :return numpy.ndarray
        """
        return (input_data - numpy.mean(input_data)) / numpy.mean(input_data)

    def forward(self, input_data: numpy.ndarray) -> numpy.ndarray:
        """
        This method has to be overridden
        :param input_data: numpy.ndarray:
        :return numpy.ndarray
        """
        pass

    def backward(self, output_data: numpy.ndarray, target_data: numpy.ndarray) -> int:
        """
        This method has to be overridden
        :param output_data: numpy.ndarray: 
        :param target_data: numpy.ndarray: 
        :return int
        """
        pass

    def update_weights(self, parallel: bool = False, batch_size: int = 1):
        """
        Updates the weights and the biases of every layer of the network when a batch is complete or when the current
        epoch is over.
        #  TODO: test the parallel implementation
        :param batch_size: int (Default value = 1)
        :param parallel: bool:  (Default value = False)
        """
        if all(
            numpy.count_nonzero(self.structure[layer]["weight_update"]) == 0
            for layer in self.structure
        ):
            return

        if parallel is False:
            logger("Updating weights for sequential execution", self.verbose)
            for layer in [
                self.structure[layers]
                for layers in self.structure
                if self.structure[layers]["type"] != "input"
            ]:
                # W_new(i) = W_old(i) - decay * weight_update (=S(i) * aT)
                logger(
                    "Updating layer "
                    + layer["name"]
                    + "'s weight: "
                    + "\nWeights updates: "
                    + str(layer["weight_update"])
                    + "\nWeight before: "
                    + str(layer["weight"]),
                    self.verbose,
                )

                layer["weight"] = (
                    layer["weight"] - self.learning_rate * layer["weight_update"]
                )
                layer["weight_update"] = numpy.zeros(layer["weight_update"].shape)
                layer["biases"] = (
                    layer["weight"] - self.learning_rate * layer["bias_update"]
                )
                layer["bias_update"] = numpy.zeros(layer["bias_update"].shape)

                logger("\nAfter: " + str(layer["weight"]), self.verbose)
        else:
            logger("Updating weights for parallel execution", self.verbose)
            for layer in [
                layers
                for layers in self.structure
                if self.structure[layers]["type"] != "input"
            ]:
                logger(
                    "Updating layer "
                    + layer
                    + "'s weight of shared structure: \nBefore: "
                    + str(self.parallel_structure[layer]["weight"]),
                    self.verbose,
                )

                self.parallel_structure[layer]["weight"] = (
                    self.parallel_structure[layer]["weight"]
                    - (self.learning_rate * self.structure[layer]["weight_update"])
                    / batch_size
                )
                self.structure[layer]["weight_update"] = numpy.zeros(
                    self.structure[layer]["weight_update"].shape
                )
                logger(
                    "\nAfter: " + str(self.parallel_structure[layer]["weight"]),
                    self.verbose,
                )
            self.structure = self.parallel_structure.copy()
        logger("Weights updated!", Verbosity.DEBUG)

    def parallel_train(
        self,
        input_data: numpy.ndarray,
        target_data: numpy.ndarray,
        validation_input: numpy.ndarray,
        validation_target: numpy.ndarray,
        epochs: int,
        batch_size: int = 1,
        parallel_batches: int = 4,
    ):
        """
        Parallel implementation of train function
        # TODO: test the parallel implementation of the train function
        :param input_data: numpy.ndarray: 
        :param target_data: numpy.ndarray: 
        :param validation_input: numpy.ndarray: 
        :param validation_target: numpy.ndarray: 
        :param epochs: int: 
        :param batch_size: int:  (Default value = 1)
        :param parallel_batches: int:  (Default value = 4)

        """
        logger(
            "Starting parallel training. Parameters: "
            "\n -epochs"
            + str(epochs)
            + "\n -batch size "
            + str(batch_size)
            + "\n - parallel batches: "
            + str(parallel_batches),
            self.verbose,
        )
        self.parallel_structure = multiprocessing.Manager().dict(self.structure)
        for epoch in range(epochs):
            processors = []
            for batches in range(parallel_batches):
                p = multiprocessing.Process(
                    target=self.train,
                    args=(
                        input_data,
                        target_data,
                        validation_input,
                        validation_target,
                        1,
                        batch_size,
                        True,
                    ),
                )
                p.start()
                processors.append(p)

            for element in processors:
                element.join()
        print("Finished training!")

    def check_validation_data(
        self,
        input_data: numpy.ndarray,
        target_data: numpy.ndarray,
        validation_input: numpy.ndarray,
        validation_target: numpy.ndarray,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Checks that the validation data are present or, if not, if there are enough input data splits them into
        train data and validation data
        :param validation_target: numpy.ndarray:
        :param input_data: numpy.ndarray: 
        :param target_data: numpy.ndarray: 
        :param validation_input: numpy.ndarray: 
        :param validation_target: numpy.ndarray: 
        :return Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """
        logger(
            "Checking arrays sizes (Before): "
            "\n -input size: "
            + str(input_data.shape)
            + "\n -target size: "
            + str(target_data.shape)
            + "\n -validation input size: "
            + str(validation_input.shape)
            + "\n -validation target size: "
            + str(validation_target.shape),
            self.verbose,
        )
        if len(validation_input) == 0:
            validation_input = input_data[int(len(input_data) * 0.85) :]
            input_data = input_data[: int(len(input_data) * 0.85)]
        else:
            number_of_cases = int(validation_input.size / input_data.shape[1])
            validation_input = numpy.reshape(
                validation_input, (number_of_cases, input_data.shape[1], 1)
            )

        if len(validation_target) == 0:
            validation_target = target_data[int(len(target_data) * 0.85) :]
            target_data = target_data[: int(len(input_data) * 0.85)]
        else:
            number_of_cases = int(validation_target.size / target_data.shape[1])
            validation_target = numpy.reshape(
                validation_target, (number_of_cases, target_data.shape[1], 1)
            )

            logger(
                "Checking arrays sizes (After): "
                "\n -input size: "
                + str(input_data.shape)
                + "\n -target size: "
                + str(target_data.shape)
                + "\n -validation input size: "
                + str(validation_input.shape)
                + "\n -validation target size: "
                + str(validation_target.shape),
                self.verbose,
            )
        return input_data, target_data, validation_input, validation_target

    def check_input_target_size(
        self, input_data: numpy.ndarray, target_data: numpy.ndarray,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Checks (and reshapes if needed) the shape of the input and target datasets
        :param input_data: numpy.ndarray: 
        :param target_data: numpy.ndarray: 
        :return Tuple[numpy.ndarray, numpy.ndarray]
        """
        input_size = self.input_layers[0]["nodes"]
        number_pairs = int(input_data.size / input_size)
        target_size = int(target_data.size / number_pairs)

        input_data = numpy.reshape(input_data, (number_pairs, input_size, 1))
        target_data = numpy.reshape(target_data, (number_pairs, target_size, 1))
        logger(
            "Input data shape: "
            + str(input_data.shape)
            + "\nTarget data shape: "
            + str(target_data.shape),
            Verbosity.DEBUG,
        )
        return input_data, target_data

    def train(
        self,
        input_data: numpy.ndarray,
        target_data: numpy.ndarray,
        validation_input: numpy.ndarray = numpy.array([]),
        validation_target: numpy.ndarray = numpy.array([]),
        epochs: int = 1,
        batch_size: int = 1,
        parallel: bool = False,
    ):
        """
        Main cycle for the training process. Iteravely, the forward and backward passes are called, and, considering
        the batch sizes, the weights are updated. This is repeated for however many epochs are specified.
        :param input_data: numpy.ndarray: 
        :param target_data: numpy.ndarray: 
        :param validation_input: numpy.ndarray:  (Default value = numpy.array([]))
        :param validation_target: numpy.ndarray:  (Default value = numpy.array([]))
        :param epochs: int:  (Default value = 1)
        :param batch_size: int:  (Default value = 1)
        :param parallel: bool:  (Default value = False)

        """
        logger(
            "STARTING TRAINING. paramerers: "
            + "\n -epochs: "
            + str(epochs)
            + "\n -batch size: "
            + str(batch_size)
            + "\n -parallel: "
            + str(parallel),
            self.verbose,
        )

        input_data, target_data = self.check_input_target_size(input_data, target_data)

        logger("Training:", Verbosity.DEBUG)
        for epoch in range(1, epochs + 1):
            logger("Epoch " + str(epoch), Verbosity.DEBUG)
            batch = 0
            total_correct = 0
            total_value = 0
            total_loss = 0.0
            for input_element, target_element in zip(input_data, target_data):
                output_element = self.forward(input_element)
                if (output_element == target_element).all() or (
                    self.output_layers[0]["activation"] == ActivationFunctions.SOFTMAX
                    and numpy.amax(output_element) == numpy.amax(target_element)
                ):
                    total_correct += 1
                step_loss = self.backward(output_element, target_element)
                batch += 1
                total_value += 1
                total_loss += step_loss
                logger("Step n°. " + str(total_value) + "\nStep loss: " + str(step_loss),Verbosity.DEBUG,)
                if ((total_value / int(input_data.shape[0])) * 100) % 10 == 0:
                    logger(
                        "Epoch "
                        + str(epoch)
                        + ": "
                        + str(((total_value / int(input_data.shape[0])) * 100))
                        + "% complete!",
                        Verbosity.DEBUG,
                    )

                if batch == batch_size:
                    batch = 0
                    logger("updating weights", self.verbose)
                    self.update_weights(parallel, batch_size)

            self.update_weights(parallel, batch_size)

            logger(
                "Epoch loss: "
                + str(total_loss / total_value)
                + ", epoch accuracy: "
                + str((total_correct / total_value) * 100),
                Verbosity.DEBUG,
            )

            if self.validation(validation_input, validation_target) is True:
                break

        logger("Finished training!", Verbosity.DEBUG)

    def validation(
        self, validation_input: numpy.ndarray, validation_target: numpy.ndarray
    ) -> bool:
        """
        Performs a validation check of the network and, if a threshold is passed, interrupts the training process
        :param validation_input: numpy.ndarray:
        :param validation_target: numpy.ndarray:
        :return bool
        """
        logger("VALIDATION CHECK", self.verbose)
        if (
            len(validation_input) == 0
            or len(validation_target) == 0
            or len(validation_input) != len(validation_target)
        ):
            logger("No validation data provided", self.verbose)
            return False

        loss = 0
        for data, target in zip(validation_input, validation_target):
            output_element = self.forward(data)
            loss += self.loss_function.value.compute(output_element, target)
        logger("Loss on validation data: " + str(loss), self.verbose)
        if loss < self.validation_threshold:
            return True
        else:
            return False

    def test(self, test_input: numpy.ndarray, test_target: numpy.ndarray):
        """
        Performs the test process, by executing only the forward pass and retrieving the loss
        :param test_input: numpy.ndarray:
        :param test_target: numpy.ndarray:
        """
        logger("STARTING TESTING", self.verbose)
        test_input, test_target = self.check_input_target_size(test_input, test_target)
        loss = 0
        total_correct = 0
        total_value = 0
        for data, target in zip(test_input, test_target):
            output_element = self.forward(data)
            loss += self.loss_function.value.compute(output_element, target)
            total_value += 1
            if (output_element == target).all() or (
                self.output_layers[0]["activation"] == ActivationFunctions.SOFTMAX
                and numpy.amax(output_element) == numpy.amax(target)
            ):
                total_correct += 1

        logger(
            "Testing loss: "
            + str(loss / total_value)
            + ", testing accuracy: "
            + str((total_correct / total_value) * 100), self.verbose
        )
