import numpy
import multiprocessing
from RDL.NeuralNetworks.activationFunctions import ActivationFunctions
from RDL.NeuralNetworks.lossFunctions import LossFunctions
from RDL.configs.debug import logger, Verbosity
import pprint


class BaseNeuralNetwork:
    """ """
    def __init__(self, verbose: Verbosity=Verbosity.RELEASE):
        self.context = {}
        self.structure = {}
        self.loss_function = None
        self.loss = None
        self.input_layers = []
        self.output_layers = []
        self.learning_rate = 0.1
        self.normalization = True
        self.standardization = True
        self.parallel_structure = multiprocessing.Manager().dict()
        self.validation_threshold = 0.001
        self.verbose = verbose

    def add_input_layer(self, layer_name: str, input_nodes: int):
        """

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
    ):
        """

        :param layer_name: str: 
        :param input_layer: str: 
        :param nodes: int: 
        :param activation_function: ActivationFunctions: 

        """
        self.structure[layer_name] = {
            "name": layer_name,
            "type": "hidden",
            "nodes": nodes,
            "activation": activation_function,
            "input_layer": input_layer,
        }
        logger("Added hidden layer: ", self.verbose)
        logger(self.structure[layer_name], self.verbose)

    def add_output_layer(
            self,
            layer_name: str,
            input_layer: str,
            nodes: int,
            activation_function: ActivationFunctions = ActivationFunctions.LINEAR,
    ):
        """

        :param layer_name: str: 
        :param input_layer: str: 
        :param nodes: int: 
        :param activation_function: ActivationFunctions:  (Default value = ActivationFunctions.LINEAR)

        """
        self.structure[layer_name] = {
            "name": layer_name,
            "type": "output",
            "activation": activation_function,
            "nodes": nodes,
            "input_layer": input_layer,
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


        """
        if self.loss_function is None:
            print("Error: loss function missing or not recognized.")
            exit()

        if not (0 < len([layer for layer in self.structure
                       if self.structure[layer]["type"] == "input"]) < 2):
            print("Error: supported only one input layer.")
            exit()

        if not (0 < len([layer for layer in self.structure
                if self.structure[layer]["type"] == "output"]) < 2):
            print("Error: supported only one output layer.")
            exit()

        for layer in self.structure:
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
        """ """
        self.sanity_check()
        new_structure = dict()
        for layer in self.structure:
            input_layer = ""
            output_layer = ""
            activation_function = ""
            weights = None
            weights_update = None
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
                weights = numpy.ones(
                    (
                        self.structure[layer]["nodes"],
                        self.structure[input_layer]["nodes"],
                    )
                )
                weights_update = numpy.zeros(weights.shape)
                biases = numpy.ones((self.structure[layer]["nodes"], 1))

            new_structure[layer] = {
                "name": layer,
                "type": self.structure[layer]["type"],
                "nodes": self.structure[layer]["nodes"],
                "weight": weights,
                "weight_update": weights_update,
                "bias": biases,
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

    def weights_initialization(self):
        """ """
        # TODO
        pass

    def prepare_data(self, input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

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
    def normalize_data(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        minimum = numpy.min(input_data)
        maximum = numpy.max(input_data)

        for rows in range(input_data.shape[0]):
            for columns in range(input_data.shape[1]):
                input_data[rows][columns] = (input_data[rows][columns] - minimum) / (
                        maximum - minimum
                )
        return input_data

    @staticmethod
    def standardize_data(input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        mean = numpy.mean(input_data)
        std = numpy.std(input_data)
        for rows in range(input_data.shape[0]):
            for columns in range(input_data.shape[1]):
                input_data[rows][columns] = (input_data[rows][columns] - mean) / std
        return input_data

    def forward(self, input_data: numpy.array) -> numpy.array:
        """

        :param input_data: numpy.array: 

        """
        pass

    def backward(self, output_data: numpy.array, target_data: numpy.array) -> int:
        """

        :param output_data: numpy.array: 
        :param target_data: numpy.array: 

        """
        pass

    def update_weights(self, parallel: bool = False, batch_size: int=1):
        """

        :param parallel: bool:  (Default value = False)

        """
        if parallel is False:
            logger("Updating weights for sequential execution", self.verbose)
            for layer in [
                self.structure[layers]
                for layers in self.structure
                if layers["type"] != "input"
            ]:
                # W_new(i) = W_old(i) - decay * S(i) * input(i)T
                logger("Updating layer " + layer["name"] + "'s weight: \nBefore: " + str(layer["weight"]), self.verbose)
                layer["weight"] = (
                        layer["weight"] - (self.learning_rate * layer["weight_update"]) / batch_size
                )
                logger("\nAfter: " + str(layer["weight"]), self.verbose)
                layer["weight_update"] = numpy.zeros(layer["weight_update"].shape)
        else:
            logger("Updating weights for parallel execution", self.verbose)
            for layer in [
                layers for layers in self.structure if layers["type"] != "input"
            ]:
                logger("Updating layer " + layer + "'s weight of shared structure: \nBefore: "
                       + str(self.parallel_structure[layer]["weight"]), self.verbose)

                self.parallel_structure[layer]["weight"] = (
                        self.parallel_structure[layer]["weight"]
                        - (self.learning_rate * self.structure[layer]["weight_update"]) / batch_size
                )
                self.structure[layer]["weight_update"] = numpy.zeros(
                    self.structure[layer]["weight_update"].shape
                )
                logger("\nAfter: " + str(self.parallel_structure[layer]["weight"]), self.verbose)
            self.structure = self.parallel_structure.copy()

    def parallel_train(
            self,
            input_data: numpy.array,
            target_data: numpy.array,
            validation_input: numpy.array,
            validation_target: numpy.array,
            epochs: int,
            batch_size: int = 1,
            parallel_batches: int = 4,
    ):
        """

        :param input_data: numpy.array: 
        :param target_data: numpy.array: 
        :param validation_input: numpy.array: 
        :param validation_target: numpy.array: 
        :param epochs: int: 
        :param batch_size: int:  (Default value = 1)
        :param parallel_batches: int:  (Default value = 4)

        """
        logger("Starting parallel training. Parameters: "
               "\n -epochs" + str(epochs) +
               "\n -batch size " + str(batch_size) +
               "\n - parallel batches: " + str(parallel_batches), self.verbose)
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

    def check_input_validation_data(
            self,
            input_data: numpy.array,
            target_data: numpy.array,
            validation_input: numpy.array,
            validation_target: numpy.array,
    ):
        """

        :param input_data: numpy.array: 
        :param target_data: numpy.array: 
        :param validation_input: numpy.array: 
        :param validation_target: numpy.array: 

        """
        logger("Checking arrays sizes (Before): "
               "\n -input size: " + str(input_data.shape) +
               "\n -target size: " + str(target_data.shape) +
               "\n -validation input size: " + str(validation_input.shape) +
               "\n -validation target size: " + str(validation_target.shape), self.verbose
               )
        if len(validation_input) == 0:
            validation_input = input_data[int(len(input_data) * 0.85) :]
            input_data = input_data[: int(len(input_data) * 0.85)]
        if len(validation_target) == 0:
            validation_target = target_data[int(len(target_data) * 0.85) :]
            target_data = target_data[: int(len(input_data) * 0.85)]

        rows, columns = input_data.shape
        if rows == 1 or columns == 1:
            input_data = numpy.array([input_data])
            target_data = numpy.array([target_data])

        logger("Checking arrays sizes (After): "
               "\n -input size: " + str(input_data.shape) +
               "\n -target size: " + str(target_data.shape) +
               "\n -validation input size: " + str(validation_input.shape) +
               "\n -validation target size: " + str(validation_target.shape), self.verbose
               )
        return input_data, target_data, validation_input, validation_target

    def train(
            self,
            input_data: numpy.array,
            target_data: numpy.array,
            validation_input: numpy.array = numpy.array([]),
            validation_target: numpy.array = numpy.array([]),
            epochs: int = 1,
            batch_size: int = 1,
            parallel: bool = False,
    ):
        """

        :param input_data: numpy.array: 
        :param target_data: numpy.array: 
        :param validation_input: numpy.array:  (Default value = numpy.array([]))
        :param validation_target: numpy.array:  (Default value = numpy.array([]))
        :param epochs: int:  (Default value = 1)
        :param batch_size: int:  (Default value = 1)
        :param parallel: bool:  (Default value = False)

        """
        logger("STARTING TRAINING. paramerers: " +
               "\n -epochs: " + str(epochs) +
               "\n -batch size: " + str(batch_size) +
               "\n -parallel: " + str(parallel), self.verbose)
        (
            input_data,
            target_data,
            validation_input,
            validation_target,
        ) = self.check_input_validation_data(
            input_data, target_data, validation_input, validation_target
        )

        print("Training:")
        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            batch = 0
            total_correct = 0
            total_value = 0
            total_loss = 0.0
            for input_element, target_element in zip(input_data, target_data):
                input_row, input_column = input_element.shape
                if input_row == 1:
                    input_element = input_element.reshape(input_column, input_row)

                output_element = self.forward(input_element)
                logger("Forward pass: \n -Input data: " + str(input_element) +
                       "\n -Output data: " + str(output_element), self.verbose)

                step_loss = self.backward(output_element, target_element)
                total_loss += step_loss
                batch += 1
                total_value += 1

                logger("Step loss: " + str(step_loss), self.verbose)
                if output_element == target_element or (
                        self.output_layers[0]["activation"] == ActivationFunctions.SOFTMAX
                        and numpy.argmax(output_element, axis=1)
                        == numpy.argmax(target_element, axis=1)
                ):
                    total_correct += 1

                print("Step loss: " + str(step_loss))
                if total_value % int(len(input_data) / 10) == 0:
                    print(
                        "Epoch "
                        + str(total_value % int(len(input_data) / 10))
                        + " complete!"
                    )

                if batch == batch_size:
                    batch = 0
                    logger("updating weights", self.verbose)
                    self.update_weights(parallel, batch_size)

            self.update_weights(parallel, batch_size)

            print(
                "Epoch loss: "
                + str(round(total_loss / total_value, 3))
                + ", epoch accuracy: "
                + str(round(total_correct / total_value, 3))
            )

            if self.validation(validation_input, validation_target) is True:
                break

        return self.structure

    def validation(self, validation_input: numpy.array, validation_target: numpy.array):
        """

        :param validation_input: numpy.array: 
        :param validation_target: numpy.array: 

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
            loss += self.loss_function.forward(output_element, target)
        logger("Loss on validation data: " + str(loss), self.verbose)
        if loss < self.validation_threshold:
            return True
        else:
            return False

    def test(self, test_input: numpy.array, test_target: numpy.array):
        """

        :param test_input: numpy.array: 
        :param test_target: numpy.array: 

        """
        logger("STARTING TESTING", self.verbose)
        loss = 0
        total_correct = 0
        total_value = 0
        print("Testing")
        for data, target in zip(test_input, test_target):
            output_element = self.forward(data)
            loss += self.loss_function.forward(output_element, target)
            total_value += 1
            if output_element == target or (
                    self.output_layers[0]["activation"] == ActivationFunctions.SOFTMAX
                    and numpy.argmax(output_element, axis=1) == numpy.argmax(target, axis=1)
            ):
                total_correct += 1

        print(
            "Testing loss: "
            + str(round(loss / total_value, 3))
            + ", testing accuracy: "
            + str(round(total_correct / total_value, 3))
        )
