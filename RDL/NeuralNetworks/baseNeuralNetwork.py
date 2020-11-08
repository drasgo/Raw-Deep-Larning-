import numpy
import multiprocessing
from RDL.NeuralNetworks.activationFunctions import ActivationFunctions


class BaseNeuralNetwork:

    def __init__(self):
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

    def add_input_layer(self, layer_name: str, input_nodes: int):
        self.structure[layer_name] = {
            "name": layer_name,
            "type": "input",
            "nodes": input_nodes
        }

    def add_layer(self, layer_name: str, input_layer: str, nodes: int, activation_function: ActivationFunctions):
        self.structure[layer_name] = {
            "name": layer_name,
            "type": "hidden",
            "nodes": nodes,
            "activation": activation_function,
            "input_layer": input_layer
        }

    def add_output_layer(self,
                         layer_name: str,
                         input_layer: str,
                         nodes: int,
                         activation_function: ActivationFunctions=ActivationFunctions.LINEAR):
        self.structure[layer_name] = {
            "name": layer_name,
            "type": "output",
            "activation": activation_function,
            "nodes": nodes,
            "input_layer": input_layer
        }

    def add_standardization(self, standardize: bool = True):
        self.standardization = standardize

    def add_normalization(self, normalize: bool = True):
        self.normalization = normalize

    def add_operation(self, operation):
        # TODO: Not supported yet
        pass

    def add_loss_function(self, loss_function):
        self.loss_function = loss_function

    def add_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate

    def sanity_check(self):
        """
        This method performs a sanity control of the structure. Orderly, it checks that:
        - a loss function is provided
        - there is only one Input layer
        - there is only one Output layer
        - every layer has an input layer (data comes from a past layer), except Input layer
        - every layer has an ouput layer (data goes to another layer), except Output layer

        """
        if self.loss_function is None:
            print("Error: loss function missing or not recognized.")
            exit()

        if not(0 < len([layer for layer in self.structure if self.structure[layer]["type"] == "input"]) < 2):
            print("Error: supported only one input layer.")
            exit()

        if not(0 < len([layer for layer in self.structure if self.structure[layer]["type"] == "output"]) < 2):
            print("Error: supported only one output layer.")
            exit()

        for layer in self.structure:
            if self.structure[layer]["type"] != "input" and self.structure[layer]["input_layer"] not in self.structure:
                print("Error: in layer " + layer + " input layer specified is " +
                      self.structure[layer]["input_layer"] + " but no layer with this name exists.")
                exit()

            if self.structure[layer]["type"] != "output" and \
                not any("input_layer" in self.structure[out_layer] and
                                self.structure[out_layer]["input_layer"] == layer for out_layer in self.structure):
                    print("Error: for layer " + layer + " no output layer found. Possible error designing structure.")
                    exit()

    def commit_structure(self):
        self.sanity_check()
        new_structure = dict()
        for layer in self.structure:
            input_layer = ""
            output_layer = ""
            activation_function = ""
            weights = None
            biases = None

            if self.structure[layer]["type"] != "output":
                output_layer = [out_layer for out_layer in self.structure
                                if "input_layer" in self.structure[out_layer] and
                                self.structure[out_layer]["input_layer"] == layer]

                if len(output_layer) == 0:
                    print("Error: for layer " + layer + " no output layer found. Possible error designing structure.")
                    exit()

                output_layer = output_layer[0]

            if self.structure[layer]["type"] != "input":
                if self.structure[layer]["input_layer"] not in self.structure:
                    print("Error: in layer " + layer + " input layer specified is " +
                          self.structure[layer]["input_layer"] + " but no layer with this name exists.")
                    exit()

                activation_function = self.structure[layer]["activation"]
                input_layer = self.structure[layer]["input_layer"]
                weights = numpy.ones((self.structure[layer]["nodes"], self.structure[input_layer]["nodes"]))
                biases = numpy.ones((self.structure[layer]["nodes"], 1))

            new_structure[layer] = {
                "name": layer,
                "type": self.structure[layer]["type"],
                "nodes": self.structure[layer]["nodes"],
                "weight": weights,
                "bias": biases,
                "data": None,
                "sensitivity": None,
                "weight_update": None,
                "activation": activation_function,
                "input_layer": input_layer,
                "output_layer": output_layer
            }
            if new_structure[layer]["type"] == "input":
                self.input_layers.append(new_structure[layer])
            if new_structure[layer]["type"] == "output":
                self.output_layers.append(new_structure[layer])

        self.structure = new_structure

    def weights_initialization(self):
        # TODO
        pass

    def prepare_data(self, input_data: numpy.Array) -> numpy.Array:
        if self.normalization is True:
            input_data = self.normalize_data(input_data)

        if self.standardization is True:
            input_data = self.standardize_data(input_data)

        return input_data

    @staticmethod
    def normalize_data(input_data: numpy.Array) -> numpy.Array:
        minimum = numpy.min(input_data)
        maximum = numpy.max(input_data)
        for rows in range(input_data):
            for columns in range(input_data[rows]):
                input_data[rows][columns] = (input_data[rows][columns] - minimum) / (maximum - minimum)
        return input_data

    @staticmethod
    def standardize_data(input_data: numpy.Array) -> numpy.Array:
        mean = numpy.mean(input_data)
        std = numpy.std(input_data)
        for rows in range(input_data):
            for columns in range(input_data[rows]):
                input_data[rows][columns] = (input_data[rows][columns] - mean) / std
        return input_data

    def forward(self, input_data: numpy.Array) -> numpy.Array:
        pass

    def backward(self, output_data: numpy.Array, target_data: numpy.Array) -> int:
        pass

    def update_weights(self, parallel: bool=False):
        if parallel is False:
            for layer in [self.structure[layers] for layers in self.structure if layers["type"] != "input"]:
                # W_new(i) = W_old(i) - decay * S(i) * input(i)T
                layer["weight"] = layer["weight"] - self.learning_rate * layer["weight_update"]
                layer["weight_update"] = numpy.zeros(layer["weight_update"].shape)
        else:
            for layer in [layers for layers in self.structure if layers["type"] != "input"]:
                self.parallel_structure[layer]["weight"] = self.parallel_structure[layer]["weight"] - \
                                                           self.learning_rate * self.structure[layer]["weight_update"]
                self.structure[layer]["weight_update"] = numpy.zeros(self.structure[layer]["weight_update"].shape)
            self.structure = self.parallel_structure.copy()

    def parallel_train(self,
                       input_data: numpy.Array,
                       target_data: numpy.Array,
                       validation_input: numpy.Array,
                       validation_target: numpy.Array,
                       epochs: int,
                       batch_size: int=1,
                       parallel_batches: int=1):
        self.parallel_structure = multiprocessing.Manager().dict(self.structure)

        for epoch in range(epochs):
            processors = []
            for batches in range(parallel_batches):
                p = multiprocessing.Process(target=self.train, args=(
                    input_data,
                    target_data,
                    validation_input,
                    validation_target,
                    1,
                    batch_size,
                    True
                    ))
                p.start()
                processors.append(p)

            for element in processors:
                element.join()

    def train(self,
              input_data: numpy.Array,
              target_data: numpy.Array,
              validation_input: numpy.Array,
              validation_target: numpy.Array,
              epochs: int=1,
              batch_size: int=1,
              parallel: bool=False):

        if len(validation_input) == 0:
            validation_input = input_data[:int(len(input_data)*0.85)]
        if len(validation_target) == 0:
            validation_target = target_data[:int(len(target_data)*0.85)]

        print("Training:")
        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            batch = 0
            total_correct = 0
            total_value = 0
            total_loss = 0.0
            for input_element, target_element in zip(input_data, target_data):
                output_element = self.forward(input_element)
                step_loss = self.backward(output_element, target_element)
                total_loss += step_loss
                batch += 1
                total_value += 1

                if output_element == target_element or \
                    (self.output_layers[0]["activation"] == ActivationFunctions.SOFTMAX and
                     numpy.argmax(output_element, axis=1) == numpy.argmax(target_element, axis=1)):
                    total_correct += 1

                print("Step loss: " + str(step_loss))
                if total_value % int(len(input_data) / 10) == 0:
                    print("Epoch " + str(total_value % int(len(input_data) / 10)) + " complete!")

                if batch == batch_size:
                    batch = 0
                    self.update_weights(parallel)

            self.update_weights(parallel)

            print("Epoch loss: " + str(round(total_loss/total_value, 3)) +
                  ", epoch accuracy: " + str(round(total_correct/total_value, 3)))

            if self.validation(validation_input, validation_target) is True:
                break

        return self.structure

    def validation(self, validation_input: numpy.Array, validation_target: numpy.Array):
        if len(validation_input) == 0 or len(validation_target) == 0 or len(validation_input) != len(validation_target):
            return False

        loss = 0
        for data, target in zip(validation_input, validation_target):
            output_element = self.forward(data)
            loss += self.loss_function.forward(output_element, target)

        if loss < self.validation_threshold:
            return True
        else:
            return False

    def test(self, test_input: numpy.Array, test_target: numpy.Array):
        loss = 0
        total_correct = 0
        total_value = 0
        print("Testing")
        for data, target in zip(test_input, test_target):
            output_element = self.forward(data)
            loss += self.loss_function.forward(output_element, target)
            total_value += 1

            if output_element == target or \
                    (self.output_layers[0]["activation"] == ActivationFunctions.SOFTMAX and
                     numpy.argmax(output_element, axis=1) == numpy.argmax(target, axis=1)):
                total_correct += 1

        print("Testing loss: " + str(round(loss / total_value, 3)) +
              ", testing accuracy: " + str(round(total_correct / total_value, 3)))
