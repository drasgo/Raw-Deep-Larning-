import numpy
from RDL.NeuralNetworks.activationFunctions import ActivationFunctions


class BaseNeuralNetwork:

    def __init__(self):
        self.context = {}
        self.structure = {}
        self.loss_function = None
        self.loss = None
        self.input_layers = []
        self.output_layers = []
        self.training_decay = 0.1

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

    def add_operation(self, operation):
        # TODO: Not supported yet
        pass

    def add_loss_function(self, loss_function):
        self.loss_function = loss_function

    def commit_structure(self):
        if self.loss_function is None:
            print("Error: loss function missing or not recognized.")
            exit()

        new_structure = dict()
        # for input_layer in [self.structure[input_layers] for input_layers in self.structure if "input" in input_layers]:
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
                "activation": activation_function,
                "input_layer": input_layer,
                "output_layer": output_layer
            }
            if new_structure[layer]["type"] == "input":
                self.input_layers.append(new_structure[layer])
            if new_structure[layer]["type"] == "output":
                self.output_layers.append(new_structure[layer])

        if not(0 < len(self.input_layers) < 2):
            print("Error: supported only one input layer.")
            exit()

        if not(0 < len(self.output_layers) < 2):
            print("Error: supported only one output layer.")
            exit()

        self.structure = new_structure

    def forward(self, input_data: numpy.Array):
        pass

    def backward(self, output_data: numpy.Array, target_data: numpy.Array):
        pass
