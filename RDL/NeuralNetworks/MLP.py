from RDL.NeuralNetworks.baseNeuralNetwork import BaseNeuralNetwork
import numpy


class MLP(BaseNeuralNetwork):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        # Add check for input data. E.g. correct size, all numeric values, etc.
        # input_flag = True
        for input_layer in [layer for layer in self.structure if self.structure[layer]["type"] == "input"]:
            # TODO: implement multiple input layers

            prev_layer = self.structure[input_layer]
            prev_layer["data"] = input_data
            while True:
                curr_layer = self.structure[prev_layer["output_layer"]]
                output_data = numpy.dot(curr_layer["weight"], prev_layer["data"]) + curr_layer["bias"]
                print("\n\n\n")
                print("layer " + curr_layer["name"] + ". result: ")
                import pprint
                pprint.pprint(output_data)
                if curr_layer["type"] == "output":
                    curr_layer["data"] = output_data
                    break
                else:
                    # curr_layer["data"] = curr_layer["activation"].forward(output_data)
                    curr_layer["data"] = output_data
                    prev_layer = curr_layer

    def backward(self):
        pass