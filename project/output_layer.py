import numpy as np

from project.activation_functions import softmax
from project.layer import Layer
from project.neuron import Neuron


class OutputLayer(Layer):
    def __init__(self, input_num: int, neuron_num: int):
        super().__init__()
        self.neuron_num = neuron_num
        self.input_num = input_num
        self.neurons = np.array([Neuron(input_num) for _ in range(neuron_num)])

    def forward(self, inputs):  # I only want to obtain weighted sums and not call any activation function
        weighted_sums = np.array([neuron.forward(inputs, activation_func=softmax) for neuron in self.neurons])
        print(weighted_sums)
        self.output = softmax(weighted_sums) # now I call softmax function for entire array
        return self.output



data = [1,2, 3, 4]

layer = OutputLayer(4, 2)
print(layer.forward(data))

