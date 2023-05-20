import numpy as np

from project.activation_functions import softmax
from project.layer import Layer
from project.neuron import Neuron


class OutputLayer(Layer):
    def __init__(self, input_num: int, neuron_num: int):
        super().__init__()
        self.neuron_num = neuron_num
        self.input_num = input_num
        self.neurons = [Neuron(input_num) for _ in range(neuron_num)]

    def forward(self, inputs):  # I only want to obtain weighted sums and not call any activation function
        self.inputs = inputs
        weighted_sums = [neuron.forward(inputs) for neuron in self.neurons]
        self.output = softmax(weighted_sums)  # now I call softmax function for entire array
        return self.output

    #
    # def mean_loss(self):
    #     summ = 0
    #     for neuron, value in zip(self.neurons, self.true_values):
    #         summ += neuron.loss_function(value)
    #     return summ / len(self.neurons)

    def loss(self, index):  # zwraca pochodną funckji straty
        return (1 / 2 * len(self.true_values)) * (self.forward(self.inputs)[index] - self.true_values[index])
