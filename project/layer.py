
import numpy as np
from project.neuron import Neuron


class Layer:
    def __init__(self, input_num: int, neuron_num: int):
        self.neuron_num = neuron_num
        self.input_num = input_num
        self.neurons = np.array([Neuron(input_num) for _ in range(neuron_num)])
        self.output = None

    def forward(self, inputs):
        self.output = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return self.output

    def weighted_sum(self):
        return np.array([neuron.weighted_sum for neuron in self.neurons])

    def update_weights(self, new_weights):
        for x in range(len(self.neurons)):
            self.neurons[x].update_weight(new_weights[x])

    def get_weights(self):
        arr = np.array([neuron.weights for neuron in self.neurons])
        return arr

