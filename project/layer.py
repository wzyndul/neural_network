from abc import ABC, abstractmethod

# abstrakcyjna klasa z której dziedziczy hidden layer i output layer
import numpy as np


class Layer(ABC):
    def __init__(self):
        self.neuron_num = None
        self.input_num = None
        self.neurons = None
        self.output = None
        self.true_values = None
        self.inputs = None

    def set_true_values(self, values):
        self.true_values = values

    def weighted_sum(self):
        return np.array([neuron.weighted_sum for neuron in self.neurons])

    def update_weights(self, new_weights):
        for x in range(len(self.neurons)):  # teraz tylko dla po jednym neuronie
            self.neurons[x].update_weight(new_weights[x])

    def get_weights(self):
        arr = np.array([neuron.weights for neuron in self.neurons])
        return arr

    @abstractmethod
    def forward(self, inputs):
        pass
