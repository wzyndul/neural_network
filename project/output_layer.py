import numpy as np

from project.activation_functions import softmax
from project.layer import Layer
from project.loss_function import categorical_cross_entropy
from project.neuron import Neuron


class OutputLayer(Layer):
    def __init__(self, input_num: int, neuron_num: int):
        super().__init__()
        self.neuron_num = neuron_num
        self.input_num = input_num
        self.neurons = np.array([Neuron(input_num) for _ in range(neuron_num)])

    def forward(self, inputs):  # I only want to obtain weighted sums and not call any activation function
        weighted_sums = np.array([neuron.forward(inputs, activation_func=softmax) for neuron in self.neurons])
        self.output = softmax(weighted_sums)  # now I call softmax function for entire array
        return self.output

    def calculate_loss(self, true_values):
        loss = categorical_cross_entropy(true_values, self.output)

    # w ostatniej warstwie wywołujemy inną funkcje aktywacji i tutaj zrobiłem tak, że właśnie dla każdego neuronu
    # podaje jako funkcje aktywacji softmax i w neuronie jest warunek, że jesli jest taka funkcja aktywacji
    # to zwraca tylko weighted_sum i potem dla takiej całej tabli wag dla wszystkich neuronów wywołuje już właśnie
    # softmax


