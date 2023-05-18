# inicjalizacja wag neuronu
# obliczanie sumy wejsc,
# aktywowanie funkcji aktywacji
# mozna na sztywno bias = 0, i najwyżej zmieniac, albo jakims parametrem boolowskim czy cso

import numpy as np
from activation_functions import sigmoid, softmax


class Neuron:
    def __init__(self, inputs_num: int):
        """
        Initializes Neuron, generates random weights for given number of inputs.

        :param inputs_num: specifies how many inputs Neuron will have.
        :type inputs_num: int
        """
        self.neuron_inputs = None
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=inputs_num)  # na razie zawsze losuje
        self.bias = 0  # określić jeszcze ten parametr i dodac do opisu metody

    def forward(self, inputs, activation_func=sigmoid) -> float:
        """
        Calculates the weighted sum by performing the np.dot operation.
        Then, the output is calculated using the specified activation function.

        :param inputs: numpy.ndarray which contains inputs.
        :type inputs: numpy.ndarray
        :param activation_func: activation function to be applied, defaults to sigmoid.
        :type activation_func: function, optional
        :return: output of the activation function.
        :rtype: float
        """
        weighted_sum = np.dot(self.weights, inputs) + self.bias  # na razie zawsze dodaje zero
        if activation_func.__name__ == 'softmax':
            return weighted_sum
        output = activation_func(weighted_sum)
        return output


# poniżej tylko jakieś testy są

# dot product work like :
# a = [1, 2, 3]
# b = [2, 3, 4]
# dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]   = 20
#

# neuron = Neuron(3)
# neuron.set_inputs([1, 2, 3])
# print(neuron.neuron_inputs)
# print(neuron.weights)
# print(neuron.forward())
