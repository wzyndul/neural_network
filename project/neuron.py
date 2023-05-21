import numpy as np
from activation_functions import sigmoid


class Neuron:
    def __init__(self, inputs_num: int):
        self.neuron_inputs = None
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=inputs_num)  # na razie zawsze losuje
        self.bias = 0  # określić jeszcze ten parametr i dodac do opisu metody
        self.output = 0
        self.weighted_sum = 0

    def forward(self, inputs, activation_func=sigmoid) -> float:
        self.weighted_sum = np.dot(self.weights, inputs) # bez biasu na razie
        if activation_func.__name__ == 'softmax':
            return self.weighted_sum
        self.output = activation_func(self.weighted_sum)
        return self.output

    def loss_function(self, true_value):
        return np.mean((self.output - true_value) ** 2)

    def update_weight(self, new_weights):
        self.weights = new_weights

# dot product work like :
# a = [1, 2, 3]
# b = [2, 3, 4]
# dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]   = 20
#
