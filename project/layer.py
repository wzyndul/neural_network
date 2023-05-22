
import numpy as np
from project.neuron import Neuron


class Layer:
    def __init__(self, input_num: int, neuron_num: int, bias: bool):
        """
        Initialize a layer with a specific number of input and neuron.

        :param input_num: The number of inputs.
        :type input_num: int
        :param neuron_num: The number of neuron in this layer.
        :type neuron_num: int
        :param bias: information if biases will be generated.
        :type bias: bool
        """
        self.neuron_num = neuron_num
        self.input_num = input_num
        self.neurons = np.array([Neuron(input_num, bias) for _ in range(neuron_num)])
        self.output = None

    def forward(self, inputs):
        """
        Forward propagate inputs through the layer and compute the output.

        :param inputs: The input values.
        :type inputs: array-like
        :return: The computed output of the layer.
        :rtype: numpy.ndarray
        """
        self.output = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return self.output

    def weighted_sum(self):
        """
        Retrieve the weighted sum values of the neurons in the layer.

        :return: The array of weighted sum values.
        :rtype: numpy.ndarray
        """
        return np.array([neuron.weighted_sum for neuron in self.neurons])

    def update_weights(self, new_weights):
        """
        Update the weights of the neurons in the layer.

        :param new_weights: The new weights.
        :type new_weights: array-like
        """
        for x in range(len(self.neurons)):
            self.neurons[x].update_weight(new_weights[x])

    def get_weights(self):
        """
        Retrieve the weights of the neurons in the layer.

        :return: The array of weights.
        :rtype: numpy.ndarray
        """
        return np.array([neuron.weights for neuron in self.neurons])

    def get_biases(self):
        return np.array([neuron.bias for neuron in self.neurons])

    def update_biases(self, new_biases):
        for x in range(len(self.neurons)):
            self.neurons[x].update_bias(new_biases[x])

