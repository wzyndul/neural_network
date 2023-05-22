import numpy as np
from project.functions import sigmoid_derivative, mean_squared_error
from project.layer import Layer


class Network:
    def __init__(self, layer_num, input_num, bias):
        self.layer_num = layer_num
        self.input_num = input_num
        self.layer_list = []
        self.bias_on = bias
        inputs_layer = input_num
        for i in range(layer_num):
            neuron_num = int(input(f"podaj ilość neuronów w {i + 1} warstwie ukrytej: "))
            self.layer_list.append(Layer(inputs_layer, neuron_num, bias))
            inputs_layer = neuron_num
        self.layer_list.append(Layer(inputs_layer, int(input("podaj ilość neuronów w warstwie ulrytej: ")), bias))

        derivatives = [np.zeros((input_num, self.layer_list[0].neuron_num))]
        derivatives.extend([np.zeros((layer.neuron_num, self.layer_list[i + 1].neuron_num)) for i, layer in
                            enumerate(self.layer_list[:-1])])
        self.derivatives = derivatives

        activations = [np.zeros(input_num)]
        activations.extend([np.zeros(layer.neuron_num) for layer in self.layer_list])
        self.activations = activations

        weights = [np.zeros((input_num, self.layer_list[0].neuron_num))]
        weights.extend([np.zeros((layer.neuron_num, self.layer_list[i + 1].neuron_num)) for i, layer in
                        enumerate(self.layer_list[:-1])])
        self.weights = weights

        biases = []
        biases.extend([np.zeros(layer.neuron_num) for layer in self.layer_list])
        self.biases = biases

        derivatives_biases = []
        derivatives_biases.extend([np.zeros(layer.neuron_num) for layer in self.layer_list])
        self.derivatives_biases = derivatives_biases

    def forward(self, input_vector):
        self.activations[0] = np.array(input_vector)
        next_activation = self.activations[0]
        for x in range(len(self.layer_list)):
            next_activation = self.layer_list[x].forward(next_activation)
            self.activations[x + 1] = next_activation
            self.weights[x] = self.layer_list[x].get_weights()
            self.biases[x] = self.layer_list[x].get_biases()

    def back_propagation(self, error):
        for x in reversed(range(len(self.derivatives))):
            activation = self.activations[x + 1]
            delta = error * sigmoid_derivative(activation)
            delta_resh = delta.reshape(delta.shape[0], -1).T
            curr_act = self.activations[x]
            curr_act_resh = curr_act.reshape(curr_act.shape[0], -1)
            self.derivatives[x] = np.dot(curr_act_resh, delta_resh)

            error = np.dot(delta, self.weights[x])

            self.derivatives_biases[x] = delta

    def update_weights(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights -= derivatives.T * learning_rate
        self.update_neurons_weights()

    def update_biases(self, learning_rate):
        for i in range(len(self.biases)):
            biases = self.biases[i]
            derivatives_biases = self.derivatives_biases[i]
            biases -= derivatives_biases * learning_rate
        self.update_neurons_biases()

    def update_neurons_weights(self):
        for x in range(len(self.layer_list)):
            self.layer_list[x].update_weights(self.weights[x])

    def update_neurons_biases(self):
        for x in range(len(self.layer_list)):
            self.layer_list[x].update_biases(self.biases[x])

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_errors = 0
            for j, input in enumerate(inputs):
                target = targets[j]

                self.forward(input)

                error = self.activations[-1] - target  # to bede outputy sieci
                self.back_propagation(error)
                self.update_weights(learning_rate)
                if self.bias_on:
                    self.update_biases(learning_rate)

                sum_errors += mean_squared_error(target, self.activations[-1])

            print(f"Error: {sum_errors / len(inputs)} at epoch {i + 1}")
