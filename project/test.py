import numpy as np
from random import random


class MLP(object):
    """A Multilayer Perceptron class."""

    def __init__(self, num_inputs, hidden_layers, num_outputs, use_bias=True, momentum=None):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, number of outputs,
            whether to use bias or not, and momentum value.

        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
            use_bias (bool): Whether to use bias or not
            momentum (float): Momentum value (between 0 and 1)
        """
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.use_bias = use_bias
        self.momentum = momentum
        self.layers = [num_inputs] + hidden_layers + [num_outputs]
        self.initialize_weights()

    def initialize_weights(self):

        # create random connection weights for the layers
        weights = []
        biases = []
        velocities = []  # momentum velocities

        for i in range(len(self.layers) - 1):
            w = np.random.rand(self.layers[i], self.layers[i + 1])
            weights.append(w)
            if self.use_bias:
                b = np.zeros(self.layers[i + 1])
                biases.append(b)
            velocities.append(np.zeros((self.layers[i], self.layers[i + 1])))

        self.weights = weights
        self.biases = biases
        self.velocities = velocities

        # save derivatives per layer
        derivatives = []
        for i in range(len(self.layers) - 1):
            d = np.zeros((self.layers[i], self.layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(self.layers)):
            a = np.zeros(self.layers[i])
            activations.append(a)
        self.activations = activations

    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.

        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropagation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # add bias if enabled
            if self.use_bias:
                net_inputs += self.biases[i]

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # save the activations for backpropagation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations

    def back_propagate(self, error):
        """Backpropagates an error signal.
        Args:
            error (ndarray): The error to backpropagate.
        Returns:
            error (ndarray): The final error of the input
        """

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):
            # get activation for previous layer
            activations = self.activations[i + 1]

            # apply sigmoid derivative function
            delta = error * self._sigmoid_derivative(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0], -1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropagate the next error
            error = np.dot(delta, self.weights[i].T)

        # update weights and biases using momentum
        for i in range(len(self.velocities)):
            self.velocities[i] = (self.momentum * self.velocities[i]) + (1 - self.momentum) * self.derivatives[i]
            self.weights[i] += self.velocities[i]
            if self.use_bias:
                self.biases[i] += np.mean(self.derivatives[i], axis=0)

    def train(self, inputs, targets, epochs, learning_rate):
        """Trains model by running forward prop and backprop
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network!
                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                # now perform gradient descent on the derivatives
                # (this will update the weights and biases)
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(inputs), i + 1))

        print("Training complete!")
        print("=====")

    def gradient_descent(self, learning_rate):
        """Learns by descending the gradient
        Args:
            learning_rate (float): How fast to learn.
        """
        # update the weights and biases by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """

        y = 1.0 / (1 + np.exp(-x))
        return y

    def _sigmoid_derivative(self, x):
        """Sigmoid derivative function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """
        return x * (1.0 - x)

    def _mse(self, target, output):
        """Mean Squared Error loss function
        Args:
            target (ndarray): The ground truth
            output (ndarray): The predicted values
        Returns:
            (float): Output
        """
        return np.average((target - output) ** 2)


if __name__ == "__main__":

    # create a dataset to train a network for the sum operation
    x = np.array([[random() / 2 for _ in range(3)] for _ in range(1000)])
    y = np.array([[i[0] + i[1] + i[2]] for i in x])

    data = []
    with open('Data/iris.csv', 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                values = line.split(',')
                features = [float(value) for value in values[:4]]
                if values[4] == 'Iris-setosa':
                    target = [1, 0, 0]
                if values[4] == 'Iris-versicolor':
                    target = [0, 1, 0]
                if values[4] == 'Iris-virginica':
                    target = [0, 0, 1]
                data.append(features + target)

    data = np.array(data)

    training_data = data[:130]
    test_data = data[130:]


    items = training_data[:, :4]
    targets = training_data[:, 4:7]

    items_test = test_data[:, :4]
    targets_test = test_data[:, 4:7]

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(4, [5], 3, False, 0)

    # train network
    mlp.train(items, targets, 120, 0.3)

    # create dummy data


    for j, input in enumerate(items_test):
        target = targets_test[j]
        output = mlp.forward_propagate(input)

            # activate the network!
        print(f"{output} wynik sieci\n {target} wynik wzorcowy")



    # print()
    # print("Our network believes that {}  {}  {}".format(output[0], output[1], output[2]))
