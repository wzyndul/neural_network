import numpy as np


def sigmoid(x: float):
    """
    Calculates the sigmoid function for a given input.
    It maps any real-valued number to a value between 0 and 1.

    :param x: Input value.
    :type x: float
    :return: Output of the sigmoid function.
    :rtype: float
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Computes the softmax function for a 1D array of values.

    Softmax function calculates the probabilities for each element in the input array,
    where the probabilities sum up to 1.

    :param x: Input array of shape (n,)
    :type x: numpy.ndarray
    :return: Array of softmax probabilities
    :rtype: numpy.ndarray
    """
    x_max = np.max(x)
    x_exp = np.exp(x - x_max)
    probabilities = x_exp / np.sum(x_exp)
    return probabilities


def softmax_derivative(x):
    softmax_output = softmax(x)
    softmax_jacobian = np.diag(softmax_output) - np.outer(softmax_output, softmax_output)
    return softmax_jacobian


def sigmoid_derivative(x):
    return np.exp(-x) / ((np.exp(-x) + 1) ** 2)

# print(sigmoid_derivative(-0.1842302012614773))

