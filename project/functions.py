import numpy as np


def sigmoid(x):
    """
    Calculates the sigmoid function for a given input.
    It maps any real-valued number to a value between 0 and 1.

    :param x: Input value.
    :type x: float or array like
    :return: Output of the sigmoid function.
    :rtype: float or array like
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Compute the derivative of the sigmoid function for a given input.

    :param x: (float or array-like): Input value(s) for which to compute the derivative.
    :type x: float / array-like
    :return: derivative of given value(s)
    :rtype: float / array-like
    """
    return x * (1.0 - x)


def mean_squared_error(target, output):
    """
    Compute the mean squared error (MSE) between target and output values.

    :param target: The target values.
    :type target: array-like
    :param output: The output values.
    :type output: array-like
    :return: The mean squared error between the target and output values.
    :rtype: float
    """
    return np.average((output - target) ** 2)
