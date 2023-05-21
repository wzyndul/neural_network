import numpy as np


def sigmoid(x):
    """
    Calculates the sigmoid function for a given input.
    It maps any real-valued number to a value between 0 and 1.

    :param x: Input value.
    :type x: float
    :return: Output of the sigmoid function.
    :rtype: float
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Sigmoid derivative function
    Args:
        x (float): Value to be processed
    Returns:
        y (float): Output
    """
    return x * (1.0 - x)


def mean_squared_error(target, output):
    """Mean Squared Error loss function
    Args:
        target (ndarray): The ground truth
        output (ndarray): The predicted values
    Returns:
        (float): Output
    """
    return np.average((output - target) ** 2)
