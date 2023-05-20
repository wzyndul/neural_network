import numpy as np


def categorical_cross_entropy(y_true, y_pred):
    """
    Computes the categorical cross-entropy loss between the true labels and predicted probabilities.

    :param y_true: True labels (encoded as integers)
    :type y_true: numpy.ndarray
    :param y_pred: Predicted probabilities
    :type y_pred: numpy.ndarray
    :return: Categorical cross-entropy loss
    :rtype: float
    """
    epsilon = 1e-7  # small constant to avoid numerical instability
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = 0.0
    for y_t, y_p in zip(y_true, y_pred):
        loss += -np.log(y_p) * y_t

    return loss


