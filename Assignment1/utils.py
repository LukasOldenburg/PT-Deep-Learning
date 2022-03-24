import numpy as np


def load_data(noise):
    """ Function loads and returns data concering a certain noise parameter"""

    if noise == 0.2:
        x_train = np.loadtxt('./two_moon_0.2/train.txt')[:, 0:2]
        y_train = np.loadtxt('./two_moon_0.2/train.txt')[:, 2]
        x_test = np.loadtxt('./two_moon_0.2/test.txt')[:, 0:2]
        y_test = np.loadtxt('./two_moon_0.2/test.txt')[:, 2]
    elif noise == 0.3:
        x_train = np.loadtxt('./two_moon_0.3/train.txt')[:, 0:2]
        y_train = np.loadtxt('./two_moon_0.3/train.txt')[:, 2]
        x_test = np.loadtxt('./two_moon_0.3/test.txt')[:, 0:2]
        y_test = np.loadtxt('./two_moon_0.3/test.txt')[:, 2]
    elif noise == 0.9:
        x_train = np.loadtxt('./two_moon_0.9/train.txt')[:, 0:2]
        y_train = np.loadtxt('./two_moon_0.9/train.txt')[:, 2]
        x_test = np.loadtxt('./two_moon_0.9/test.txt')[:, 0:2]
        y_test = np.loadtxt('./two_moon_0.9/test.txt')[:, 2]
    else:
        raise ValueError('No valid value for noise parameter.')
    return x_train, y_train, x_test, y_test


def minmax_scale(x):
    """Function scales each column (feature) of the input data to a range [0,1]"""
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    for i in range(x.shape[1]):
        for k in range(x.shape[0]):
            x[k, i] = (x[k, i] - x_min[i]) / (x_max[i] - x_min[i])
    return x


def sigmoid(x):
    """Applies logistic function to an input vector"""
    out = 1 / (1 + np.exp(-x))
    return out


def sigmoid_back(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def relu(x):
    return np.maximum(x, np.zeros_like(x))


def relu_back(do, x):
    out = np.copy(do)
    out[x.transpose() < 0] = 0
    return out.transpose()
