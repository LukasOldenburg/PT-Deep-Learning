import numpy as np
from utils import sigmoid, relu, sigmoid_back, relu_back


class NeuralNetwork:
    """ Neural Network class is able to create and train a one- or two hidden layer feed-forward
        network with adaptable units per layer.

        Inputs:
        hidden_layer: int 1 or 2 - choose number of hidden layers in the network
        layer_units: list [int, int] - choose number of units for each hidden layer
        lr: float - define learning rate for optimization step
        epochs: int - number of epochs to train the network
        """

    def __init__(self, hidden_layer, layer_units, lr=0.0001, epochs=100):
        self.hidden_layer = hidden_layer
        self.layer_units = layer_units
        self.lr = lr
        self.epochs = epochs
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.b1 = None
        self.b2 = None
        self.b3 = None

    def init_network(self, x):
        """ Initializes network with random weights between [0,1] and a bias of 1 for every neuron"""
        self.w1 = np.random.rand(x.shape[1], self.layer_units[0])
        self.w2 = np.random.rand(self.layer_units[0], self.layer_units[1]) if self.hidden_layer == 2 else None
        self.w3 = np.random.rand(self.layer_units[1], 1) if self.hidden_layer == 2 else np.random.rand(self.layer_units[0], 1)

        self.b1 = np.zeros((self.layer_units[0]))
        self.b2 = np.zeros((self.layer_units[1])) if self.hidden_layer == 2 else None
        self.b3 = np.zeros(1)

    def training_step(self, x, y):
        """ Performs forward and backward propagation through the network and updates weights.
            Loss and accuracy are computed for every training step. """
        m = x.shape[0]

        # network forward pass
        z1 = np.dot(x, self.w1) + self.b1
        o1 = relu(z1)
        z2 = np.dot(o1, self.w2) + self.b2 if self.hidden_layer == 2 else z1
        o2 = relu(z2)
        z3 = np.dot(o2, self.w3) + self.b3
        y_pred = sigmoid(z3).reshape(-1)

        # loss and accuracy measurement
        loss = np.sum((y_pred - y) ** 2) / m
        y_pred_list = [1 if y_pred[j] > 0.5 else 0 for j in range(0, y_pred.shape[0])]
        acc = np.round((np.count_nonzero(y_pred_list == y) / y.shape[0]) * 100, 3)

        # compute derivates (backward pass)
        dloss = (2 * (y_pred - y)).reshape(-1,1)
        dz3 = dloss * sigmoid_back(z3)
        dw3 = np.dot(dz3.transpose(), o2) / m
        db3 = dz3.sum(axis=0, keepdims=True) / m
        do2 = np.dot(self.w3, dz3.transpose())

        # decides wether one or two hidden layer are used
        if self.hidden_layer == 2:
            dz2 = relu_back(do2, z2)
            dw2 = np.dot(dz2.transpose(), o1)
            db2 = np.sum(dz2, axis=0, keepdims=True) / m
            do1 = np.dot(self.w2, dz2.transpose())
        else:
            dz2 = dz3
            dw2 = dw3
            db2 = db3
            do1 = do2

        dz1 = relu_back(do1, z1)
        dw1 = np.dot(dz1.transpose(), x) / m
        db1 = dz1.sum(axis=0, keepdims=True) / m

        # optimization step
        self.w1 = self.w1 - self.lr * dw1.transpose()
        self.w2 = self.w2 - self.lr * dw2.transpose() if self.hidden_layer == 2 else None
        self.w3 = self.w3 - self.lr * dw3.transpose()
        self.b1 = self.b1 - self.lr * db1.reshape(-1)
        self.b2 = self.b2 - self.lr * db2.reshape(-1) if self.hidden_layer == 2 else None
        self.b3 = self.b3 - self.lr * db3.reshape(-1)

        return loss, acc

    def predict(self, x_test, y_test):
        """Predicts output of a previous trained network by evaluating it on the test set."""

        z1 = np.dot(x_test, self.w1) + self.b1
        o1 = relu(z1)
        z2 = np.dot(o1, self.w2) + self.b2 if self.hidden_layer == 2 else z1
        o2 = relu(z2)
        z3 = np.dot(o2, self.w3) + self.b3

        y_pred = sigmoid(z3).reshape(-1)
        y_pred_list = [1 if y_pred[j] > 0.5 else 0 for j in range(0, y_pred.shape[0])]
        acc = np.round((np.count_nonzero(y_pred_list == y_test) / y_test.shape[0]) * 100, 3)

        return acc






