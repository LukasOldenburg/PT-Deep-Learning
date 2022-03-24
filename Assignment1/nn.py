import numpy as np
from utils import sigmoid, relu, sigmoid_back, relu_back
from utils import load_data


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
        self.w2 = np.random.rand(self.layer_units[0], self.layer_units[1])
        self.w3 = np.random.rand(self.layer_units[1], 1)

        self.b1 = np.ones((self.layer_units[0], ))
        self.b2 = np.ones((self.layer_units[1], 1))
        self.b3 = np.ones((1, 1))

    def training_step(self, x, y):
        """ """
        m = x.shape[0]

        z1 = np.dot(x, self.w1)
        o1 = relu(z1)
        z2 = np.dot(o1, self.w2) if self.hidden_layer == 2 else z1
        o2 = relu(z2)
        z3 = np.dot(o2, self.w3)
        y_pred = sigmoid(z3).reshape(-1)

        loss = np.sum((y_pred - y) ** 2) / m

        # compute derivates
        dloss = (2 * (y_pred - y)).reshape(-1,1)
        dz3 = dloss * sigmoid_back(z3)
        dw3 = np.dot(dz3.transpose(), o2) / m
        db3 = dz3.sum(axis=0, keepdims=True) / m
        do2 = np.dot(self.w3, dz3.transpose())

        # todo place if statement in case of one hidden layer
        dz2 = relu_back(do2, z2)
        dw2 = np.dot(dz2.transpose(), o1)
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        do1 = np.dot(self.w2, dz2.transpose())

        dz1 = relu_back(do1, z1)
        dw1 = np.dot(dz1.transpose(), x) / m
        db1 = dz1.sum(axis=0, keepdims=True) / m

        # optimization step
        self.w1 = self.w1 - self.lr * dw1.transpose()
        self.w2 = self.w2 - self.lr * dw2.transpose()
        self.w3 = self.w3 - self.lr * dw3.transpose()
        self.b1 = self.b1 - self.lr * db1.transpose()
        self.b1 = self.b2 - self.lr * db2.transpose()
        self.b1 = self.b3 - self.lr * db3.transpose()

        return loss


x_train, y_train, x_test, y_test = load_data(0.2)
NN = NeuralNetwork(hidden_layer=2, layer_units=[5, 5], lr=0.005, epochs=100)
NN.init_network(x_train)
for i in range(NN.epochs):
    loss = NN.training_step(x_train, y_train)
    print(loss)

