import numpy as np
from utils import sigmoid

class LogisticRegression:
    """ Logistic Regression class is able to create and train a binary classifier which fits
        a logistic function to data with the gradient descent algorithm and cross a entropy loss.

        Inputs:
        lr: float - learning rate used during optimization step
        epochs: int - number of epochs to train the model"""

    def __init__(self, lr=0.0001, epochs=100):
        self.epochs = epochs # number of training epochs
        self.lr = lr # learning rate
        self.w = None # weights

    def weight_data_init(self, x):
        """ Initializes weights and add bias term"""
        w = np.zeros(x.shape[1] + 1)
        x = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
        return w, x

    def loss(self, x, y, w):
        """Computes loss function output with cross entropy"""
        l1 = np.dot(y, np.log(sigmoid(np.dot(w, x.transpose()))))
        l2 = np.dot((1 - y), np.log(1 - np.dot(w, x.transpose())))
        loss = (-1 / x.shape[0]) * (l1 + l2)
        return loss

    def grad_desc(self, x, y):
        """Gradient descent algorithm with weight updates and loss/accuracy measurement.
            Model parameters are stored and function returns loss values and accuracy for every epoch during training."""
        self.w, _ = self.weight_data_init(x)
        w, x = self.weight_data_init(x)

        loss_list = ['loss:']
        acc_list = ['accuracy:']
        for i in range(1, self.epochs):
            w = w - self.lr * (np.dot(x.transpose(), sigmoid(np.dot(w, x.transpose())) - y))
            loss_list.append(np.round(self.loss(x, y, w), 3))  # loss for tracking purpose
            y_pred = sigmoid(np.dot(w, x.transpose()))
            y_pred_list = [1 if y_pred[j] > 0.5 else 0 for j in range(0, y_pred.shape[0])]
            acc_list.append(
                np.round((np.count_nonzero(y_pred_list == y) / x.shape[0]) * 100, 3))  # accuracy for tracking purpose
        self.w = w
        return loss_list, acc_list

    def predict(self, x_test, y_test):
        """ Function returns accuracy measurement for a trained model on the test dataset."""
        x_test = self.weight_data_init(x_test)[1] # add bias term to test data
        y_pred = sigmoid(np.dot(x_test, self.w))
        y_pred_list = [1 if y_pred[i] > 0.5 else 0 for i in range(len(y_pred))]

        # compute accuracy as recall (TruePositives / (TruePositives + FalseNegatives))
        recall_acc = np.round((np.count_nonzero(y_pred_list == y_test) / x_test.shape[0]) * 100, 3)
        return recall_acc
