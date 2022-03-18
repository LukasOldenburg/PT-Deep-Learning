import numpy as np


class LogisticRegression:
    """ Performs logistic regression """

    def __init__(self, lr=0.0001, epochs=100):
        self.epochs = epochs # number of training epochs
        self.lr = lr # learning rate
        self.w = None # weights

    def weight_init(self, x):
        """ Initializes weights and add bias term"""
        w = np.zeros(x.shape[1] + 1)
        x = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
        return w, x

    def logistic_func(self, x):
        """Applies logistic function to an input vector"""
        logist = 1 / (1 + np.exp(-x))
        return logist

    def loss(self, x, y, w):
        """Computes loss function output"""
        l1 = np.dot(y, np.log(self.logistic_func(np.dot(w, x.transpose()))))
        l2 = np.dot((1 - y), np.log(1 - np.dot(w, x.transpose())))
        loss = (-1 / x.shape[0]) * (l1 + l2)
        return loss

    def grad_desc(self, x, y):
        """Gradient descent algorithm with weight updates and loss/accuracy measurement"""
        self.w, _ = self.weight_init(x)
        w, x = self.weight_init(x)
        loss_list = ['loss:']
        acc_list = ['accuracy:']
        for i in range(1, self.epochs):
            w = w - self.lr * (np.dot(x.transpose(), self.logistic_func(np.dot(w, x.transpose())) - y))
            loss_list.append(np.round(self.loss(x, y, w), 3))  # loss for tracking purpose
            y_pred = self.logistic_func(np.dot(w, x.transpose()))
            y_pred_list = [1 if y_pred[j] > 0.5 else 0 for j in range(0, y_pred.shape[0])]
            acc_list.append(
                np.round((np.count_nonzero(y_pred_list == y) / x.shape[0]) * 100, 3))  # accuracy for tracking purpose
        self.w = w
        return loss_list, acc_list

    def predict(self, x_test):
        y_pred_list = []
        # y_pred = self.logistic_func(np.dot(self.weight_init(x_test), self.w))
        # y_pred_list = [1 if y_pred[i] > 0.5 else 0 for i in y_pred]
        # return y_pred_list
