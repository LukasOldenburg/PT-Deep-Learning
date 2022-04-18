import numpy as np
import matplotlib.pyplot as plt


def plot_loss(train_loss, val_loss, visualize=True):

    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()

    if visualize:
        plt.show()
