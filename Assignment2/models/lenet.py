import torch.nn as nn


class LeNet(nn.Module):

    def __init__(self):

        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1), padding=0)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1), padding=0)
        self.pool = nn.MaxPool2d((2,2), stride=2, padding=0)