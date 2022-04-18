import torch.nn as nn


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1), padding=0)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1), padding=0)
        self.pool = nn.MaxPool2d((2,2), stride=2, padding=0)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()

        self.convnet = nn.Sequential(self.conv1, self.pool, self.conv2, self.pool)

    def forward(self, x):
        out_convnet = self.convnet(x)
        out = self.flatten(out_convnet)
        out = self.relu(self.fc1(out))
        return self.softmax(self.fc2(out))




