from torch import nn
import torch

class AlexNet(nn.Module):

    def __init__(self, global_params=None):
        super(AlexNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(5,5), stride=(1,1), padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=(5,5), stride=(1,1), padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 2304),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2304, 10),
            nn.Dropout(p=0.5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.convnet(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out