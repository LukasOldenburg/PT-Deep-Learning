from torch import nn
import torch
# test2

class AlexNet(nn.Module):

    def __init__(self, dataset='mnist'):
        super(AlexNet, self).__init__()

        self.dataset = dataset

        # dataset specific linear layers
        self.conv_layer1 = nn.Conv2d(1, 96, kernel_size=(5,5), stride=(1,1), padding=2) if self.dataset == 'mnist' else nn.Conv2d(3, 96, kernel_size=(5,5), stride=(1,1), padding=2)
        self.lin_layer1 = nn.Linear(1024, 2304) if self.dataset == 'mnist' else nn.Linear(2304, 4096)
        self.lin_layer2 = nn.Linear(2304, 10) if self.dataset == 'mnist' else nn.Linear(4096, 10)

        self.convnet = nn.Sequential(
            self.conv_layer1,
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
            self.lin_layer1,
            nn.ReLU(),
            nn.Dropout(p=0.5),
            self.lin_layer2,
            nn.Dropout(p=0.5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.convnet(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out