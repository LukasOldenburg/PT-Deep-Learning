import torch, torchvision
from models.lenet import LeNet
from train_test import train_step, val_step, final_eval
from torchvision import models
from utils import plot_loss

# ---------------- define training parameters ----------------
batch_size = 64
train_val_split = True
train_epochs = 2
learning_rate = 1e-5
schedule = False

# setting device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- create datasets and dataloader ----------------
test_data = torchvision.datasets.MNIST(root='./data',
                                       train=False,
                                       download=True,
                                       transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=True)

train_data = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        download=True,
                                        transform=torchvision.transforms.ToTensor())

# split test set into validation and test set if its desired
if train_val_split:
    train_size = int(len(train_data) * 0.9)
    val_size = int(len(train_data) * 0.1)
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=True)
else:
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=batch_size,
                                             shuffle=True)

# ---------------- perform network training with LeNet ----------------
lenet = LeNet()
train_loss = ['train loss']
val_loss = ['validation loss']
for ep in range(0, train_epochs):
    loss_train = train_step(ep=ep, train_loader=train_loader, model=lenet, learning_rate=learning_rate,
                            device=device, scheduler=schedule)
    loss_validation = val_step(val_loader=val_loader, model=lenet, device=device)
    train_loss.append(loss_train)
    val_loss.append(loss_validation)
final_eval(train=True, dataloader=test_loader, model=lenet, device=device)
final_eval(train=False, dataloader=train_loader, model=lenet, device=device)
plot_loss(train_loss[1:], val_loss[1:], visualize=True)

resnet18 = models.resnet18(pretrained=True)

