import matplotlib.pyplot as plt
import torch
from models.alexnet import AlexNet
from training_validtion import train_step, val_step, final_eval
from utils import plot_loss
import sys
from torchvision import transforms
import torchvision
from torchvision import models

# sys.stdout = open("out.txt", "w")

# ---------------- define training parameters ----------------
batch_size = 64
train_val_split = True
train_epochs = 50
learning_rate = 1e-6
schedule = False
# setting device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- create datasets and dataloader ----------------
test_data = torchvision.datasets.CIFAR10(root='./data',
                                         train=False,
                                         download=True,
                                         transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=True)

train_data = torchvision.datasets.CIFAR10(root='./data',
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

print('----------------- Starting CIFAR10 AlexNet Training -----------------')

batch_size = 64
train_epochs = 1
learning_rate = 1e-2

alexnet = AlexNet(dataset='cifar')
train_loss = ['train loss']
val_loss = ['validation loss']
for ep in range(0, train_epochs):
    loss_train = train_step(ep=ep, train_loader=train_loader, model=alexnet, learning_rate=learning_rate,
                            device=device, scheduler=schedule)
    loss_validation = val_step(val_loader=val_loader, model=alexnet, device=device)
    train_loss.append(loss_train)
    val_loss.append(loss_validation)
final_eval(train=True, dataloader=test_loader, model=alexnet, device=device)
final_eval(train=False, dataloader=train_loader, model=alexnet, device=device)
fig = plot_loss(train_loss[1:], val_loss[1:], visualize=False)

plt.savefig("AlexNet_MNIST.pdf")
torch.save(alexnet.state_dict(), "AlexNet_MNIST")

print('----------------- Starting CIFAR10 resnet18 Training -----------------')

batch_size = 64
train_epochs = 1
learning_rate = 1e-6

resnet18 = models.resnet18(pretrained=True)
train_loss = ['train loss']
val_loss = ['validation loss']
for ep in range(0, train_epochs):
    loss_train = train_step(ep=ep, train_loader=train_loader, model=resnet18, learning_rate=learning_rate,
                            device=device, scheduler=schedule)
    loss_validation = val_step(val_loader=val_loader, model=resnet18, device=device)
    train_loss.append(loss_train)
    val_loss.append(loss_validation)
final_eval(train=True, dataloader=test_loader, model=resnet18, device=device)
final_eval(train=False, dataloader=train_loader, model=resnet18, device=device)
fig = plot_loss(train_loss[1:], val_loss[1:], visualize=False)

plt.savefig("resnet18_Cifar.pdf")
torch.save(resnet18.state_dict(), "resnet18_Cifar")

print('----------------- Starting CIFAR10 resnet50 Training -----------------')

batch_size = 64
train_epochs = 1
learning_rate = 1e-6

resnet50 = models.resnet50(pretrained=True)
train_loss = ['train loss']
val_loss = ['validation loss']
for ep in range(0, train_epochs):
    loss_train = train_step(ep=ep, train_loader=train_loader, model=resnet50, learning_rate=learning_rate,
                            device=device, scheduler=schedule)
    loss_validation = val_step(val_loader=val_loader, model=resnet50, device=device)
    train_loss.append(loss_train)
    val_loss.append(loss_validation)
final_eval(train=True, dataloader=test_loader, model=resnet50, device=device)
final_eval(train=False, dataloader=train_loader, model=resnet50, device=device)
fig = plot_loss(train_loss[1:], val_loss[1:], visualize=False)

plt.savefig("resnet50_Cifar.pdf")
torch.save(resnet50.state_dict(), "resnet50_Cifar")

print('----------------- Starting CIFAR10 resnet101 Training -----------------')

batch_size = 64
train_epochs = 1
learning_rate = 1e-6

resnet101 = models.resnet101(pretrained=True)
train_loss = ['train loss']
val_loss = ['validation loss']
for ep in range(0, train_epochs):
    loss_train = train_step(ep=ep, train_loader=train_loader, model=resnet101, learning_rate=learning_rate,
                            device=device, scheduler=schedule)
    loss_validation = val_step(val_loader=val_loader, model=resnet101, device=device)
    train_loss.append(loss_train)
    val_loss.append(loss_validation)
final_eval(train=True, dataloader=test_loader, model=resnet101, device=device)
final_eval(train=False, dataloader=train_loader, model=resnet101, device=device)
fig = plot_loss(train_loss[1:], val_loss[1:], visualize=False)

plt.savefig("resnet101_Cifar.pdf")
torch.save(resnet101.state_dict(), "resnet101_Cifar")

sys.stdout.close()