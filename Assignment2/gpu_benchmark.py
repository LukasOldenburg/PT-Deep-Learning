import torch
from training_validtion import train_step, val_step, final_eval
from torchvision import transforms
import torchvision
from torchvision import models
import time
import json

train_time_dict = {'MNIST_resnet18': [], 'MNIST_resnet18_CPU': [], 'Cifar_resnet101': []}
avg_steps = 5

# ---------------- define training parameters ----------------
batch_size = 64
train_val_split = True
train_epochs = 10
learning_rate = 1e-5
schedule = False
# setting device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- create datasets and dataloader ----------------
test_data = torchvision.datasets.MNIST(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transforms.ToTensor())
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

# ---------------- perform network training with ResNet18 ----------------
for i in range(0, avg_steps):
    print('----------------- Starting MNIST ResNet18 Training -----------------')
    start = time.time()
    resnet18 = models.resnet18(pretrained=True)
    train_loss = ['train loss']
    val_loss = ['validation loss']
    for ep in range(0, train_epochs):
        loss_train = train_step(ep=ep, train_loader=train_loader, model=resnet18, learning_rate=learning_rate,
                                device=device, scheduler=schedule, rgb_channel=True)
        loss_validation = val_step(val_loader=val_loader, model=resnet18, device=device, rgb_channel=True)
        train_loss.append(loss_train)
        val_loss.append(loss_validation)

    train_time_dict['MNIST_resnet18'].append(time.time() - start)


# ---------------- define training parameters ----------------
batch_size = 64
train_val_split = True
train_epochs = 10
learning_rate = 1e-5
schedule = False
# setting device accordingly
device = torch.device('cpu')

# ---------------- create datasets and dataloader ----------------
test_data = torchvision.datasets.MNIST(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transforms.ToTensor())
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

# ---------------- perform network training with ResNet18 ----------------
for i in range(0, avg_steps):
    print('----------------- Starting MNIST ResNet18 Training -----------------')
    start = time.time()
    resnet18 = models.resnet18(pretrained=True)
    train_loss = ['train loss']
    val_loss = ['validation loss']
    for ep in range(0, train_epochs):
        loss_train = train_step(ep=ep, train_loader=train_loader, model=resnet18, learning_rate=learning_rate,
                                device=device, scheduler=schedule, rgb_channel=True)
        loss_validation = val_step(val_loader=val_loader, model=resnet18, device=device, rgb_channel=True)
        train_loss.append(loss_train)
        val_loss.append(loss_validation)

    train_time_dict['MNIST_resnet18_CPU'].append(time.time() - start)

# ---------------- define training parameters ----------------
batch_size = 64
train_val_split = True
train_epochs = 10
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

print('----------------- Starting CIFAR10 resnet101 Training -----------------')

resnet101 = models.resnet101(pretrained=True)
train_loss = ['train loss']
val_loss = ['validation loss']
for i in range(0, avg_steps):
    start = time.time()
    for ep in range(0, train_epochs):
        loss_train = train_step(ep=ep, train_loader=train_loader, model=resnet101, learning_rate=learning_rate,
                                device=device, scheduler=schedule)
        loss_validation = val_step(val_loader=val_loader, model=resnet101, device=device)
        train_loss.append(loss_train)
        val_loss.append(loss_validation)
    train_time_dict['Cifar_resnet101'].append(time.time() - start)


with open('benchmark.txt', 'w') as file:
    file.write(json.dumps(train_time_dict))  # use `json.loads` to do the reverse
