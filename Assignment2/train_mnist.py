import torch, torchvision

# ---------------- define training parameters ----------------
batch_size = 64
test_val_split = True

# setting device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- create datasets and dataloader ----------------
trainset = torchvision.datasets.MNIST(root='./data',
                                      train=True,
                                      download=True,
                                      transform=torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True)
test_data = torchvision.datasets.MNIST(root='./data',
                                       train=False,
                                       download=True,
                                       transform=torchvision.transforms.ToTensor())
# split test set into validation and test set if its desired
if test_val_split:
    val_size = int(len(test_data) * 0.6)
    test_size = int(len(test_data) * 0.4)
    valset, testset = torch.utils.data.random_split(test_data, [val_size, test_size])
else:
    testset = test_data
    valset = test_data

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=True)
valloader = torch.utils.data.DataLoader(valset,
                                        batch_size=batch_size,
                                        shuffle=True)

A = 1
