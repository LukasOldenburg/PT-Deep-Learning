import torch
from torch import nn
import numpy as np


def train_step(ep, train_loader, model, device, learning_rate, scheduler=False):
    model.train()
    model.to(device)
    train_loss = 0

    # set optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # set model properties to training mode on respective device and reset gradients
    optimizer.zero_grad()
    console_loss = 0

    for batch_idx, data in enumerate(train_loader):
        x, y = data
        x.to(device)
        y.to(device)

        # define loss function and feed itr with predicted and ground truth values
        ce = nn.CrossEntropyLoss()
        y_pred = model(x)
        loss = ce(y_pred, y)

        # perform backward propagation and update step
        loss.backward()
        optimizer.step()

        # print loss and accuracy measurements
        log_interval = int(len(train_loader) / 3)
        console_loss += loss
        train_loss += loss.detach().numpy()
        if batch_idx > 0 and batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                ep, batch_idx * train_loader.batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), console_loss / log_interval))
            console_loss = 0

        # schedule learning rate if desired
        if scheduler:
            scheduler.step()

    return train_loss / len(train_loader)


def val_step(val_loader, model, device):
    model.eval()
    model.to(device)
    corr = 0
    val_loss = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            x, y = data
            x.to(device)
            y.to(device)
            out = model(x)

            ce = nn.CrossEntropyLoss()
            loss = ce(out, y)
            val_loss += loss.detach().numpy()

            # transforming probability into class prediction
            _, y_pred = torch.max(out, dim=1)
            corr += (y_pred == y).sum()

    # compute and print accuracy
    acc = corr / len(val_loader.dataset) * 100
    print('Validation Accuracy: {:.2f}%\n'.format(acc))

    return val_loss / len(val_loader)


def final_eval(train, dataloader, model, device):
    model.eval()
    model.to(device)
    cl = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)  # tuple with all classes in the dataset
    corr_pred = {class_idx: 0 for class_idx in cl}
    sum_pred = {class_idx: 0 for class_idx in cl}
    corr = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            x, y = data
            x.to(device)
            y.to(device)

            out = model(x)

            # transforming probability into class prediction
            _, y_pred = torch.max(out, dim=1)
            corr += (y_pred == y).sum()

            # creating dictionary entries for every class
            for y, y_pred in zip(y, y_pred):
                if y == y_pred:
                    corr_pred[cl[y]] += 1
                sum_pred[cl[y]] += 1

    # printing average accuracy over all classes
    dataset = 'train' if train else 'test'
    print('#### Final Evaluation on {} set ####'.format(dataset))
    avg_acc = corr / len(dataloader.dataset) * 100
    print('Average Accuracy on {} : {:.2f}%'.format(dataset, avg_acc))

    # printing class dependent accuracies
    for class_idx, count in corr_pred.items():
        acc = 100 * float(count) / sum_pred[class_idx]
        print('Accuracy for class {}: {:.2f}%'.format(class_idx, acc))
