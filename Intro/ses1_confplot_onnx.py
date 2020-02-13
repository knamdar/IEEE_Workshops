# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 01:24:08 2019

@author: Ernest (Khashayar) Namdar
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import Plot_ConfMat
import matplotlib.pyplot as plt
import torch.onnx


def dset():
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())
    return train_dataset, test_dataset


class CNN(nn.Module):
    """CREATE MODEL CLASS"""
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1,
                              out_channels=16,
                              kernel_size=3,
                              stride=1,
                              padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16,
                              out_channels=32,
                              kernel_size=3,
                              stride=1,
                              padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        # FC
        out = self.fc1(out)

        return out


def train(train_loader, test_loader, model, optimizer, criterion):
    iteration = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iteration += 1

        # Calculate Accuracy
        correct = 0
        total = 0
        pred = []
        lbl = []
        # Iterate through test dataset
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = Variable(images.cuda())
            else:
                images = Variable(images)
            #print(images.size())

            # Forward pass only to get logits/output
            outputs = model(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)

            # Total correct predictions
#            if torch.cuda.is_available():
#                correct += (predicted.cpu() == labels.cpu()).sum()
#                pred.append(predicted.cpu())
#                lbl.append(labels.cpu())
#            else:
#                correct += (predicted == labels).sum()
#                pred.append(predicted)
#                lbl.append(labels)
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
                pred.extend(predicted.cpu().tolist())
                lbl.extend(labels.cpu().tolist())
            else:
                correct += (predicted == labels).sum()
                pred.extend(predicted.tolist())
                lbl.extend(labels.tolist())
        accuracy = 100 * correct / total
        # Plot_ConfMat.plot_confusion_matrix(lbl, pred, [i for i in range(10)])
        # https://matplotlib.org/examples/color/colormaps_reference.html
        Plot_ConfMat.plot_confusion_matrix(lbl, pred, [i for i in range(10)],
                                           normalize=True,
                                           title="We Love IEEE",
                                           cmap=plt.cm.YlGn)

        # Print Loss
        print(f"Epoch: {epoch+1}. "
              f"Loss: {loss.item()}. "
              f"Accuracy: {accuracy}")
    print("total iters:", iteration)
    # Create the right input shape (e.g. for an image)
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    torch.onnx.export(model, dummy_input, "onnx_PyT_MNIST.onnx")
    return lbl, pred


print("outside if __name__")
if __name__ == "__main__":
    print("inside if __name__")
    print(__name__)
    train_dataset, test_dataset = dset()  # LOADING DATASET

    batch_size = 100
    num_epochs = 2

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    model = CNN()
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    lbl, pred = train(train_loader, test_loader, model, optimizer, criterion)
