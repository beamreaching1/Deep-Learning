# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:40:20 2021

@author: Cayden
"""

import torch
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np


class FunctionData(Dataset):
    def __init__(self):
        self.X = []

        a = float(0.00)

        for i in range(8001):
            self.X.append(a)
            a += float(0.0001)
            a = round(a, 4)

        self.fn = []

        for i in self.X:
            if i != 0.0:
                self.fn.append(math.sin(5*math.pi*i)/(5*math.pi*i))
            else:
                self.fn.append(1.0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.fn[idx]

dataset = FunctionData()

plt.plot(dataset.X, dataset.fn)
plt.show()


train, test = torch.utils.data.random_split(dataset, [2401, 5600],generator=torch.Generator().manual_seed(42))

trainloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
testloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 467)
        self.fc3 = nn.Linear(467, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc3(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
print(net.count_parameters())

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, target]
        inputs, target = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, target.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

net.cpu()
Y = []
X = []

for i, data in enumerate(testloader, 0):
    inputs, target = data[0], data[1]
    outputs = net(inputs.float())
    X.append(inputs.item())
    Y.append(outputs.item())
    
order = np.argsort(X)
xs = np.array(X)[order]
ys = np.array(Y)[order]

plt.plot(xs, ys, "-r")
plt.plot(dataset.X, dataset.fn, "-b")
plt.show()