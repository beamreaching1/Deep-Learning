# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:21:46 2021

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


train, test = torch.utils.data.random_split(dataset, [2400, 5601],generator=torch.Generator().manual_seed(42))

trainloader = torch.utils.data.DataLoader(train, batch_size=30, shuffle=True)
testloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(30, 15)
        self.fc2 = nn.Linear(15, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
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

epochLoss = []
grad_norm = []
lastLoss = 0.0

for epoch in range(100):  # loop over the dataset multiple times

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
        if epoch == 0:
            grad_all = 0.0
        
            for p in net.parameters():
                grad = 0.0
                if p.grad is not None:
                    grad = (p.grad.cpu().data.numpy() ** 2).sum()
                grad_all += grad
            
            grad_norm.append([grad_all ** 0.5, loss.item()])
        
        
        
        
    print('[%d, %5d] loss: %.8f' %
          (epoch + 1, i, running_loss / len(trainloader) ))
    epochLoss.append(running_loss / len(trainloader))
    
    for p in net.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad
        
    print("Gradient Normal: ",grad_all ** 0.5)

optimizer = optim.Adam(net.parameters(), lr=0.0001)
print(net.count_parameters())

epochLoss = []
grad_norm = []


for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, target]
        inputs, target = data[0].to(device), data[1].to(device)

        # forward + backward + optimize
        outputs = net(inputs.float())
        grad_all = 0.0
    
        for p in net.parameters():
            grad = 0.0
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy() ** 2).sum()
            grad_all += grad
        optimizer.zero_grad()
        grad_all = grad_all ** 0.50

        output = torch.tensor([grad_all], requires_grad=True).to(device)
        loss = (criterion(outputs, target.float()))
        output.backward()
        loss.backward()
        
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        lastLoss = loss.item()
        
        
    print('[%d, %5d] loss: %.8f' %
          (epoch + 1, i, running_loss / len(trainloader) ))
    epochLoss.append(running_loss / len(trainloader))
   

    for p in net.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad
        
    print("Gradient Normal: ",grad_all ** 0.5)
    grad_norm.append(grad_all ** 0.5)
    if grad_all ** 0.5 < 0.001:
        break

"""
def gradNormal(net):
    grad_all = 0.0
    for p in net.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad
    return torch.Tensor([grad_all ** 0.5])
"""