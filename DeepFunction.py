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

trainloader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 37)
        self.fc2 = nn.Linear(37, 26)
        self.fc3 = nn.Linear(26, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
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
l1w = []
lw = []
grad_norm = []

for epoch in range(2):  # loop over the dataset multiple times

    """if epoch % 3 == 0:
        p = []
        for params in net.parameters():
            p.append(params)
        
        p = p[:4]
        
        layer1Weights = torch.reshape(p[0], (2,int(p[0].numel()/2)))
        a = torch.sum(layer1Weights[0])
        b = torch.sum(layer1Weights[1])
        
        l1w.append([a,b])
        
        for x in range(len(p)):
            p[x] = torch.reshape(p[x], (2,int(p[x].numel()/2)))
        for x in range(len(p)-1):
            a = torch.cat((p[0][0],p[x+1][0]))
            b = torch.cat((p[0][1],p[x+1][1]))
            p[0] = torch.stack((a,b))
        
        a = torch.sum(p[0])
        b = torch.sum(p[1])
        lw.append(([a,b]))"""

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


plt.title("Loss Vs Epoch", loc="center")
plt.plot(epochLoss, "-r", label='DeepModel')
plt.legend(loc='upper right')
plt.show()

"""
net.cpu()
Y = []
X = []

for i, data in enumerate(testloader, 0):
    inputs, target = data[0], data[1]
    outputs = net(inputs.float())
    X.append(inputs.item())
    Y.append(outputs.item())
    
order = np.argsort(X)
xd = np.array(X)[order]
yd = np.array(Y)[order]

plt.title("Graph Comparison", loc="center")
plt.plot(xd, yd, "-r", label='DeepModel')
plt.plot(dataset.X, dataset.fn, "-b",label='GroundTruth')
plt.legend(loc='upper right')
plt.show()"""