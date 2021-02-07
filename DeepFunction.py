# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:21:46 2021

@author: Cayden
"""

import torch
import math
import matplotlib.pyplot as plt

x = []

a = float(0.00)

for i in range(2001):
    x.append(a)
    a += float(0.001)
    a = round(a, 3)

fn = []

for i in x:
    if i != 0.0:
        fn.append(math.sin(5*math.pi*i)/(5*math.pi*i))
    else:
        fn.append(1.0)

plt.plot(fn)
plt.show()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

dataset = merged_list = tuple(zip(x, fn))

train, test = torch.utils.data.random_split(dataset, [601, 1400],generator=torch.Generator().manual_seed(42))

trainloader = torch.utils.data.DataLoader(train, batch_size=30, shuffle=True)
testloader = torch.utils.data.DataLoader(test, batch_size=1024, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 190)
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(190, 1)

    def forward(self, x):
        x = self.drop(F.elu(self.fc1(x)))
        x = self.drop(F.logsigmoid(x))
        x = self.fc2(F.gelu(x))
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(net.count_parameters())

for epoch in range(6):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0