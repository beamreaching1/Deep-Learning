import torch
import torchvision
import torchvision.transforms as transforms
import random

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

trainset.targets = [random.randint(0, 9) for _ in range(len(trainset))]

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)

testset.targets = [random.randint(0, 9) for _ in range(len(testset))]

testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(49, 10)

    def forward(self, x):
        x = F.max_pool2d(x, 4)
        x = x.view(-1, 49)
        x = self.fc1(x)
        return F.log_softmax(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

epochTrainLoss = []
epochTestLoss = []

for epoch in range(500):  # loop over the dataset multiple times
   
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        if i > 10:
            break
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
    print('[%d, %5d] Train Loss: %.8f' %
          (epoch + 1, i + 1, running_loss / len(trainloader)))
    epochTrainLoss.append(running_loss / len(trainloader))
    
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels.float())
            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epochTestLoss.append(running_loss / len(trainloader))
    print('[%d, %5d] Test Loss: %.8f' %
          (epoch + 1, i + 1, running_loss / len(trainloader)))
print('Finished Training')
