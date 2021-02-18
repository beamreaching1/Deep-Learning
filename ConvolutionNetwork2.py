import torch
import torchvision
import torchvision.transforms as transforms
#import random

#torch.manual_seed(random.randint(0, 1000))
#torch.cuda.manual_seed_all(random.randint(0, 1000))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.fc1 = nn.Linear(1690, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 1690)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epochLoss = []
epochAccuracy = []
l1w = []
lw = []

for epoch in range(40):  # loop over the dataset multiple times
    
    """if epoch % 3 == 0:
        p = []
        for params in net.parameters():
            p.append(params)
        
        layer1Weights = torch.reshape(p[0], (2,int(p[0].numel()/2)))
        (U, S, V)=torch.pca_lowrank(layer1Weights, q=1, center=True)
        l1w.append(([U[0].item(),U[1].item(), S.item()]))
        
        for x in range(len(p)):
            p[x] = torch.reshape(p[x], (2,int(p[x].numel()/2)))
        for x in range(len(p)-1):
            a = torch.cat((p[0][0],p[x+1][0]))
            b = torch.cat((p[0][1],p[x+1][1]))
            p[0] = torch.stack((a,b))
        
        (U, S, V)=torch.pca_lowrank(p[0], q=1, center=True)
        lw.append(([U[0].item(),U[1].item(), S.item()]))"""
        
    """if epoch % 3 == 0:
        p = []
        for params in net.parameters():
            p.append(params)
        
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
        
        grad_all = 0.0
        
        for p in net.parameters():
            grad = 0.0
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy() ** 2).sum()
            grad_all += grad
        
    print('[%d, %5d] loss: %.8f' %
          (epoch + 1, i + 1, running_loss / len(trainloader)))
    epochLoss.append(running_loss / len(trainloader))
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epochAccuracy.append((100 * correct / total))

print('Finished Training')

"""
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))"""