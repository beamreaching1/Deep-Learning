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


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epochLoss = []
epochAccuracy = []

for epoch in range(40):  # loop over the dataset multiple times
   
    running_loss = 0.0
    for i in range(10000):
        

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        
        grad_all = 0.0
    
        for p in net.parameters():
            grad = 0.0
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy() ** 2).sum()
            grad_all += grad
        
        loss = criterion(torch.tensor([grad_all ** 0.5], requires_grad=True), torch.Tensor([0.00001]))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
            
    print('[%d, %5d] loss: %.8f' %
          (epoch + 1, i + 1, running_loss / len(trainloader)))
    epochLoss.append(running_loss / len(trainloader))
    grad_all = 0.0
    
    for p in net.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad
        
    print("Gradient Normal: ",grad_all ** 0.5)