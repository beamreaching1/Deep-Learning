import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.utils as vutils

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch

n_epochs = 100
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 8
latent_dim = 200
n_classes = 10
img_size = 64
channels = 3

modelPath = "./save/ACGAN64.pickle"

device = torch.device("cpu")
cuda = False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Initialize generator and discriminator
generator = Generator()

# Configure data loader
transform = transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)


FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


#Load model
print("Trying to load model: "+modelPath)
savedModel = torch.load(modelPath)

generator.load_state_dict(savedModel['generator'])

if cuda:
    generator.to(device)

print("Beginning set generation.")

fakeSet = []
realSet = []
for i in range(40):
    z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
    gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
    gen_imgs = generator(z, gen_labels)
    gen_imgs = list(torch.split(gen_imgs,1,dim=0))
    real_batch = next(iter(dataloader))
    real = real_batch[0].to(device)[:64].cpu()
    real = list(torch.split(real,1,dim=0))
    for a in gen_imgs:
        fakeSet.append(a)
    for a in real:
        realSet.append(a)

for index, i in enumerate(fakeSet):
    vutils.save_image(i,"./fakeAC/"+str(index)+".jpeg")

for index, i in enumerate(realSet):
    vutils.save_image(i,"./realAC/"+str(index)+".jpeg")

print("Finished.")