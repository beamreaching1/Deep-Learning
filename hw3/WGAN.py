# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:14:01 2021

@author: Cayden
"""
from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
import numpy as np

# Set random seed for reproducibility
seed = 42
print("Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 200

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 500

# Learning rate for optimizers
lr = 0.0005

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

#Specific Class 0)airplane 1)automobile 2)bird 3)cat 4)deer 5)dog 6)frog 7)horse 8)ship 9)truck
useClass = False
classToUse = 0

#Load model
loadModel = True
modelPath = "./save/WGAN39.pickle"
# We can use an image folder dataset the way we have it setup.
# Create the dataset
transform = transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

if useClass:
    idx = np.where(np.array(dataset.targets) == classToUse)
    dataset.data = dataset.data[idx]


# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)

#Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = -1

one = torch.tensor(1, dtype=torch.float).to(device)
mone = (one * -1)

# Setup Adam optimizers for both G and D
optimizerD = optim.AdamW(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.AdamW(netG.parameters(), lr=lr, betas=(beta1, 0.999))

def gP(netD, real_data, synth_data, batch_size, gp_lambda):
    alpha = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    alpha = alpha.expand(batch_size, input.size(1), input.size(2), input.size(3))
    alpha = (alpha.contiguous().view(batch_size, 3, 64, 64)).to(device)

    interpolates = (alpha * real_data + ((1 - alpha) * synth_data)).to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    critic_interpolates = netD(interpolates)

    grad_outputs = (torch.ones(critic_interpolates.size())).to(device)

    gradients = autograd.grad(outputs=critic_interpolates, inputs=interpolates,
                              grad_outputs=grad_outputs, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda

    return gradient_penalty

#Load model
if loadModel:
    print("Found Model, Loading: "+modelPath)
    savedModel = torch.load(modelPath, map_location=device)
    
    netG.load_state_dict(savedModel['netG'])
    netD.load_state_dict(savedModel['netD'])

    optimizerG.load_state_dict(savedModel['optimG'])
    optimizerD.load_state_dict(savedModel['optimD'])


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        for p in netD.parameters():
            p.requires_grad = True
        input = data[0].to(device)
        batch_size = input.size(0)

        ##############################################
        # (1) Update D network: maximize log(D(x)) - log(D(G(z)))
        ##############################################
        # Set discriminator gradients to zero.

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        netD.zero_grad()

        # Train with real
        real_output = netD(input)
        lossReal = real_output.mean()
        D_x = real_output.mean().item()
        lossReal.backward(mone)

        # Generate fake image batch with G
        fake_images = netG(noise)

        # Train with fake
        fake_output = netD(fake_images.detach())
        lossFake = fake_output.mean()
        D_G_z1 = fake_output.mean().item()
        lossFake.backward(one)


        gradient_penalty = gP(netD, input, fake_images,
                                                      batch_size, 10)

        gradient_penalty.backward()
        # Update D
        optimizerD.step()

        lossD = lossFake - lossReal + gradient_penalty

        # Train the generator every n_critic iterations.
        if (i + 1) % 5 == 0:
            for p in netD.parameters():
                p.requires_grad = False

            # Generate fake image batch with G
            fake_images = netG(noise)
            fake_output = netD(fake_images)

            # Set generator gradients to zero
            netG.zero_grad()

            lossG = fake_output.mean()
            D_G_z2 = fake_output.mean().item()
            lossG.backward(mone)
            optimizerG.step()

            if i == 774:
                # Output training stats
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader), lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
                # Plot the fake images from the last epoch
                vutils.save_image(img_list[-1],"./"+"last2"+".jpeg")

            # Save Losses for plotting later
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

# Model state storage
savePath = os.path.join("./save/", "WGAN.pickle")
modelDict = {'netG': netG.state_dict(),
                'netD': netD.state_dict(),
                'optimG': optimizerG.state_dict(),
                'optimD': optimizerD.state_dict()}

torch.save(modelDict, savePath)

netG.eval()

print("Beginning set generation.")
fakeSet = []
realSet = []
for i in range(40):
    fake = netG(torch.randn(64, nz, 1, 1, device=device)).detach().cpu()
    fake = list(torch.split(fake,1,dim=0))
    real_batch = next(iter(dataloader))
    real = real_batch[0].to(device)[:64].cpu()
    real = list(torch.split(real,1,dim=0))
    for a in fake:
        fakeSet.append(a)
    for a in real:
        realSet.append(a)

for index, i in enumerate(fakeSet):
    vutils.save_image(i,"./fake2/"+str(index)+".jpeg")

for index, i in enumerate(realSet):
    vutils.save_image(i,"./real2/"+str(index)+".jpeg")

print("Finished.")
