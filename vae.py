import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])
image_size = 64
batch_size = 64

trainset = MNIST('./data', download=True, train=True,
                 transform=transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(0.5,0.5)]))
validset = MNIST('./data', download=True, train=False,
                 transform=transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(0.5,0.5)]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size = 6, stride = 2, padding = 2) #32
    self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2) #15
    self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 2) #7
    self.conv4 = nn.Conv2d(128, 256, kernel_size = 4) #4
    self.conv5 = nn.Conv2d(256, 256, kernel_size = 4) #1
    self.bn1 = nn.BatchNorm2d(32, affine = True)
    self.bn2 = nn.BatchNorm2d(64, affine = True)
    self.bn3 = nn.BatchNorm2d(128, affine = True)
    self.bn4 = nn.BatchNorm2d(256, affine = True)
    self.average = nn.Linear()
    self.distribution =  nn.Linear()
    self.parameterization = 

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.conv4(x)
    x = self.bn4(x)
    x = self.conv5(x)
    x = torch.flatten(x, start_dim = 1)
    aver = self.average(x)
    dist = self.distribution(x)
    std = torch.exp(0.5 * aver)
    z = torch.randn_like(std)
    return z * std + dist

class Decoder(nn.Module):
  def __init__(self):



device = 'cuda' if torch.cuda.is_available() else 'cpu'
Encoder = Encoder().to(device)
EncoderOptim = optim.Adam(Encoder.parameters(), lr = 0.00015, betas=(0.5, 0.999))
criterion = nn.BCELoss.to(device)

for epoch in range(30):
  for _, data in enumerate(trainloader, 0):
    images, labels = data
    encoder.zero_grads()