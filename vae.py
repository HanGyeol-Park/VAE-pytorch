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
    self.average = nn.Linear(256,1)
    self.distribution =  nn.Linear(256,1)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = F.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = F.relu(x)
    x = self.conv4(x)
    x = self.bn4(x)
    x = F.relu(x)
    x = self.conv5(x)
    x = torch.flatten(x, start_dim = 1)
    aver = self.average(x)
    dist = self.distribution(x)
    std = torch.exp(0.5 * aver)
    z = torch.randn_like(std)
    return z * std + dist

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.conv1 = nn.ConvTranspose2d(1, 512, kernel_size = 6, stride = 1, padding = 0) #6
    self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 2) #10
    self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 2) #18
    self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 2) #34
    self.conv5 = nn.ConvTranspose2d(64, 1, kernel_size = 4, stride = 2, padding = 3) # 64
    self.bn1 = nn.BatchNorm2d(512, affine = True)
    self.bn2 = nn.BatchNorm2d(256, affine = True)
    self.bn3 = nn.BatchNorm2d(128, affine = True)
    self.bn4 = nn.BatchNorm2d(64, affine = True)
    self.tanh = nn.Tanh()

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = F.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = F.relu(x)
    x = self.conv4(x)
    x = F.relu(x)
    x = self.conv5(x)
    x = self.tanh(x)
    return x


class VAE(nn.Module):
  def __init__(self):
    super(VAE, self).__init__()
    self.encode = Encoder()
    self.decode = Decoder()

  def forward(self, x, text = 'train'):
    if text == 'train':
      x = self.encode(x)
      x = x.reshape(64,1,1,1)
      x = self.decode(x)
      return x
    else:
      x = self.decode(x)
      return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
VAENet = VAE().to(device)
VAEOptim = optim.Adam(VAENet.parameters(), lr = 0.00015, betas=(0.5, 0.999))
criterion = nn.MSELoss().to(device)

for epoch in range(30):
  VAENet.train()
  for i, data in enumerate(trainloader, 0):
    images, labels = data
    VAENet.zero_grad()
    real_image = images.to(device)
    
    gen_images = VAENet(real_image)
    loss = criterion(real_image, gen_images)
    loss.backward()
    VAEOptim.step()

    if i % 150 == 0:
      print(f"loss: {loss.item()}")
      plt.imshow(gen_images[0][0].detach().cpu(), cmap=plt.cm.gray_r)
      plt.show()

  with torch.no_grad():
    VAENet.eval()
    plt.imshow(gen_images[0][0].detach().cpu(), cmap=plt.cm.gray_r)
    plt.show()