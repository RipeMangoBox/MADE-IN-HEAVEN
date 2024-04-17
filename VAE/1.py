import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from torchvision import datasets, transforms
from typing import List, Callable, Union, Any, TypeVar, Tuple
from sklearn.manifold import TSNE
import pandas as pd


class CVAE(nn.Module):
  def __init__(self, config: dict = None, num_classes: int = None, all_class_mean_mu: torch.Tensor = None):
    super(CVAE, self).__init__()

    self.image_size = config.image_size
    self.input_dim = config.input_channel
    self.layer_sizes = config.layer_sizes
    self.fc_dim = config.fc_dim
    self.latent_dim = config.latent_dim

    self.beta = 1

    self.num_classes = num_classes
    self.all_class_mean_mu = all_class_mean_mu

    self.encoder = Encoder(self.image_size, self.input_dim, self.layer_sizes, self.fc_dim, self.latent_dim, self.num_classes)
    self.decoder = Decoder(self.image_size, self.input_dim, self.layer_sizes[::-1], self.latent_dim, self.num_classes)

  def reparameterize(self, mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

  def forward(self, x, label):
    x = x.view(x.shape[0], x.shape[1], self.image_size * self.image_size)
    mu, log_var = self.encoder(x, label)
    z = self.reparameterize(mu, log_var)
    return self.decoder(z, label), mu, log_var

  def calc_loss(self, x, recon_x, mu, log_var, label):
    mu_mean = self.all_class_mean_mu[label].view(-1, self.latent_dim).detach()
    # print(f"mu_mean: {mu_mean.shape}, mu: {mu.shape}")
    recons_loss = F.mse_loss(recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction="sum").div(x.shape[0])
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu** 2 - log_var.exp(), dim = 1), dim = 0)
    return recons_loss + self.beta * kld_loss
  
class Encoder(nn.Module):
    def __init__(self,
         image_size: int,
         input_dim: int,
         layer_sizes: List,
         fc_dim: int,
         latent_dim: int,
         num_classes: int):
        super().__init__()


        self.num_classes = num_classes
        self.layer_sizes = layer_sizes.copy()
        self.layer_sizes[0] += self.num_classes


        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.mu = nn.Linear(self.layer_sizes[-1], latent_dim)
        self.log_var = nn.Linear(self.layer_sizes[-1], latent_dim)

    def forward(self, x, c=None):

        c = idx2onehot(c, 10, x.size(1))
        x = torch.cat((x, c), dim=-1).squeeze()

        x = self.MLP(x)

        mu = self.mu(x)
        log_vars = self.log_var(x)

        return mu, log_vars

class Decoder(nn.Module):
  def __init__(self,
         image_size: int,
         output_dim: int,
         layer_sizes: List,
         latent_dim: int,
         num_classes: int):
    super().__init__()

    self.MLP = nn.Sequential()

    input_size = latent_dim + num_classes

    for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
        self.MLP.add_module(
            name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
        if i+1 < len(layer_sizes):
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        else:
            self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

  def forward(self, z, c):

    c = idx2onehot(c, n=10, channels=0)
    z = torch.cat((z, c), dim=-1).squeeze()

    x = self.MLP(z)

    return x


def idx2onehot(idx, n, channels = 1):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    if channels:
        onehot = onehot.unsqueeze(1).expand(-1, channels, -1)
        
    return onehot


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


MNIST_config = Config(
    image_size = 28,
    layer_sizes = [784, 256],
    fc_dim = 128,
    latent_dim = 64,
    batch_size = 128,
    epochs = 10,
    input_channel = 1,

    n_cols = 8,
    n_rows = 8

)

CRIFA10_config = Config(
    image_size = 32,
    layer_sizes = [32, 64],
    fc_dim = 128,
    latent_dim = 64,
    batch_size = 128,
    epochs = 15,
    input_channel = 3,

    n_cols = 14,
    n_rows = 14

)

transform=transforms.Compose([
    transforms.ToTensor()
])

dataset_opt = 'CIFAR10'


if dataset_opt == 'MNIST':
    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                        transform=transform)
    args = MNIST_config

elif dataset_opt == 'CIFAR10':
    dataset1 = datasets.CIFAR10('./data/', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.CIFAR10('./data/', train=False,
                        transform=transform)
    args = CRIFA10_config

# 超参
image_size, layer_sizes, fc_dim, latent_dim, batch_size, epochs, input_channel, n_cols, n_rows = vars(args).values()

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)


# 获取所有标签
all_labels = set()
for _, label in dataset1:
    all_labels.add(label)

all_labels = torch.Tensor(list(all_labels)).float().view(-1, 1).cuda()
label_projection = nn.Linear(1, latent_dim).cuda()
all_class_mean_mu = label_projection(all_labels)

model = CVAE(args, len(all_labels), all_class_mean_mu).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print_freq = 200
for epoch in range(epochs):
    print("Start training epoch {}".format(epoch,))
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda().unsqueeze(1)
        
        # print(labels.shape, images.shape)
        recon, mu, log_var = model(images, labels)
        loss = model.calc_loss(images, recon, mu, log_var, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
            print("\t [{}/{}]: loss {}".format(i, len(train_loader), loss.item()))