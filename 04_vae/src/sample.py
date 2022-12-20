import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils as utils
from torchvision import datasets, transforms
import os

warnings.simplefilter('ignore')

class VAE(nn.Module):
    def __init__(self, latent_dim, input_dim=28*28):
        super(VAE, self).__init__()
        hidden_dim = 100
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim*3)
        self.encoder_fc2 = nn.Linear(hidden_dim*3, hidden_dim)
        self.encoder_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim*3)
        self.decoder_fc_output = nn.Linear(hidden_dim*3, input_dim)

    def encoder(self, x):
        x.view(-1, self.input_dim)
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))
        mean = self.encoder_fc_mu(x)
        log_var = self.encoder_fc_var(x)
        return mean, log_var

    def decoder(self, z):
        y = F.relu(self.decoder_fc1(z))
        y = F.relu(self.decoder_fc2(y))
        y = torch.sigmoid(self.decoder_fc_output(y))
        return y

    @staticmethod
    def reparametrization(mean, log_var):
        # reparametrization trick
        epsilon = torch.randn(mean.shape)
        return mean + epsilon * torch.exp(0.5 * log_var)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        mean, log_var = self.encoder(x)
        z = self.reparametrization(mean, log_var)
        x_out = self.decoder(z)

        # Appendix B from VAE paper:
        # 1/2 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        # Formula 10 from VAE paper:
        # 1/L * sum(log(p(x(i)|z(l,i))))
        reconstruction = F.binary_cross_entropy(x_out, x, reduction="sum")

        lower_bound = -KLD + reconstruction
        return lower_bound, z, x_out

download=True
trans = transforms.ToTensor()
train_set = datasets.MNIST(root="./", train=True, transform=trans, download=download)
valid_set = datasets.MNIST(root="./", train=False,transform=trans)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=128, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=128, shuffle=False)

epoch=20
latent_dims=6
vae = VAE(latent_dims,28*28)
vae.train()
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

loss_list = []
for i in range(epoch):
    losses = []
    for x, t in train_loader:
        loss, z, y = vae(x)
        vae.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy())
    loss_list.append(np.average(losses))
    print(f"Epoch: {i}, loss: {np.average(losses)}")

    torch.save(vae.state_dict(), os.path.join(
                'sample/', 'net-{}.pth'.format(epoch)
            ))