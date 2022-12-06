from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Net, self).__init__()
        self.enc1 = nn.Linear(x_dim, h_dim)

        self.mean = nn.Linear(h_dim, z_dim)
        self.log_var = nn.Linear(h_dim, z_dim)

        self.dec1 = nn.Linear(z_dim, h_dim)
        self.dec2 = nn.Linear(h_dim, x_dim)

    def encode(self, x):
        x = F.relu(self.enc1(x))
        mean = self.mean(x)
        log_var = self.log_var(x)

        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(log_var/2)
        eps = torch.rand_like(std)

        return mean + eps*std

    def decode(self, z):
        y = F.relu(self.dec1(z))
        y = torch.sigmoid(self.dec2(y))

        return y

    def forward(self, x):
        mean, log_var = self.encode(x)

        z = self.reparameterize(mean, log_var)
        reconstruct = self.decode(z)

        return reconstruct, mean, log_var

if __name__ == '__main__':
    net = Net()

    batch_size = 5
    summary(net, input_size=(batch_size, 4))