import os
import torch
import torch.nn as nn
from dataset import get_mnist_dataset
from model import Net
import torch.nn.functional as F

def train(dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/'
    epochs = 50
    save_step = 100
    save_epoch = 1
    batch_size = 128
    shuffle = True

    x_dim = 28*28
    h_dim = 400
    z_dim = 20

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    net = Net(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim).to(device)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=0.95
    )

    for epoch in range(1, epochs+1):
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device).view(-1, x_dim)

            reconstruct, mean, log_var = net(images)

            reconst_loss = F.binary_cross_entropy(reconstruct, images)
            kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

            loss = reconst_loss + kl_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % save_step == 0:
                print("step: [{}/{}], Loss: {:.4f}".format(i, len(data_loader), loss.item()))

        print('Epoch: [{}/{}], Loss: {:.4f}'.format(epoch, epochs, loss.item()))
        print('==========================')
        scheduler.step()

        if epoch % save_epoch == 0:
            torch.save(net.state_dict(), os.path.join(
                model_path, 'net-{}.pth'.format(epoch)
            ))

if __name__ == '__main__':
    train_dataset, _ = get_mnist_dataset()
    train(train_dataset)