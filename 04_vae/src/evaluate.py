from model import Net
import torch
import torch.nn as nn
from dataset import get_mnist_dataset
import matplotlib.pyplot as plt
import numpy as np

def reconstruct(dataset, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    shuffle = True

    x_dim = 28*28
    h_dim = 100
    z_dim = 20

    fig = plt.figure(figsize=(20, 6))

    with torch.no_grad():

        net = Net(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim)
        net.load_state_dict(torch.load(model_path))
        net.eval()

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

        for i, (images, _) in enumerate(data_loader):
            if i > 0: break

            for i, im in enumerate(images.view(-1, 28, 28).detach().numpy()[:10]):
                ax = fig.add_subplot(3, 10, i+1)
                ax.imshow(im, 'gray')
            plt.savefig("original.png")

            images = images.to(device).view(-1, x_dim)
            reconstruct, _, _ = net(images)

            for i, im in enumerate(reconstruct.view(-1, 28, 28).detach().numpy()[:10]):
                ax = fig.add_subplot(3, 10, i+1)
                ax.imshow(im, 'gray')
            plt.savefig("reconstruct.png")

def plot_z(dataset, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    shuffle = True
    fig = plt.figure(figsize=(21,21))

    visible_label = 7

    x_dim = 28*28
    h_dim = 100
    z_dim = 20

    data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
    
    with torch.no_grad():

        net = Net(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim)
        net.load_state_dict(torch.load(model_path))
        net.eval()

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

        for i, (images, labels) in enumerate(data_loader):
            if i > 0: break

            visible_label = (labels==visible_label).nonzero().squeeze()[0]
            plt.imshow(images[visible_label][0].numpy(), 'gray')
            plt.savefig("original_z.png")

            images = images.to(device).view(-1, x_dim)
            mean, log_var = net.encode(images)
            vis_z = net.reparameterize(mean, log_var)

            print('----------------------------')
            print(vis_z.shape)
            print('----------------------------')

            reconstruct = net.decode(vis_z).reshape(-1, 28, 28)[visible_label].cpu().detach().numpy()
            plt.imshow(reconstruct, 'gray')
            plt.savefig("reconstruct_z.png")

            fig = plt.figure(figsize=(21,21))
            scale = 3
            i = 1
            for x in np.linspace(-scale, scale, 21):
                for y in np.linspace(-scale, scale, 21):
                    vis_z[visible_label][0] = y
                    vis_z[visible_label][1] = x
                    ax = fig.add_subplot(21,21,i, xticks=[], yticks=[])
                    x_out = net.decode(vis_z)
                    x_out = x_out.cpu().detach().numpy()
                    ax.imshow(x_out.reshape(-1,28,28)[visible_label], "gray")
                    i = i+1

            fig.savefig(f"vae_z.png")


if __name__ == '__main__':
    _, test_dataset = get_mnist_dataset()

    model_paths = [
        # 'models/net-1.pth',
        # 'models/net-5.pth',
        # 'models/net-10.pth',
        'models/net-20.pth',
    ]

    for path in model_paths:
        print(path)
        # reconstruct()(test_dataset, path)
        plot_z(test_dataset, path)