from model import Net
import torch
import torch.nn as nn
from dataset import get_mnist_dataset
import matplotlib.pyplot as plt

def evaluate(dataset, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    shuffle = True

    x_dim = 28*28
    h_dim = 400
    z_dim = 20

    fig = plt.figure(figsize=(20, 6))

    with torch.no_grad():

        net = Net(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim)
        net.load_state_dict(torch.load(model_path))

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


if __name__ == '__main__':
    _, test_dataset = get_mnist_dataset()

    model_paths = [
        # 'models/net-1.pth',
        # 'models/net-5.pth',
        # 'models/net-10.pth',
        'models/net-30.pth',
    ]

    for path in model_paths:
        print(path)
        evaluate(test_dataset, path)