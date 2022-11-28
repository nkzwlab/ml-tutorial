import os
import torch
import torch.nn as nn
from dataset import get_mnist_dataset
from model import Net

def train(dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/'
    epochs = 10
    save_step = 100
    save_epoch = 1
    batch_size = 256
    shuffle = True

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=0.95
    )

    for epoch in range(1, epochs+1):
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)

            loss = criterion(outputs, labels)
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