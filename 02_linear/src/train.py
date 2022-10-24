import os
import torch
import torch.nn as nn
from dataset import get_iris_dataset
from model import Net

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/'
    epochs = 30
    log_epoch = 5
    save_epoch = 10
    batch_size = 10
    shuffle = True

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    dataset = get_iris_dataset()
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    for epoch in range(1, epochs+1):
        for i, (features, labels) in enumerate(data_loader):
            features = features.to(device)
            labels = labels.to(device)

            outputs = net(features)

            loss = criterion(outputs, labels)
            net.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch)
        if epoch % log_epoch == 0:
            print('Epoch: [{}/{}], Loss: {:.4f}'.format(epoch, epochs, loss.item()))

        if epoch+1 % save_epoch == 0:
            torch.save(net.state_dict(), os.path.join(
                model_path, 'net-{}.ckpt'.format(epoch)
            ))

if __name__ == '__main__':
    train()