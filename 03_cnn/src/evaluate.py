from model import Net
import torch
import torch.nn as nn
from dataset import get_mnist_dataset
import statistics

def evaluate(dataset, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    shuffle = True

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        accs = []

        net = Net()
        net.load_state_dict(torch.load(model_path))

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

        losses = []
        accs = []
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            _, predicts = outputs.max(1)
            
            acc = torch.sum(predicts == labels) / len(labels)
            accs.append(acc)

    average_loss = statistics.mean(losses)
    average_acc = torch.tensor(accs).mean()
    print('Loss: {:.3f}'.format(average_loss))
    print('Accuracy: {:.1f}%'.format(average_acc * 100))

if __name__ == '__main__':
    _, test_dataset = get_mnist_dataset()

    model_paths = [
        'models/net-1.pth',
        'models/net-5.pth',
        'models/net-10.pth',
    ]

    for path in model_paths:
        print(path)
        evaluate(test_dataset, path)