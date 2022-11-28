from model import Net
import torch
from dataset import get_iris_dataset

def evaluate(dataset, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 10
    shuffle = True

    with torch.no_grad():
        accs = []

        net = Net()
        net.load_state_dict(torch.load(model_path))

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            y = torch.argmax(y.to(device), dim=1)
            predict = net(x)

            predict = torch.argmax(predict, dim=1)

            acc = torch.sum(predict == y) / len(y)
            accs.append(acc)

    average_acc = torch.tensor(accs).mean()
    print('Accuracy: {:.1f}%'.format(average_acc * 100))

if __name__ == '__main__':
    dataset = get_iris_dataset()

    model_paths = [
        'models/net-10.pth',
        'models/net-20.pth',
        'models/net-30.pth',
    ]

    for path in model_paths:
        evaluate(dataset, path)