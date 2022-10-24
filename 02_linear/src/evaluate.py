from model import Net
import torch
from dataset import get_iris_dataset

def evaluate(data_loader, device):
    with torch.no_grad():
        accs = []

        net = Net()
        net.load_state_dict(torch.load('models/net-30.pth'))

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/'
    epochs = 30
    log_epoch = 5
    save_epoch = 10
    batch_size = 10
    shuffle = True

    dataset = get_iris_dataset()
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    evaluate(data_loader, device)