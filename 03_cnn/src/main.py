from train import train
from evaluate import evaluate
from dataset import get_mnist_dataset

if __name__ == '__main__':
    train_dataset, test_dataset= get_mnist_dataset()
    train(train_dataset)

    model_paths = [
        'models/net-1.pth',
        'models/net-5.pth',
        'models/net-10.pth',
    ]

    for path in model_paths:
        evaluate(test_dataset, path)