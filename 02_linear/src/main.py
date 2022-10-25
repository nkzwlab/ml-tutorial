from dataset import get_iris_dataset
from train import train
from evaluate import evaluate
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dataset = get_iris_dataset()
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.8)
    train(train_dataset)

    model_paths = [
        'models/net-10.pth',
        'models/net-20.pth',
        'models/net-30.pth',
    ]

    for path in model_paths:
        evaluate(test_dataset, path)