import torch
import torch.utils.data as data
from sklearn.datasets import load_iris
import pandas as pd

class IrisDataset(data.Dataset):
    def __init__(self, df, features, labels):
        self.features = df[features].values
        self.labels = df[labels].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor(self.labels[idx])

        return feature, label

def get_iris_dataset():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    label_nums = {
        'setosa': 0,
        'versicolor': 1,
        'virginica': 2,
    }

    df['labels'] = iris.target_names[iris.target]
    df['labels'] = df['labels'].map(label_nums)
    print(df)

    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    labels = ['labels']

    return IrisDataset(df, features, labels)

if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    label_nums = {
        'setosa': 0,
        'versicolor': 1,
        'virginica': 2,
    }

    df['labels'] = iris.target_names[iris.target]
    df['labels'] = df['labels'].map(label_nums)
    print(df)

    dataset = get_iris_dataset()
    print(len(dataset))
    print(dataset[0])
