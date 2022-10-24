import torch
import torch.utils.data as data
from sklearn.datasets import load_iris

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