import torch
from torch.utils.data import Dataset


class MMCDataset(Dataset):
    def __init__(self, x, x_length, y):
        self.x = torch.from_numpy(x)
        self.x_length = torch.from_numpy(x_length)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.x_length[idx], self.y[idx]

