from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, x_train, y_train):
        super().__init__()
        self.x_train = torch.from_numpy(x_train)
        self.x_train = self.x_train.to(torch.float32)
        self.y_train = torch.from_numpy(y_train)
        self.y_train = self.y_train.to(torch.long)

    def __getitem__(self, index):
        x = self.x_train[index]
        y = self.y_train[index]
        return x, y

    def __len__(self):
        return len(self.x_train)
