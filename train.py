from model import NeuralNetwork
import torch
from dataset.mydataset import MyDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import ShuffleSplit
import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import read_csv
from torch.nn import functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_loader, model, optimizer):
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    losses = []
    for batch in train_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y = F.one_hot(y, 2)
        y = y.to(torch.float32)
        with torch.cuda.amp.autocast():
            output = model(x)
            loss = criterion(output, y)
        losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return losses


def validation(model, val_loader):
    model.eval()
    num_correct = 0
    num_samples = 0
    for batch in val_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            output = model(x)
        y_pred = torch.argmax(output)
        if y_pred.item() == y.item():
            num_correct += 1
        num_samples += 1

    return num_correct / num_samples


def cross_validation(model, x_train, y_train, batch_size, num_epochs):
    model.to(device)
    writer = SummaryWriter()
    dataset = MyDataset(x_train, y_train)
    kf = ShuffleSplit(n_splits=10, test_size=0.10, random_state=0)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=1e-3, weight_decay=1 * 1e-4)
    model_save_dir = './models/'
    os.makedirs(model_save_dir, exist_ok=True)
    num_models = len(os.listdir(model_save_dir))
    for _fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        train_dataset = Subset(dataset, train_index)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_dataset = Subset(dataset, val_index)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        for epoch in tqdm.tqdm(range(num_epochs)):
            losses = train(train_loader, model, optimizer)
            acc = validation(model, val_loader)
            writer.add_scalar('train_loss', np.mean(losses), _fold * num_epochs + epoch + 1)
            writer.add_scalar('val_accuracy', acc, _fold * num_epochs + epoch + 1)
    torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_{num_models}.pt'))


def main():
    csv_path = './heart.csv'
    x_train, x_test, y_train, y_test = read_csv.read_csv(csv_path)
    model = NeuralNetwork(input_features=x_train.shape[1], num_classes=2)
    cross_validation(model, x_train, y_train, 16, 100)


if __name__ == '__main__':
    main()
