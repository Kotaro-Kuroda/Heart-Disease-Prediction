from model import NeuralNetwork
import torch
from dataset.mydataset import MyDataset
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import os
import read_csv
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(model, x_test, y_test):
    dataset = MyDataset(x_test, y_test)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    num_correct = 0
    num_samples = 0
    confusion_matrix = np.zeros((2, 2))
    for batch in tqdm.tqdm(dataloader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            output = model(x)
        y_pred = torch.argmax(output)
        confusion_matrix[y.item()][y_pred.item()] += 1

    return confusion_matrix


def main():
    csv_path = './heart.csv'
    x_train, x_test, y_train, y_test = read_csv.read_csv(csv_path)
    model = NeuralNetwork(input_features=x_train.shape[1], num_classes=2)
    model.load_state_dict(torch.load('./models/model_14.pt'))
    model.to(device)
    confusion_matrix = test(model, x_test, y_test)
    acc = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    print(acc)
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True)
    plt.show()


if __name__ == '__main__':
    main()
