import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % (size / 2) == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            # print(f'pred is {pred}, y is {y}')
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(
        f"Test avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    train_dataset = np.load('train_dataset.npz')
    train_T, train_x, train_y = torch.from_numpy(train_dataset['T']), torch.from_numpy(
        train_dataset['x']).float(), torch.from_numpy(train_dataset['y']).float()
    train_dataloader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=32, shuffle=True)  # need to convert tensor x and y from dtype int32 to float

    test_dataset = np.load('test_dataset.npz')
    test_T, test_x, test_y = torch.from_numpy(test_dataset['T']), torch.from_numpy(
        test_dataset['x']).float(), torch.from_numpy(test_dataset['y']).float()
    test_dataloader = DataLoader(TensorDataset(test_x, test_y),
                                 batch_size=32, shuffle=True)

    model = MLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-3)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        # test_loop(test_dataloader, model, loss_fn)
    print("Done!")
