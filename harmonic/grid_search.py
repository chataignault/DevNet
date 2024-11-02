import os
import torch
from torch import nn, Tensor
from torch import optim
from tqdm import tqdm

import numpy as np
from aeon.datasets import load_from_tsfile

from development.so import so
from development.gl import gl
from development.he import he

from models.attention_development import MultiheadAttentionDevelopment, AttentionDevelopmentConfig, GroupConfig
from torch.utils.data import DataLoader


def to_one_hot(y, num_classes=6):
    return np.eye(num_classes)[y.astype(int)]


# Convert to soft probabilities
def to_soft_probabilities(y_one_hot, temperature=0.2):
    exp_values = np.exp(y_one_hot / temperature)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)



def train_model(fm: nn.Module, train_loader: DataLoader, nepochs: int, learning_rate: float):
    fm.train()
    optimizer = optim.Adam(fm.parameters(), lr=learning_rate)
    lossx = []
    for epoch in tqdm(range(nepochs)):
        for x, y in train_loader:
            optimizer.zero_grad()
            y_hat = fm(x)
            loss = torch.sum((y - y_hat) ** 2) / len(y)
            loss.backward()
            lossx.append(loss.item())
            optimizer.step()

        print(f"Epoch : {epoch} | Loss {lossx[-1]} | gradient {0.}")

    return fm, lossx


def train_sample_accuracy(fm: nn.Module, train_loader: DataLoader, y_train: Tensor):
    fm.eval()
    n_true_prediction = 0
    preds, trues = [], []
    for x, y in train_loader:
        y_hat = fm(x)
        y_pred = torch.argmax(y_hat, axis=1)
        y_true = torch.argmax(y, axis=1)
        n_true_prediction += torch.sum(y_pred == y_true).detach().cpu().numpy()
        preds = np.concatenate([preds, y_pred.detach().cpu().numpy()])
        trues = np.concatenate([trues, y_true.detach().cpu().numpy()])
    
    return n_true_prediction / len(y_train)


def test_sample_accuracy(fm: nn.Module, test_loader: DataLoader, y_test: Tensor):
    fm.eval()
    n_true_prediction = 0
    preds, trues = [], []
    for x, y in test_loader:
        y_hat = fm(x)
        y_pred = torch.argmax(y_hat, axis=1)
        y_true = torch.argmax(y, axis=1)
        n_true_prediction += torch.sum(y_pred == y_true).detach().cpu().numpy()
        preds = np.concatenate([preds, y_pred.detach().cpu().numpy()])
        trues = np.concatenate([trues, y_true.detach().cpu().numpy()])

    return n_true_prediction / len(y_test)


class PDevBaggingBiLSTM(nn.Module):
    def __init__(
        self,
        dropout: float,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        multidev_config: AttentionDevelopmentConfig,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.atdev = MultiheadAttentionDevelopment(
            dropout=dropout,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            multidev_config=multidev_config,
        )
        head_sizes = [g.channels * g.dim**2 for g in multidev_config.groups]
        inter_dim = sum(head_sizes)
        self.lin1 = nn.Linear(inter_dim, inter_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(inter_dim, out_dim)

    def forward_partial(self, x: Tensor):
        x, _ = self.lstm(x)
        sx = self.atdev(x)
        return sx

    def forward(self, x: Tensor):
        x, _ = self.lstm(x)
        sx = self.atdev(x)
        sx_flat = [s.view(len(s), -1) for s in sx]
        sc = torch.cat(sx_flat, axis=-1)
        y = self.lin1(sc)
        y = self.relu(y)
        y = self.lin2(y)
        return y
    

if __name__ == "__main__":

    n_epochs = 10
    hidden_size = 10
    n_heads = 4

    learning_rate = 1e-3

    data_dir = os.path.join(os.getcwd(), "..", "data", "WalkingSittingStanding")
    train_file = "WalkingSittingStanding_TRAIN.ts"
    test_file = "WalkingSittingStanding_TEST.ts"


    tsx_train, y_train_labels = load_from_tsfile(os.path.join(data_dir, train_file))
    tsx_test, y_test_labels = load_from_tsfile(os.path.join(data_dir, test_file))
    # Convert labels to one-hot encoded vectors

    batch_size = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # Apply transformations
    y_train = to_soft_probabilities(to_one_hot(y_train_labels.astype(float)))
    y_test = to_soft_probabilities(to_one_hot(y_test_labels.astype(float)))

    tsx_train = Tensor(tsx_train).swapaxes(1, 2).to(device)
    tsx_test = Tensor(tsx_test).swapaxes(1, 2).to(device)

    # Convert back to PyTorch tensors
    y_train = torch.logit(torch.tensor(y_train, dtype=torch.float32)).to(device)
    y_test = torch.logit(torch.tensor(y_test, dtype=torch.float32)).to(device)


    # Create DataLoader for training data
    train_dataset = torch.utils.data.TensorDataset(tsx_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device=device),
    )
    test_dataset = torch.utils.data.TensorDataset(tsx_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device=device),
    )


    multidev_config = AttentionDevelopmentConfig(
        n_heads=2,
        groups=[
            GroupConfig(group=so, dim=3, channels=3),
            GroupConfig(group=so, dim=4, channels=3),
        ],
    )


    model = PDevBaggingBiLSTM(
        dropout=.05,
        input_dim=3,
        hidden_dim=4,
        out_dim=6,
        multidev_config=multidev_config,
    ).to(device)

    model.train()

    model, lossx = train_model(model, train_loader, n_epochs, learning_rate)

    train_acc = train_sample_accuracy(
        model, train_loader, y_train
    )

    test_acc = test_sample_accuracy(
        model,
        test_loader,
        y_test
    )

    print(f"Train accuracy: {train_acc} | Test accuracy: {test_acc}")

