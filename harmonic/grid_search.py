# TODO before running
# - check the data path
# - check that path develooment repo is locally installed

# inf1.6xlarge 24 vCPUs, 48GB RAM


import os
import torch
from torch import nn, Tensor
from torch import optim
from tqdm import tqdm
import logging
import pandas as pd

import numpy as np
from aeon.datasets import load_from_tsfile

from development.so import so
from development.gl import gl
from development.he import he

from models.attention_development import (
    MultiheadAttentionDevelopment,
    AttentionDevelopmentConfig,
    GroupConfig,
)
from torch.utils.data import DataLoader


def initialise_logger(log_dir, log_file_name, log_level):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create the full path for the log file
    log_file_path = os.path.join(log_dir, log_file_name)

    # Create logger
    logger = logging.getLogger("SO_grid_search")
    logger.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )

    # Create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def to_one_hot(y, num_classes=6):
    return np.eye(num_classes)[y.astype(int)]


# Convert to soft probabilities
def to_soft_probabilities(y_one_hot, temperature=0.2):
    exp_values = np.exp(y_one_hot / temperature)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def train_model(
    fm: nn.Module, train_loader: DataLoader, nepochs: int, learning_rate: float
):
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
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
        )
        self.atdev = MultiheadAttentionDevelopment(
            dropout=dropout,
            input_dim=(1 + int(bidirectional)) * hidden_dim,
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
    log_dir = "logs"
    log_file_name = "SO_grid_search.log"
    log_level = logging.INFO
    logger = initialise_logger(log_dir, log_file_name, log_level)

    n_epochs = 10
    learning_rate = 1e-3
    batch_size = 256

    data_dir = os.path.join(os.getcwd(), "..", "data", "WalkingSittingStanding")
    train_file = "WalkingSittingStanding_TRAIN.ts"
    test_file = "WalkingSittingStanding_TEST.ts"

    tsx_train, y_train_labels = load_from_tsfile(os.path.join(data_dir, train_file))
    tsx_test, y_test_labels = load_from_tsfile(os.path.join(data_dir, test_file))
    # Convert labels to one-hot encoded vectors

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

    channel_range = range(2, 5)
    dim_range = range(3, 10)
    hidden_size_range = [4, 6, 10, 15, 20]
    heads_range = range(2, 3)
    lstm_is_bidirectional = [True, False]

    # nchannels = channel_range[0]
    # dim = dim_range[0]
    # n_heads = heads_range[0]
    # hidden_size = hidden_size_range[0]
    # bidirectional = lstm_is_bidirectional[0]

    res = pd.DataFrame(
        columns=["train_acc", "test_acc"],
        index=pd.MultiIndex.from_product(
            [
                channel_range,
                dim_range,
                heads_range,
                hidden_size_range,
                lstm_is_bidirectional,
            ],
            names=["nchannels", "dim", "n_heads", "hidden_size", "bidirectional"],
        ),
    ).iloc[:2]

    for nchannels, dim, n_heads, hidden_size, bidirectional in res.index.to_series():
        if not hidden_size % n_heads == 0:
            continue

        logger.info(
            f">>> Starting grid search with : nchannels={nchannels}, dim={dim}, hidden_size={hidden_size}, nheads={n_heads}, bidirectional={bidirectional}"
        )

        multidev_config = AttentionDevelopmentConfig(
            n_heads=n_heads,
            groups=[
                GroupConfig(group=so, dim=dim, channels=nchannels)
                for _ in range(n_heads)
            ],
        )

        model = PDevBaggingBiLSTM(
            dropout=0.05,
            input_dim=3,
            hidden_dim=hidden_size,
            out_dim=6,
            multidev_config=multidev_config,
            bidirectional=bidirectional,
        ).to(device)

        model.train()

        model, lossx = train_model(model, train_loader, n_epochs, learning_rate)

        train_acc = train_sample_accuracy(model, train_loader, y_train)

        test_acc = test_sample_accuracy(model, test_loader, y_test)

        logger.info(f"Train accuracy: {train_acc} | Test accuracy: {test_acc}")

        res.loc[(nchannels, dim, n_heads, hidden_size, bidirectional), :] = [
            train_acc,
            test_acc,
        ]

    res.to_csv(os.path.join(log_dir, "grid_search_results.csv"))
