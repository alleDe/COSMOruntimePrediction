from itertools import tee
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


torch.manual_seed(0)


def pairwise(seq):
    a, b = tee(seq)
    next(b, None)
    return zip(a, b)


def create_model(input_size, output_size, cfg):
    """Creates a model from a dict, e.g.
        {
            "layers": [{
                "size": 40,
                "dropout": 0.3,
            }, {
                "size": 30,
                "dropout": 0.0,
            }],
        }
    Each element of "layers" array is a hidden layer, where "size" is the
    number of neurons and "dropout" is the dropout. Each layer is a linear
    layer with a ReLU activation function.

    Return a nn.Sequential container.
    """
    layers = cfg["layers"]
    if len(layers):
        seq = [
            nn.Linear(input_size, int(layers[0]["size"])),
            nn.ReLU(),
        ]
        for l, nl in pairwise(layers):
            if l["dropout"] > 0.0:
                seq += [
                    nn.Dropout(l["dropout"]),
                ]

            seq += [
                nn.Linear(int(l["size"]), int(nl["size"])),
                nn.ReLU(),
            ]

        if layers[-1]["dropout"] > 0.0:
            seq += [
                nn.Dropout(layers[-1]["dropout"]),
            ]

        seq += [
            nn.Linear(int(layers[-1]["size"]), 1),
        ]
        return nn.Sequential(*seq)
    else:
        return nn.Linear(input_size, 1)


class CosmoPredictor:
    """Generic COSMO predictor.

    The architecture of the model is passes in the constructor ("config"), e.g.
        {
            "model": {
                "layers": ...
            },
            "loss": "mae",
            "optimizer": {
                "name": "adam",
                "lr": 0.01,
                "weight_decay": 0.0,
            },
        }
    """
    def __init__(self, input_size, config, logger=None):
        self.config = config
        # model
        self.model = create_model(input_size, 1, config["model"])
        # loss
        self.criterion = {
            "mae": nn.L1Loss(reduction="sum"),
            "mse": nn.MSELoss(reduction="sum"),
        }[self.config["loss"]]
        # optimizer
        self.optimizer = {
            "adam": optim.Adam,
            "sgd": optim.SGD,
        }[self.config["optimizer"]["name"]](
            self.model.parameters(),
            lr=self.config["optimizer"]["lr"],
            weight_decay=self.config["optimizer"]["weight_decay"],
        )
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger()

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)

    def fit(self, data, max_epochs):
        """Train and validate the model for max_epochs epochs.

        Return a DataFrame with the following columns:
        - epoch
        - train_loss
        - val_loss
        - val_mae
        - val_mse
        - val_mre
        """
        metrics = []
        for epoch in range(max_epochs):
            self.logger.info("Epoch %s", epoch)

            train_res = self.train_epoch(data.train_dataloader())
            val_res = self.val_epoch(data.val_dataloader())

            metrics.append([
                epoch,
                train_res["train_loss"],
                val_res["val_loss"],
                val_res["val_mae"],
                val_res["val_mre"],
                val_res["val_mse"],
            ])

        return pd.DataFrame(metrics, columns=[
            "epoch",
            "train_loss",
            "val_loss",
            "val_mae",
            "val_mre",
            "val_mse",
        ])

    def test(self, data):
        """Evaluation of the model.

        Return a DataFrame with the following columns:
        - target
        - prediction
        """
        targets = []
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch in data.test_dataloader():
                x, y = batch
                z = self.model(x)
                yexp = np.exp(y[:, 0].numpy())
                zexp = np.exp(z[:, 0].numpy())
                targets = np.concatenate([targets, yexp])
                predictions = np.concatenate([predictions, zexp])

        return pd.DataFrame({
            "target": targets,
            "prediction": predictions,
        })

    def train_epoch(self, dataloader):
        # train
        train_loss_acc = 0.0
        train_count = 0
        self.model.train()
        for batch in dataloader:
            res = self.train_step(batch)
            train_loss_acc += res["train_loss"]
            train_count += res["train_count"]

        train_loss = train_loss_acc / train_count
        return {
            "train_loss": train_loss,
        }

    def val_epoch(self, dataloader):
        if dataloader is None:
            val_loss = None
            val_mae = None
            val_mre = None
            val_mse = None
        else:
            # validate
            val_loss_acc = 0.0
            val_mae_acc = 0.0
            val_mre_acc = 0.0
            val_mse_acc = 0.0
            val_count = 0
            self.model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    res = self.val_step(batch)
                    val_loss_acc += res["val_loss"]
                    val_mae_acc += res["val_mae_acc"]
                    val_mre_acc += res["val_mre_acc"]
                    val_mse_acc += res["val_mse_acc"]
                    val_count += res["val_count"]

            val_loss = val_loss_acc / val_count
            val_mae = val_mae_acc / val_count
            val_mre = val_mre_acc / val_count
            val_mse = val_mse_acc / val_count

        return {
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_mre": val_mre,
            "val_mse": val_mse,
        }

    def train_step(self, batch):
        x, y = batch
        z = self.model(x)
        loss = self.criterion(z, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "train_loss": loss.item(),
            "train_count": len(y),
        }

    def val_step(self, batch):
        x, y = batch
        z = self.model(x)
        loss = self.criterion(z, y)
        yexp = np.exp(y.double())
        zexp = np.exp(z.double())
        err = zexp - yexp
        mae = err.abs().sum()
        mre = (err.abs()/yexp).sum()
        mse = (err*err).sum()
        return {
            "val_loss": loss.item(),
            "val_mae_acc": mae.item(),
            "val_mre_acc": mre.item(),
            "val_mse_acc": mse.item(),
            "val_count": len(y),
        }


class CosmoData:
    def __init__(self, train_df, val_df, test_df, batch_size="full"):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.input_size = self.train_df.shape[1] - 1
        self.batch_size = batch_size
        self.scaler = MinMaxScaler().fit(train_df.iloc[:, :-1])

    def __dataframe_to_dataloader(self, df, is_train=False):
        x, y = self.__dataframe_to_scaled_array(df)
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()
        if self.batch_size == "full" or not is_train:
            batch_size = len(x)
        else:
            batch_size = self.batch_size
        return DataLoader(TensorDataset(x, y),
                          batch_size=batch_size,
                          shuffle=is_train)

    def __dataframe_to_scaled_array(self, df):
        return (
            self.scaler.transform(df.iloc[:, :-1]),
            np.log(df.iloc[:, -1:]).values,
        )

    def train_dataloader(self):
        return self.__dataframe_to_dataloader(self.train_df, True)

    def val_dataloader(self):
        if self.val_df is not None:
            return self.__dataframe_to_dataloader(self.val_df)

    def test_dataloader(self):
        return self.__dataframe_to_dataloader(self.test_df)

    def train_set(self):
        return self.__dataframe_to_scaled_array(self.train_df)

    def val_set(self):
        return self.__dataframe_to_scaled_array(self.val_df)

    def test_set(self):
        return self.__dataframe_to_scaled_array(self.test_df)
