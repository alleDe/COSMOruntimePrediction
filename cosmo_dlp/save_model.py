import pickle
import json
import io

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from cosmo_dlp.model import CosmoPredictor, CosmoData


def train_and_save(name):
    with open(f"data/hpopt/{name}.pk", "rb") as fp:
        trials = pickle.load(fp)

    config = trials.best_trial["result"]["config"]
    hp_df = pd.read_csv(io.StringIO(trials.best_trial["result"]["csv"]))
    mean_mse = hp_df.groupby("epoch")["val_mse"].mean()
    best_epoch = mean_mse.idxmin()

    data = CosmoData(
        pd.read_csv(f"data/input/{name}_train.csv"),
        pd.read_csv(f"data/input/{name}_train.csv").sample(1),
        pd.read_csv(f"data/input/{name}_test.csv"),
    )
    model = CosmoPredictor(data.input_size, config)
    model.fit(data, best_epoch)
    model.save(f"data/model/{name}_model.pth")


def test(name):
    model = torch.load(f"data/model/{name}_model.pth")
    scaler = MinMaxScaler().fit(
        pd.read_csv(f"data/input/{name}_train.csv").iloc[:, :-1]
    )
    df = pd.read_csv(f"data/input/{name}_test.csv")
    x = torch.from_numpy(scaler.transform(df.iloc[:, :-1])).float()
    model.eval()
    with torch.no_grad():
        df["predicted"] = np.exp(model(x).numpy())

    df.to_csv(f"data/output/{name}_result.csv", index=False)

if __name__ == '__main__':
    train_and_save("DGV_PT2")
    test("DGV_PT2")
