import argparse
import os
import logging

import torch
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split

from .kfold import run_kfold, run_fold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pytorch-threads", type=int)
    parser.add_argument("--kfold-pool-size", type=int, default=1)
    parser.add_argument("--disable-kfold", action="store_true")
    parser.add_argument("trainset")
    parser.add_argument("experimentfile")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.experimentfile) as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    train_df = pd.read_csv(args.trainset)

    if args.num_pytorch_threads:
        torch.set_num_threads(args.num_pytorch_threads)

    if args.disable_kfold:
        train_df, val_df = train_test_split(train_df, test_size=0.3, random_state=42, shuffle=True)
        res = run_fold(0, config, train_df, val_df)
    else:
        res = run_kfold(config, train_df, args.kfold_pool_size)

    basedir = os.path.dirname(args.experimentfile)
    basename = os.path.basename(args.experimentfile).split(".")[0]
    basepath = f"{basedir}/{basename}"
    res.to_csv(f"{basepath}.csv", index=False)
    grp = res.groupby("epoch")
    print(grp[["val_mae", "val_mre", "val_mse"]].mean())
    mean = grp[["train_loss", "val_loss"]].mean()
    mean.plot(logy=True, grid=True)
    plt.savefig(f"{basepath}.png")
