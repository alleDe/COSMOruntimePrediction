import math
from functools import partial
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.base import scope

from .kfold import run_kfold


def search_space(input_size):
    return {
        "kfold": 10,
        "batch": hp.choice("batch", [64, 128, 256, "full"]),
        "loss": hp.choice("loss", ["mse", "mae"]),
        "optimizer": {
            "name": hp.choice("optimizer", ["adam", "sgd"]),
            "lr": hp.loguniform("lr", math.log(0.001), math.log(0.1)),
            "weight_decay": hp.uniform("weight_decay", 0.01, 0.1)
        },
        "epochs": 500,
        "model": hp.choice("model", [{
            "layers": [{
                "size": scope.int(hp.quniform(f"size-{i}-{j}", input_size, input_size * 7, 1)),
                "dropout": hp.choice(f"dropout-{i}-{j}", [0.0, 0.3]),
            } for j in range(1, i+1)]
        } for i in range(2, 4)])
    }


def objective(train_df, num_concurrent_fold, config):
    losses = run_kfold(config, train_df, num_concurrent_fold)
    loss = np.sqrt(losses.groupby("epoch").val_mse.mean()).min()
    status = STATUS_FAIL if math.isnan(loss) or math.isinf(loss) else STATUS_OK
    return {
        "loss": loss,
        "status": status,
        "csv": losses.to_csv(),
        "config": config,
    }


def run_trials(num_trials, train_df, trialpath, num_concurrent_fold):
    best = fmin(
        partial(objective, train_df, num_concurrent_fold),
        space=search_space(train_df.shape[1]-1),
        algo=tpe.suggest,
        show_progressbar=True,
        max_evals=num_trials,
        trials_save_file=trialpath,
    )

    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pytorch-threads", type=int, default=1)
    parser.add_argument("--num-concurrent-fold", type=int, default=1)
    parser.add_argument("--num-trials", type=int)
    parser.add_argument("trialfile")
    parser.add_argument("trainfile")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.num_pytorch_threads:
        torch.set_num_threads(args.num_pytorch_threads)

    run_trials(args.num_trials, pd.read_csv(args.trainfile),
               args.trialfile, args.num_concurrent_fold)
