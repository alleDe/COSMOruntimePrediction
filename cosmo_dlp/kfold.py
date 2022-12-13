from multiprocessing import Pool
import logging

import pandas as pd
from sklearn.model_selection import KFold

from .model import CosmoData, CosmoPredictor


def run_fold(nfold, config, train_df, val_df, logger=None):
    data = CosmoData(
        train_df=train_df,
        val_df=val_df,
        test_df=None,
        batch_size=config["batch"],
    )
    model = CosmoPredictor(data.input_size, config, logger)
    return model.fit(data, config["epochs"])


def run_kfold(config, train_df, pool_size=1):
    kfold = KFold(n_splits=config.get("kfold", 10), shuffle=True, random_state=42)
    args = []
    for n, (i, j) in enumerate(kfold.split(train_df)):
        args.append([
            n, config, train_df.iloc[i], train_df.iloc[j],
            logging.getLogger(f"{__name__}.kfold.{n}"),
        ])

    with Pool(pool_size) as pool:
        results = pool.starmap(run_fold, args)

    for n, result in enumerate(results):
        result["n"] = n

    return pd.concat(results)
