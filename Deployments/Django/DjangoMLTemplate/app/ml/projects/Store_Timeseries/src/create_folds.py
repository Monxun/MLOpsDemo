# src/create_folds.py
import argparse
import os
# config.py
import config 

import numpy as np
import pandas as pd
from sklearn import model_selection


def run(folds):
    # read training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # create new column labeled kfold
    df["kfold"] = -1

    # pass fold argument
    kf = model_selection.KFold(n_splits=folds, shuffle=True, random_state=42)

    for fold, (train_indices, valid_indices) in enumerate(kf.split(X=df)):
        df.loc[valid_indices, "kfold"] = fold
    
    # output to csv file
    df.to_csv(f"../input/{folds}_train_folds.csv", index=False)


if __name__ == '__main__':
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their type
    # currently, we only need fold
    parser.add_argument(
        "--folds",
        type=int
    )
    # read the arguments from the command line 
    args = parser.parse_args()

    # run the fold specified by command line arguments
    run(folds=args.folds)