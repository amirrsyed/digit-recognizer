import pandas as pd
import numpy as np

def load_training_data(path):
    df = pd.read_csv(path)

    X = df.drop("label", axis=1).values
    y = df["label"].values

    # Normalize
    X = X / 255.0

    # Reshape for CNN
    X = X.reshape(-1, 28, 28, 1)

    return X, y


def load_test_data(path):
    df = pd.read_csv(path)

    X = df.values / 255.0
    X = X.reshape(-1, 28, 28, 1)

    return X
