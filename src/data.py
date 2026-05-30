"""Carga y división del dataset de calidad de vino."""

import pandas as pd
from sklearn.model_selection import train_test_split

WINE_URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)


def load_data(url=WINE_URL):
    return pd.read_csv(url, sep=";")


def split_features_target(df, target="quality"):
    y = df[target]
    x = df.drop(columns=[target])
    return x, y


def split_train_test(x, y, test_size=0.25, random_state=123456):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
