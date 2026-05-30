"""Orquestador: entrena los modelos del homework y persiste el mejor.

Este script no se ejecuta durante los tests. Sirve como punto de entrada
para reproducir el entrenamiento manualmente:

    python -m src.main
"""

import os
import pickle

from src.data import load_data, split_features_target, split_train_test
from src.metrics import compute_metrics, report
from src.models import ELASTICNET_GRID, KNN_GRID, train_elasticnet, train_knn

MODEL_PATH = "models/estimator.pkl"


def save_estimator(estimator, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(estimator, f)


def load_estimator(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_best(x_train, y_train, x_test, y_test):
    candidates = []

    for alpha, l1_ratio in ELASTICNET_GRID:
        est = train_elasticnet(x_train, y_train, alpha=alpha, l1_ratio=l1_ratio)
        mse = compute_metrics(y_test, est.predict(x_test))["mse"]
        candidates.append((mse, est))

    for k in KNN_GRID:
        est = train_knn(x_train, y_train, n_neighbors=k)
        mse = compute_metrics(y_test, est.predict(x_test))["mse"]
        candidates.append((mse, est))

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def main():
    df = load_data()
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = split_train_test(x, y)

    best = find_best(x_train, y_train, x_test, y_test)
    report("train", y_train, best.predict(x_train))
    report("test", y_test, best.predict(x_test))

    save_estimator(best)
    print(f"Mejor estimador guardado en {MODEL_PATH}: {best}")


if __name__ == "__main__":
    main()
