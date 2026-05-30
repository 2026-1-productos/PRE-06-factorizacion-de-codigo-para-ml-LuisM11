"""Entrenamiento de los estimadores propuestos en homework/."""

from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor

ELASTICNET_GRID = [
    (0.5, 0.5),
    (0.2, 0.2),
    (0.1, 0.1),
    (0.1, 0.05),
    (0.3, 0.2),
]

KNN_GRID = [3, 5, 7, 9, 11]


def train_elasticnet(x_train, y_train, alpha=0.5, l1_ratio=0.5, random_state=12345):
    estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    estimator.fit(x_train, y_train)
    return estimator


def train_knn(x_train, y_train, n_neighbors=5):
    estimator = KNeighborsRegressor(n_neighbors=n_neighbors)
    estimator.fit(x_train, y_train)
    return estimator
