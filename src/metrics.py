"""Métricas comunes a todos los modelos de regresión."""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def report(name, y_true, y_pred):
    m = compute_metrics(y_true, y_pred)
    print(f"{name}:  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")
    return m
