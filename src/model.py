import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> object:
    """Train a LightGBM regressor for track_popularity prediction."""
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            __import__("lightgbm").early_stopping(50, verbose=False),
            __import__("lightgbm").log_evaluation(0),
        ],
    )
    return model


def predict(model: object, X: pd.DataFrame) -> np.ndarray:
    """Generate predictions from a trained model."""
    preds = model.predict(X)
    return np.clip(preds, 0, 100)


def save_model(model: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)
