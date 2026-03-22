import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    seed: int = 42,
) -> object:
    """Train a LightGBM regressor."""
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=-1,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=5.0,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            __import__("lightgbm").early_stopping(200, verbose=False),
            __import__("lightgbm").log_evaluation(0),
        ],
    )
    return model


def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    seed: int = 42,
) -> object:
    """Train a CatBoost regressor."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=0,
        early_stopping_rounds=200,
        task_type="CPU",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
    )
    return model


def train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    seed: int = 42,
) -> dict:
    """Train LightGBM + CatBoost ensemble. Returns dict of models."""
    lgbm_model = train_lgbm(X_train, y_train, X_valid, y_valid, seed=seed)
    catboost_model = train_catboost(X_train, y_train, X_valid, y_valid, seed=seed)
    return {"lgbm": lgbm_model, "catboost": catboost_model}


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Generate predictions from a trained model (single or ensemble dict)."""
    if isinstance(model, dict):
        preds_lgbm = model["lgbm"].predict(X)
        preds_catboost = model["catboost"].predict(X)
        preds = 0.5 * preds_lgbm + 0.5 * preds_catboost
    else:
        preds = model.predict(X)
    return np.clip(preds, 0, 100)


def save_model(model: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)
