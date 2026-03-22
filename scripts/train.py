"""Training entrypoint.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --seed 0 --n-folds 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import load_train
from src.evaluate import get_cv_splitter, log_experiment, metric_fn
from src.features import build_features_stateless, fit_encodings
from src.model import predict, save_model, train
from src.utils import Timer, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    cfg = Config(seed=args.seed, n_folds=args.n_folds)
    set_seed(cfg.seed)

    with Timer("load data"):
        raw_df = load_train(cfg)

    splitter = get_cv_splitter(cfg)
    fold_scores = []
    oof_preds = np.zeros(len(raw_df))
    feature_cols = None

    # Use raw_df index for splitting
    dummy_X = np.arange(len(raw_df)).reshape(-1, 1)
    y_all = raw_df[cfg.target_col].values

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(dummy_X, y_all)):
        with Timer(f"fold {fold}"):
            # Split raw data FIRST, then fit encodings only on train fold
            raw_train = raw_df.iloc[train_idx].reset_index(drop=True)
            raw_valid = raw_df.iloc[valid_idx].reset_index(drop=True)

            # Fit encodings on train fold only (no leakage!)
            encodings = fit_encodings(raw_train, cfg)

            # Apply encodings to both splits
            df_train, feature_cols = build_features_stateless(raw_train, cfg, encodings)
            df_valid, _ = build_features_stateless(raw_valid, cfg, encodings)

            X_train = df_train[feature_cols]
            y_train = df_train[cfg.target_col].values
            X_valid = df_valid[feature_cols]
            y_valid = df_valid[cfg.target_col].values

            model = train(X_train, y_train, X_valid, y_valid)
            preds = predict(model, X_valid)
            oof_preds[valid_idx] = preds

            score = metric_fn(y_valid, preds)
            fold_scores.append(score)
            print(f"Fold {fold}: R2 = {score:.4f}")

            save_model(model, cfg.models_dir / f"model_fold{fold}.pkl")

    mean_score = np.mean(fold_scores)
    print(f"\nCV Mean R2: {mean_score:.4f} (+/- {np.std(fold_scores):.4f})")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # Save OOF predictions
    cfg.oof_dir.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame({cfg.id_col: raw_df[cfg.id_col], "oof_pred": oof_preds})
    oof_df.to_csv(cfg.oof_dir / "oof_predictions.csv", index=False)

    log_experiment(
        cfg,
        {
            "experiment": f"v3_seed{cfg.seed}_folds{cfg.n_folds}",
            "seed": cfg.seed,
            "n_folds": cfg.n_folds,
            "fold_scores": fold_scores,
            "mean_score": float(mean_score),
            "n_features": len(feature_cols),
        },
    )

    # Retrain models on full data for prediction
    print("\nRetraining on full data for final models...")
    full_encodings = fit_encodings(raw_df, cfg)
    df_full, _ = build_features_stateless(raw_df, cfg, full_encodings)
    X_full = df_full[feature_cols]
    y_full = df_full[cfg.target_col].values

    # Save full-data encodings for predict.py
    import pickle
    enc_path = cfg.models_dir / "encodings.pkl"
    enc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(enc_path, "wb") as f:
        pickle.dump(full_encodings, f)
    print(f"Encodings saved to {enc_path}")

    # Train final models (one per fold seed for diversity, but here just one)
    final_model = train(X_full, y_full, X_full, y_full)
    save_model(final_model, cfg.models_dir / "model_full.pkl")
    print("Full model saved.")


if __name__ == "__main__":
    main()
