"""Inference entrypoint. Loads saved models and generates predictions.

Usage:
    uv run python scripts/predict.py
    uv run python scripts/predict.py --model-dir outputs/models
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import load_test, load_train
from src.features import build_features_stateless, fit_encodings
from src.model import load_model, predict
from src.submit import create_submission
from src.utils import Timer, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(seed=args.seed)
    set_seed(cfg.seed)
    model_dir = Path(args.model_dir) if args.model_dir else cfg.models_dir

    # Load pre-computed encodings (saved by train.py)
    enc_path = model_dir / "encodings.pkl"
    if enc_path.exists():
        print(f"Loading encodings from {enc_path}")
        with open(enc_path, "rb") as f:
            encodings = pickle.load(f)
    else:
        # Fallback: fit encodings on train data
        print("No saved encodings found, fitting on train data...")
        with Timer("load train (for encoders)"):
            train_df = load_train(cfg)
            encodings = fit_encodings(train_df, cfg)

    with Timer("load test data"):
        df = load_test(cfg)

    with Timer("build features"):
        df, feature_cols = build_features_stateless(df, cfg, encodings)

    # Load models and ensemble
    # Prefer full model if available
    full_model_path = model_dir / "model_full.pkl"
    fold_model_paths = sorted(model_dir.glob("model_fold*.pkl"))

    if full_model_path.exists():
        print(f"Using full model: {full_model_path}")
        model = load_model(full_model_path)
        ensemble_preds = predict(model, df[feature_cols])
    elif fold_model_paths:
        print(f"Ensembling {len(fold_model_paths)} fold models")
        all_preds = []
        for path in fold_model_paths:
            model = load_model(path)
            preds = predict(model, df[feature_cols])
            all_preds.append(preds)
        ensemble_preds = np.mean(all_preds, axis=0)
    else:
        print(f"No models found in {model_dir}")
        sys.exit(1)

    submission_path = create_submission(
        cfg,
        df[cfg.id_col].tolist(),
        ensemble_preds.tolist(),
        id_col=cfg.id_col,
        target_col=cfg.target_col,
    )
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
