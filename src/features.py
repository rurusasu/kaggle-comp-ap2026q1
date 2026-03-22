import numpy as np
import pandas as pd

from src.config import Config

# Smoothing factor for target encoding
SMOOTHING = 10


def _compute_target_encoding(
    train_col: pd.Series, target: pd.Series, global_mean: float
) -> dict:
    """Compute smoothed target encoding map from training data."""
    stats = pd.DataFrame({"col": train_col, "target": target}).groupby("col")["target"].agg(["mean", "count"])
    smooth_mean = (stats["count"] * stats["mean"] + SMOOTHING * global_mean) / (stats["count"] + SMOOTHING)
    return smooth_mean.to_dict()


def _compute_freq_encoding(col: pd.Series) -> dict:
    """Compute frequency encoding map."""
    return col.value_counts().to_dict()


def build_features_stateless(df: pd.DataFrame, cfg: Config, encodings: dict) -> pd.DataFrame:
    """Build features using pre-computed encodings (no fitting).

    Args:
        df: Raw dataframe.
        cfg: Config.
        encodings: Dict with keys 'target_enc', 'freq_enc', 'global_mean'.
    """
    out = df.copy()

    target_enc = encodings["target_enc"]
    freq_enc = encodings["freq_enc"]
    global_mean = encodings["global_mean"]

    # --- Date features from track_album_release_date ---
    out["release_date"] = pd.to_datetime(out["track_album_release_date"], errors="coerce")
    out["release_year"] = out["release_date"].dt.year.fillna(2000).astype(int)
    out["release_month"] = out["release_date"].dt.month.fillna(1).astype(int)
    out["release_day"] = out["release_date"].dt.day.fillna(1).astype(int)
    ref_date = pd.Timestamp("2020-01-01")
    out["release_days_since_ref"] = (out["release_date"] - ref_date).dt.days.fillna(0).astype(int)

    # --- Frequency encoding for high-cardinality categoricals ---
    # NOTE: removed track_id from freq encoding (too leaky, maps directly to target)
    freq_cols = ["track_artist", "track_album_id", "playlist_id"]
    for col in freq_cols:
        col_str = out[col].astype(str)
        freq_map = freq_enc.get(col, {})
        out[f"{col}_freq"] = col_str.map(freq_map).fillna(1).astype(int)

    # --- Target encoding for categoricals ---
    # NOTE: removed track_id target encoding (pure leakage)
    # NOTE: removed track_album_id target encoding (high leakage risk with sparse categories)
    target_enc_cols = ["playlist_genre", "playlist_subgenre", "track_artist"]
    for col in target_enc_cols:
        col_str = out[col].astype(str)
        enc_map = target_enc.get(col, {})
        out[f"{col}_target_enc"] = col_str.map(enc_map).fillna(global_mean)

    # --- Label encoding for genre/subgenre ---
    for col in ["playlist_genre", "playlist_subgenre"]:
        out[f"{col}_label"] = out[col].astype("category").cat.codes

    # --- Audio feature interactions ---
    out["energy_loudness"] = out["energy"] * out["loudness"]
    out["danceability_valence"] = out["danceability"] * out["valence"]
    out["energy_danceability"] = out["energy"] * out["danceability"]
    out["acousticness_energy"] = out["acousticness"] * out["energy"]
    out["speechiness_danceability"] = out["speechiness"] * out["danceability"]
    out["tempo_energy"] = out["tempo"] * out["energy"]
    out["duration_min"] = out["duration_ms"] / 60000.0
    out["loudness_squared"] = out["loudness"] ** 2

    # --- Collect feature columns ---
    feature_cols = (
        list(cfg.numeric_features)
        + ["release_year", "release_month", "release_day", "release_days_since_ref"]
        + [f"{c}_freq" for c in freq_cols]
        + [f"{c}_target_enc" for c in target_enc_cols]
        + ["playlist_genre_label", "playlist_subgenre_label"]
        + [
            "energy_loudness",
            "danceability_valence",
            "energy_danceability",
            "acousticness_energy",
            "speechiness_danceability",
            "tempo_energy",
            "duration_min",
            "loudness_squared",
        ]
    )

    keep_cols = [cfg.id_col] + feature_cols
    if cfg.target_col in out.columns:
        keep_cols.append(cfg.target_col)

    return out[keep_cols], feature_cols


def fit_encodings(df: pd.DataFrame, cfg: Config) -> dict:
    """Fit target and frequency encodings on training data.

    Returns dict with 'target_enc', 'freq_enc', 'global_mean'.
    """
    global_mean = df[cfg.target_col].mean()

    # Frequency encodings
    freq_cols = ["track_artist", "track_album_id", "playlist_id"]
    freq_enc = {}
    for col in freq_cols:
        freq_enc[col] = _compute_freq_encoding(df[col].astype(str))

    # Target encodings
    target_enc_cols = ["playlist_genre", "playlist_subgenre", "track_artist"]
    target_enc = {}
    for col in target_enc_cols:
        target_enc[col] = _compute_target_encoding(df[col].astype(str), df[cfg.target_col], global_mean)

    return {
        "target_enc": target_enc,
        "freq_enc": freq_enc,
        "global_mean": global_mean,
    }


# --- Legacy interface for backward compat (used by predict.py) ---

_cached_encodings: dict = {}


def build_features(df: pd.DataFrame, cfg: Config, is_train: bool = True) -> pd.DataFrame:
    """Legacy wrapper. For prediction pipeline only."""
    global _cached_encodings
    if is_train:
        _cached_encodings = fit_encodings(df, cfg)
    result_df, _ = build_features_stateless(df, cfg, _cached_encodings)
    return result_df
