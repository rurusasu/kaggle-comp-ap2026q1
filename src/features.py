import numpy as np
import pandas as pd

from src.config import Config

# Module-level state for stateful transforms
_target_encodings: dict[str, dict] = {}
_freq_encodings: dict[str, dict] = {}
_global_target_mean: float = 0.0


def build_features(df: pd.DataFrame, cfg: Config, is_train: bool = True) -> pd.DataFrame:
    """Build features from raw DataFrame.

    v2: target encoding, frequency encoding, date features, interactions.
    Stateful transforms are fit only when is_train=True.
    """
    global _target_encodings, _freq_encodings, _global_target_mean
    out = df.copy()

    # --- Date features from track_album_release_date ---
    out["release_date"] = pd.to_datetime(out["track_album_release_date"], errors="coerce")
    out["release_year"] = out["release_date"].dt.year.fillna(2000).astype(int)
    out["release_month"] = out["release_date"].dt.month.fillna(1).astype(int)
    out["release_day"] = out["release_date"].dt.day.fillna(1).astype(int)
    # Days since a reference date (2020-01-01)
    ref_date = pd.Timestamp("2020-01-01")
    out["release_days_since_ref"] = (out["release_date"] - ref_date).dt.days.fillna(0).astype(int)

    # --- Frequency encoding for high-cardinality categoricals ---
    freq_cols = ["track_artist", "track_album_id", "track_id", "playlist_id"]
    for col in freq_cols:
        col_str = out[col].astype(str)
        if is_train:
            freq_map = col_str.value_counts().to_dict()
            _freq_encodings[col] = freq_map
        else:
            freq_map = _freq_encodings[col]
        out[f"{col}_freq"] = col_str.map(freq_map).fillna(1).astype(int)

    # --- Target encoding for categoricals (with smoothing) ---
    target_enc_cols = ["playlist_genre", "playlist_subgenre", "track_artist", "track_album_id"]
    if is_train:
        _global_target_mean = out[cfg.target_col].mean()

    smoothing = 10  # smoothing factor
    for col in target_enc_cols:
        col_str = out[col].astype(str)
        if is_train:
            stats = out.groupby(col_str)[cfg.target_col].agg(["mean", "count"])
            # Smoothed target encoding: (count * mean + smoothing * global_mean) / (count + smoothing)
            smooth_mean = (stats["count"] * stats["mean"] + smoothing * _global_target_mean) / (
                stats["count"] + smoothing
            )
            _target_encodings[col] = smooth_mean.to_dict()
        enc_map = _target_encodings[col]
        out[f"{col}_target_enc"] = col_str.map(enc_map).fillna(_global_target_mean)

    # --- Label encoding for genre/subgenre (keep as additional feature) ---
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

    return out[keep_cols]
