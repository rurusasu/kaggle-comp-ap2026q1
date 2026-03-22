import numpy as np
import pandas as pd

from src.config import Config


def _compute_freq_encoding(col: pd.Series) -> dict:
    """Compute frequency encoding map."""
    return col.value_counts().to_dict()


def build_features_stateless(df: pd.DataFrame, cfg: Config, encodings: dict) -> pd.DataFrame:
    """Build features using pre-computed encodings (no fitting, no target encoding).

    Args:
        df: Raw dataframe.
        cfg: Config.
        encodings: Dict with keys 'freq_enc', 'label_enc'.
    """
    out = df.copy()

    freq_enc = encodings["freq_enc"]
    label_enc = encodings["label_enc"]

    # --- Date features from track_album_release_date ---
    out["release_date"] = pd.to_datetime(out["track_album_release_date"], errors="coerce")
    out["release_year"] = out["release_date"].dt.year.fillna(2000).astype(int)
    out["release_month"] = out["release_date"].dt.month.fillna(1).astype(int)
    out["release_day"] = out["release_date"].dt.day.fillna(1).astype(int)
    ref_date = pd.Timestamp("2020-01-01")
    out["release_days_since_ref"] = (out["release_date"] - ref_date).dt.days.fillna(0).astype(int)
    out["release_dayofweek"] = out["release_date"].dt.dayofweek.fillna(0).astype(int)

    # --- Frequency encoding for high-cardinality categoricals ---
    freq_cols = ["track_artist", "track_album_id", "playlist_id"]
    for col in freq_cols:
        col_str = out[col].fillna("__MISSING__").astype(str)
        freq_map = freq_enc.get(col, {})
        out[f"{col}_freq"] = col_str.map(freq_map).fillna(1).astype(int)

    # Log-frequency (frequency can be very skewed)
    for col in freq_cols:
        out[f"{col}_log_freq"] = np.log1p(out[f"{col}_freq"])

    # --- Label encoding for categoricals ---
    for col in ["playlist_genre", "playlist_subgenre"]:
        enc_map = label_enc.get(col, {})
        out[f"{col}_label"] = out[col].fillna("__MISSING__").astype(str).map(enc_map).fillna(-1).astype(int)

    # Label encoding for high-cardinality (track_artist, track_album_id)
    for col in ["track_artist", "track_album_id"]:
        enc_map = label_enc.get(col, {})
        out[f"{col}_label"] = out[col].fillna("__MISSING__").astype(str).map(enc_map).fillna(-1).astype(int)

    # --- Audio feature interactions ---
    out["energy_loudness"] = out["energy"] * out["loudness"]
    out["danceability_valence"] = out["danceability"] * out["valence"]
    out["energy_danceability"] = out["energy"] * out["danceability"]
    out["acousticness_energy"] = out["acousticness"] * out["energy"]
    out["speechiness_danceability"] = out["speechiness"] * out["danceability"]
    out["tempo_energy"] = out["tempo"] * out["energy"]
    out["duration_min"] = out["duration_ms"] / 60000.0
    out["loudness_squared"] = out["loudness"] ** 2
    out["energy_squared"] = out["energy"] ** 2
    out["valence_energy"] = out["valence"] * out["energy"]
    out["instrumentalness_acousticness"] = out["instrumentalness"] * out["acousticness"]
    out["liveness_energy"] = out["liveness"] * out["energy"]
    out["tempo_danceability"] = out["tempo"] * out["danceability"]
    out["loudness_energy_ratio"] = out["loudness"] / (out["energy"] + 1e-8)

    # --- Collect feature columns ---
    feature_cols = (
        list(cfg.numeric_features)
        + ["release_year", "release_month", "release_day", "release_days_since_ref", "release_dayofweek"]
        + [f"{c}_freq" for c in freq_cols]
        + [f"{c}_log_freq" for c in freq_cols]
        + ["playlist_genre_label", "playlist_subgenre_label"]
        + ["track_artist_label", "track_album_id_label"]
        + [
            "energy_loudness",
            "danceability_valence",
            "energy_danceability",
            "acousticness_energy",
            "speechiness_danceability",
            "tempo_energy",
            "duration_min",
            "loudness_squared",
            "energy_squared",
            "valence_energy",
            "instrumentalness_acousticness",
            "liveness_energy",
            "tempo_danceability",
            "loudness_energy_ratio",
        ]
    )

    keep_cols = [cfg.id_col] + feature_cols
    if cfg.target_col in out.columns:
        keep_cols.append(cfg.target_col)

    return out[keep_cols], feature_cols


def fit_encodings(df: pd.DataFrame, cfg: Config) -> dict:
    """Fit frequency and label encodings on training data (NO target encoding).

    Returns dict with 'freq_enc', 'label_enc'.
    """
    # Frequency encodings
    freq_cols = ["track_artist", "track_album_id", "playlist_id"]
    freq_enc = {}
    for col in freq_cols:
        freq_enc[col] = _compute_freq_encoding(df[col].fillna("__MISSING__").astype(str))

    # Label encodings (deterministic mapping based on training data)
    label_enc = {}
    for col in ["playlist_genre", "playlist_subgenre", "track_artist", "track_album_id"]:
        str_vals = df[col].fillna("__MISSING__").astype(str).unique()
        unique_vals = sorted(str_vals)
        label_enc[col] = {v: i for i, v in enumerate(unique_vals)}

    return {
        "freq_enc": freq_enc,
        "label_enc": label_enc,
    }


# --- Legacy interface for backward compat ---

_cached_encodings: dict = {}


def build_features(df: pd.DataFrame, cfg: Config, is_train: bool = True) -> pd.DataFrame:
    """Legacy wrapper. For prediction pipeline only."""
    global _cached_encodings
    if is_train:
        _cached_encodings = fit_encodings(df, cfg)
    result_df, _ = build_features_stateless(df, cfg, _cached_encodings)
    return result_df
