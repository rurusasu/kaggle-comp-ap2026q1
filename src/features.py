import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import Config

# Module-level encoders so they persist between train and test calls
_label_encoders: dict[str, LabelEncoder] = {}


def build_features(df: pd.DataFrame, cfg: Config, is_train: bool = True) -> pd.DataFrame:
    """Build features from raw DataFrame.

    Encodes categorical columns and selects numeric audio features.
    Stateful transforms (label encoders) are fit only when is_train=True.
    """
    global _label_encoders
    out = df.copy()

    for col in cfg.categorical_features:
        if is_train:
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str))
            _label_encoders[col] = le
        else:
            le = _label_encoders[col]
            # Handle unseen labels by mapping to -1
            mapping = {v: i for i, v in enumerate(le.classes_)}
            out[col] = out[col].astype(str).map(mapping).fillna(-1).astype(int)

    feature_cols = list(cfg.numeric_features) + list(cfg.categorical_features)
    keep_cols = [cfg.id_col] + feature_cols
    if cfg.target_col in out.columns:
        keep_cols.append(cfg.target_col)

    return out[keep_cols]
