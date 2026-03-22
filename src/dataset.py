import pandas as pd

from src.config import Config


def load_train(cfg: Config) -> pd.DataFrame:
    path = cfg.raw_dir / "base_train.csv"
    df = pd.read_csv(path, index_col=0)
    df.index.name = cfg.id_col
    df = df.reset_index()
    return df


def load_test(cfg: Config) -> pd.DataFrame:
    path = cfg.raw_dir / "base_val.csv"
    df = pd.read_csv(path, index_col=0)
    df.index.name = cfg.id_col
    df = df.reset_index()
    return df
