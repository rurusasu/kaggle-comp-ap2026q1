# AP2026Q1

## Competition Info

- **URL:** https://www.kaggle.com/competitions/ap2026q1
- **Deadline:** 2026-06-17 23:59 UTC
- **Prize:** Kudos
- **Category:** Community

## Task

Spotify の楽曲データから `track_popularity` を予測する回帰タスク。

## Data

- `base_train.csv` — 26,266行 × 24カラム（`track_popularity` を含む）
- `base_val.csv` — 6,567行 × 23カラム（`track_popularity` なし = 予測対象）
- 特徴量: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, playlist_genre, playlist_subgenre 等

## Evaluation

- 回帰タスク（具体的な metric は Kaggle ページで確認）

## Submission Format

- Kaggle ページで確認

---

# Kaggle Competition Workspace

## Structure

- `src/config.py` — All configuration (paths, params, seed). Change settings HERE, not in other modules.
- `src/dataset.py` — Stateless data I/O. `load_train()` / `load_test()` return raw DataFrames.
- `src/features.py` — Feature engineering. Stateful transforms (fit on train only).
- `src/model.py` — Model train/predict/save/load.
- `src/evaluate.py` — CV splitter, metrics, experiment logging. Owns all writes to `logs/`.
- `src/submit.py` — Generates timestamped submission CSVs.
- `src/utils.py` — `set_seed()`, `Timer`.
- `scripts/train.py` — Training entrypoint. Runs full CV pipeline.
- `scripts/predict.py` — Inference entrypoint. Loads saved models, generates submission.

## Conventions

- Format with ruff (line-length=120, Python 3.14)
- Type hints encouraged
- Config changes go in `src/config.py` only
- Experiment logs go in `logs/` via `src/evaluate.py` only

## Commands

- `task setup` — Install deps + download data
- `task train` — Train models
- `task predict` — Generate predictions
- `task submit` — Submit to Kaggle
- `task lint` — Check code style
- `task test` — Run tests
