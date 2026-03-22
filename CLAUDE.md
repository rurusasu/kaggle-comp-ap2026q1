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

## Submission History

| Version | Score | Method | Notes |
|---|---|---|---|
| v1 | 0.04442 | LightGBM regressor, CV R²=0.2420 | Baseline, label-encoded genres |

## Lessons Learned

1. **CV R² と Public Score の乖離**: ローカル CV R²=0.2420 に対し public score=0.04442。評価指標が R² ではなく別の metric（RMSE 等）の可能性が高い。docs/evaluation.md を再確認すべき。
2. **特徴量の余地**: playlist_genre/subgenre の label encoding は情報損失が大きい。target encoding や frequency encoding を試す価値あり。
3. **track_popularity の分布**: 0-100 の範囲で偏りがある可能性。分布を確認してから前処理を検討すべき。

### Next Steps

- 評価指標を正確に確認（RMSE? MAE? R²?）
- target encoding for categorical features
- track_artist の frequency/target encoding
- track_album_release_date から年/月を抽出
- LightGBM のハイパーパラメータチューニング

## Documentation

**IMPORTANT: Before starting any implementation work, you MUST read the relevant docs first.**

- [docs/overview.md](docs/overview.md) — Competition description, goal, background
- [docs/evaluation.md](docs/evaluation.md) — Evaluation metric, scoring methodology
- [docs/submission.md](docs/submission.md) — Submission format, file structure, requirements
- [docs/timeline.md](docs/timeline.md) — Important dates and deadlines
- [docs/rules.md](docs/rules.md) — Full competition rules
- [docs/prizes.md](docs/prizes.md) — Prize structure

### Required Reading Order

1. Before EDA or feature engineering → read `overview.md` and `evaluation.md`
2. Before building submission pipeline → read `submission.md`
3. Before using external data or models → read `rules.md`
4. Before final submission → read `timeline.md` to confirm deadlines

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
