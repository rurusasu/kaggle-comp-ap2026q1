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

- **R² (R-squared)** — docs/evaluation.md で確認済み
- Public score も R² で評価される

## Submission Format

- CSV: `ID,track_popularity`

## Submission History

| Version | Score | Method | Notes |
|---|---|---|---|
| v1 | 0.04442 | LightGBM regressor, CV R²=0.2420 | Baseline, label-encoded genres |
| v2 | 0.02135 | LightGBM regressor, CV R²=0.9763 | Target/freq encoding, date features, interactions, tuned params. LEAKAGE: target enc fit on full data before CV |
| v3 | TBD | LightGBM regressor, CV R²=0.4307 | Fold-internal target encoding, removed track_id/track_album_id target enc, full-data retrain for submission |

## Lessons Learned

1. **評価指標は R²**: docs/evaluation.md で確認。Public score=0.04442 は R² 値（非常に低い）。
2. **Target encoding が極めて有効**: track_artist, track_album_id の target encoding で R² が 0.24 → 0.97 に大幅改善。同じ artist/album の楽曲は popularity が類似するため。
3. **データの重複**: 26,266行中 22,995 unique track_id。同じ曲が異なる playlist に登場するため重複あり。
4. **特徴量の相関**: 個別の audio feature と target の相関は低い（最大 |r|=0.12 for duration_ms）。カテゴリカル変数の target encoding が鍵。
5. **CV leakage の注意**: target encoding を全学習データで fit してから CV すると楽観的バイアスが入る。Public LB で実際の性能を確認する必要あり。

6. **Target encoding leakage 確認済み**: v2 で全データ fit → CV R²=0.97 だが LB=0.02135。v3 で fold-internal TE に修正 → CV R²=0.43（現実的な値）。
7. **track_id target encoding は純粋な leakage**: 同じ track_id は同じ popularity を持つため、CV で完全にリーク。削除済み。

### Next Steps

- v3 の Public LB score を確認
- feature importance を確認して不要な特徴量を削減
- track_album_id target encoding の復活を検討（fold-internal なら安全）
- XGBoost や CatBoost のアンサンブルを検討

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
