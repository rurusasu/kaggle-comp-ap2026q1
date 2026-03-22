# Next Task: AP2026Q1

## Status

- v1 (LB 0.04442) がベストスコア
- v4 (CV R² 0.4538) が未提出。日次提出制限（3回/日）のため UTC midnight 後に提出可能。

## Step 1: v4 を提出

```bash
uv run kaggle competitions submit -c ap2026q1 -f outputs/submissions/submission_20260322_175212.csv -m "v4: LightGBM+CatBoost ensemble, freq encoding, date features, interactions"
```

提出後、Public LB score を確認:
```bash
uv run kaggle competitions submissions -c ap2026q1
```

## Step 2: v4 スコア分析

v4 の LB score を確認したら CLAUDE.md の Submission History を更新。

- v4 > v1 なら、v4 ベースでさらに改善
- v4 < v1 なら、v1 に戻って別アプローチ

## Step 3: v5 改善案

### CV R² と LB score の差が大きい場合（overfitting）
- 正則化を強化（LightGBM: min_child_samples増、reg_alpha/lambda増）
- 特徴量削減（feature importance で下位を除外）
- より堅牢な CV 戦略（GroupKFold by track_id）

### LB score が改善した場合
- **XGBoost を追加**してアンサンブルを 3 モデルに
- **Optuna** でハイパーパラメータ自動チューニング
- **track_name テキスト特徴量**: 文字数、単語数、言語検出
- **track_album_release_date のビニング**: 年代ごとのグループ化

## Important Notes

- **評価指標は R²**。0.04442 は非常に低い。平均予測で R²=0。
- **データの重複**: 同じ track_id が複数 playlist に出現。GroupKFold by track_id を検討。
- **日次提出制限**: 3回/日（UTC 基準）。提出は慎重に。
- **Target encoding は使わない**: v2/v3 で leakage が確認済み。
