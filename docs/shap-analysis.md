# SHAP explanations & cross-dataset evaluation

Read-only analysis of the **existing saved model** (sklearn
`HistGradientBoostingClassifier`). Nothing here retrains, tunes, thresholds, or
forces predictions — SHAP only reads the model, and the cross-dataset step reuses
the Dataset Testing path (`ml.predict_batch`). `shap` is an analysis-only
dependency, lazy-imported so the web app never depends on it.

## Why real FTP brute force reads as Benign — SHAP

```bash
# from webapp_django/
python manage.py shap_report --output ../validation/results/shap --cic-sample 300
```

`shap.TreeExplainer` supports this model exactly (additivity error ~2e-14). It
attributes each prediction to the 30 features in the model's raw-margin space,
comparing two populations:

- **CIC-IDS2018 FTP-BruteForce** rows the model classifies **correctly**;
- **real captured FTP flows** the model classifies **Benign** (from
  `sample_data/real_pcap/ftp_bruteforce/*.pcap`, extracted by the live pipeline).

**Global importance (CIC FTP):** `Fwd Seg Size Min`, `Dst Port`,
`Init Fwd Win Byts` dominate the FTP-BruteForce decision.

**Mean SHAP toward Benign — real vs CIC (top drivers):**

| feature | CIC→Benign | real→Benign | Δ | CIC median | real median |
|---|---|---|---|---|---|
| Fwd Seg Size Min | −1.48 | +0.73 | **+2.21** | 40 | 20 |
| Init Fwd Win Byts | −0.83 | +0.80 | **+1.63** | 26883 | 8134 |
| Dst Port | −1.31 | −0.15 | +1.16 | 21 | 21 |

**Interpretation.** The model learned CIC-specific *artifact* values as the
FTP-BruteForce signature: `Fwd Seg Size Min = 40` (options on every forward
packet) and `Init Fwd Win Byts = 26883`. Real FTP traffic carries ordinary TCP
values (`Fwd Seg Size Min = 20`, `Init Fwd Win Byts = 8134/65535`), so those same
features now push the flow **toward Benign**. This is a train/serve distribution
mismatch on the very features the model relies on — not a live-extraction bug
(feature parity was independently verified).

Outputs: `validation/results/shap/` — `global_importance_*.csv`,
`cic_vs_real_toward_benign.csv`, `sample_*_top_features.csv`, `summary.json`.

## Cross-dataset evaluation (same Dataset Testing path)

```bash
python manage.py cross_dataset_eval --input ../sample_data --output ../validation/results/cross_dataset
```

Runs labelled datasets already in the repo through `ml.predict_batch`:

| Dataset | Rows | Accuracy | Macro-F1 |
|---|---|---|---|
| CIC-IDS**2017**-style (`foreign_dataset_ids2017_style.csv`) | 3,561 | **0.896** | 0.871 |
| Aggregate over per-attack CSVs (2018) | 14,209 | 0.909 | 0.902 |
| CIC-IDS2018 held-out baseline | 40,000 | 0.980 | 0.883 |

The decisive contrast for FTP-BruteForce:

| Source | FTP-BruteForce recall |
|---|---|
| CIC-shaped CSV (`ftp_bruteforce.csv`) | **0.885** |
| **Real PCAP capture** | **0.000** |

Same model, same class: it detects FTP-BruteForce on CIC-shaped data but not on
real captures — exactly what SHAP predicts. The model generalises across
CICFlowMeter-derived datasets (2017-style: 0.90) but not to real packet captures
of the artifact-defined classes.

Outputs: `validation/results/cross_dataset/` — `per_dataset.csv`,
`aggregate_per_class.csv`, `confusion_matrix.csv`, `summary.json`.

## Does retraining appear justified?

Yes — the evidence (SHAP + real-PCAP baseline + cross-dataset) shows the failure
is concentrated on classes whose CIC features are capture artifacts (FTP-BruteForce
especially), while the model is otherwise healthy. Realistic-traffic
retraining/augmentation, using features extracted by this same live pipeline, is
the principled next step. **No retraining is done here** — this is the evidence to
decide on.
