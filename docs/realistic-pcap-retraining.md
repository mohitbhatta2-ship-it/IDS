# Realistic-PCAP retraining experiment

**Experimental candidate only — the production/baseline model is never touched.**
This branch (`claude/realistic-pcap-retraining`) explores whether adding
realistically captured FTP traffic to the CIC training distribution improves the
model's real-world FTP-BruteForce generalisation, without regressing CIC
performance.

> **Scientific caveat, stated up front.** Only **5 real PCAPs** are available
> (3 FTP brute-force, 2 benign; **42 flows**, 34 FTP + 8 Benign). This is
> **exploratory evidence only** — not a statistically strong dataset. We do not
> oversample or synthesise flows to make it look larger, and any result obtained
> by training and testing on the same capture is explicitly rejected.

## Motivation & baseline evidence

The frozen `claude/real-pcap-validation` baseline established: real FTP
brute-force flows are all predicted **Benign** (real-PCAP accuracy 0.190,
macro-F1 0.160, **FTP recall 0.000**) while CIC held-out is healthy (0.980 /
0.883). SHAP showed the model relies on CIC-specific artifact features
(`Fwd Seg Size Min=40`, `Init Fwd Win Byts=26883`); cross-dataset evaluation
agreed. This experiment asks: does adding real FTP flows change that?

## Data used & labels

Real flows come **only** from the committed captures, labelled by folder
(ground truth, never the model's prediction):

| Capture | Label | Flows |
|---|---|---|
| `benign/benign_01.pcap` | Benign | 2 |
| `benign/benign_03.pcap` | Benign | 6 |
| `ftp_bruteforce/ftp_01.pcap` | FTP-BruteForce | 3 |
| `ftp_bruteforce/ftp_02.pcap` | FTP-BruteForce | 15 |
| `ftp_bruteforce/ftp_03.pcap` | FTP-BruteForce | 16 |

## Extraction pipeline & feature parity

Real flows are extracted through the **existing validated Live-Capture path**
(`pcap_validation.replay_pcap` → the `CaptureSession` flow engine →
`calculate_features`). No second feature implementation. Every flow is validated
to carry **exactly the 30 `ml.FEATURES`, all finite**; a flow missing any feature
is reported and **excluded**, never zero-filled (0 invalid here). Feature order
matches `ml.FEATURES`.

## Model & "only the data changes"

Candidates are `HistGradientBoostingClassifier` trained with the **same tuned
hyperparameters** as the baseline (`HistGradientBoosting_Tuned_best_params.json`,
`hgb_` prefix stripped) and the **same 30 features**, seed 42. A from-scratch
retrain on CIC alone reproduces the baseline exactly (CIC 0.9803 / 0.8627), so
comparing `CIC` vs `CIC + real` is a clean A/B where only the training data
differs.

## Candidates

- **Baseline** — the frozen saved model, unchanged.
- **Control** — retrain on CIC only (isolates the training pipeline; must match baseline).
- **Candidate B** — CIC + real, **unweighted**. (31 real FTP flows against ~7,968
  CIC FTP rows are numerically negligible, so this is expected to be inert.)
- **Candidate C** — CIC + real, real-FTP **sample-weighted ×250** so the real FTP
  mass ≈ the CIC FTP class support (~7,968). This is a documented, reproducible
  `sample_weight` (not row duplication). Candidate C is the saved artifact.

## Group-aware split (leakage control)

Evaluation is **leave-one-capture-out (LOCO)**: for each real capture, the
candidate is trained on CIC + all *other* real captures and tested on the
held-out capture. **No PCAP appears in both train and test.** The held-out
capture's flows give the uncontaminated real-world estimate. Self-scoring the
training flows would be contaminated and is **not** reported. With only 3 FTP and
2 benign captures, each held-out fold is tiny — reported per-fold and pooled, with
the limitation stated.

## Evaluation protocol & metrics

Every candidate is measured on **CIC held-out** (regression check) and, via LOCO,
on **held-out real captures**. Reported: accuracy, macro precision/recall/F1,
weighted-F1, confusion matrix, per-class P/R/F1/support, FTP-BruteForce recall,
Benign recall, prediction distribution, confidence. The headline comparison is
**baseline real FTP recall vs candidate (LOCO) real FTP recall**, alongside the
CIC regression.

## Reproduce

```bash
# from webapp_django/
python manage.py retrain_experiment --output ../validation/results/retraining
```

Evidence is written to `validation/results/retraining/` (see §Evidence). The
candidate artifact is saved to `validation/models/realistic_pcap_candidate/`
(`model.pkl`, `metadata.json`) — **never** to `webapp_data/Results/Models/`.

## Results

All candidates use seed 42, tuned HGB params, the 30 `ml.FEATURES`. Real-world
numbers are **leave-one-capture-out (held-out)** — never self-scored.

**Baseline vs Candidate C (CIC + real, FTP ×250):**

| Metric | Baseline | Candidate C | Change |
|---|---|---|---|
| CIC accuracy | 0.9803 | 0.9811 | +0.0008 |
| CIC macro-F1 | 0.8627 | 0.8733 | +0.0106 |
| Real-PCAP accuracy (LOCO) | 0.1905 | 0.9048 | +0.7143 |
| Real-PCAP macro-F1 (LOCO) | 0.1600 | 0.8586 | +0.6986 |
| **FTP-BruteForce recall** | **0.000** | **0.912** | **+0.912** |
| Benign recall | 1.000 | 0.875 | −0.125 |

**Control** (retrain on CIC only) reproduces the baseline (CIC 0.9804 / 0.8599),
confirming the training pipeline is not the variable.

**Candidate B** (CIC + real, *unweighted*): even without upweighting, LOCO FTP
recall rises to **0.882** (30/34) with CIC unchanged (0.9807 / 0.8601) — adding
the flows at all is what helps; the ×250 weight only nudges recall further.

**Per-capture leave-one-out (Candidate C):**

| Held-out capture | Label | Flows | Recall | Predictions |
|---|---|---|---|---|
| benign_01 | Benign | 2 | 1.000 | 2 Benign |
| benign_03 | Benign | 6 | 0.833 | 5 Benign, **1 FTP (false positive)** |
| ftp_01 | FTP-BruteForce | 3 | **0.000** | 3 Benign |
| ftp_02 | FTP-BruteForce | 15 | 1.000 | 15 FTP |
| ftp_03 | FTP-BruteForce | 16 | 1.000 | 16 FTP |

**SHAP (baseline vs Candidate C).** The candidate **still relies on the same
features** — `Fwd Seg Size Min` importance only falls 0.738 → 0.681 and
`Init Fwd Win Byts` actually rises. It did not abandon the artifact features; it
learned to *also* map the real region (`Fwd Seg Size Min=20`,
`Init Fwd Win Byts=8134`) to FTP. **Feature distribution** confirms the two
regions remain distinct (CIC FTP `Fwd Seg Size Min=40` vs real `20`;
`Init Fwd Win Byts` 26883 vs 8134).

### Did the candidate improve real FTP detection? — Yes, but read the caveats.

- **Real FTP recall 0.000 → 0.912 (LOCO), with no CIC regression** (CIC macro-F1
  actually +0.0106). That is a large, uncontaminated improvement.
- **But it is not robust generalisation.** Held-out `ftp_01` (3 flows) still
  scores 0.000 — 2 of 3 FTP captures transfer, one does not. The gain comes from
  the model memorising the narrow real-capture region (SHAP shows the same
  feature dependence), so it may not transfer to different FTP tools, OSes, or
  networks.
- **Small cost:** Benign recall 1.000 → 0.875 (one benign flow flagged FTP).

### Did CIC performance regress? — No.

CIC accuracy 0.9803 → 0.9811 and macro-F1 0.8627 → 0.8733 (both slightly up).

### Is the evidence strong enough to replace the baseline? — No.

With only **5 captures / 42 flows**, one held-out FTP capture failing, a benign
false positive, and SHAP showing unchanged feature dependence, this is a
**promising exploratory signal, not production-grade evidence**. The candidate
should **remain experimental**. A production-validation phase would need many
more diverse real captures (multiple tools/OSes/networks) and a proper held-out
test set before replacing the baseline.

## Limitations

- **5 PCAPs / 42 flows** — exploratory only; per-fold held-out sets are 2–16
  flows. Confidence intervals would be very wide; we report point estimates and
  refuse to manufacture confidence.
- Only FTP-BruteForce and Benign are represented in the real captures; the other
  13 classes are evaluated on CIC only.
- LOCO trains a fresh model per fold, so the "candidate" real-world number is an
  average over small held-out captures, not a single fixed model's score on a
  large independent set.

## Evidence files (`validation/results/retraining/`)

`experiment_summary.csv`, `baseline_vs_candidate.csv`, `real_pcap_metrics.csv`,
`cic_metrics.csv`, `per_class_metrics.csv`, `confusion_matrix_baseline.csv`,
`confusion_matrix_candidate.csv`, `capture_split_manifest.csv`,
`training_manifest.csv`, `feature_distribution_comparison.csv`,
`prediction_distribution.csv`, `confidence_comparison.csv`,
`shap_comparison.csv`, `experiment_metadata.json` (seeds, params, sources, split
strategy, feature list).
