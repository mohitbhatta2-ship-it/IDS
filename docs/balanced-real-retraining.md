# Balanced real-PCAP retraining experiment

**Candidate only. Production model and Candidate 2 are frozen and byte-for-byte
unchanged (verified before == after). No candidate is promoted, and nothing is
merged.** Candidate artifacts live under `validation/models/balanced-real-candidate/`;
evidence under `validation/results/balanced_real_retraining/`.

## Question

Can a **larger, class-balanced** set of real FTP-BruteForce + real Benign flows
(v1 + v2 + targeted benign, 1272 flows) added to CIC produce a model that detects
real FTP attacks **while keeping benign false positives acceptable**, on the FROZEN
independent 36-PCAP test set?

## Data separation (strict)

- **Training real sources:** v1 + v2 + targeted benign — **1272 flows (826 FTP,
  446 Benign)**, 0 invalid. Existing 30-feature `pcap_validation` extraction;
  exactly 30 ordered finite features, no zero-fill.
- **v2 leave-one-capture-out:** the real-PCAP validation estimate (each v2 capture
  held out; v1 + targeted always training).
- **Independent 36-PCAP test set:** FROZEN — never used for training, weighting,
  threshold/feature selection, or **candidate selection** (content-hash disjoint
  from the training corpora; a guard aborts otherwise). The primary candidate was
  selected on **CIC held-out + v2 LOCO only**, then evaluated once on the
  independent set. The production decision threshold was not changed and no
  heuristic was added.

Same tuned HGB family/seed as every prior experiment. All candidate models are
stored outside `webapp_data/Results/Models/`.

## Candidates

| # | Candidate | Weighting |
|---|---|---|
| 1 | CIC-only control | — (must reproduce baseline) |
| 2 | CIC + real, unweighted | ftp×1, benign×1 |
| 3 | CIC + real, moderate FTP | ftp×5, benign×1 |
| 4 | CIC + real, balanced | ftp×3, benign×5.56 → equal real-class mass |

## Results — CIC held-out + v2 LOCO (selection signals only)

| Candidate | CIC acc / macro-F1 | v2-LOCO FTP recall | v2-LOCO Benign recall | v2-LOCO macro-F1 |
|---|---|---|---|---|
| CIC-only control | 0.9804 / 0.8599 (reproduces) | 0.000 (all v2) | 0.956 (all v2) | — |
| unweighted | 0.9804 / 0.8653 | 0.936 | 0.612 | 0.790 |
| moderate_ftp | 0.9810 / 0.8618 | 0.978 | 0.530 | 0.793 |
| balanced | 0.9804 / 0.8607 | 0.881 | 0.809 | **0.816** |

No CIC regression for any candidate. The **CIC-only control detects no real FTP**
(recall 0.000) — CIC alone cannot do this task. Selected on best v2-LOCO macro-F1:
**`balanced`**.

## Final independent test (frozen 36-PCAP set) — the truth

| Model | FTP recall | Benign recall | Benign FP rate | macro-F1 | accuracy |
|---|---|---|---|---|---|
| production | 0.000 | 0.974 | 0.026 | 0.123 | 0.181 |
| Candidate 2 | 0.754 | 0.718 | 0.282 | 0.672 | 0.748 |
| CIC-only control | 0.000 | 1.000 | 0.000 | 0.127 | 0.186 |
| unweighted | 0.526 | 0.859 | 0.141 | 0.556 | 0.588 |
| **moderate_ftp** | **0.819** | 0.551 | 0.449 | 0.661 | 0.769 |
| **balanced (selected)** | 0.193 | **0.974** | **0.026** | 0.338 | 0.338 |

Capture-level bootstrap 95% CIs in `bootstrap_ci_results.csv`.

### The definitive finding: a coupled trade-off frontier

The four CIC+real candidates map out a **trade-off frontier**, not a solution:

- **moderate_ftp** reaches **FTP recall 0.819 (>0.70 ✓)** but benign recall
  collapses to 0.551 and benign FP explodes to **0.449**.
- **balanced** drives benign FP down to **0.026** and benign recall to **0.974
  (>0.90 ✓)**, but FTP recall collapses to **0.193**.
- **unweighted** and **Candidate 2** sit between these corners.

**No candidate achieves FTP recall >0.70 AND benign recall >0.90 at the same
time.** The two ends satisfy one target each and badly miss the other. This is the
strongest evidence yet that the benign FP rate and the FTP recall are **coupled**:
they cannot be independently improved by re-weighting the same data in the same
feature space.

### Why (SHAP)

The selected candidate's global top features remain **`Fwd Seg Size Min`,
`Init Fwd Win Byts`, Dst Port** — `selected_artifact_dependent = true`. Every
candidate still separates FTP from Benign using the **CIC-specific artifact
features**, and (from `final_robustness`) real benign and real FTP **overlap** on
exactly those features. So any decision boundary drawn there trades one class's
errors for the other's — precisely the frontier observed.

## Verdict — DO NOT PROMOTE

The selected candidate (`balanced`) fails the balanced target: **FTP recall 0.193
< 0.70**. Reviewing all candidates, none simultaneously satisfies the promotion
criteria:

| Criterion | Required | best candidate meeting it |
|---|---|---|
| FTP recall | >0.70 | moderate_ftp (0.819) |
| Benign recall | >0.90 | balanced (0.974) |
| Benign FP ≪ Candidate 2 (0.282) | yes | balanced (0.026) |
| No CIC regression | yes | all |

The FTP-recall and benign-recall criteria are met by **different, opposite**
candidates. **The current feature space / data is insufficient** to produce a model
that is simultaneously a good real-FTP detector and a low-false-positive benign
classifier. **DO NOT promote anything.** `final_verdict.json → promote: false`.

### What would be needed (honest)

Because the limitation is the **feature space** (real benign and real FTP overlap
on the CIC artifacts), more real data of the same 30-feature shape will keep
landing on this frontier. Breaking it likely requires either features that
separate the two real regions (beyond the CIC-artifact set), or genuinely
different real FTP traffic whose signature does not coincide with benign on those
features — not just more benign or more weighting.

## Reproduce

```bash
# from webapp_django/
python manage.py balanced_real_retrain
python manage.py test predictor.tests.BalancedRetrainModuleTests \
                      predictor.tests.BalancedRetrainResultsTests
```

## Integrity

Production model, Candidate 2, `ml.py` / `live_capture.py` / `pcap_validation.py`,
and the v1/v2/independent/targeted PCAPs verified unchanged (before == after). No
independent PCAP entered training. Capture-level LOCO with per-fold no-flow-overlap
assertions. Candidate models stored outside `webapp_data/`. Reproducibility test
confirms same seed + data → identical predictions.

## Files (`validation/results/balanced_real_retraining/`)

`candidate_metrics.csv`, `baseline_metrics.csv`, `candidate_comparison.csv`,
`independent_test_metrics.csv`, `independent_per_capture_metrics.csv`,
`per_capture_v2loco_metrics.csv`, `confusion_*.csv`, `confidence_distribution.csv`,
`bootstrap_ci_results.csv`, `shap_comparison.csv`, `leakage_validation.json`,
`model_hashes_before_after.json`, `training_metadata.json`, `final_verdict.json`.
