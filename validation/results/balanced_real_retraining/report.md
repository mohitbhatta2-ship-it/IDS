# Balanced real-PCAP retraining — report

**Candidate only. Production model and Candidate 2 frozen and byte-for-byte
unchanged (verified). No auto-promotion, no merge.** The independent 36-PCAP test
set was NEVER used for training, weighting, threshold/feature selection, or
candidate selection — the candidate was selected on CIC held-out + v2 LOCO only,
then evaluated once on the independent set.

## Data (approved training corpora only)

Combined real training flows = **1272** (FTP
826, Benign
446) from **v1 + v2 + targeted benign**. The
independent set is disjoint (content-hash) and frozen. Existing 30-feature
`pcap_validation` extraction; exactly 30 ordered finite features, no zero-fill.

## Candidate 1 — CIC-only control

CIC 0.9804/0.8599 (reproduces baseline:
**True**); on all v2 real flows FTP recall 0.0000 /
benign recall 0.9563 — confirms CIC alone does not detect
real FTP.

## Candidates — CIC + real (CIC held-out + v2 LOCO; selection signals only)

| candidate | ftp_weight | benign_weight | cic_accuracy | cic_macro_f1 | v2_loco_ftp_recall | v2_loco_benign_recall | v2_loco_macro_f1 | v2_loco_accuracy |
|---|---|---|---|---|---|---|---|---|
| unweighted | 1.0000 | 1.0000 | 0.9804 | 0.8653 | 0.9359 | 0.6120 | 0.7898 | 0.8665 |
| moderate_ftp | 5.0000 | 1.0000 | 0.9810 | 0.8618 | 0.9776 | 0.5301 | 0.7931 | 0.8817 |
| balanced | 3.0000 | 5.5561 | 0.9804 | 0.8607 | 0.8808 | 0.8087 | 0.8158 | 0.8653 |

Selected on CIC+v2-LOCO: **`balanced`** (rule: no CIC regression, then best
v2-LOCO macro-F1).

## Final independent test (frozen 36-PCAP set)

| model | ftp_recall | ftp_precision | benign_recall | benign_fp_rate | macro_f1 | accuracy |
|---|---|---|---|---|---|---|
| production | 0.0000 | 0.0000 | 0.9744 | 0.0256 | 0.1230 | 0.1810 |
| candidate2 | 0.7544 | 0.9214 | 0.7179 | 0.2821 | 0.6717 | 0.7476 |
| cic_only | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.1268 | 0.1857 |
| unweighted | 0.5263 | 0.9424 | 0.8590 | 0.1410 | 0.5560 | 0.5881 |
| moderate_ftp | 0.8187 | 0.8889 | 0.5513 | 0.4487 | 0.6612 | 0.7690 |
| balanced | 0.1930 | 0.9706 | 0.9744 | 0.0256 | 0.3377 | 0.3381 |

Bootstrap 95% CIs in `bootstrap_ci_results.csv`.

### Headline — Candidate 2 vs selected (independent)

| Metric | production | Candidate 2 | selected `balanced` |
|---|---|---|---|
| FTP recall | 0.0000 | 0.7544 | 0.1930 |
| Benign recall | 0.9744 | 0.7179 | 0.9744 |
| Benign FP rate | 0.0256 | 0.2821 | 0.0256 |
| macro-F1 | 0.1230 | 0.6717 | 0.3377 |

## SHAP (selected `balanced`)

- Candidate 2 global-top5 artifact-dependent: **True**
- Selected candidate global-top5 artifact-dependent: **True**
- Still leans on `Fwd Seg Size Min` / `Init Fwd Win Byts`: see `shap_comparison.csv`.

## Verdict — DO NOT PROMOTE

selected `balanced` independent: FTP recall 0.193, benign recall 0.974, benign FP 0.026 (Cand2 0.282), macro-F1 0.338, CIC non-regressed=True. Unmet criteria: FTP recall>0.70.

Promotion criteria (all required): {"ftp_recall_gt_0.70": false, "benign_recall_gt_0.90": true, "benign_fp_much_lower_than_cand2": true, "cic_non_regressed": true}

**No candidate meets the balanced target on the independent test; at this scale/feature space the real benign and real FTP regions overlap on the CIC-artifact features, so lowering benign FP costs FTP recall. The current feature space/data is insufficient -- DO NOT promote.**

### Caveats
- Independent test small (36 captures / 420 flows), loopback-only, 2 real classes.
- Selection on CIC+v2-LOCO; independent evaluated once; CIs wide; no significance claim.

## Integrity

Production model, Candidate 2, ml.py/live_capture.py/pcap_validation.py, and the
v1/v2/independent/targeted PCAPs verified unchanged (before==after). No independent
PCAP entered training. Capture-level LOCO with per-fold no-flow-overlap assertions.

## Files

`candidate_metrics.csv`, `baseline_metrics.csv`, `candidate_comparison.csv`,
`independent_test_metrics.csv`, `independent_per_capture_metrics.csv`,
`per_capture_v2loco_metrics.csv`, `confusion_*.csv`, `confidence_distribution.csv`,
`bootstrap_ci_results.csv`, `shap_comparison.csv`, `leakage_validation.json`,
`model_hashes_before_after.json`, `training_metadata.json`, `final_verdict.json`.
