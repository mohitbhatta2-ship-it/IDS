# Targeted-benign retraining — report

**Candidate only. Production model and Candidate 2 are frozen and byte-for-byte
unchanged (verified). No promotion by default, no merge.** The independent 36-PCAP
test set was NEVER used for training, weighting, hyperparameter/threshold
selection, or candidate selection — the primary candidate was selected using CIC
held-out + v2 LOCO only, then evaluated once on the independent set.

## Data roles

- **Training:** CIC `balanced_train_selected` + v2 real (83 captures, weight 1) +
  **41 targeted benign captures** (weight w_b).
- **v2 LOCO:** held-out v2 capture per fold (targeted benign always training
  augmentation) — the real-PCAP validation estimate.
- **Independent test:** `validation/independent_real_pcaps/` — FROZEN, evaluated
  once.

## Candidates (CIC held-out + v2 LOCO; selection signals only)

| candidate | benign_weight | cic_accuracy | cic_macro_f1 | v2_loco_accuracy | v2_loco_macro_f1 | v2_loco_ftp_recall | v2_loco_benign_recall |
|---|---|---|---|---|---|---|---|
| targeted_none | 1.0000 | 0.9803 | 0.8603 | 0.8981 | 0.8536 | 0.9225 | 0.8087 |
| targeted_moderate | 5.0000 | 0.9797 | 0.8545 | 0.8888 | 0.8407 | 0.9151 | 0.7923 |
| targeted_strong | 15.0000 | 0.9801 | 0.8637 | 0.8970 | 0.8578 | 0.9046 | 0.8689 |

Primary candidate selected on CIC+v2-LOCO: **`targeted_strong`** (rule: no CIC regression,
v2-LOCO FTP recall ≥0.85, then max v2-LOCO benign recall).

## Final independent test (frozen 36-PCAP set)

| model | ftp_recall | ftp_precision | benign_recall | benign_fp_rate | macro_f1 | accuracy |
|---|---|---|---|---|---|---|
| production | 0.0000 | 0.0000 | 0.9744 | 0.0256 | 0.1230 | 0.1810 |
| candidate2 | 0.7544 | 0.9214 | 0.7179 | 0.2821 | 0.6717 | 0.7476 |
| targeted_none | 0.5556 | 0.9500 | 0.8718 | 0.1282 | 0.5787 | 0.6143 |
| targeted_moderate | 0.2018 | 0.9583 | 0.9615 | 0.0385 | 0.3427 | 0.3429 |
| targeted_strong | 0.3509 | 0.9231 | 0.8718 | 0.1282 | 0.4390 | 0.4476 |

Capture-level bootstrap 95% CIs (36 captures,
420 flows, 2000 resamples) in
`bootstrap_ci_results.csv`.

## Headline comparison — Candidate 2 vs selected new candidate (independent)

| Metric | production | Candidate 2 | new `strong` |
|---|---|---|---|
| FTP recall | 0.0000 | 0.7544 | 0.3509 |
| Benign recall | 0.9744 | 0.7179 | 0.8718 |
| Benign FP rate | 0.0256 | 0.2821 | 0.1282 |
| macro-F1 | 0.1230 | 0.6717 | 0.4390 |
| accuracy | 0.1810 | 0.7476 | 0.4476 |

## SHAP (selected candidate `targeted_strong`)

- Candidate 2 global top-5 artifact-dependent: **True**
- New candidate global top-5 artifact-dependent: **True**
- New candidate top features on independent BENIGN flows: Dst Port;Fwd Seg Size Min;Init Fwd Win Byts;Fwd IAT Min;Flow Pkts/s

## Verdict — DO NOT PROMOTE

benign FP 0.282->0.128 (+0.154); FTP recall 0.754->0.351; macro-F1 0.672->0.439; CIC non-regressed=True. Does not meet all promotion criteria (needs meaningful FP drop, retained FTP recall, no CIC regression, and macro-F1 not worse).

- benign FP change (Cand2→new): **+0.154**
- FTP recall change (Cand2→new): **-0.404**
- CIC non-regressed: **True**  ·  promote: **False**

### Caveats
- Independent test small (36 captures / 420 flows), loopback-only, 2 real classes.
- Candidate selected on CIC+v2-LOCO; independent test evaluated once.
- No statistical-significance claim; CIs are wide.

## Integrity

Production model, Candidate 2, ml.py / live_capture.py / pcap_validation.py, and
the v1/v2/independent/targeted PCAPs verified unchanged before == after. Exactly 30
ordered finite features, no zero-fill. No independent PCAP entered training.

## Files

`candidate_metrics.csv`, `baseline_metrics.csv`, `per_class_metrics.csv`,
`per_capture_v2loco_metrics.csv`, `independent_test_metrics.csv`,
`independent_per_capture_metrics.csv`, `weighting_comparison.csv`,
`confusion_*.csv`, `confidence_distribution.csv`, `bootstrap_ci_results.csv`,
`shap_results.csv`, `leakage_validation.json`, `model_hashes_before_after.json`,
`training_metadata.json`, `final_verdict.json`.
