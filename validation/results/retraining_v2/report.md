# Realistic-PCAP retraining v2 — report

**Experimental only. Production model frozen and byte-for-byte unchanged
(sha256 verified before==after). No promotion, no merge.** Candidates live under
`validation/models/realistic_pcap_candidate_v2/`; evidence here.

## Baseline (production model)

| Eval | Accuracy | Macro-F1 | FTP recall | Benign recall |
|---|---|---|---|---|
| CIC held-out | 0.9803 | 0.8627 | — | — |
| Real (854 flows) | 0.1885 | 0.0904 | 0.000 | 0.880 |

## Candidate 1 — CIC-only reproduction

Full-CIC retrain (seed 42): CIC acc 0.9804 / macro-F1
0.8599. Reproduces baseline: **True** — confirms the
training pipeline is not the variable.

## Candidate 2 — CIC + real (weighting comparison, LOCO)

Capture-level leave-one-capture-out (no PCAP in train and test). CIC held-out from
full-CIC candidates; real-PCAP LOCO from a fixed 30000-row
stratified CIC subsample (matched subsample-only control CIC macro-F1
0.8101). See `weighting_comparison.csv`,
`candidate_metrics.csv`, `per_capture_metrics.csv`.

| weighting | real_weight | real_total_weight | cic_accuracy | cic_macro_f1 | loco_accuracy | loco_macro_f1 | loco_ftp_recall | loco_benign_recall |
|---|---|---|---|---|---|---|---|---|
| none | 1.0000 | 854.0000 | 0.9809 | 0.8650 | 0.8993 | 0.8534 | 0.9285 | 0.7923 |
| moderate | 6.0000 | 5124.0000 | 0.9801 | 0.8638 | 0.8864 | 0.8310 | 0.9285 | 0.7322 |
| strong | 24.0000 | 20496.0000 | 0.9803 | 0.8608 | 0.8841 | 0.8302 | 0.9210 | 0.7486 |

## Leakage / integrity

All automated checks pass: **True**. See `leakage_validation.json`
(no PCAP in train+test, labels from folders only, exactly 30 ordered finite
features, no zero-fill, every flow traceable to its PCAP, candidate artifacts
separate from production, production model unchanged).

## SHAP (production vs none candidate)

Top production features: Dst Port, Init Fwd Win Byts, Fwd Seg Size Min, Flow IAT Min, Fwd Pkts/s. Candidate still depends on CIC artifact features (`Fwd Seg Size Min`/`Init Fwd Win Byts`): **True**. See `shap_comparison.csv`.

## Verdict — promising but insufficient evidence

FTP recall improves (+0.928) with CIC non-regressed, but benign recall drops to 0.7923497267759563 (false positives) -- not ready.

**Best candidate:** cic+real_none (LOCO FTP recall 0.9284649776453056,
benign recall 0.7923497267759563, CIC macro-F1
0.8650 vs baseline 0.8627).
**Not promoted.**

### Caveats
- Only 83 real captures (loopback lab), 2 real classes (Benign, FTP-BruteForce).
- LOCO uses a fixed 30k stratified CIC subsample for tractability; full-CIC candidates confirm CIC regression separately.
- No statistical-significance claim: sample too small; point estimates only.

## Files

`baseline_metrics.csv`, `candidate_metrics.csv`, `weighting_comparison.csv`,
`per_class_metrics.csv`, `per_capture_metrics.csv`, `statistical_comparison.csv`,
`confusion_baseline_{cic,real}.csv`, `confusion_loco_*.csv`,
`schema_validation.json`, `leakage_validation.json`, `training_metadata.json`,
`final_verdict.json`, `shap_comparison.csv`.
