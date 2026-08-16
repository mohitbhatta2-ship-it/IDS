# Independent real-PCAP test — final evaluation of Candidate 2

**These test PCAPs were NEVER used for training, retraining, fine-tuning, sample
weighting, threshold selection, hyperparameter selection, feature selection,
candidate selection, or SHAP-based tuning.** Both models are frozen; the
independent set is used only to evaluate them. Candidate 2 was not modified after
seeing these results.

## Test corpus

36 fresh PCAPs (18 Benign, 18
FTP-BruteForce), 36 distinct scenarios,
4 clients, 3 server implementations,
3 environments. All verified: True.

## Headline (independent test)

| Model | FTP recall | Benign recall | Benign FP rate | macro-F1 | accuracy |
|---|---|---|---|---|---|
| production | 0.0000 | 0.9744 | 0.0256 | 0.1230 | 0.1810 |
| Candidate 2 | 0.7544 | 0.7179 | 0.2821 | 0.6717 | 0.7476 |

**Capture-level bootstrap 95% CIs** (36 captures,
420 flows, 2000 resamples):

| Model | FTP recall (95% CI) | Benign recall (95% CI) | Accuracy (95% CI) |
|---|---|---|---|
| production | 0.000 [0.000, 0.000] | 0.973 [0.925, 1.000] | 0.191 [0.088, 0.346] |
| Candidate 2 | 0.751 [0.595, 0.898] | 0.718 [0.646, 0.789] | 0.745 [0.619, 0.861] |

## Confidence analysis

- Production: confidently-wrong (≥0.90) = 191
  (FTP→Benign 191); mean conf when
  wrong 0.8520.
- Candidate 2: confidently-wrong (≥0.90) = 6
  (Benign→FTP 6); mean conf when
  wrong 0.6937.

## SHAP (frozen models, diagnosis only)

- Production global top-5: Dst Port, Init Fwd Win Byts, Fwd Seg Size Min, Flow IAT Min, Fwd Pkts/s
- Candidate 2 global top-5: Fwd Seg Size Min, Init Fwd Win Byts, Dst Port, Fwd IAT Min, Fwd Pkts/s
- **Candidate 2 still depends on CIC artifact features (`Fwd Seg Size Min`/`Init Fwd Win Byts`): True**

## Three-way comparison (training vs prior LOCO vs independent)

| source | data_role | production_accuracy | production_macro_f1 | candidate_accuracy | candidate_macro_f1 | production_ftp_recall | candidate_ftp_recall | production_benign_recall | candidate_benign_recall |
|---|---|---|---|---|---|---|---|---|---|
| A_CIC_heldout | training/validation distribution | 0.9803 | 0.8627 | 0.9809 | 0.8650 | nan | nan | nan | nan |
| B_v2_realistic_LOCO | prior leave-one-capture-out eval | 0.1885 | 0.0904 | 0.8993 | 0.8534 | 0.0000 | 0.9285 | 0.8798 | 0.7923 |
| C_independent_test | COMPLETELY INDEPENDENT final test | 0.1810 | 0.1230 | 0.7476 | 0.6717 | 0.0000 | 0.7544 | 0.9744 | 0.7179 |

## Leakage / independence — all pass: True

Independent PCAPs disjoint from v1 and v2 (by content hash), no duplicates, labels
from folders only, exactly 30 ordered finite features, no zero-fill, every flow
traceable to its PCAP. Frozen artifacts (production + Candidate 2 + v2 evidence +
v1/v2 datasets) verified unchanged before == after.

## FINAL VERDICT — PROMISING BUT INSUFFICIENT EVIDENCE

Candidate 2 improves FTP recall to 0.754 vs production 0.000 on independent data, but benign recall 0.718 is low / the lower CI bound is 0.595, per-capture consistency 0.89, artifact-dependent=True. Not sufficient to promote.

**Candidate is NOT promoted** (`promote=false`).

### Caveats
- Independent test is small: 36 captures / 420 flows, loopback lab, 2 classes.
- CIs are wide; no strong statistical-significance claim is made.
- Two server implementations (custom + pyftpdlib); single-host (no separate interface).

## Files

`baseline_metrics.csv`, `candidate_metrics.csv`, `confusion_matrix_*.csv`,
`per_class_metrics.csv`, `per_capture_metrics.csv`, `prediction_distribution.csv`,
`confidence_distribution.csv`, `bootstrap_or_ci_results.csv`, `shap_comparison.csv`,
`three_way_comparison.csv`, `test_set_manifest_summary.csv`,
`leakage_validation.json`, `model_hashes_before_after.json`, `final_verdict.json`.
