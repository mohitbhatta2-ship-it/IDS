# FTP behavioural-feature experiment - report

**Candidate only. Production model and Candidate 2 frozen and byte-for-byte
unchanged (verified). No auto-promotion, no merge.** The existing 30-feature
`pcap_validation` pipeline is unchanged; behavioural features are added only in this
experimental module. The independent 36-PCAP set was NEVER used for feature
selection, tuning, weighting, or candidate selection (hash-guarded); the candidate
was selected on CIC held-out + v2 LOCO, then evaluated once on the independent set.

## New features

15 FTP control-channel behavioural features (`ftp_behavioral.BEHAV_FEATURES`) read
directly from cleartext FTP commands/responses: login attempts, failed logins
(530), successful logins (230), failed-login ratio, has-successful-auth, data
commands, data-setup responses (150), command variety, control connections /
reconnects, etc. Computed per capture from the PCAP; CIC flows (no PCAP) carry
**NaN** (HGB native missing-value handling -- not zero-fill).

## Step 1 - separability gate (training corpora only: v1+v2+targeted)

Best single-feature separability |AUC-0.5|: **1.000**
(`ftp_failed_logins`); behavioural-only 5-fold CV macro-F1:
**0.992** over 140 captures.
**Gate passed: True** -- the behavioural features strongly separate real
FTP-BruteForce from real benign (benign always authenticates successfully and does
data transfers; brute force is repeated failed auth). See `separability_analysis.csv`.

## Step 2 - augmented candidates (CIC held-out + v2 LOCO; selection signals only)

| candidate | ftp_weight | benign_weight | cic_accuracy | cic_macro_f1 | v2_loco_ftp_recall | v2_loco_benign_recall | v2_loco_macro_f1 |
|---|---|---|---|---|---|---|---|
| aug_unweighted | 1.0000 | 1.0000 | 0.9809 | 0.8637 | 0.9940 | 1.0000 | 0.9931 |
| aug_balanced | 3.0000 | 5.5561 | 0.9806 | 0.8655 | 0.9925 | 1.0000 | 0.9914 |

Selected on CIC+v2-LOCO: **`aug_unweighted`**.

## Final independent test (frozen 36-PCAP set)

| Model | FTP recall | Benign recall | Benign FP | macro-F1 | accuracy |
|---|---|---|---|---|---|
| production | 0.0000 | 0.9744 | 0.0256 | 0.1230 | 0.1810 |
| Candidate 2 | 0.7544 | 0.7179 | 0.2821 | 0.6717 | 0.7476 |
| control (30-feat, CIC+real) | 0.5263 | 0.8590 | 0.1410 | 0.5560 | 0.5881 |
| **aug_unweighted (30+behavioural)** | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

FTP recall 95% CI (aug): [1.000,
1.000]; benign recall 95% CI:
[1.000,
1.000]. See `bootstrap_ci_results.csv`.

## SHAP (selected `aug_unweighted`)

- Uses behavioural features in top-5: **True**; still CIC-artifact in top-5: **True**
- Behavioural-feature importance share: **0.14**
- Global top-8: Dst Port, Fwd Seg Size Min, ftp_failed_logins, Init Fwd Win Byts, Fwd Pkts/s, Fwd Header Len, Fwd IAT Min, Flow Pkts/s

## Verdict - PROMISING - FRONTIER BROKEN (validate further; DO NOT auto-promote)

aug_unweighted independent: FTP recall 1.000, benign recall 1.000, benign FP 0.000 (Cand2 0.282), macro-F1 1.000, CIC ok=True. All balanced-target criteria met on the untouched independent test -- the behavioural features break the FTP-recall/benign-FP trade-off.

Balanced-target criteria (all required to break the frontier): {"ftp_recall_gt_0.70": true, "benign_recall_gt_0.90": true, "benign_fp_much_lower_than_cand2": true, "cic_non_regressed": true}

**The behavioural features separate real FTP from real benign and, on the untouched independent test, let the augmented model achieve high FTP recall AND high benign recall simultaneously -- breaking the trade-off the 30 CIC features could not. This is a promising direction; it is NOT auto-promoted.**

Promote: **False** (never auto-promoted).

### Caveats
- Separation is strong partly because captures are cleanly scenario-labelled (benign always authenticates, brute force always fails); real-world traffic can be messier (benign typos, brute force that eventually succeeds).
- Behavioural features are per-capture session context attached to each flow.
- Independent corpus small (36 captures / 420 flows), loopback-only; CIs wide; no significance claim.
- Not auto-promoted: needs broader real traffic and a fresh independent test before any rollout.

## Integrity

Production model, Candidate 2, ml.py/live_capture.py/pcap_validation.py, and the
v1/v2/independent/targeted PCAPs verified unchanged (before==after). No independent
PCAP entered training. Capture-level LOCO with per-fold no-flow-overlap assertions.
CIC behavioural features are NaN (not zero-fill). Candidate models stored outside
webapp_data.

## Files

`separability_analysis.csv`, `candidate_metrics.csv`, `candidate_comparison.csv`,
`independent_test_metrics.csv`, `independent_per_capture_metrics.csv`,
`confusion_*.csv`, `bootstrap_ci_results.csv`, `shap_comparison.csv`,
`leakage_validation.json`, `model_hashes_before_after.json`, `training_metadata.json`,
`final_verdict.json`.
