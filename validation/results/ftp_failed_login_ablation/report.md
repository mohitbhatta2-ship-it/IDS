# Is `ftp_failed_logins` causing the benign false positives? - controlled ablation

**Candidates only; production frozen (production, Candidate 2, the 45-feature candidate,
ml.py/live_capture.py/pcap_validation.py/ftp_behavioral.py unchanged, before==after). The
training data is HELD CONSTANT (CIC + v1 + v2 + targeted + robust_train); only the feature
set varies. Selection used CIC held-out + LOCO only; the frozen vsFTPD corpus was
evaluated once, for reporting. No promotion, no merge.** Branch
`claude/final-independent-ftp-validation`.

## Candidates (data identical, features varied)

- **C1** full 45 (current frozen candidate)
- **C2** 44 = 45 - `ftp_failed_logins`
- **C3** 43 = 45 - `ftp_failed_logins` - `ftp_failed_login_ratio`
- **C4** 40 = 30 packet + positive-evidence behavioural (no `ftp_failed_logins`,
  `ftp_failed_login_ratio`, `ftp_error_responses_5xx`, `ftp_login_attempts`, `ftp_total_commands`)

Selected via CIC + LOCO: **C3_no_failed_or_ratio_43** (CIC-gate=True (no CIC regression), LOCO macro-F1 0.9948 (best gate-passer)).

## CIC held-out (no-regression gate)
| Model | FTP recall | Benign recall | accuracy |
|---|---|---|---|
| production | 0.8835 | 0.9855 | 0.9803 |
| candidate2 | 0.8936 | 0.9866 | 0.9809 |
| C1_full_45 | 0.8715 | 0.9863 | 0.9808 |
| C2_no_failed_logins_44 | 0.8715 | 0.9867 | 0.9811 |
| C3_no_failed_or_ratio_43 | 0.8956 | 0.9858 | 0.9804 |
| C4_alt_no_failure_volume_40 | 0.8976 | 0.9855 | 0.9802 |

## FINAL vsFTPD test (cleartext, evaluated once)
| Model | FTP recall | Benign recall | FPR | FTP precision | macro-F1 | accuracy |
|---|---|---|---|---|---|---|
| production | 0.0000 | 0.9836 | 0.0164 | 0.0000 | 0.2094 | 0.4580 |
| candidate2 | 0.0857 | 0.8525 | 0.1475 | 0.4000 | 0.3644 | 0.4427 |
| C1_full_45 | 1.0000 | 0.8525 | 0.1475 | 0.8861 | 0.9300 | 0.9313 |
| C2_no_failed_logins_44 | 1.0000 | 0.8361 | 0.1639 | 0.8750 | 0.9220 | 0.9237 |
| C3_no_failed_or_ratio_43 | 1.0000 | 0.8689 | 0.1311 | 0.8974 | 0.9379 | 0.9389 |
| C4_alt_no_failure_volume_40 | 0.6000 | 0.9016 | 0.0984 | 0.8750 | 0.7379 | 0.7405 |

### Bootstrap 95% CIs
| Model | FTP recall [95% CI] | Benign recall [95% CI] | FPR [95% CI] |
|---|---|---|---|
| C1_full_45 | 1.000 [1.000,1.000] | 0.845 [0.654,0.982] | 0.155 [0.018,0.346] |
| C3_no_failed_or_ratio_43 | 1.000 [1.000,1.000] | 0.863 [0.681,1.000] | 0.137 [0.000,0.319] |

### Robustness across scenarios (recall by family; columns C1/C2/C3/C4)
| family | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| mistype | 0.6667 | 0.5833 | 0.7500 | 0.9167 |
| gave_up | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| eventual_success | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| clean | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| activity | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| fast_brute | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| slow_brute | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| multi_conn | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## SHAP (selected C3_no_failed_or_ratio_43, cleartext test)

- largest feature `Dst Port` at **0.167**; behavioural **0.120**; CIC artifacts **0.415**; `ftp_failed_logins` present: **False**
- top: Dst Port, Fwd Seg Size Min, Init Fwd Win Byts, ftp_error_responses_5xx, Fwd IAT Min, Fwd Header Len

## FTPS / TLS (encrypted; behavioural unavailable)
| Model | FTP recall | Benign recall |
|---|---|---|
| production | 0.0000 | 1.0000 |
| candidate2 | 0.3750 | 1.0000 |
| C1_full_45 | 0.0000 | 1.0000 |
| C2_no_failed_logins_44 | 0.0000 | 1.0000 |
| C3_no_failed_or_ratio_43 | 0.0000 | 1.0000 |
| C4_alt_no_failure_volume_40 | 0.0417 | 1.0000 |

## Verdict - BENIGN IMPROVED WITHOUT LOSING FTP -- BUT TARGET NOT FULLY MET

Removing ftp_failed_logins improves the benign side (recall 0.852->0.869, FPR 0.148->0.131) while holding FTP recall (1.000); but the target thresholds are not all met (benign>=0.90? False; FPR<=0.10? False) -- gave_up 0.0.

- target_met: **False**  ·  removing_failed_logins_helps_benign:
  **True**  ·  removing_failed_logins_damages_ftp:
  **False**  ·  cic_no_regression:
  **True**  ·  promote: **False**

Robustness (selected): mistype 0.7500, gave_up
0.0000, eventual_success 1.0000.

### Caveats
- Data held constant (CIC+v1+v2+targeted+robust_train); only the feature set varies.
- Frozen vsFTPD corpus evaluated once for reporting; never used for training/selection/tuning.
- Loopback/vsFTPD single-container lab; small test corpus (wide CIs).
- No promotion, no threshold/heuristic change, no merge.

## Files
`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`per_scenario_family_metrics.csv`, `per_scenario_metrics.csv`, `per_client_metrics.csv`,
`per_server_metrics.csv`, `per_environment_metrics.csv`, `bootstrap_cis.json`,
`shap_top_features.csv`, `ftps_encrypted_metrics.csv`, `confusion_*.csv`,
`leakage_validation.json`, `model_hashes_before_after.json`, `final_verdict.json`.
