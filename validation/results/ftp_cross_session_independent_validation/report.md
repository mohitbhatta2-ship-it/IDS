# Final independent validation of the cross-session FTP detector - report

**All models frozen (production, Candidate 2, the 45-feature candidate, C3, the
cross-session candidate, and ml.py/live_capture.py/pcap_validation.py/ftp_behavioral.py/
ftp_cross_session.py verified before==after). This 2nd independent corpus is TEST-ONLY --
never training/selection/tuning. No promotion, no merge.** Branch
`claude/ftp-cross-session-independent-validation`.

## What this test is

A SECOND genuinely-independent corpus, different from every prior one and from the first
vsFTPD test: **pure-ftpd** server (new), **ncftp** client (new), a new **netns/subnet**
(10.88.0.0/24) and ports (2222/2323), **fresh scenario code** with deliberate
**single- vs multi-session** structure, plus **FTPS**. It tests whether the cross-session
detector's benefit holds on a different server/client/network -- and whether it still
catches **single-session** attacks (which could expose over-reliance on session count).

## CIC held-out (no-regression check)
| Model | FTP recall | Benign recall | accuracy |
|---|---|---|---|
| production | 0.8835 | 0.9855 | 0.9803 |
| candidate2 | 0.8936 | 0.9866 | 0.9809 |
| C1_current_45 | 0.8715 | 0.9863 | 0.9808 |
| C3_controlled_43 | 0.8956 | 0.9858 | 0.9804 |
| cross_session_58 | 0.8715 | 0.9858 | 0.9803 |

## FINAL cleartext evaluation (2nd independent corpus, evaluated once)
| Model | FTP recall | Benign recall | FPR | FTP precision | macro-F1 | accuracy |
|---|---|---|---|---|---|---|
| production | 0.0000 | 0.9778 | 0.0222 | 0.0000 | 0.2173 | 0.4835 |
| candidate2 | 0.0000 | 0.7556 | 0.2444 | 0.0000 | 0.2720 | 0.3736 |
| C1_current_45 | 0.6739 | 0.9778 | 0.0222 | 0.9688 | 0.8205 | 0.8242 |
| C3_controlled_43 | 0.8913 | 0.9778 | 0.0222 | 0.9762 | 0.9340 | 0.9341 |
| cross_session_58 | 0.8043 | 1.0000 | 0.0000 | 1.0000 | 0.9003 | 0.9011 |

### Bootstrap 95% CIs
| Model | FTP recall [95% CI] | Benign recall [95% CI] | FPR [95% CI] |
|---|---|---|---|
| C1_current_45 | 0.663 [0.472,0.794] | 0.974 [0.895,1.000] | 0.026 [0.000,0.105] |
| C3_controlled_43 | 0.881 [0.667,1.000] | 0.974 [0.895,1.000] | 0.026 [0.000,0.105] |
| cross_session_58 | 0.789 [0.500,0.978] | 1.000 [1.000,1.000] | 0.000 [0.000,0.000] |

## With / without the cross-session block (frozen models)

- with cross (cross_session_58): FTP 0.8043,
  benign 1.0000,
  FPR 0.0000.
- without cross (C1 45): FTP 0.6739,
  benign 0.9778,
  FPR 0.0222.

## SHAP (cross candidate, 2nd corpus)

- cross-session block share **0.071** (top cross feature `ftpx_failure_rate_across_sessions`); behavioural **0.040**; CIC artifacts **0.419**; largest single feature `Dst Port` **0.179** (single-feature shortcut: False)
- top: Dst Port, Fwd Seg Size Min, Init Fwd Win Byts, ftpx_failure_rate_across_sessions, Fwd Pkts/s, Fwd Header Len

## FTPS / TLS (encrypted; behavioural + cross-session unavailable)
| Model | FTP recall | Benign recall |
|---|---|---|
| production | 0.0000 | 1.0000 |
| candidate2 | 0.0000 | 1.0000 |
| C1_current_45 | 0.0000 | 1.0000 |
| C3_controlled_43 | 0.0000 | 1.0000 |
| cross_session_58 | 0.0000 | 1.0000 |

## Verdict — NOT EFFECTIVE

On the 2nd independent corpus the cross-session candidate did not meet the target (FTP 0.804>=0.90? benign 1.000>=0.90? FPR 0.000<=0.10?; CIC no-regression True; single-feature shortcut False). See failure mode.

- cic_no_regression: **True**  ·  single_feature_shortcut:
  **False**  ·  cross_block_contributes_vs_c1:
  **True**  ·  promote: **False**

### What it proves / does not prove
Proves the cross-session detector's behaviour on a **different server (pure-ftpd), client
(ncftp), and network** with fresh scenarios and single/multi-session structure. Does NOT
prove generalisation to a different OS/host, the public internet, or attackers who sustain
long single-session brute force (bounded here by pure-ftpd's failure delay).

### Caveats
- Different server (pure-ftpd), client (ncftp), netns/subnet (10.88.0.x) and fresh scenario code, but still a single-container lab -- not a separate OS/host or the public internet.
- Pure-FTPd's escalating failure delay bounds attempt counts, so single-session brute force is shorter here than a real attacker could sustain.
- The corpus was evaluated once, for reporting; never training/selection/tuning.
- Small corpus (wide CIs). No promotion, no threshold change, no merge.

## Files
`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `per_scenario_family_metrics.csv`,
`per_session_structure_metrics.csv`, `per_scenario_metrics.csv`, `per_client_metrics.csv`,
`bootstrap_cis.json`, `shap_top_features.csv`, `ftps_encrypted_metrics.csv`,
`confusion_*.csv`, `leakage_validation.json`, `model_hashes_before_after.json`,
`final_verdict.json`.
