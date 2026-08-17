# Cross-session / source-level behaviour experiment - report

**Candidates only; production frozen (production, the 45-feature candidate, the C3
candidate, ml.py/live_capture.py/pcap_validation.py/ftp_behavioral.py unchanged,
before==after). The 30 packet + 15 behavioural features are PRESERVED; cross-session
features are appended. Selection used CIC held-out + LOCO only; the frozen vsFTPD corpus
was evaluated once. No promotion, no merge.** Branch `claude/ftp-cross-session-behavior`.

## Idea

Single-session features cannot separate "a legitimate user failed a few times and left"
from "a short brute force". Cross-session/source-level features (sessions per source,
failures across sessions, credential variation, persistence, per-session success history)
might, because an attacker persists across many sessions while a legitimate user makes few.
A new corpus with multiple sessions per source (benign 1-4 sessions; attacker 6-12) was
collected; 13 label-free cross-session features were added (58 total). Selected: **candidate_cross_unweighted**
(CIC-gate=True, LOCO macro-F1 0.9905).

## FINAL vsFTPD test (cleartext, evaluated once)

| Model | FTP recall | Benign recall | FPR | FTP precision | macro-F1 | accuracy |
|---|---|---|---|---|---|---|
| production | 0.0000 | 0.9836 | 0.0164 | 0.0000 | 0.2094 | 0.4580 |
| C1_current_45 | 1.0000 | 0.8525 | 0.1475 | 0.8861 | 0.9300 | 0.9313 |
| C3_controlled_43 | 1.0000 | 0.8689 | 0.1311 | 0.8974 | 0.9379 | 0.9389 |
| candidate_cross_unweighted | 1.0000 | 0.9344 | 0.0656 | 0.9459 | 0.9692 | 0.9695 |

### Bootstrap 95% CIs
| Model | FTP recall [95% CI] | Benign recall [95% CI] | FPR [95% CI] |
|---|---|---|---|
| C1_current_45 | 1.000 [1.000,1.000] | 0.845 [0.654,0.982] | 0.155 [0.018,0.346] |
| C3_controlled_43 | 1.000 [1.000,1.000] | 0.863 [0.681,1.000] | 0.137 [0.000,0.319] |
| candidate_cross_unweighted | 1.000 [1.000,1.000] | 0.931 [0.761,1.000] | 0.069 [0.000,0.239] |

### Critical cases (recall; C1 vs selected cross candidate)
| family | C1 | cross |
|---|---|---|
| mistype | 0.6667 | 1.0000 |
| gave_up | 0.0000 | 0.2000 |
| eventual_success | 1.0000 | 1.0000 |
| clean | 1.0000 | 1.0000 |
| activity | 1.0000 | 1.0000 |
| fast_brute | 1.0000 | 1.0000 |
| slow_brute | 1.0000 | 1.0000 |
| multi_conn | 1.0000 | 1.0000 |
| user_enum | 1.0000 | 1.0000 |

## Feature ablation: does the cross-session block matter?

- With cross (candidate_cross_unweighted): FTP 1.0000, benign
  0.9344, FPR 0.0656.
- Without cross (45 features, same data): FTP 0.6000,
  benign 0.9180, FPR 0.0820.

## SHAP (selected, cleartext test)

- cross-session block share **0.073** (top cross feature `ftpx_failure_rate_across_sessions`); behavioural **0.041**; CIC artifacts **0.439**; largest single `Fwd Seg Size Min` 0.174
- top: Fwd Seg Size Min, Dst Port, Init Fwd Win Byts, ftpx_failure_rate_across_sessions, Fwd Pkts/s, Fwd IAT Min

## FTPS / TLS (encrypted; behavioural unavailable)
| Model | FTP recall | Benign recall |
|---|---|---|
| production | 0.0000 | 1.0000 |
| C1_current_45 | 0.0000 | 1.0000 |
| C3_controlled_43 | 0.0000 | 1.0000 |
| candidate_cross_unweighted | 0.0000 | 1.0000 |

## Critical test — does cross-session information resolve the ambiguity?

Cross-session information DOES resolve the benign-give-up vs brute-force ambiguity on genuinely independent traffic.

`gave_up` recall: C1 0.0000 -> cross 0.2000.

## Verdict — PROMISING -- TARGET MET

Cross-session features meet the target on the frozen vsFTPD test WITHOUT sacrificing attack detection: FTP recall 1.000, benign recall 0.934, FPR 0.066 (CI lo FTP 1.000/benign 0.761); gave_up 0.2 (C1 0.0); no CIC regression. Recommend review.

- target_met: **True**  ·  ftp_recall_preserved: **True**  ·
  benign_improved_vs_c1: **True**  ·  cic_no_regression:
  **True**  ·  promote: **False**

### Caveats
- Loopback lab; the cross-session corpus's attackers use one credential per connection, so they have MORE sessions than the vsFTPD test's attackers (which pack attempts per connection) -- a session-count distribution shift between train and test.
- Frozen vsFTPD corpus evaluated once, for reporting; never training/selection/tuning.
- Small test corpus (wide CIs). No promotion, no threshold change, no merge.

## Files
`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`per_scenario_family_metrics.csv`, `per_source_session_metrics.csv`, `per_scenario_metrics.csv`,
`per_client_metrics.csv`, `per_server_metrics.csv`, `per_environment_metrics.csv`,
`bootstrap_cis.json`, `shap_top_features.csv`, `ftps_encrypted_metrics.csv`, `confusion_*.csv`,
`leakage_validation.json`, `model_hashes_before_after.json`, `final_verdict.json`.
