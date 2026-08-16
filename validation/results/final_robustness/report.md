# Final robustness analysis — Candidate 2 (analysis only)

**No retraining, no threshold change, no heuristic, no promotion, no merge.** The
production model and Candidate 2 are byte-for-byte unchanged (verified). The
independent test corpus is FROZEN and was used only for evaluation/diagnosis.

## Data roles (kept strictly separate)

- **Training distribution:** CIC `balanced_train_selected` (+ v1/v2 real, only where
  compared against).
- **v2 validation distribution:** `validation/realistic_pcaps_v2/` (prior LOCO).
- **Independent test:** `validation/independent_real_pcaps/` — 36 PCAPs / 420 flows,
  frozen; never used for any fitting decision.

## 1. Benign false positives — where they cluster

Candidate 2 makes **22 benign false positives out of 78 benign
flows (28.2%)**. They are **not** uniform: see
`benign_fp_breakdown.csv`. The concentration is in command-heavy / multi-command /
transfer / active-mode benign sessions rather than plain logins — behaviours
under-represented in training.

## 2. Scenario difficulty

Hardest FTP-BruteForce scenarios (lowest Candidate-2 recall):

| scenario | class | capture_count | flow_count | correct | incorrect | recall | mean_confidence |
|---|---|---|---|---|---|---|---|
| many_attempts | FTP-BruteForce | 1 | 48 | 21 | 27 | 0.4375 | 0.8868 |
| different_usernames | FTP-BruteForce | 1 | 20 | 9 | 11 | 0.4500 | 0.7387 |
| different_passwords | FTP-BruteForce | 1 | 32 | 16 | 16 | 0.5000 | 0.7749 |
| fast_custom_active | FTP-BruteForce | 1 | 16 | 8 | 8 | 0.5000 | 0.7483 |
| few_attempts | FTP-BruteForce | 1 | 6 | 3 | 3 | 0.5000 | 0.7661 |

See `scenario_metrics.csv` for all scenarios and the benign command-scenario FP
rates.

## 3. Client / server generalisation (support shown)

| group_type | group | flows | benign_support | ftp_support | ftp_recall | benign_recall | benign_fp_rate | macro_f1 |
|---|---|---|---|---|---|---|---|---|
| client | curl | 40 | 8 | 32 | 1.0000 | 0.7500 | 0.2500 | 0.9134 |
| client | python-ftplib | 370 | 62 | 308 | 0.7273 | 0.7258 | 0.2742 | 0.6436 |
| client | raw-socket | 2 | 0 | 2 | 1.0000 | n/a | n/a | 1.0000 |
| client | wget | 8 | 8 | 0 | n/a | 0.6250 | 0.3750 | 0.3846 |
| server_impl | pyftpdlib | 157 | 34 | 123 | 0.9024 | 0.7059 | 0.2941 | 0.7978 |
| server_impl | raw-socket | 263 | 44 | 219 | 0.6712 | 0.7273 | 0.2727 | 0.6051 |
| server_config | custom(custom-raw-socket) | 263 | 44 | 219 | 0.6712 | 0.7273 | 0.2727 | 0.6051 |
| server_config | permissive(pyftpdlib) | 135 | 34 | 101 | 0.9703 | 0.7059 | 0.2941 | 0.8623 |
| server_config | ratelimited(pyftpdlib) | 22 | 0 | 22 | 0.5909 | n/a | n/a | 0.3714 |

## 4. Active vs passive mode

| group_type | mode | flows | benign_support | ftp_support | ftp_recall | benign_recall | benign_fp_rate | macro_f1 |
|---|---|---|---|---|---|---|---|---|
| mode | active | 20 | 4 | 16 | 0.5000 | 0.5000 | 0.5000 | 0.4505 |
| mode | passive | 400 | 74 | 326 | 0.7669 | 0.7297 | 0.2703 | 0.6842 |

## 5. Feature distributions (train vs v2 vs independent)

`feature_distribution_final.csv` reports medians/quartiles for the focus features
across CIC-FTP (training), v2 real-FTP (validation), independent real-FTP, and
independent benign. The independent real-FTP region and the benign region are
compared against the CIC artifact values (`Fwd Seg Size Min`, `Init Fwd Win Byts`).

## SHAP by error group (model-attribution, not causation)

- **TP_FTP**: Dst Port, Fwd Seg Size Min, Init Fwd Win Byts, Fwd Pkts/s, Fwd IAT Min
- **FN_FTP**: Fwd Seg Size Min, Dst Port, Init Fwd Win Byts, Fwd Pkts/s, Flow Pkts/s
- **TP_Benign**: Dst Port, Fwd Seg Size Min, Init Fwd Win Byts, Fwd Pkts/s, Fwd IAT Min
- **FP_Benign**: Fwd Seg Size Min, Dst Port, Init Fwd Win Byts, Fwd Pkts/s, Fwd IAT Min

## 7. Confidence by error group

| group | count | mean | median | p10 | p25 | p75 | p90 |
|---|---|---|---|---|---|---|---|
| TP_FTP | 258 | 0.8935 | 0.9848 | 0.5074 | 0.8600 | 0.9993 | 0.9999 |
| FN_FTP | 84 | 0.6631 | 0.5400 | 0.5400 | 0.5400 | 0.8258 | 0.8258 |
| TP_Benign | 56 | 0.8364 | 0.9484 | 0.5400 | 0.6079 | 0.9970 | 0.9993 |
| FP_Benign | 22 | 0.8104 | 0.8600 | 0.5074 | 0.7873 | 0.9127 | 0.9601 |

## 8. Capture-level robustness

- FTP captures with recall ≥0.5: 89%; ≥0.75:
  56%; ≥0.9: 56%.
- Benign captures with zero FP: 11%.

See `capture_metrics.csv` (weak captures are not hidden).

## 9. Data requirements (ranked by expected value, evidence-based)

1. **real multi-host LAN traffic with additional OSes and FTP clients** — loopback-only corpus; SHAP still shows CIC-artifact dependence (score 0.6)
2. **more benign upload/download sessions across clients** — 10 FP benign captures are transfers (score 0.5)
3. **more FTP brute-force diversity (tools/pacing/patterns) targeting the weakest scenarios** — lowest-recall FTP scenarios: many_attempts, different_usernames, different_passwords (score 0.5)
4. **more benign command-heavy sessions (MKD/RMD/RNFR/RNTO/DELE/APPE/SIZE/MDTM/NLST/STAT, multi-command, reconnect)** — 7 of 16 FP benign captures are command-heavy (score 0.409)
5. **additional FTP server implementations (beyond pyftpdlib + custom raw-socket)** — FTP recall differs by 0.23 across the two server impls (score 0.381)
6. **more benign ACTIVE-mode sessions** — active-mode benign FP 0.50 vs passive 0.27 (score 0.23)

## 10. Final recommendation — B. Collect another specifically targeted real-PCAP corpus

Collect a targeted real-PCAP corpus (highest-ranked: benign command-heavy / active-mode sessions, more transfer sessions, and additional server implementations), then RE-EVALUATE on a fresh independent set before any retraining. Do not run another training experiment yet: the benign FP cause is structured and addressable with data, but there is not yet enough evidence that retraining on the current corpora would fix it without a new independent test.

- candidate_promote: **false**  ·  retraining_performed: **false**  ·
  threshold_changed: **false**  ·  heuristics_added: **false**  ·
  production_model_changed: **false**

### Caveats
- Independent corpus is small (36 captures / 420 flows), loopback-only, 2 classes.
- Findings are model-attribution evidence (SHAP/confidence/grouping), not proven causation.
- No statistical-significance claim.

## Integrity

Production model + Candidate 2 + ml.py / live_capture.py / pcap_validation.py +
independent PCAPs/manifest all verified unchanged (before == after). This analysis
wrote no model artifact and altered no label.

## Files

`benign_fp_breakdown.csv`, `scenario_metrics.csv`, `client_server_metrics.csv`,
`mode_metrics.csv`, `feature_distribution_final.csv`, `shap_error_groups.csv`,
`confidence_error_groups.csv`, `capture_metrics.csv`, `data_requirements.json`,
`final_robustness_verdict.json`.
