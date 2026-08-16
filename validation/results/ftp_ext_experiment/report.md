# Extended-behavioural retraining experiment - report

**Candidates only; production frozen (production, Candidate 2, the 45-feature robust
candidate, ml.py/live_capture.py/pcap_validation.py/ftp_behavioral.py unchanged,
verified before==after). The 30 packet features and 15 behavioural features are
PRESERVED; EXT only adds features. The frozen vsFTPD corpus is the final test only --
never training/selection/tuning. No promotion, no merge.** Branch
`claude/final-independent-ftp-validation`.

## Idea

The independent vsFTPD test showed the behavioural model false-positives on benign
failed-login sessions. Separability analysis (approved corpora) found the discriminating
*dynamics* -- inter-attempt pacing, password reuse, small attempt counts, graceful QUIT,
post-success activity -- were absent from the earlier corpora. This experiment adds 12
label-free EXT features and a realistic benign failed-login training corpus, then
re-tests on the frozen vsFTPD corpus.

Selected: **candidate_ext_unweighted** (CIC-gate=True, LOCO macro-F1 0.9957); selection used CIC held-out + LOCO only.

## FINAL vsFTPD test (cleartext)

| Model | FTP recall | Benign recall | FPR | FTP precision | macro-F1 | accuracy |
|---|---|---|---|---|---|---|
| production | 0.0000 | 0.9836 | 0.0164 | 0.0000 | 0.2094 | 0.4580 |
| candidate2 | 0.0857 | 0.8525 | 0.1475 | 0.4000 | 0.3644 | 0.4427 |
| robust45 | 1.0000 | 0.8525 | 0.1475 | 0.8861 | 0.9300 | 0.9313 |
| candidate_ext_unweighted | 1.0000 | 0.8361 | 0.1639 | 0.8750 | 0.9220 | 0.9237 |

### Bootstrap 95% CIs
| Model | FTP recall [95% CI] | Benign recall [95% CI] |
|---|---|---|
| production | 0.000 [0.000,0.000] | 0.984 [0.947,1.000] |
| robust45 | 1.000 [1.000,1.000] | 0.845 [0.654,0.982] |
| candidate_ext_unweighted | 1.000 [1.000,1.000] | 0.829 [0.604,1.000] |

### Per scenario family (selected)
| family | label | flows | recall |
|---|---|---|---|
| activity | Benign | 20 | 1.0000 |
| clean | Benign | 8 | 1.0000 |
| eventual_success | FTP-BruteForce | 7 | 1.0000 |
| fast_brute | FTP-BruteForce | 21 | 1.0000 |
| gave_up | Benign | 5 | 0.0000 |
| mistype | Benign | 12 | 0.5833 |
| multi_conn | FTP-BruteForce | 28 | 1.0000 |
| multi_user | Benign | 8 | 1.0000 |
| reconnect | Benign | 8 | 1.0000 |
| slow_brute | FTP-BruteForce | 9 | 1.0000 |
| user_enum | FTP-BruteForce | 5 | 1.0000 |

## Single-feature dependence -- drop-one ablation (retrained, eval on vsFTPD test)

| dropped feature | FTP recall (Δ) | benign recall (Δ) | FPR |
|---|---|---|---|
| `ftp_failed_logins` | 1.0000 (+0.000) | 0.9180 (+0.082) | 0.0820 |
| `ftpx_fail_run_before_success` | 1.0000 (+0.000) | 0.8525 (+0.016) | 0.1475 |
| `ftpx_distinct_passwords` | 1.0000 (+0.000) | 0.8525 (+0.016) | 0.1475 |
| `ftpx_interattempt_mean_s` | 1.0000 (+0.000) | 0.8525 (+0.016) | 0.1475 |
| `ftpx_post_auth_commands` | 1.0000 (+0.000) | 0.8361 (+0.000) | 0.1639 |
| `ftp_user_commands` | 0.6714 (-0.329) | 0.9180 (+0.082) | 0.0820 |

## SHAP (selected candidate, cleartext test)

- largest single feature: `Dst Port` at **0.176**; `ftp_failed_logins` **0.042**; all application-layer **0.120**; new EXT **0.013**; CIC artifacts **0.431**
- top: Dst Port, Fwd Seg Size Min, Init Fwd Win Byts, ftp_failed_logins, Fwd Pkts/s, Fwd Header Len, Fwd IAT Min

## FTPS / TLS (encrypted; behavioural unavailable)

| Model | FTP recall | Benign recall |
|---|---|---|
| production | 0.0000 | 1.0000 |
| candidate2 | 0.3750 | 1.0000 |
| robust45 | 0.0000 | 1.0000 |
| candidate_ext_unweighted | 0.0000 | 1.0000 |

## Verdict -- TARGET NOT MET

EXT features did not fix the benign false positives on the frozen vsFTPD test (FTP 1.000, benign 0.836, FPR 0.164). See limitation.

- target_met: **False**  ·  CIC no-regression: **True**  ·
  single-feature dependence: **True**  ·  promote: **False**

### Remaining limitation
A benign user who fails to authenticate and leaves (gave_up) produces control- and network-level traffic that is near-identical to a short failed brute force; no label-free feature fully separates them. Pacing helps only when benign is human-paced AND the attacker is not; on the vsFTPD test both are shaped by the server's failure-delay timing.

### Recommended next experiment
Either (a) collect benign failed-login traffic from MULTIPLE real servers so pacing/activity signals generalise beyond one server's timing, or (b) adopt an explicit product policy that 'many failed logins with no success' is treated as suspicious regardless of intent (accepting gave_up as a boundary case), or (c) add source-reputation / cross-session features (repeated connections from one source over time).

### Caveats
- Loopback + vsFTPD single-container lab; benign corpus uses pyftpdlib/custom (vsftpd held out).
- Frozen vsFTPD corpus used for final evaluation ONLY -- never training/selection/tuning.
- No promotion, no threshold/heuristic change, no merge.

## Files
`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`drop_one_ablation.csv`, `per_scenario_family_metrics.csv`, `per_scenario_metrics.csv`,
`per_client_metrics.csv`, `per_server_metrics.csv`, `per_environment_metrics.csv`,
`bootstrap_cis.json`, `shap_top_features.csv`, `ftps_encrypted_metrics.csv`,
`confusion_*.csv`, `leakage_validation.json`, `model_hashes_before_after.json`,
`experiment_metadata.json`, `final_verdict.json`.
