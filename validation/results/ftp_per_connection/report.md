# Per-connection + cross-session FTP detector - report

**Candidates only; production frozen (production, Candidate 2, C3, the cross-session model,
ml.py/live_capture.py/pcap_validation.py/ftp_behavioral.py unchanged, before==after). 30
packet features preserved; per-connection + cross-session features appended. Selection used
CIC held-out + LOCO only; the fresh proftpd corpus was evaluated once. No promotion, no
merge.** Branch `claude/ftp-per-connection-detector`.

## Idea

The cross-session model failed its 2nd independent test by missing single-session attacks
(it leaned on sessions-per-source). This candidate adds **per-connection** brute-force
features -- attempts / failures / rate / credential-variation *within a connection* -- so an
attack is detectable regardless of connection count, keeping cross-session features as
supporting context. Training explicitly includes packed single-session attacks across the
full session-count spectrum.

## FINAL fresh proftpd test (evaluated once)

| Model | FTP recall | Benign recall | FPR | macro-F1 | accuracy |
|---|---|---|---|---|---|
| production | 0.0000 | 0.9773 | 0.0227 | 0.2606 | 0.5972 |
| candidate2 | 0.0000 | 0.7727 | 0.2273 | 0.3208 | 0.4722 |
| C3_43 | 0.8571 | 0.9773 | 0.0227 | 0.9254 | 0.9306 |
| cross_session_58 | 0.8214 | 1.0000 | 0.0000 | 0.9241 | 0.9306 |
| candidate_pc_67 | 0.8929 | 1.0000 | 0.0000 | 0.9552 | 0.9583 |
| ablation_no_cross_54 | 0.8214 | 1.0000 | 0.0000 | 0.9241 | 0.9306 |
| ablation_no_per_connection_56 | 0.8571 | 1.0000 | 0.0000 | 0.9398 | 0.9444 |

### Bootstrap 95% CIs
| Model | FTP recall [95% CI] | Benign recall [95% CI] | FPR [95% CI] |
|---|---|---|---|
| C3_43 | 0.850 [0.600,1.000] | 0.974 [0.900,1.000] | 0.026 [0.000,0.100] |
| cross_session_58 | 0.810 [0.522,1.000] | 1.000 [1.000,1.000] | 0.000 [0.000,0.000] |
| candidate_pc_67 | 0.885 [0.714,1.000] | 1.000 [1.000,1.000] | 0.000 [0.000,0.000] |

### Attack recall by sessions-per-source (the decisive test)
Columns: candidate_pc / cross / C3 / no-cross / no-per-connection.

| sessions/source | n | pc | cross | C3 | no-cross | no-pc |
|---|---|---|---|---|---|---|
| 1 | 5 | 0.8000 | 0.8000 | 1.0000 | 0.8000 | 0.8000 |
| 2-3 | 6 | 0.6667 | 0.3333 | 0.3333 | 0.3333 | 0.5000 |
| 4-6 | 17 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Single-session attack recall: **candidate_pc 0.8000**, cross
0.8000, no-cross 0.8000, no-per-connection
0.8000.

## SHAP (candidate, fresh proftpd corpus)

- per-connection block share **0.018** (top: `ftppc_mean_conn_duration_s`); cross-session block **0.048**; CIC artifacts **0.412**
- `ftpx_sessions_per_source`: share **0.000** (rank 61) -- session count is NOT the main signal
- largest single feature `Dst Port` **0.186** (single-feature shortcut: False)
- top: Dst Port, Fwd Seg Size Min, Init Fwd Win Byts, ftpx_failure_rate_across_sessions, Fwd IAT Min, Fwd Header Len

## Ablation checks
- Removing cross-session features keeps single-session detection: **False**
  (no-cross single-session recall 0.8000).
- Removing per-connection features drops single-session detection: **False**
  (no-per-connection single-session recall 0.8000) -- shows the per-connection block is what enables it.
- Session count is the main signal: **False**; single-feature shortcut: **False**.

## Verdict -- PROMISING -- NEEDS MORE DATA

The per-connection candidate BREAKS the session-count shortcut (sessions_per_source SHAP rank 61, share ~0) and is the best model (FTP recall 0.893 vs cross 0.821 / C3 0.857; benign 1.000, FPR 0.000; packed single-session attacks detected). But it misses the strict targets on this small fresh corpus (FTP 0.893>=0.90? False; single-session attack recall 0.8>=0.90? False) -- the residual is the attacker-eventual-success-in-ONE-session case, network-indistinguishable from a benign mistype. Needs more/larger independent data (CI lo FTP 0.714); do not tune on the test.

- target_met: **False**  ·  cic_no_regression: **True**  ·
  session_count_main: **False**  ·  single_feature_shortcut:
  **False**  ·  promote: **False**

### Caveats
- Fresh server (proftpd), client (ncftp), netns/subnet (10.99.0.x) and scenarios, but still a single-container lab -- not a separate OS/host or the public internet.
- proftpd's MaxLoginAttempts still bounds single-session length; a real attacker could pack more.
- Corpus evaluated once, for reporting; never training/selection/tuning.
- Small corpus (wide CIs). No promotion, no threshold change, no merge.

## Files
`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`per_scenario_family_metrics.csv`, `per_session_structure_metrics.csv`, `bootstrap_cis.json`,
`shap_top_features.csv`, `confusion_*.csv`, `leakage_validation.json`,
`model_hashes_before_after.json`, `final_verdict.json`.
