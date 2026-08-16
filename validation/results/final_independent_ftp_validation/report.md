# Final independent FTP validation - report

**All models frozen (production, Candidate 2, robust behavioural candidate, ablation
models, ml.py/live_capture.py/pcap_validation.py verified before==after). This corpus
is TEST-ONLY -- never training, weighting, threshold, feature, or hyperparameter
selection. No promotion, no merge.** Branch `claude/final-independent-ftp-validation`.

## What this test proves (and what it does not)

**Different stack:** real **vsFTPd** server (never used before), **lftp** client (new),
a genuine **non-loopback** private network (veth pair to an isolated `ip netns`,
`10.77.0.0/24`), and **FTPS/TLS** captured separately. Fresh scenario code (not the
prior generators). So this tests whether the robust candidate generalises to a
different server implementation, a different client, and a different network path.

**It does NOT prove** generalisation to a different OS, physical hosts, the public
internet, or FTP dialects beyond vsftpd -- this is still a single-container lab.

## Cleartext evaluation (behavioural features meaningful)

| Model | FTP recall | Benign recall | FPR | FTP precision | macro-F1 | accuracy |
|---|---|---|---|---|---|---|
| production | 0.0000 | 0.9836 | 0.0164 | 0.0000 | 0.2094 | 0.4580 |
| candidate2 | 0.0857 | 0.8525 | 0.1475 | 0.4000 | 0.3644 | 0.4427 |
| robust_behavioural | 1.0000 | 0.8525 | 0.1475 | 0.8861 | 0.9300 | 0.9313 |
| ablation_30only | 0.0857 | 1.0000 | 0.0000 | 1.0000 | 0.4069 | 0.5115 |
| ablation_behavioural_only | 1.0000 | 0.8197 | 0.1803 | 0.8642 | 0.9140 | 0.9160 |
| ablation_no_failed_logins | 1.0000 | 0.8361 | 0.1639 | 0.8750 | 0.9220 | 0.9237 |

## Bootstrap 95% CIs (capture-level resampling)

| Model | FTP recall [95% CI] | Benign recall [95% CI] | FPR [95% CI] |
|---|---|---|---|
| production | 0.000 [0.000, 0.000] | 0.984 [0.947, 1.000] | 0.016 [0.000, 0.053] |
| candidate2 | 0.087 [0.000, 0.233] | 0.851 [0.773, 0.929] | 0.149 [0.071, 0.227] |
| robust_behavioural | 1.000 [1.000, 1.000] | 0.845 [0.654, 0.982] | 0.155 [0.018, 0.346] |

## With vs without behavioural features / `ftp_failed_logins` ablation (cleartext)

- With behavioural (robust): FTP recall **1.0000**.
- 30 packet-only: **0.0857**;
  behavioural-only (15): **1.0000**.
- WITHOUT `ftp_failed_logins` (44): **1.0000**
  (drop **0.0000**).

## SHAP dependence (robust candidate, cleartext)

- `ftp_failed_logins`: share **0.037** (rank 5)
- successful-auth features (`ftp_has_successful_auth`+`ftp_successful_logins`): **0.002**
- CIC artifacts: **0.428**; all behavioural: **0.116**
- largest single application-layer feature: `ftp_failed_logins` at **0.037**
- top features: Dst Port, Fwd Seg Size Min, Init Fwd Win Byts, Fwd IAT Min, ftp_failed_logins, Fwd Header Len

## Does the robust model still depend on...?

- **`ftp_failed_logins`**: False (SHAP share
  0.0365, removal drops FTP recall by
  0.0000).
- **successful authentication**: False
  (attacker-eventual-success recall 1.0000 -- not fooled by the 230).
- **CIC artifacts**: False
  (share 0.4276).
- **any single application-layer feature**: False
  (max 0.0365 on
  `ftp_failed_logins`).

## FTPS / TLS (encrypted -- behavioural features UNAVAILABLE)

Behavioural features are 0/unavailable under TLS (encrypted control channel); these numbers reflect the 30 packet features only, and behavioural models effectively fall back to their CIC-artifact behaviour on encrypted traffic.

| Model | FTP recall | Benign recall | pred FTP/Benign |
|---|---|---|---|
| production | 0.0000 | 1.0000 | 0/31 |
| candidate2 | 0.3750 | 1.0000 | 9/22 |
| robust_behavioural | 0.0000 | 1.0000 | 0/31 |
| ablation_30only | 0.0417 | 1.0000 | 1/30 |
| ablation_behavioural_only | 0.0000 | 1.0000 | 0/31 |
| ablation_no_failed_logins | 0.0000 | 1.0000 | 0/31 |

## Verdict -- NOT ROBUST

On genuinely different traffic the robust candidate does NOT meet the criteria (FTP 1.000>=0.70? benign 0.852>=0.90? FPR 0.148<=0.10?; beats C2=True; failed-login dependence=False; single-feature=False). See metrics.

- beats_candidate2: **True**  ·  meets_thresholds: **False**  ·
  CI-lower FTP recall **1.000**, benign **0.654**  ·
  promote: **False**

### Caveats
- Single-container lab: real vsftpd + real veth/netns (non-loopback) but no separate OS/VM or physical host; one server implementation; cleartext + one FTPS config.
- FTPS captures analysed separately: behavioural features genuinely unavailable (encrypted).
- This corpus was used for evaluation ONLY -- never training/weighting/threshold/feature/HP selection.
- No promotion, no threshold/heuristic change, no merge.

## Files

`cleartext_metrics.csv`, `confusion_*.csv`, `per_scenario_metrics.csv`,
`per_scenario_family_metrics.csv`, `per_client_metrics.csv`, `per_server_metrics.csv`,
`per_environment_metrics.csv`, `confidence_distribution.csv`, `bootstrap_cis.json`,
`ftps_encrypted_metrics.csv`, `shap_top_features.csv`, `leakage_validation.json`,
`model_hashes_before_after.json`, `evaluation_metadata.json`, `final_verdict.json`.
