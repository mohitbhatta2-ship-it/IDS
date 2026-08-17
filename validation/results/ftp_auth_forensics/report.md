# Auth-forensics FTP detector - report

**Candidates only; production frozen (production, Candidate 2, C3, the cross-session model,
the per-connection candidate, ml.py/live_capture.py/pcap_validation.py/ftp_behavioral.py/
ftp_cross_session.py unchanged, before==after). 30 packet features preserved; per-connection,
cross-session and auth-forensics features appended. Selection used CIC held-out + LOCO only;
the fresh pure-ftpd corpus was evaluated once. No promotion, no merge.** Branch
`claude/ftp-auth-forensics`.

## Idea

Every prior detector could not separate a benign user who **mistypes then logs in** from an
attacker who **fails then succeeds** in a single session -- failure count and session count
are identical. The auth-forensics features quantify what the failed passwords *look like*: a
human mistypes (edit-distance-**close** to the correct password) while an attacker guesses a
dictionary (edit-distance-**far**). Features: distinct failed passwords, min/mean Levenshtein
distance from failed attempts to the successful one, edit-distance between consecutive
attempts, password length spread / reuse, inter-attempt timing. `ftp_failed_logins` is NOT in
the feature set (C3 drops it), so the model cannot lean on failure count.

## FINAL fresh pure-ftpd test (evaluated once)

| Model | FTP recall | Benign recall | FPR | macro-F1 | accuracy |
|---|---|---|---|---|---|
| production | 0.0000 | 0.9130 | 0.0870 | 0.2545 | 0.6176 |
| candidate2 | 0.0000 | 0.7826 | 0.2174 | 0.3462 | 0.5294 |
| C3_43 | 1.0000 | 0.9565 | 0.0435 | 0.9671 | 0.9706 |
| cross_session_58 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| per_connection_67 | 1.0000 | 0.9565 | 0.0435 | 0.9671 | 0.9706 |
| candidate_af_77 | 0.9091 | 1.0000 | 0.0000 | 0.9656 | 0.9706 |
| ablation_no_forensic_67 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| ablation_no_editdistance_74 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

### Bootstrap 95% CIs
| Model | FTP recall [95% CI] | Benign recall [95% CI] | FPR [95% CI] |
|---|---|---|---|
| per_connection_67 | 1.000 [1.000,1.000] | 0.950 [0.786,1.000] | 0.050 [0.000,0.214] |
| candidate_af_77 | 0.898 [0.600,1.000] | 1.000 [1.000,1.000] | 0.000 [0.000,0.000] |
| ablation_no_forensic_67 | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 0.000 [0.000,0.000] |

### The decisive case: benign mistype vs attacker dictionary-then-success
Recall by scenario family (columns: candidate_af / per-connection / no-edit-distance ablation).

| scenario family (label) | candidate_af | per-connection | no-edit-distance |
|---|---|---|---|
| mistype/typo-then-success (Benign) | 1.0000 | 1.0000 | 1.0000 |
| dict_success/dict-then-success (FTP) | 0.6667 | 1.0000 | 1.0000 |

- Forensic features improve the decisive case over per-connection: **False**
- Edit-distance features carry the separation (removing them hurts): **False**

## SHAP (candidate, fresh pure-ftpd corpus)

- forensic block share **0.006** (top forensic: `ftpaf_mean_interattempt_s`); edit-distance features share **0.000** (top: `ftpaf_mean_editdist_fail_to_success`); CIC artifacts **0.390**
- `ftpx_sessions_per_source`: share **0.000** (rank 73) -- session count is NOT the main signal
- `ftp_failed_logins` in feature set: **False** (F43 drops it -- the model cannot lean on it)
- largest single feature `Dst Port` **0.174** (single-feature shortcut: False)
- top: Dst Port, Fwd Seg Size Min, Init Fwd Win Byts, Fwd Header Len, Flow IAT Mean, Flow Pkts/s

## Ablation / shortcut checks
- Session count is the main signal: **False**; single-feature shortcut:
  **False**.
- `ftp_failed_logins` in the feature set: **False** (excluded by design).
- `ftpx_sessions_per_source` SHAP rank: **73**; edit-distance SHAP share:
  **0.0002**.

## Verdict -- NOT EFFECTIVE -- DISTINCTION NOT RELIABLY OBSERVABLE

On the fresh pure-ftpd corpus the auth-forensics features do NOT reliably separate a benign mistype-then-success from a single-session dictionary-then-success attack (benign typo recall 1.0, attacker dict-success recall 0.6667; vs per-connection 1.0/1.0; edit-distance carries it: False). Failure count and session structure are identical for the two, and once an attacker's guess happens to succeed in one session the wire trace is not reliably distinguishable from a human who mistyped. RECOMMENDATION: handle this at the product level -- e.g. post-login step-up verification (MFA / device or IP reputation), server-side rate-limiting and lockout, and alerting on the successful login's context -- rather than continuing to tune traffic features. Report the negative result honestly; no promotion.

- target_met: **False**  ·  cic_no_regression: **True**  ·
  session_count_main: **False**  ·  single_feature_shortcut:
  **False**  ·  promote: **False**

### Caveats
- Benign 'mistypes' are synthetic single-edit typos; a real user could paste a wholly wrong saved password (edit-far) and look like a dictionary guess -- a fundamental ambiguity.
- Fresh server (pure-ftpd), new netns/subnet (10.111.0.x)/port and new scenarios, but still a single-container lab -- not a separate OS/host or the public internet.
- pure-ftpd's own anti-bruteforce delay/lockout bounds single-session attack length.
- Cleartext control channel only; under FTPS the passwords are encrypted and these features are unavailable (handled separately).
- Corpus evaluated once, for reporting; never training/selection/tuning.
- Small corpus (wide CIs). No promotion, no threshold change, no merge.

## Files
`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`per_scenario_family_metrics.csv`, `bootstrap_cis.json`, `shap_top_features.csv`,
`confusion_*.csv`, `leakage_validation.json`, `model_hashes_before_after.json`,
`final_verdict.json`.
