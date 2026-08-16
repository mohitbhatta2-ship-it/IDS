# FTP behavioural-feature experiment

**Candidate only. Production model and Candidate 2 are frozen and byte-for-byte
unchanged (verified). No auto-promotion, no merge, no production change.** The
existing 30-feature `pcap_validation` pipeline is untouched; the behavioural
features are added only in an experimental module. Candidate artifacts live under
`validation/models/ftp-behavioral-candidate/`; evidence under
`validation/results/ftp_behavioral/`.

## Question

Every prior experiment hit a **coupled trade-off frontier**: re-weighting the 30
CIC features could raise FTP recall *or* lower benign FP, never both, because real
benign and real FTP overlap on the CIC packet/timing artifacts. Can **FTP
application-layer behavioural features** — read from the control channel itself —
break that frontier?

## New features (experimental, additive)

15 features (`ftp_behavioral.BEHAV_FEATURES`) parsed from the cleartext FTP control
channel: login attempts, **failed logins (530)**, **successful logins (230)**,
**failed-login ratio**, **has-successful-auth**, USER count, distinct commands,
total commands, data-setup responses (150), data commands, 5xx/2xx counts, control
connections, reconnects, commands-per-connection. Measured directly from each PCAP
— no fabrication, no zero-fill, no label information.

Feature set = **30 `ml.FEATURES` + 15 behavioural = 45**, attached per capture.
**CIC flows have no PCAP, so their behavioural features are `NaN`** (genuinely
not-computable; HGB handles NaN natively — this is *not* zero-fill).

## Step 1 — separability gate (training corpora only: v1 + v2 + targeted)

Before training anything, we tested whether the behavioural features separate real
FTP-BruteForce from real Benign:

- Best single feature: **`ftp_failed_logins`, AUC = 1.000**;
  `ftp_failed_login_ratio`, `ftp_error_responses_5xx` also AUC 1.000;
  `ftp_has_successful_auth` / `ftp_successful_logins` separate inversely (AUC 0.000).
- Behavioural-only 5-fold CV macro-F1 = **0.992** over 140 captures.

**Gate passed.** This reflects the real semantics: benign users authenticate
successfully and do activity; brute force is repeated failed authentication. See
`separability_analysis.csv`.

## Step 2 — augmented candidates (CIC held-out + v2 LOCO; selection signals only)

| Candidate | CIC acc / macro-F1 | v2-LOCO FTP recall | v2-LOCO Benign recall | v2-LOCO macro-F1 |
|---|---|---|---|---|
| aug_unweighted | 0.9809 / 0.8637 | 0.994 | 1.000 | 0.993 |
| aug_balanced | 0.9806 / 0.8655 | 0.993 | 1.000 | 0.991 |

No CIC regression, and — unlike the 30-feature models — **both FTP recall and
benign recall are high at once** in v2 LOCO. Selected on best v2-LOCO macro-F1:
**`aug_unweighted`** (selection never used the independent test).

## Final independent test (frozen 36-PCAP set) — the frontier breaks

| Model | FTP recall | Benign recall | Benign FP | macro-F1 | accuracy |
|---|---|---|---|---|---|
| production | 0.000 | 0.974 | 0.026 | 0.123 | 0.181 |
| Candidate 2 | 0.754 | 0.718 | 0.282 | 0.672 | 0.748 |
| control (30-feat, CIC+real) | 0.526 | 0.859 | 0.141 | 0.556 | 0.588 |
| **aug_unweighted (30+behavioural)** | **1.000** | **1.000** | **0.000** | **1.000** | **1.000** |

FTP recall 95% CI **[1.000, 1.000]**, benign recall 95% CI **[1.000, 1.000]**
(capture-level bootstrap). The augmented model satisfies **all** balanced-target
criteria (FTP recall >0.70, benign recall >0.90, benign FP ≪ Candidate 2, no CIC
regression) that no 30-feature candidate could — the trade-off frontier is broken.

## SHAP — how it does it (and what it still leans on)

- The model **uses the behavioural signal**: `ftp_failed_logins` is the **3rd most
  important feature globally** and on the test FTP flows. `uses_behavioural = true`.
- It **did not abandon the CIC artifacts**: `Fwd Seg Size Min`, `Init Fwd Win Byts`
  remain top (`still_cic_artifact_top = true`) — they are still needed for the
  15-class CIC task (where behavioural features are NaN). Behavioural-feature
  importance share ≈ **0.14**.

So the added ~14% behavioural signal — chiefly "how many logins failed" — is what
supplies the FTP-vs-benign discrimination the CIC artifacts could not, without
disturbing CIC performance.

## Verdict — PROMISING, FRONTIER BROKEN; **NOT auto-promoted** (`promote: false`)

The hypothesis is confirmed: **FTP application-layer behavioural features separate
real FTP brute force from real benign where the CIC packet-statistics cannot**, and
on the untouched independent test the augmented model achieves high FTP recall AND
high benign recall simultaneously. This is a genuinely promising direction.

### But read the caveats — a perfect score is a reason for MORE scrutiny, not less

- **The separation is near-perfect partly because the captures are cleanly
  scenario-labelled**: every benign session authenticates successfully; every
  brute-force session only fails. A single feature (`ftp_failed_logins`) already
  achieves AUC 1.0. Real-world traffic is messier — benign users mistype
  (fail-then-succeed, still handled by `has_successful_auth`), truncated benign
  sessions that never complete auth would look attack-like, and attackers who
  eventually guess a password would be missed.
- **Cleartext FTP only.** These features are read from the unencrypted control
  channel. **FTPS/encrypted FTP hides it entirely** — behavioural features would be
  unavailable (NaN), and the model would fall back to the CIC artifacts, i.e. the
  frontier would return. Production FTP is often encrypted.
- **Small, loopback-only corpus** (36 independent captures / 420 flows), two server
  implementations, no third-party traffic. 100% here is not evidence of production
  robustness.
- **Behavioural features are per-capture session context** attached to each flow;
  at true streaming detection time the control session must be observed.

### Honest conclusion

This experiment answers the research question the earlier ones raised: the
FTP-recall/benign-FP coupling was a property of the **feature space** (CIC packet
statistics), not an inherent limit — **semantic application-layer features break
it**. That is the valuable finding. It is **not** a production-ready detector: it
needs diverse, messy, and encrypted real traffic and a fresh independent test
before any promotion. **Nothing is promoted.**

## Reproduce

```bash
# from webapp_django/
python manage.py ftp_behavioral_experiment
python manage.py test predictor.tests.FtpBehaviouralExtractorTests \
                      predictor.tests.BehaviouralRetrainModuleTests \
                      predictor.tests.BehaviouralResultsTests
```

## Integrity

Production model, Candidate 2, `ml.py` / `live_capture.py` / `pcap_validation.py`,
and the v1/v2/independent/targeted PCAPs verified unchanged (before == after). No
independent PCAP entered training (hash-guarded). Capture-level LOCO with per-fold
no-flow-overlap assertions. CIC behavioural features are NaN (not zero-fill).
Candidate models stored outside `webapp_data/`. Reproducibility test confirms same
seed + data → identical predictions.

## Files (`validation/results/ftp_behavioral/`)

`separability_analysis.csv`, `candidate_metrics.csv`, `candidate_comparison.csv`,
`independent_test_metrics.csv`, `independent_per_capture_metrics.csv`,
`confusion_*.csv`, `bootstrap_ci_results.csv`, `shap_comparison.csv`,
`leakage_validation.json`, `model_hashes_before_after.json`,
`training_metadata.json`, `final_verdict.json`, `report.md`.
