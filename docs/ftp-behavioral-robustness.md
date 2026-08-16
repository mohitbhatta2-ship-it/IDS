# FTP behavioural-model robustness stress-test + ablation

**Frozen models only. The production model, Candidate 2, and the 45-feature
behavioural candidate are byte-for-byte unchanged (verified). Ablation models are
diagnostic, trained on the approved corpora only. The messy corpus is TEST-ONLY —
never used for training, selection, or tuning. No promotion, no threshold/heuristic
change, no merge.** Branch `claude/ftp-behavioral-robustness`.

## Why this test

The behavioural experiment scored 1.000/1.000 on a **clean** corpus where benign
sessions always authenticated and brute force always failed — so a single feature
(`ftp_failed_logins`) could separate the classes. A perfect score is a reason for
scrutiny. This experiment collects deliberately **messy** real FTP traffic where the
failed/successful-login proxy breaks, and runs a **4-way ablation** to test whether
`ftp_failed_logins` is a label shortcut.

## The messy corpus (test-only)

**34 fresh PCAPs (20 Benign, 14 FTP-BruteForce), 170 flows, 0 incomplete/zero-filled**,
hash-disjoint from v1/v2/independent/targeted (leakage all_pass), new addresses
(127.0.0.11-13), new ports (2330/2340), multi-user servers. Adversarial families:

- **mistype** — benign user fails 1–3 passwords, then logs in successfully.
- **gave_up** — benign user fails, then disconnects (never authenticates → *looks
  like brute force*).
- **eventual_success** — attacker keeps guessing and *eventually hits a correct
  password* (→ *looks partially benign*).
- **incomplete** — session cut before any auth outcome (no control evidence).
- **multi_user**, activity, transfers, reconnect, active mode, curl/wget/raw-socket.

## Evaluation + ablation (all on the messy corpus)

| Model | FTP recall | Benign recall | Benign FP | macro-F1 | accuracy |
|---|---|---|---|---|---|
| production (30) | 0.000 | 0.978 | 0.022 | 0.244 | 0.529 |
| Candidate 2 (30) | 0.885 | 0.794 | 0.206 | 0.835 | 0.835 |
| **behav_full (30+15)** | **1.000** | **0.935** | **0.065** | **0.965** | **0.965** |
| ablation: 30 only | 0.436 | 0.848 | 0.152 | 0.634 | 0.659 |
| ablation: behavioural only (15) | 1.000 | 0.913 | 0.087 | 0.953 | 0.953 |
| ablation: 30+15 **without `ftp_failed_logins`** | 0.590 | 0.870 | 0.130 | 0.730 | 0.741 |

## Behaviour on the adversarial families (behav_full model)

| Family | Label | flows | recall | note |
|---|---|---|---|---|
| **gave_up** | Benign | 4 | **0.500** | 2/4 benign-that-fails flagged as attack (the shortcut showing) |
| **mistype** | Benign | 14 | **0.714** | 4/14 flagged as attack |
| **eventual_success** | FTP | 32 | **1.000** | attacker-that-succeeds still caught — not fooled by the 230 |
| **incomplete** | Benign | 4 | 1.000 | no auth evidence → correctly benign |
| multi_user / activity / reconnect / typo_user | Benign | — | 1.000 | robust |

## Is `ftp_failed_logins` a label shortcut? — Partly, but not purely

Three lines of evidence, read honestly:

1. **Ablation:** removing `ftp_failed_logins` drops FTP recall **1.000 → 0.590** —
   the model **relies heavily** on it for FTP detection.
2. **SHAP (messy corpus):** `ftp_failed_logins` is the **4th** feature at only
   **~8% importance share**; the top three are the CIC artifacts (`Dst Port`,
   `Fwd Seg Size Min`, `Init Fwd Win Byts`). The model is **not dominated by one
   feature** on average, and `behavioural only` (all 15) still scores 1.000/0.913 —
   so *other* behavioural features carry signal too.
3. **Failure mode:** the errors concentrate exactly where the proxy is wrong —
   benign sessions that merely fail to authenticate (`gave_up` 0.50, `mistype`
   0.71) are partly flagged as attacks, because the model treats "failed logins
   present" as attack-like.

**Reconciliation:** `ftp_failed_logins`'s *average* importance is modest (8%) but it
is *decisive for the FTP class* (it flips FTP flows off the artifact-based benign
default), which is why removing it halves FTP recall. Crucially, the model is **not
purely a `ftp_failed_logins` shortcut**: it still catches attacker-eventual-success
(1.000, not fooled by the successful 230), handles incomplete evidence, and the full
behavioural set beats every 30-feature model even on messy data. But it **over-relies**
on failed-login count and therefore raises false positives on benign users who fail
to log in.

## Verdict — PROMISING BUT NEEDS MORE DATA

On messy traffic the behavioural model is **substantially better than the
30-feature models** (FTP recall 1.000 vs Candidate 2's 0.885; benign recall 0.935 vs
0.794; benign FP 0.065 vs 0.206) and is **not a pure label shortcut**. But it is
**not yet robust**: benign sessions that fail authentication (mistype/gave-up) are
partly misclassified as attacks, and FTP detection leans on `ftp_failed_logins`.

`final_verdict.json`: `verdict = "PROMISING BUT NEEDS MORE DATA"`,
`is_label_shortcut = false`, `is_robust = false`, `promote = false`.

### What the failure mode says is needed

The model was trained only on **clean** data (benign never fails, brute force never
succeeds), so it never learned that *a few* failed logins can be benign. It needs
training traffic that includes **benign failed logins (mistypes, give-ups)** and
**attacks that succeed**, so it learns the discriminator that actually matters —
*sustained* failure / attempt-rate / no-eventual-success — rather than "any failed
login → attack". This is a data problem, not a feature-space dead end: the
behavioural direction is promising; it needs messier training data (and a fresh
independent test) before any promotion.

### Caveats
- Messy corpus is small (34 captures / 170 flows), loopback-only, two server
  implementations, cleartext FTP only. Results are diagnostic, not final.
- The messy corpus was never used for training/selection/tuning.
- No threshold change, no heuristic, no promotion.

## Reproduce

```bash
# from webapp_django/
python manage.py collect_robustness_pcaps
python3 ../validation/robustness_pcaps/verify_pcaps.py   # (if present)
python manage.py ftp_behavioral_robustness
python manage.py test predictor.tests.RobustnessCorpusArtifactTests \
                      predictor.tests.RobustnessResultsTests
```

## Integrity

Production model, Candidate 2, the 45-feature candidate,
`ml.py`/`live_capture.py`/`pcap_validation.py`, and the v1/v2/independent/targeted
PCAPs verified unchanged (before == after). The messy corpus is hash-disjoint from
all prior corpora and used for evaluation/ablation only. Ablation models stored
under `validation/models/ftp-behavioral-robustness/`. The `custom_ftp_server`
multi-user extension is backward-compatible (existing capture tests pass).

## Files (`validation/results/ftp_behavioral_robustness/`)

`ablation_and_model_metrics.csv`, `per_scenario_family_metrics.csv`,
`per_scenario_metrics.csv`, `per_client_metrics.csv`, `per_server_metrics.csv`,
`confidence_distribution.csv`, `confusion_*.csv`, `shap_top_features.csv`,
`leakage_validation.json`, `model_hashes_before_after.json`, `training_metadata.json`,
`final_verdict.json`, `report.md`.
