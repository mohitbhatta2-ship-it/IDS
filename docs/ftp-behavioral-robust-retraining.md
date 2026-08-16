# FTP behavioural robust-retraining

**Candidates only. The production model, Candidate 2, the clean-trained 45-feature
behavioural candidates, and `ml.py` / `live_capture.py` / `pcap_validation.py` are
byte-for-byte unchanged (verified before == after). The 34-PCAP messy corpus is the
FINAL independent TEST — never used for training, weighting, hyperparameter/threshold/
feature selection. No promotion, no merge.** Branch
`claude/ftp-behavioral-robust-retraining`.

## The problem this addresses

The behavioural stress-test (`docs/ftp-behavioral-robustness.md`) showed the
clean-trained 45-feature model over-relied on `ftp_failed_logins`. It was trained only
on **clean** data (benign always authenticated; brute force always failed), so it
learned "any failed login → attack". On messy traffic that broke:

- benign users who fail then give up (`gave_up`) — recall **0.50**;
- benign users who mistype then succeed (`mistype`) — recall **0.71**;
- removing `ftp_failed_logins` halved FTP recall (1.000 → 0.590).

The fix hypothesis: train on **messy real traffic whose two classes carry overlapping
failed-login counts**, so the model must learn behavioural *context* rather than the raw
failed-login count.

## What was collected — a messy TRAINING corpus (`validation/robust_train_pcaps`)

**48 fresh real loopback PCAPs (30 Benign, 18 FTP-BruteForce), 224 flows, 0
incomplete/zero-filled.** NEW addresses (127.0.0.14-16 / eth0) and NEW ports
(2430/2440), so it is **hash-disjoint from the frozen 34-PCAP TEST corpus and every
prior corpus** (leakage `all_pass = True`). Genuine FTP interactions (pyftpdlib +
custom raw-socket multi-user servers; ftplib / curl / wget / raw-socket clients;
passive + active mode) captured with `tcpdump`. Labels come from the scenario folder,
never a prediction.

The corpus deliberately makes the failed-login count **overlap** across classes:

| | failed logins 0 | 1 | 2 | 3 | 4+ |
|---|---|---|---|---|---|
| Benign | 82 | 30 | 6 | 14 | 0 |
| FTP-BruteForce | 0 | 2 | 4 | 10 | 76 |

- Benign flows with ≥1 failed login: **50** (mistype / give-up / typo-user / reconnect).
- Attacker flows with ≥1 failed login: **92**; attackers that **eventually authenticate**: **42**.

So "1–3 failed logins" appears in **both** classes, and "successful auth" appears in
both — a model cannot separate the classes on either proxy alone.

## Method (leakage-controlled)

- **Train on:** CIC (30 packet features + NaN behavioural — genuinely missing, HGB
  handles NaN natively) + real `v1 + v2 + targeted + robust_train`.
- **Candidates:** the existing 30-feature control (Candidate 2, frozen); the existing
  30+15 behavioural model (clean-trained, frozen); the 30+15 model retrained WITH the
  messy corpus, **unweighted** and **balanced** weighting.
- **Selection** used **only** the CIC held-out split and capture-level LOCO on the
  approved training corpora (188 captures). The 34-PCAP messy corpus was **not**
  consulted for selection or tuning.
- **Final independent test** = the frozen 34-PCAP messy corpus.
- **4-way ablation** (30 only / behavioural only / 30+behavioural / 30+behavioural
  **without `ftp_failed_logins`**), all now trained WITH the messy data.
- **SHAP** (TreeExplainer) on the selected candidate over the 34-PCAP test.

Selected candidate: **`candidate_robust_unweighted`** (higher LOCO macro-F1: 0.9937 vs
0.9915 balanced; both give identical perfect test results).

## Results

### CIC held-out (no-regression check)

| Model | FTP recall | Benign recall | macro-F1 | accuracy |
|---|---|---|---|---|
| production (30) | 0.884 | 0.986 | 0.863 | 0.9803 |
| Candidate 2 (30) | 0.894 | 0.987 | 0.865 | 0.9809 |
| behav_clean (30+15) | 0.900 | 0.986 | 0.864 | 0.9809 |
| **candidate_robust_unweighted** | **0.871** | **0.986** | **0.862** | **0.9808** |
| candidate_robust_balanced | 0.876 | 0.985 | 0.859 | 0.9795 |

No accuracy regression (0.9808 vs production 0.9803). CIC FTP recall dips slightly vs
the clean-trained model (0.900 → 0.871) but stays above the production-minus-0.02 gate.

### Capture-level LOCO on approved training corpora (pooled, 188 captures)

| Weighting | FTP recall | Benign recall | macro-F1 | accuracy |
|---|---|---|---|---|
| unweighted | 0.993 | 0.995 | 0.9937 | 0.9940 |
| balanced | 0.993 | 0.990 | 0.9915 | 0.9920 |

### FINAL independent test — frozen 34-PCAP messy corpus

| Model | FTP recall | Benign recall | Benign FP | macro-F1 | accuracy |
|---|---|---|---|---|---|
| production (30) | 0.000 | 0.978 | 0.022 | 0.244 | 0.529 |
| Candidate 2 (30) | 0.885 | 0.794 | 0.206 | 0.835 | 0.835 |
| behav_clean (30+15, clean-trained) | 1.000 | 0.935 | 0.065 | 0.965 | 0.965 |
| **candidate_robust_unweighted** | **1.000** | **1.000** | **0.000** | **1.000** | **1.000** |
| candidate_robust_balanced | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |

Confusion (selected): Benign 92/92, FTP-BruteForce 78/78 — **zero errors**.

### The families that used to fail (selected candidate)

| Family | Label | flows | clean-trained recall | **robust-trained recall** |
|---|---|---|---|---|
| gave_up (benign fails then quits) | Benign | 4 | 0.50 | **1.00** |
| mistype (benign fails then succeeds) | Benign | 14 | 0.71 | **1.00** |
| eventual_success (attacker succeeds) | FTP | 32 | 1.00 | **1.00** |
| incomplete / multi_user / activity / reconnect / typo_user | Benign | — | ~1.00 | **1.00** |

The two weakness families (`gave_up`, `mistype`) are now fully corrected.

## Is the `ftp_failed_logins` shortcut broken? — Yes

| Ablation (all trained WITH messy data) | FTP recall on 34-PCAP test |
|---|---|
| 30 only | 0.487 |
| behavioural only (15) | 1.000 |
| 30 + behavioural (full, selected) | 1.000 |
| 30 + behavioural **without `ftp_failed_logins`** | **1.000** |

Removing `ftp_failed_logins` now costs **0.000** FTP recall (clean-trained model lost
0.410). **SHAP** on the selected candidate: `ftp_failed_logins` is **rank 9 at 3.0%**
share (clean-trained: rank 4, ~8%); the top three remain CIC artifacts (`Dst Port`
17.5%, `Fwd Seg Size Min` 16.1%, `Init Fwd Win Byts` 9.8%). The model separates the
classes on broad behavioural + flow context, not the failed-login count.

## Verdict — PROMISING, IMPROVED, MEETS CRITERIA (still: DO NOT auto-promote)

`final_verdict.json`: `verdict = "PROMISING -- IMPROVED, MEETS CRITERIA"`,
`meets_promotion_criteria = true`, `promote = false`.

Against the stated criteria, on the frozen messy test the selected candidate:

- improves on Candidate 2 on all three metrics (FTP 1.000 ≥ 0.885, benign 1.000 ≥
  0.794, FP 0.000 ≤ 0.206) ✓
- FTP recall 1.000 ≥ 0.70 ✓, benign recall 1.000 ≥ 0.90 ✓, benign FP 0.000 ≤ 0.10 ✓
- no CIC regression (accuracy 0.9808 vs 0.9803) ✓
- no major `ftp_failed_logins` dependence (3% share; 0.000 ablation drop) ✓
- meaningful across every messy scenario family (all 1.000) ✓

All criteria are met — but promotion is a **human decision**, and the perfect scores
must be read with the caveats below.

### Caveats (read the perfect scores honestly)

- **Shared generation methodology.** The messy TRAIN corpus (`robust_train_capture`)
  and the messy TEST corpus (`robustness_capture`) are hash/address/port/user-disjoint
  but are produced by the *same family of scenario functions*. The 1.000/1.000 result
  therefore shows the model generalises across new addresses/ports/users/servers for the
  *same behavioural scenarios* — it is **not** evidence of generalisation to
  attacker/benign behaviours or FTP dialects the scenario set never contained. The gain
  over the clean-trained model (benign 0.935 → 1.000; gave_up 0.50 → 1.00) is real and in
  the expected direction; the absolute 1.000 is optimistic.
- Loopback-only, two server implementations, cleartext FTP; small corpora.
- The 34-PCAP corpus was used for final reporting only — never for training/selection/tuning.
- **Before any promotion**, test on genuinely independent real FTP traffic (different
  capture methodology, real servers, FTPS/TLS, non-loopback).
- No threshold change, no heuristic, no promotion, no merge.

## Reproduce

```bash
# from webapp_django/
python manage.py collect_robust_train_pcaps          # messy TRAINING corpus (real FTP)
python manage.py ftp_behavioral_robust_retraining    # train candidates, eval, ablation, SHAP
python manage.py test predictor.tests.RobustTrainCaptureFrameworkTests \
                      predictor.tests.RobustTrainCorpusArtifactTests \
                      predictor.tests.RobustRetrainResultsTests
```

## Integrity

Production, Candidate 2, the clean-trained behavioural candidates,
`ml.py`/`live_capture.py`/`pcap_validation.py`, and the
v1/v2/independent/targeted/robustness PCAP corpora verified unchanged (before == after).
The messy TRAIN corpus is hash-disjoint from the 34-PCAP TEST corpus and all prior
corpora. Candidate + ablation models live under
`validation/models/ftp-behavioral-robust-retraining/`; evidence under
`validation/results/ftp_behavioral_robust_retraining/`.

## Files (`validation/results/ftp_behavioral_robust_retraining/`)

`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`per_scenario_family_metrics.csv`, `per_scenario_metrics.csv`, `per_client_metrics.csv`,
`per_server_metrics.csv`, `confidence_distribution.csv`, `confusion_*.csv`,
`shap_top_features.csv`, `leakage_validation.json`, `model_hashes_before_after.json`,
`training_metadata.json`, `final_verdict.json`, `report.md`.
