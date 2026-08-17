# Is `ftp_failed_logins` causing the benign false positives? — a controlled ablation

**Candidates only; production frozen. The production model, Candidate 2, the 45-feature
behavioural candidate, and `ml.py` / `live_capture.py` / `pcap_validation.py` /
`ftp_behavioral.py` are byte-for-byte unchanged (verified before == after). Training data
is HELD CONSTANT; only the feature set varies. Selection used CIC held-out + capture-level
LOCO on the approved corpora only; the frozen vsFTPD corpus was evaluated exactly once, for
reporting. No promotion, no merge.** Branch `claude/final-independent-ftp-validation`.

## Design (a clean, confounder-free ablation)

Every candidate is trained on the *same* approved corpora the current model used —
CIC + v1 + v2 + targeted + robust_train — so any difference is attributable to the
**feature set alone**, not a data change.

| Candidate | Features | What is removed |
|---|---|---|
| **C1** | 45 | none (the current frozen candidate, evaluated as-is) |
| **C2** | 44 | `ftp_failed_logins` |
| **C3** | 43 | `ftp_failed_logins` + `ftp_failed_login_ratio` |
| **C4** | 40 | all direct failure-volume encoders: `ftp_failed_logins`, `ftp_failed_login_ratio`, `ftp_error_responses_5xx`, `ftp_login_attempts`, `ftp_total_commands` |

Selection (CIC + LOCO only, never the vsFTPD test) picked **C3** (LOCO macro-F1 0.9948,
CIC no-regression). All of C2/C3/C4 pass the CIC no-regression gate.

## CIC held-out (no regression)

| Model | FTP recall | Benign recall | accuracy |
|---|---|---|---|
| production | 0.884 | 0.986 | 0.9803 |
| C1 full 45 | 0.872 | 0.986 | 0.9808 |
| C2 (−failed_logins) | 0.872 | 0.987 | 0.9811 |
| C3 (−failed_logins −ratio) | 0.896 | 0.986 | 0.9804 |
| C4 (−all failure volume) | 0.898 | 0.986 | 0.9802 |

No candidate regresses on CIC.

## FINAL vsFTPD test (cleartext, evaluated once)

| Model | FTP recall | Benign recall | FPR | FTP precision | macro-F1 | accuracy |
|---|---|---|---|---|---|---|
| production (30) | 0.000 | 0.984 | 0.016 | 0.000 | 0.209 | 0.458 |
| Candidate 2 (30) | 0.086 | 0.852 | 0.148 | 0.400 | 0.364 | 0.443 |
| **C1 full 45 (current)** | 1.000 | 0.852 | 0.148 | 0.886 | 0.930 | 0.931 |
| **C2 (−failed_logins)** | 1.000 | 0.836 | 0.164 | 0.875 | 0.922 | 0.924 |
| **C3 (−failed_logins −ratio)** | **1.000** | **0.869** | **0.131** | 0.897 | 0.938 | 0.939 |
| **C4 (−all failure volume)** | **0.600** | **0.902** | **0.098** | 0.875 | 0.738 | — |

Bootstrap 95% CIs: C1 benign 0.845 [0.654, 0.982]; C3 benign 0.863 [0.681, 1.000], FPR
0.137 [0.000, 0.319] — the improvement is real but within overlapping CIs on this small
corpus. Per family (C3): `mistype` 0.667 → **0.750**, `eventual_success` 1.000,
`gave_up` **0.000** (unchanged); every brute-force family stays 1.000.

## Verdict — BENIGN IMPROVED WITHOUT LOSING FTP, BUT TARGET NOT FULLY MET

`final_verdict.json`: `verdict = "BENIGN IMPROVED WITHOUT LOSING FTP -- BUT TARGET NOT
FULLY MET"`, `target_met = false`, `promote = false`. Target (FTP ≥ 0.90 AND benign ≥ 0.90
AND FPR ≤ 0.10, no CIC regression) is not met.

## What the ablation actually shows — the honest, complete answer

**`ftp_failed_logins` is only *part* of a broader dependence on "failure volume", and that
dependence is simultaneously the cause of the benign false positives AND the signal that
makes brute-force detection work.**

1. **Removing `ftp_failed_logins` alone (C2) does *not* help** — benign actually drops
   slightly (0.852 → 0.836). The model routes around it using the strongly-correlated
   `ftp_failed_login_ratio` and `ftp_error_responses_5xx`. So the earlier drop-one hint (in
   a different 57-feature model with different training data) does **not** reproduce as a
   clean single-feature fix here.

2. **Removing `ftp_failed_logins` *and* the failure *ratio* (C3) helps modestly** — benign
   0.852 → 0.869, FPR 0.148 → 0.131, FTP recall held at 1.000, no CIC regression. This is
   the best trade-off, but SHAP shows the model has simply promoted the *next* failure
   proxy: `ftp_error_responses_5xx` is now its 4th-most-important feature (top three remain
   CIC artifacts; behavioural share 12%, CIC-artifact share 42%).

3. **Removing *all* failure-volume features (C4) finally fixes the benign side** —
   FPR 0.098, benign recall 0.902 — **but FTP recall collapses to 0.600**, because the
   model no longer has the very signal that distinguishes a brute force from a normal
   session. This is the trade-off in its starkest form.

**Conclusion:** failure volume (of which `ftp_failed_logins` is the most direct encoding)
is a *shared* feature — you cannot suppress the benign false positives by removing it
without also removing the brute-force detector. Because `gave_up` benign sessions are, at
the packet/control level, near-identical to a short failed brute force, no reweighting of
these features separates them. Removing the two explicit failed-login features (C3) buys a
small, honest benign improvement at no FTP cost; going further (C4) trades away FTP
detection. The promotion target is not reachable by feature removal alone.

## Trade-off, stated plainly (as requested)

- Remove nothing (C1): FTP 1.000 / benign 0.852 / FPR 0.148.
- Remove the two explicit failed-login features (C3): FTP 1.000 / benign 0.869 / FPR 0.131
  — **strictly better on the benign side at no FTP cost**, but still short of target.
- Remove all failure-volume signal (C4): benign 0.902 / FPR 0.098 (≈ target) but FTP 0.600
  — **benign fixed, FTP detection sacrificed**.

## Recommended next experiment

Feature removal cannot close the gap. The next lever is *information the current features
do not carry*:

1. **Cross-session / source-reputation features** — repeated failed connections from one
   source over time is the signal that actually separates a returning legitimate user from
   a brute-forcer; single-session PCAP features cannot see it.
2. **A cost-aware operating point** rather than argmax — accept a small, controlled benign
   FP budget and pick a threshold on that budget (out of scope here: no threshold changes).
3. **Multi-server benign failed-login data** so any residual pacing/activity signal
   generalises beyond one server's failure timing.
4. Otherwise, accept `gave_up` (many failures, no success) as a labelled boundary case that
   is legitimately treated as suspicious.

## Integrity / leakage (all verified)

Production, Candidate 2, the 45-feature candidate, and
`ml.py`/`live_capture.py`/`pcap_validation.py`/`ftp_behavioral.py` unchanged
(before == after); the frozen vsFTPD independent PCAPs unchanged. Training data held
constant across candidates; 30 packet features preserved; labels from folders; training
corpora disjoint from the frozen test; no independent-test PCAP entered training;
selection used CIC + LOCO only. Candidate models under
`validation/models/ftp-failed-login-ablation/` (never `webapp_data/`).

## Reproduce

```bash
# from webapp_django/
python manage.py ftp_failed_login_ablation --loco-caps 50
python manage.py test predictor.tests.FtpFailedLoginAblationResultsTests \
                      predictor.tests.FtpFailedLoginAblationFeatureSetTests
```

## Files (`validation/results/ftp_failed_login_ablation/`)

`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`per_scenario_family_metrics.csv`, `per_scenario_metrics.csv`, `per_client_metrics.csv`,
`per_server_metrics.csv`, `per_environment_metrics.csv`, `bootstrap_cis.json`,
`shap_top_features.csv`, `ftps_encrypted_metrics.csv`, `confusion_*.csv`,
`leakage_validation.json`, `model_hashes_before_after.json`, `final_verdict.json`.
