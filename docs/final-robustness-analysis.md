# Final robustness analysis — Candidate 2 (analysis only)

**This is a diagnosis, not a change.** No retraining, no threshold change, no
heuristic, no test-case removal, no promotion, no merge. The production model and
Candidate 2 are byte-for-byte unchanged (sha256 verified before == after), and the
independent test corpus + its labels are unaltered. Findings are **model-
attribution evidence** (grouping, feature distributions, SHAP, confidence), not
proven causation.

## Data roles — kept strictly separate

| Role | Source | Use here |
|---|---|---|
| **Training** | CIC `balanced_train_selected` (+ v1/v2 real) | compared *against* only |
| **v2 validation** | `validation/realistic_pcaps_v2/` | prior LOCO; distribution comparison |
| **Independent test** | `validation/independent_real_pcaps/` (36 PCAPs, 420 flows) | **FROZEN** — evaluation/diagnosis only, never any fitting decision |

The independent set was never used for training, weighting, threshold, feature,
model, or calibration decisions.

## Headline (frozen, reproduced exactly)

Candidate 2 on the independent test: **FTP recall 0.754**, **benign FP rate
0.282** (22 FPs / 78 benign flows), macro-F1 0.672. Production: FTP recall 0.000,
benign FP 0.026.

## 1. Where the benign false positives cluster

FPs are **not** uniform (`benign_fp_breakdown.csv`). Only **11%** of benign
captures are FP-free. They concentrate in **command-heavy and transfer** benign
sessions (multi-command, MKD/RMD/RNFR/RNTO/DELE/APPE/SIZE/MDTM/NLST/STAT,
uploads/downloads, reconnect) and in **active mode** — behaviours thin or absent
in training — rather than plain logins. The FPs are moderately confident (see §7),
so they are not just boundary noise.

## 2. Scenario difficulty

Hardest FTP-BruteForce scenarios for Candidate 2 (`scenario_metrics.csv`):
`many_attempts` (0.44), `different_usernames` (0.45), `different_passwords`
(0.50), `fast_custom_active` (0.50), `few_attempts` (0.50). The FTP improvement is
**scenario-dependent, not broad**.

## 3. Client / server generalisation (support shown)

| group | n | FTP recall | benign recall | benign FP |
|---|---|---|---|---|
| curl | 40 | 1.000 | 0.750 | 0.250 |
| python-ftplib | 370 | 0.727 | 0.726 | 0.274 |
| raw-socket (client) | 2 | 1.000 | n/a | n/a *(tiny)* |
| wget | 8 | n/a | 0.625 | 0.375 *(small)* |
| **server: pyftpdlib** | 157 | **0.902** | 0.706 | 0.294 |
| **server: raw-socket** | 263 | **0.671** | 0.727 | 0.273 |

FTP recall is **markedly lower against the raw-socket server (0.671) than
pyftpdlib (0.902)** — i.e. the improvement is partly tied to the pyftpdlib server
family it was trained on, and generalises less to a genuinely different server
implementation. Benign FP is high (~0.27–0.29) on both.

## 4. Active vs passive mode (`mode_metrics.csv`)

| mode | flows | FTP recall | benign recall | benign FP | macro-F1 |
|---|---|---|---|---|---|
| active | 20 | 0.500 | 0.500 | **0.500** | 0.451 |
| passive | 400 | 0.767 | 0.730 | 0.270 | 0.684 |

Active-mode traffic is systematically harder on every metric — **but n = 20**, so
this is a signal to investigate with more data, not a firm conclusion.

## 5. Feature distributions — the core mechanism

Medians across the three data roles (`feature_distribution_final.csv`):

| Feature | CIC-FTP (train) | v2 real-FTP | indep real-FTP | indep benign |
|---|---|---|---|---|
| **Fwd Seg Size Min** | **40** | 32 | **32** | **32** |
| Init Fwd Win Byts | 26 883 | 65 495 | 65 495 | 32 780 |
| Dst Port | 21 | 2121 | 2140 | 43 126 |
| Flow Duration (µs) | 4 | 240 | 249 | 105 |
| Flow Pkts/s | 476 883 | 58 312 | 56 329 | **366 438** |

Two things stand out, and together they explain the benign FPs:

1. **`Fwd Seg Size Min` is identical (32) for real FTP *and* real benign** — and
   both differ from CIC's 40. The feature the model leans on to flag FTP does
   **not separate real FTP from real benign**.
2. Real **benign** traffic has **high `Flow Pkts/s` (~366k)** — close to CIC-FTP's
   477k artifact and far above real FTP's ~56k. Benign flows therefore look
   "FTP-like" on exactly the CIC-derived cues the model still uses.

So real benign traffic **enters the model's learned FTP region**, producing
confident false positives, while the CIC artifact values it was originally trained
on (4 µs flows, `Fwd Seg Size Min=40`) match neither real class.

## 6. SHAP by error group (`shap_error_groups.csv`, attribution — not causation)

Top mean-|SHAP| features per Candidate-2 error group:

- **TP_FTP:** Dst Port, Fwd Seg Size Min, Init Fwd Win Byts, Fwd Pkts/s, Fwd IAT Min
- **FN_FTP (missed FTP):** Fwd Seg Size Min, Dst Port, Init Fwd Win Byts, Fwd Pkts/s, Flow Pkts/s
- **TP_Benign:** Dst Port, Fwd Seg Size Min, Init Fwd Win Byts, Fwd Pkts/s, Fwd IAT Min
- **FP_Benign:** Fwd Seg Size Min, Dst Port, Init Fwd Win Byts, Fwd Pkts/s, Fwd IAT Min

Answers to the required questions:

- **Are FPs driven by the same features that identify FTP?** **Yes** — FP_Benign
  and TP_FTP share the same top features (`Fwd Seg Size Min`, Dst Port,
  `Init Fwd Win Byts`). Benign FPs are benign flows entering the learned FTP
  region via those features.
- **Are missed FTP flows outside the learned real-FTP region?** The FN_FTP group
  leans on the same features but is not pushed over the decision boundary — i.e.
  they sit near the region edge rather than in a different feature space.
- **Does Candidate 2 still depend primarily on CIC artifacts?** **Yes** —
  `Fwd Seg Size Min` and `Init Fwd Win Byts` remain top across all groups.
- **Are benign FPs benign traffic entering the FTP region?** The attribution
  evidence supports exactly that.

## 7. Confidence by error group (`confidence_error_groups.csv`)

| group | n | mean | median | p10 | p90 |
|---|---|---|---|---|---|
| TP_FTP | 258 | 0.893 | 0.985 | 0.507 | 1.000 |
| FN_FTP | 84 | 0.663 | 0.540 | 0.540 | 0.826 |
| TP_Benign | 56 | 0.836 | 0.948 | 0.540 | 0.999 |
| **FP_Benign** | 22 | **0.810** | 0.860 | 0.507 | 0.960 |

**Benign false positives are moderately confident** (mean 0.81; ~27% at ≥0.90).
They are not merely low-confidence noise that a threshold tweak would clear — and
**the threshold was not changed**.

## 8. Capture-level robustness (`capture_metrics.csv`, weak captures not hidden)

- FTP captures with recall ≥0.5: **89%**; ≥0.75: **56%**; ≥0.9: **56%**.
- Benign captures with **zero** false positives: **11%**.

## 9. Data requirements — ranked by expected value (evidence-based)

Derived strictly from the failure patterns above (`data_requirements.json`; no
synthetic data, no external traffic):

1. **Real multi-host LAN traffic with more OSes and FTP clients** — corpus is
   loopback-only and SHAP still shows CIC-artifact dependence.
2. **More benign upload/download sessions across clients** — 10 of the FP benign
   captures are transfers.
3. **More FTP brute-force diversity (tools/pacing/patterns)** targeting the
   weakest scenarios (`many_attempts`, `different_usernames`,
   `different_passwords`).
4. **More benign command-heavy sessions** (MKD/RMD/RNFR/RNTO/DELE/APPE/SIZE/MDTM/
   NLST/STAT, multi-command, reconnect) — 7 of 16 FP benign captures are
   command-heavy.
5. **Additional FTP server implementations** — FTP recall differs by 0.23 between
   the two server implementations tested.
6. **More benign active-mode sessions** — active-mode benign FP 0.50 vs passive
   0.27 (small n).

## 10. Final recommendation — **B. Collect another specifically targeted real-PCAP corpus**

The evidence shows a **real but scenario-dependent** FTP-detection gain over
production, with a **structured, explainable** benign-FP cause: real benign
traffic overlaps the model's CIC-artifact-based FTP region on `Fwd Seg Size Min`
and `Flow Pkts/s`. That is addressable with **targeted data** (benign
command-heavy / active-mode / transfer sessions, more server implementations, and
harder brute-force scenarios), then **re-evaluated on a fresh independent set**.

It is **not** yet time to run another training experiment (option C): there is no
new data to train on that would plausibly fix the overlap, and retraining on the
current corpora has already been shown (v2 → independent) to leave the artifact
dependence and benign-FP cost. Stopping entirely (option A) would discard a
genuine, diagnosable signal. **Collect targeted data first (B), re-test, then
decide on C.**

`final_robustness_verdict.json`:
`candidate_promote=false`, `independent_test_frozen=true`,
`retraining_performed=false`, `threshold_changed=false`, `heuristics_added=false`,
`production_model_changed=false`.

### Caveats
- Independent corpus is small (36 captures / 420 flows), loopback-only, 2 classes;
  several groups (active mode, wget, raw-socket client) have tiny support and are
  flagged as such.
- All findings are model-attribution evidence, not proven causation.
- No statistical-significance claim.

## Reproduce

```bash
# from webapp_django/
python manage.py final_robustness
python manage.py test predictor.tests.FinalRobustnessTests
```

## Files (`validation/results/final_robustness/`)

`benign_fp_breakdown.csv`, `scenario_metrics.csv`, `client_server_metrics.csv`,
`mode_metrics.csv`, `feature_distribution_final.csv`, `shap_error_groups.csv`,
`confidence_error_groups.csv`, `capture_metrics.csv`, `data_requirements.json`,
`final_robustness_verdict.json`, `report.md`.
