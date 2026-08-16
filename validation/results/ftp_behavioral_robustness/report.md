# Behavioural-model robustness stress-test + ablation - report

**Frozen models only (production, Candidate 2, and the 45-feature behavioural
candidate are byte-for-byte unchanged, verified). Ablation models are diagnostic,
trained on the approved corpora (v1+v2+targeted) only. The messy corpus is TEST-ONLY
(never training/selection/tuning). No promotion, no threshold/heuristic change, no
merge.**

## Why this test

The earlier 45-feature model scored 1.000/1.000 on a CLEAN corpus (benign always
authenticated, brute force always failed) -- so a single feature `ftp_failed_logins`
could separate the classes. This corpus deliberately contains the messy cases where
that proxy breaks: benign users who mistype then succeed, benign who fail then give
up (look like brute force), and attackers who eventually guess a correct password
(look partially benign).

## Evaluation + ablation on the messy corpus

| Model | FTP recall | Benign recall | Benign FP | macro-F1 | accuracy |
|---|---|---|---|---|---|
| production | 0.0000 | 0.9783 | 0.0217 | 0.2439 | 0.5294 |
| candidate2 | 0.8846 | 0.7935 | 0.2065 | 0.8352 | 0.8353 |
| behav_full_30+15 | 1.0000 | 0.9348 | 0.0652 | 0.9646 | 0.9647 |
| ablation_30only | 0.4359 | 0.8478 | 0.1522 | 0.6343 | 0.6588 |
| ablation_behavioural_only | 1.0000 | 0.9130 | 0.0870 | 0.9529 | 0.9529 |
| ablation_no_failed_logins | 0.5897 | 0.8696 | 0.1304 | 0.7304 | 0.7412 |

## Behaviour on the adversarial families (full behavioural model)

- **Benign, fail-then-give-up** (looks like brute force): recall **0.5000** over 4 flows.
- **Benign, mistype-then-success**: recall **0.7143** over 14 flows.
- **Attacker, eventual success** (guesses a valid password): recall **1.0000** over 32 flows.
- **Benign, incomplete control evidence** (no auth outcome): recall **1.0000** over 4 flows.

## Ablation - is `ftp_failed_logins` a shortcut?

- Full (30+behavioural) FTP recall: **1.0000**; WITHOUT `ftp_failed_logins`:
  **0.5897** (drop 0.4103).
- Benign FP full: 0.0652; without `ftp_failed_logins`: 0.1304.

## SHAP on the messy corpus (does it use one feature or many?)

- `ftp_failed_logins` importance share: **0.08**; all behavioural features: **0.10**; behavioural features in top-5: **1**; dominated by ftp_failed_logins: **False**
- Top features: Dst Port, Fwd Seg Size Min, Init Fwd Win Byts, ftp_failed_logins, Fwd Header Len, Fwd Pkts/s

## Verdict - PROMISING BUT NEEDS MORE DATA

Behavioural model is better than the 30-feature models but not robust on messy cases (FTP recall 1.000, benign recall 0.935, benign FP 0.065; gave_up recall 0.5, attacker-eventual-success recall 1.0). Needs broader/messier real traffic.

- is_label_shortcut: **False**  ·  is_robust: **False**  ·  promote: **False**

### Caveats
- Messy corpus is small (loopback-only, two servers); results are diagnostic, not final.
- The messy corpus was never used for training/selection/tuning.
- No threshold change, no heuristic, no promotion.

## Integrity

Production model, Candidate 2, the 45-feature candidate, ml.py/live_capture.py/
pcap_validation.py, and the v1/v2/independent/targeted PCAPs verified unchanged
(before==after). The messy corpus is hash-disjoint from all prior corpora and was
used for evaluation only. Ablation models stored under
validation/models/ftp-behavioral-robustness/.

## Files

`ablation_and_model_metrics.csv`, `per_scenario_family_metrics.csv`,
`per_scenario_metrics.csv`, `per_client_metrics.csv`, `per_server_metrics.csv`,
`confidence_distribution.csv`, `confusion_*.csv`, `shap_top_features.csv`,
`leakage_validation.json`, `model_hashes_before_after.json`, `training_metadata.json`,
`final_verdict.json`.
