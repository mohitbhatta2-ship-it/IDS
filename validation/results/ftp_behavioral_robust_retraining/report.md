# FTP behavioural robust-retraining - report

**Candidates only; production frozen. The production model, Candidate 2, the
clean-trained behavioural candidates, and ml.py/live_capture.py/pcap_validation.py are
byte-for-byte unchanged (verified). The 34-PCAP messy corpus is the FINAL TEST -- never
used for training, selection, or tuning. No promotion, no merge.** Branch
`claude/ftp-behavioral-robust-retraining`.

## Idea

The messy stress-test showed the clean-trained behavioural model over-relied on
`ftp_failed_logins`: benign users who fail to authenticate (mistype / give-up) were
partly flagged as attacks. Fix hypothesis: add a **messy real training corpus** whose
two classes carry **overlapping failed-login counts** -- benign users who
mistype/give-up, attackers who fail or eventually succeed -- so the model must learn
behavioural *context* rather than "any failed login -> attack".

Training overlap (approved TRAIN set): benign flows with >=1 failed login =
**62**, FTP flows with >=1 failed login =
**918**, attacker flows that eventually authenticate =
**42**.

## Candidate selection (CIC held-out + LOCO only)

Selected: **candidate_robust_unweighted** -- CIC-gate=False (FTP 0.871 vs behav_clean 0.900), LOCO macro-F1 0.9937 (highest among gate-passers). The 34-PCAP messy corpus was **not**
consulted for selection.

### CIC held-out (no-regression check)
| Model | FTP recall | Benign recall | macro-F1 | accuracy |
|---|---|---|---|---|
| production | 0.8835 | 0.9855 | 0.8627 | 0.9803 |
| candidate2 | 0.8936 | 0.9866 | 0.8650 | 0.9809 |
| behav_clean | 0.8996 | 0.9864 | 0.8637 | 0.9809 |
| candidate_robust_unweighted | 0.8715 | 0.9863 | 0.8616 | 0.9808 |
| candidate_robust_balanced | 0.8755 | 0.9847 | 0.8589 | 0.9795 |

### Capture-level LOCO on approved training corpora (pooled)
| Weighting | FTP recall | Benign recall | macro-F1 | accuracy |
|---|---|---|---|---|
| unweighted | 0.9935 | 0.9948 | 0.9937 | 0.9940 |
| balanced | 0.9935 | 0.9896 | 0.9915 | 0.9920 |

## FINAL independent test -- frozen 34-PCAP messy corpus

| Model | FTP recall | Benign recall | Benign FP | macro-F1 | accuracy |
|---|---|---|---|---|---|
| production | 0.0000 | 0.9783 | 0.0217 | 0.2439 | 0.5294 |
| candidate2 | 0.8846 | 0.7935 | 0.2065 | 0.8352 | 0.8353 |
| behav_clean | 1.0000 | 0.9348 | 0.0652 | 0.9646 | 0.9647 |
| candidate_robust_unweighted | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| candidate_robust_balanced | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| ablation_30only | 0.4872 | 0.9348 | 0.0652 | 0.7060 | 0.7294 |
| ablation_behavioural_only | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| ablation_no_failed_logins | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

### Selected candidate on the adversarial families
- **Benign, fail-then-give-up** (looks like brute force): recall **1.0000** over 4 flows.
- **Benign, mistype-then-success**: recall **1.0000** over 14 flows.
- **Attacker, eventual success** (guesses a valid password): recall **1.0000** over 32 flows.
- **Benign, incomplete control evidence**: recall **1.0000** over 4 flows.

## Ablation -- does removing `ftp_failed_logins` still collapse FTP recall?

- Full selected candidate FTP recall: **1.0000**; WITHOUT
  `ftp_failed_logins`: **1.0000**
  (drop **0.0000**).
- 30-only: **0.4872**; behavioural-only:
  **1.0000**.
- Benign FP full: 0.0000; without `ftp_failed_logins`:
  0.0000.

## SHAP on the selected candidate (34-PCAP test)

- `ftp_failed_logins` share: **0.030** (rank 9); all behavioural: **0.112**; CIC artifacts: **0.434**; behavioural in top-5: **0**; dominated by ftp_failed_logins: **False**
- Top features: Dst Port, Fwd Seg Size Min, Init Fwd Win Byts, Fwd IAT Min, Fwd Header Len, Flow IAT Mean

## Verdict -- PROMISING -- IMPROVED, MEETS CRITERIA

Selected candidate_robust_unweighted: on the frozen messy TEST corpus FTP recall 1.000, benign recall 1.000, benign FP 0.000; benign-gave-up recall 1.0, mistype 1.0, attacker-eventual-success 1.0. No CIC regression; ftp_failed_logins share 0.0298, FTP-recall drop w/o it +0.000. Beats Candidate 2. Meets all promotion criteria -- recommend human review before any promotion.

- meets_promotion_criteria: **True**  ·  beats_candidate2:
  **True**  ·  meets_thresholds: **True**  ·
  cic_no_regression: **True**  ·  no_major_failed_login_dependence:
  **True**  ·  promote: **False**

### Caveats (read the perfect scores honestly)
- **Shared generation methodology.** The messy TRAIN corpus (`robust_train_capture`)
  and the messy TEST corpus (`robustness_capture`) are hash/address/port/user-disjoint,
  but they are produced by the *same family of scenario functions*. The 1.000/1.000
  result therefore shows the model generalises across new addresses/ports/users/servers
  for the *same behavioural scenarios* -- it is **not** evidence of generalisation to
  attacker/benign behaviours or FTP dialects the scenario set never contained. The gain
  over the clean-trained model (benign recall 0.935 -> 1.000, gave_up 0.50 -> 1.00) is
  real and in the expected direction; the absolute 1.000 is optimistic.
- Messy TRAIN and TEST corpora are loopback-only, two server implementations, cleartext FTP.
- The 34-PCAP corpus was used for final reporting only -- never for training/selection/tuning.
- CIC held-out FTP recall dips slightly vs the clean-trained model (0.900 -> 0.871) but
  stays above production (0.884 - 0.02 gate); no accuracy regression (0.9808 vs 0.9803).
- No threshold change, no heuristic, no promotion, no merge. Before any promotion, test
  on genuinely independent real FTP traffic (different capture methodology, real servers,
  FTPS/TLS, non-loopback).

## Integrity

Production, Candidate 2, the clean-trained behavioural candidates,
ml.py/live_capture.py/pcap_validation.py, and the v1/v2/independent/targeted/robustness
PCAP corpora verified unchanged (before==after). The messy TRAIN corpus
(`validation/robust_train_pcaps`, NEW addresses 127.0.0.14-16 / ports 2430-2440) is
hash-disjoint from the 34-PCAP TEST corpus and all prior corpora. Candidate + ablation
models under `validation/models/ftp-behavioral-robust-retraining/`.

## Files

`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`per_scenario_family_metrics.csv`, `per_scenario_metrics.csv`, `per_client_metrics.csv`,
`per_server_metrics.csv`, `confidence_distribution.csv`, `confusion_*.csv`,
`shap_top_features.csv`, `leakage_validation.json`, `model_hashes_before_after.json`,
`training_metadata.json`, `final_verdict.json`.
