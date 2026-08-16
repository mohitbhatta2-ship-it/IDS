# Targeted-benign retraining experiment

**Candidate only. The production model and Candidate 2 are frozen and byte-for-byte
unchanged (verified before == after). No candidate is promoted, and nothing is
merged.** Candidate artifacts live under `validation/models/targeted-benign-candidate/`;
evidence under `validation/results/targeted_benign_retraining/`.

## Goal

Reduce Candidate 2's high independent benign false-positive rate (0.282) **while
retaining** its real FTP-detection improvement (independent FTP recall 0.754),
by adding the 41 targeted benign captures to its training recipe.

## Strict data separation

- **Training:** CIC `balanced_train_selected` + v2 real (83 captures, weight 1) +
  **41 targeted benign captures** (weight `w_b`). Nothing else.
- **v2 leave-one-capture-out:** the real-PCAP validation estimate (each v2 capture
  held out; targeted benign is always training augmentation).
- **Independent 36-PCAP test set:** **FROZEN.** Never used for training, weighting,
  hyperparameter/threshold selection, or **candidate selection**. The primary
  candidate was selected using **CIC held-out + v2 LOCO only**, then evaluated once
  on the independent set. A guard aborts if any independent PCAP hash appears in
  the training corpora.

Same tuned HGB family/seed and the existing unchanged 30-feature `pcap_validation`
extraction as every prior experiment (exactly 30 ordered finite features, no
zero-fill; 854 v2 + 215 targeted flows, 0 invalid).

## Candidates & weighting (justified, not blindly optimised)

Benign-weight `w_b` applied to the 41 targeted benign captures only (they are the
FP-prone region; upweighting teaches the model to keep them Benign — over-weighting
risks FTP recall, so all three are compared):

| Weighting | w_b | CIC acc | CIC macro-F1 | v2-LOCO FTP recall | v2-LOCO Benign recall |
|---|---|---|---|---|---|
| none | 1 | 0.9803 | 0.8603 | 0.923 | 0.809 |
| moderate | 5 | 0.9797 | 0.8545 | 0.915 | 0.792 |
| strong | 15 | 0.9801 | 0.8637 | 0.905 | 0.869 |

**On in-distribution v2 LOCO all three look healthy** — FTP recall stays ~0.90 and
CIC does not regress. The selection rule (no CIC regression, v2-LOCO FTP recall
≥0.85, then max v2-LOCO benign recall) picks **`strong`**. Selection used **only**
CIC + v2 LOCO.

## Final independent test (frozen 36-PCAP set)

| Model | FTP recall | Benign recall | Benign FP rate | macro-F1 | accuracy |
|---|---|---|---|---|---|
| production | 0.000 | 0.974 | 0.026 | 0.123 | 0.181 |
| Candidate 2 | 0.754 | 0.718 | 0.282 | 0.672 | 0.748 |
| targeted **none** | 0.556 | 0.872 | 0.128 | 0.579 | 0.614 |
| targeted **moderate** | 0.202 | 0.962 | 0.038 | 0.343 | 0.343 |
| targeted **strong** (selected) | 0.351 | 0.872 | 0.128 | 0.439 | 0.448 |

Capture-level bootstrap 95% CIs in `bootstrap_ci_results.csv`.

### The core finding

**The targeted benign data reduces the benign FP rate but at a large cost to FTP
recall — it does not achieve "reduce FP *while* retaining FTP".** Adding the
targeted benign pushes real benign out of the model's learned FTP region, but
because real benign and real FTP **overlap** on the CIC-artifact features (the
`final_robustness` diagnosis), the same push also drops real FTP:

- **Candidate 2 → selected (`strong`):** benign FP **0.282 → 0.128** (−0.154), but
  FTP recall **0.754 → 0.351** and macro-F1 **0.672 → 0.439**.
- Even the mildest new candidate (`none`, best independent macro-F1 0.579) still
  loses a third of the FTP recall (0.754 → 0.556) for a smaller FP gain.

**The v2 LOCO was optimistic** (v2 FTP is in-distribution, so it stayed ~0.90); the
frozen independent test is what exposed the FTP-recall collapse — exactly why the
independent set is kept separate and evaluated only once.

## SHAP (selected candidate)

The selected candidate's global top features remain **`Fwd Seg Size Min`,
`Init Fwd Win Byts`, Dst Port** — `new_artifact_dependent = true`. It did **not**
decrease CIC-artifact dependence, and the benign false positives did **not** move
to a different feature space. The model simply learned a **smaller** FTP region
that also excludes some real FTP flows. See `shap_results.csv`.

## Verdict — DO NOT PROMOTE

The selected candidate reduces benign FP (0.282 → 0.128) but **fails** the
promotion criteria: FTP recall collapses (0.754 → 0.351), macro-F1 worsens
(0.672 → 0.439). A candidate is only promising if it reduces benign FPs on the
independent test **while retaining meaningful FTP recall and without CIC
regression** — this does not. `final_verdict.json → promote: false`.

### What this tells us (honestly)

The benign FP and the FTP recall are **coupled** through the shared CIC-artifact
feature region; adding benign data alone cannot separate them at this scale.
Fixing this likely needs traffic that breaks the artifact overlap (diverse real
FTP *and* benign from multiple hosts/OSes/tools so the model can learn a
non-artifact FTP signature), not more benign of the same shape — consistent with
the `final_robustness` recommendation.

### Caveats
- Independent test is small (36 captures / 420 flows), loopback-only, 2 real
  classes; bootstrap CIs are wide; no statistical-significance claim.
- The candidate was selected on CIC + v2 LOCO; the independent test was evaluated
  once. The v2-LOCO-optimal weighting (`strong`) is not the independent-optimal one
  (`none` by macro-F1) — a real illustration of why in-distribution validation
  over-estimates.

## Reproduce

```bash
# from webapp_django/
python manage.py targeted_benign_retrain
python manage.py test predictor.tests.TargetedRetrainModuleTests \
                      predictor.tests.TargetedRetrainResultsTests
```

## Integrity

Production model, Candidate 2, `ml.py` / `live_capture.py` / `pcap_validation.py`,
and the v1/v2/independent/targeted PCAPs verified unchanged (before == after). No
independent PCAP entered training. Exactly 30 ordered finite features, no zero-fill.

## Files (`validation/results/targeted_benign_retraining/`)

`candidate_metrics.csv`, `baseline_metrics.csv`, `per_class_metrics.csv`,
`per_capture_v2loco_metrics.csv`, `independent_test_metrics.csv`,
`independent_per_capture_metrics.csv`, `weighting_comparison.csv`,
`confusion_*.csv`, `confidence_distribution.csv`, `bootstrap_ci_results.csv`,
`shap_results.csv`, `leakage_validation.json`, `model_hashes_before_after.json`,
`training_metadata.json`, `final_verdict.json`, `report.md`.
