# Realistic-PCAP retraining experiment v2

**Strictly experimental. The production model is frozen and byte-for-byte
unchanged (sha256 verified before == after). No candidate is promoted, nothing is
merged, and no production file is modified.** Candidate models live under
`validation/models/realistic_pcap_candidate_v2/`; all evidence under
`validation/results/retraining_v2/`.

## Question

Does adding the **83 diversified real captures** (`validation/realistic_pcaps_v2/`,
854 flows: 671 FTP-BruteForce, 183 Benign) to the CIC training distribution
improve the model's real-world FTP-BruteForce generalisation, without regressing
CIC performance?

## Safety & isolation

- Production model, `ml.py`, `live_capture.py`, `pcap_validation.py`, Dataset
  Testing, `webapp_data/Results/Models/`, the frozen validation results, and
  `validation/realistic_pcaps_v2/` are all **untouched** (verified by `git diff`
  and a sha256 check on the production `.pkl`).
- Candidates are new `HistGradientBoostingClassifier` models with the **same
  tuned hyperparameters and 30 `ml.FEATURES`** (seed 42) — only the training data
  changes. A full-CIC retrain reproduces the baseline, so the comparison is a
  clean A/B.
- Real flows come **only** from the existing `pcap_validation.replay_pcap`
  extraction (no second feature implementation, no zero-fill); a flow missing any
  of the 30 finite features is reported and excluded. Ground truth is the PCAP
  folder — never a model prediction.

## Design

### A. Baseline
Production model evaluated on CIC held-out and on all 854 real flows: accuracy,
macro P/R/F1, weighted-F1, confusion, per-class, FTP recall, Benign recall.

### B. Candidate 1 — CIC-only reproduction
Full CIC retrain (seed 42). Must reproduce the baseline closely — confirms the
training pipeline is not the variable. Saved separately; unused by production.

### C+D. Candidate 2 — CIC + real, three weighting strategies
Real-flow `sample_weight` ∈ {**none** ×1, **moderate** ×6, **strong** ×24}
(applied to all real rows; ×6 ≈ 0.5× and ×24 ≈ 2× the CIC FTP class mass). For
each: CIC held-out (from the full-CIC candidate) and the real-PCAP estimate via
**capture-level leave-one-capture-out (LOCO)**.

### Leakage control (LOCO)
For each real capture, train on CIC + all *other* captures, test on the held-out
capture. **No PCAP — and no flow — appears in both train and test** (asserted per
fold; recorded in `leakage_validation.json`).

### Compute honesty
Saved candidates and CIC-regression numbers use the **full CIC** train set (the
honest CIC headline). The 83-fold LOCO uses a **fixed 30 000-row stratified CIC
subsample** (seed 42) for tractability, with a **matched subsample-only control**
so "did adding real data regress CIC on the same base" is apples-to-apples. This
is a stated caveat, not hidden.

### E. Leakage checks (automated)
No PCAP/flow in both train & test · labels from folders only · exactly 30 features
· order == `ml.FEATURES` · all finite · no zero-fill · every flow traceable to its
PCAP · candidate artifacts separate from production · production model unchanged.

### F. Statistical comparison
Baseline vs each candidate: absolute & relative metric changes, per-capture FTP
recall, per-capture Benign false positives, CIC regression/non-regression. **No
statistical-significance claim** — the sample is far too small; point estimates
only.

### G. SHAP
Production vs the best candidate (selected on CIC-non-regression first, then LOCO
macro-F1 — *not* on real performance alone): compares top features and reports
honestly whether the candidate still leans on the CIC-specific FTP artifacts
(`Fwd Seg Size Min`, `Init Fwd Win Byts`).

### H. Verdict
Conservative, one of: **production candidate** / **promising but insufficient
evidence** / **unsuccessful**. No automatic promotion.

## Results

**A. Baseline (production model).** CIC held-out acc **0.9803**, macro-F1
**0.8627**. On all 854 real flows: acc 0.1885, macro-F1 0.0904, **FTP-BruteForce
recall 0.000**, Benign recall 0.880 — the production model calls every real FTP
flow Benign, exactly as the v1 investigation found.

**B. Candidate 1 (CIC-only reproduction).** Full-CIC retrain (seed 42):
0.9804 / 0.8599 — **reproduces the baseline** (Δacc < 5e-3), so the training
pipeline is not the variable.

**C/D. Candidate 2 (CIC + real), leave-one-capture-out:**

| Weighting | CIC acc | CIC macro-F1 | LOCO acc | LOCO macro-F1 | LOCO FTP recall | LOCO Benign recall |
|---|---|---|---|---|---|---|
| baseline | 0.9803 | 0.8627 | — | — | **0.000** | 0.880 |
| none (×1) | 0.9809 | **0.8650** | 0.8993 | **0.8534** | **0.928** | **0.792** |
| moderate (×6) | 0.9801 | 0.8638 | 0.8864 | 0.8310 | 0.928 | 0.732 |
| strong (×24) | 0.9803 | 0.8608 | 0.8841 | 0.8302 | 0.921 | 0.749 |

**Best = none (unweighted).** Adding the real captures lifts LOCO FTP recall from
**0.000 → 0.928** with **CIC not regressed** (macro-F1 +0.0023). Upweighting real
flows does **not** help — it holds FTP recall flat while lowering Benign recall
(more false positives), so the honest finding is that *more weighting is worse*
here.

**Cost.** Benign recall drops 0.880 → **0.792** (≈9 pts of new false positives on
real benign traffic) even at the best setting. See `weighting_comparison.csv`,
`statistical_comparison.csv`, `per_capture_metrics.csv`.

**E. Leakage / integrity.** All automated checks pass (`leakage_validation.json`):
no PCAP or flow in both train & test across all 249 folds, labels from folders
only, exactly 30 ordered finite features, no zero-fill, every flow traceable to
its PCAP, candidate artifacts separate from production, production model
unchanged.

**G. SHAP (production vs best candidate).** The candidate **still depends heavily
on the CIC-specific artifact features** — `Fwd Seg Size Min` (0.744→0.746) and
`Init Fwd Win Byts` (0.745→0.729) remain top by mean |SHAP|
(`candidate_still_artifact_dependent = true`). It did not abandon the artifacts;
it learned to *also* map the real region to FTP. This mirrors the v1 result.

**H. Verdict — promising but insufficient evidence. Not promoted.** A large
uncontaminated FTP-recall gain (0.000 → 0.928) with no CIC regression is genuinely
encouraging, but: only 83 loopback captures / 2 real classes; a ~9-point Benign
false-positive cost; SHAP shows the same artifact dependence; and LOCO uses a
subsampled CIC. No statistical-significance claim is made. Production readiness
would need many more diverse real captures and a proper independent test set.

## Reproduce

```bash
# from webapp_django/
python manage.py retrain_experiment_v2
python manage.py test predictor.tests.RetrainingV2ModuleTests \
                      predictor.tests.RetrainingV2ResultsTests
```

## Limitations

- Only **83 real captures** from a **loopback lab**, **2 real classes** (Benign,
  FTP-BruteForce); the other 13 classes are CIC-only. Exploratory, not
  production-grade.
- LOCO uses a 30k CIC subsample for compute; full-CIC candidates carry the CIC
  regression numbers.
- No significance testing — the dataset cannot support it.

## Verdict policy

Given the dataset size, the experiment **does not promote** any candidate under
any outcome. The best candidate (if any) is reported as *promising but
insufficient evidence* at most; a genuine production decision needs many more
diverse real captures (multiple tools, OSes, networks) and a proper held-out test
set.
