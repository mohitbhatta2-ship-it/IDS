# Change log — August 2026

Work carried out on the CSE-CIC-IDS2018 intrusion detection project in response
to the review feedback. Each section names the pull request it landed in, so the
history can be checked directly against the repository.

---

## 1. Data size scaling experiment — PR #2, PR #3 (merged)

**Requirement:** test the same models with the same hyperparameters on small,
medium and large amounts of data.

### What was added

`C_notebooks/07_Data_Size_Scaling_Experiment.ipynb` — approximately 2,000 lines,
41 cells.

All four tuned models are retrained at three training-set sizes and scored
against one fixed test set:

| Subset | Rows | Share of training data |
|---|---|---|
| Small | 25,000 | 8.9% |
| Medium | 100,000 | 35.6% |
| Large | 281,295 | 100% |

Three controls make the comparison fair:

1. **Hyperparameters are frozen.** They are read from the `*_best_params.json`
   files written by notebooks 04 and 06. Nothing is re-tuned — re-tuning per
   subset would confound "more data" with "better tuned", and the point of the
   experiment is to isolate the effect of data volume alone.
2. **The test set never changes.** Every model at every size is scored on the
   same held-out 200,498 rows. Only the training set shrinks.
3. **Fixed random seed, stratified sampling**, so class proportions in the Small
   subset match those in the Large one.

### Results

Macro F1 (all 15 classes weighted equally):

| Model | Small | Medium | Large |
|---|---|---|---|
| **HistGradientBoosting** | **0.8218** | **0.8466** | **0.8627** |
| XGBoost | 0.7316 | 0.8182 | 0.8421 |
| RandomForest | 0.7452 | 0.7892 | 0.8085 |
| MLP | 0.7216 | 0.7521 | 0.7490 |

Accuracy:

| Model | Small | Medium | Large |
|---|---|---|---|
| XGBoost | 0.9838 | 0.9860 | **0.9862** |
| MLP | 0.9809 | 0.9829 | 0.9837 |
| HistGradientBoosting | 0.9787 | 0.9799 | 0.9803 |
| RandomForest | 0.8896 | 0.8985 | 0.9000 |

### Validity check

At the Large size the notebook independently reproduces the numbers already
published in notebooks 04 and 06, which is the evidence that the pipeline is
faithful rather than accidentally different:

| Model | Notebook 04/06 | Notebook 07 (Large) |
|---|---|---|
| HistGradientBoosting | 0.9803 / 0.8627 | 0.9803 / 0.8627 |
| XGBoost | 0.9859 / 0.8429 | 0.9862 / 0.8421 |
| RandomForest | 0.8991 / 0.8082 | 0.9000 / 0.8085 |

HistGradientBoosting matches to four decimal places. The MLP sits slightly
higher (0.7490 against 0.7373) because notebook 06's rare-class oversampling is
deliberately not repeated here — including it would mean "25,000 rows" described
a different quantity of real data for the MLP than for the tree models, and the
curves would no longer be comparable. This is documented inside the notebook.

### Findings

1. **Accuracy is close to useless on this dataset.** HistGradientBoosting moves
   from 0.9787 to 0.9803 across an eleven-fold increase in training data — a gain
   of 0.0016. Macro F1 over the same range moves 0.0409. Accuracy is saturated by
   the Benign majority class and conceals everything of interest. This is the
   strongest argument in the project for reporting macro F1 throughout.
2. **Algorithm choice outweighs data volume.** HistGradientBoosting trained on
   9% of the data (macro F1 0.8218) beats RandomForest trained on all of it
   (0.8085).
3. **The MLP degrades with more data.** 0.7216 → 0.7521 → 0.7490. It is the only
   non-monotonic curve, and it is the deep-learning model.
4. **The gains are concentrated in the rare classes.** XGBoost scores F1 0.000 on
   SQL Injection at Small — six training examples is not enough for it ever to
   predict the class — rising to 0.364 at Large. That single class accounts for
   much of its +0.1106 macro F1 gain, the largest of any model.

### Defect found and fixed

XGBoost's scikit-learn wrapper requires training labels to form a contiguous run
`0 … n-1`. SQL Injection is encoded `13` and has only 70 rows out of 281,295. Any
subset small enough to lose it leaves a hole in the label space (`[0…12, 14]`)
and `fit()` raises:

```
ValueError: Invalid classes inferred from unique values of `y`.
Expected: [0 1 2 ... 13], got [0 1 2 ... 12 14]
```

The notebook relabels to a contiguous range before fitting XGBoost and maps the
predictions back afterwards, so metrics remain in the original label space. A
class missing from a subset is still reported as a genuine result (F1 = 0, marked
`In Training Subset = False`) rather than concealed.

### Portability

Unlike notebooks 04 and 06, which hardcode `M:\IDS\c_filesnew`, notebook 07
locates its data through the `NIDS_PROJECT_PATH` environment variable or by
searching the repository. It runs on a fresh clone on any operating system
without editing code.

### Output files

| File | Contents |
|---|---|
| `Tables/data_scaling_results.csv` | One row per model and subset: all metrics and timings |
| `Tables/data_scaling_per_class_f1.csv` | Per-class F1 in long form |
| `Tables/data_scaling_gain_summary.csv` | Small → Large improvement per model |
| `Figures/data_scaling_macro_f1.png` | Macro F1 learning curves |
| `Figures/data_scaling_accuracy.png` | Accuracy learning curves |
| `Figures/data_scaling_weighted_f1.png` | Weighted F1 learning curves |
| `Figures/data_scaling_training_time.png` | Training cost against data size |
| `Figures/data_scaling_rare_class_f1.png` | Rare-class F1 curves |

---

## 2. Prediction web application — PR #4 (open)

**Requirement:** a user interface accepting the 30 features that returns the
attack type, and the ability to submit a whole dataset for classification.

Branch `feat/django-prediction-webapp`. 32 files, 3,138 lines. Django backend,
hand-written HTML/CSS/JavaScript front end, no external front-end libraries.

### Pages

**Single flow.** The 30 training features presented as a form, arranged into six
labelled groups (flow identity, rates, inter-arrival times, forward direction,
backward direction, packet size distribution). Each field shows the dataset
median as a placeholder. A preset selector fills all 30 fields with a real flow
drawn from the held-out test set — one per class — because entering 30 numbers by
hand is otherwise impractical. Returns the predicted class, its confidence, and
the three most likely classes.

**Dataset.** Upload a CSV or Parquet file holding the 30 feature columns; every
row is classified. If the file also carries a `Label` column it is treated as
ground truth and the page additionally reports accuracy, macro F1, weighted F1
and a per-class precision/recall/F1 table — so an upload doubles as an evaluation
run rather than merely a prediction run. The annotated file is downloadable with
`Predicted Class` and `Confidence` appended.

**History.** Every run is recorded through the Django ORM and listed with its
model, result and scores.

### Models

HistGradientBoosting (default, best macro F1), XGBoost, and the MLP are all
selectable. RandomForest is deliberately excluded: it is absent from the
repository, and it is also the weakest model.

### Verification

The median flow of each of the 15 classes was submitted to every model:

| Model | Correct | Macro F1 (notebook 07) |
|---|---|---|
| HistGradientBoosting | 14 / 15 | 0.8627 |
| XGBoost | 12 / 15 | 0.8421 |
| MLP | 11 / 15 | 0.7490 |

The ordering matches the models' macro F1 exactly, which is evidence that the
feature pipeline is wired correctly rather than silently reordering columns. The
misclassifications are the classes expected to be difficult — `Infilteration`
scored F1 0.27 in notebook 07 as well.

A 4,507-row upload taken from the test set scored **accuracy 0.9008, macro F1
0.8723**. The macro F1 is the comparable figure against notebook 07's 0.8627; the
accuracy is lower because the sample was drawn with roughly 400 rows per class
rather than in the natural class proportions.

Error handling was tested for missing feature columns, unsupported file types,
non-numeric values and absent model files. CSRF protection is enforced on the
prediction endpoint.

### Database

MySQL is used when `IDS_DB_ENGINE=mysql` with credentials supplied; otherwise the
application falls back to SQLite so it runs on a fresh clone with no database
server installed.

### Deployment

`render.yaml`, `Procfile`, `build.sh` and `DEPLOY.md` are included. All five
warnings from `manage.py check --deploy` are cleared, and the application refuses
to start with `DEBUG=0` and no `IDS_SECRET_KEY` so the development key cannot
reach production. `build.sh` fails the build if the model files are still Git LFS
pointers, rather than allowing a runtime failure that is harder to diagnose.

---

## 3. Outstanding defects — identified, not yet fixed

These were found during review of the existing code and remain present on `main`.

| Location | Problem |
|---|---|
| `app.py` | A 45-line Flask stub calling `render_template` for five templates that do not exist, so every route fails. The commit that added `Live/` replaced 923 lines of a working Streamlit dashboard (recoverable at `git show e47249b:app.py`). `requirements.txt` and `.devcontainer/devcontainer.json` still specify Streamlit, and Flask is not listed as a dependency. |
| `Live/live_predictor.py`, `Live/preprocess_live.py` | Hardcoded Windows paths (`M:\IDS\...`) that exist only on one team member's machine. |
| `Live/live_predictor.py` | Loads `RandomForest_Tuned.pkl`, which is excluded by `.gitignore` and absent from the repository, so the module fails at import. RandomForest is also the weakest of the four models. |
| `Live/flow_manager.py:34` | `prediction, confidence = predict(processed)` unpacks a three-element list into two names, raising `ValueError` on the first expired flow. |
| `Live/flow.py`, `Live/feature_extractor.py` | `start_time` is taken from the wall clock while `last_seen` comes from packet timestamps. The difference is negligible during live capture but makes flow duration and every inter-arrival feature meaningless when replaying a stored capture. |
| `Live/sniff_test.py:23` | `cleanup_flows()` is called for every packet and iterates the entire flow table each time. |
| `Live/validate_live_features.py:3` | Reads `train_selected.csv`, which does not exist — the repository stores Parquet. |
| Repository | No README, no tests, and `webapp_data/` largely duplicates `c_filesnew/`. |

### Feature parity — the largest open risk

The models were trained on features produced by **CICFlowMeter**.
`Live/feature_calculator.py` is a hand-written reimplementation of 30 of them. If
any definition differs, the model receives out-of-distribution input and its
predictions are meaningless even when the code runs without error. Specific
points to verify before trusting any live result:

- `Fwd Header Len` is computed as `ip.ihl * 4`, the IP header length, whereas
  CICFlowMeter measures the transport header.
- `Fwd Seg Size Min` is derived from TCP payload lengths here; CICFlowMeter's is
  a header measure.
- `FLOW_TIMEOUT` is 30 seconds against CICFlowMeter's default of 120, which
  changes flow boundaries and therefore duration and all inter-arrival statistics.
- `Init Fwd Win Byts` defaults to 0 here; the training data uses −1 when absent.

`Live/validate_live_features.py` was an attempt at exactly this check and should
be repaired into a proper parity report.
