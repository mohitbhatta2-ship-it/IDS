# NIDS — CSE-CIC-IDS2018

Network intrusion detection on the [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)
dataset. Four tuned classifiers, a study of how much training data actually
helps, CTGAN augmentation for the rare attack classes, and a Django web app that
classifies either a single network flow or an entire uploaded dataset.

**Live app → <https://nids-ids2018.onrender.com>**

The models distinguish **15 traffic classes** (benign plus 14 attack types) from
**30 flow features**. HistGradientBoosting is the default: it has the best macro
F1, which is the figure that matters here for reasons explained under
[Results](#results).

---

## Contents

| Path | What it is |
|---|---|
| `webapp_django/` | The Django web app — this is what's deployed |
| `C_notebooks/` | Jupyter notebooks: EDA, CTGAN prep, tuning, comparison, scaling |
| `webapp_data/` | Processed data, trained models (Git LFS), result tables and figures |
| `sample_data/` | Ready-to-upload CSVs for demonstrating the batch page |
| `Live/` | Live packet capture and feature extraction — **experimental, see [below](#live-capture-live)** |
| `c_filesnew/` | An earlier copy of `webapp_data/`, kept for reference |
| `CHANGES.md` | Detailed change log, including known defects |

---

## Results

All four models were tuned with Optuna and scored on the same held-out test set
of **200,498 rows** the models never saw during training.

| Model | Accuracy | Macro F1 | Train time | In the app |
|---|---:|---:|---:|:--:|
| **HistGradientBoosting** | 0.9803 | **0.8627** | 26 s | default |
| XGBoost | **0.9862** | 0.8421 | 115 s | yes |
| MLP (neural network) | 0.9837 | 0.7490 | 9,212 s | yes |
| RandomForest | 0.8991 | 0.8082 | 54 s | no — see note |

<sub>Source: `webapp_data/Results/Tables/model_comparison.csv`, independently
reproduced by notebook 07's Large run.</sub>

**Read macro F1, not accuracy.** The dataset is heavily imbalanced, so a model
can score 0.98 accuracy while handling the rare attacks badly. The gap between
XGBoost's 0.9862 accuracy and its 0.8421 macro F1 is exactly that effect.
HistGradientBoosting gives up 0.6 points of accuracy for 2 points of macro F1,
which is the better trade for an IDS — the rare classes are the interesting ones.

RandomForest is excluded from the web app deliberately: it is the weakest of the
four, and its `.pkl` is not committed.

### How much data is enough?

Notebook 07 retrains every model at three training-set sizes with frozen
hyperparameters and a fixed test set, so only data volume varies:

| Model | Small (25k) | Medium (100k) | Large (281k) | Macro F1 gain |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 0.8218 | 0.8466 | 0.8627 | +0.041 |
| XGBoost | 0.7316 | 0.8182 | 0.8421 | **+0.111** |
| RandomForest | 0.7452 | 0.7892 | 0.8085 | +0.063 |
| MLP | 0.7216 | 0.7521 | 0.7490 | +0.027 |

Accuracy barely moves across an eleven-fold increase in data (HistGradientBoosting:
0.9787 → 0.9803). Macro F1 moves a lot. XGBoost is the biggest beneficiary of
more data; the MLP essentially stops improving after 100k rows.

---

## The web app

Three pages, no front-end framework — hand-written HTML, CSS and JavaScript.

**Single flow** (`/predict/`) — the 30 features as a form, grouped into six
sections so it isn't one flat list of 30 number boxes. Every field shows the
dataset median as its placeholder, and a preset selector fills all 30 at once
with a real flow from the test set (one per class), because typing 30 numbers by
hand is impractical. Returns the predicted class, its confidence, and the three
most likely classes.

**Dataset** (`/batch/`) — upload a CSV or Parquet file with the 30 feature
columns and every row is classified. Column names don't have to match exactly:
CIC-IDS2017 spellings (`Destination Port`, `Total Length of Fwd Packets`) and
headers with stray leading spaces are mapped automatically. If the file also has
a `Label` column it's treated as ground truth, and the page reports accuracy,
macro F1 and a per-class precision/recall/F1 table — so an upload doubles as an
evaluation run.

**History** (`/history/`) — every run recorded through the Django ORM, with its
model, result and scores.

There's also a JSON endpoint at `POST /api/predict/` for a single flow.

> **A `Label` column never influences predictions.** `predict_batch()` selects
> `df[FEATURES]` — a fixed 30-name list that does not include `Label` — and only
> reads `Label` afterwards to score what was already predicted.
> `verify_per_attack_datasets.py` checks this on every run by scoring each sample
> file against its label-free twin and requiring byte-identical output. Currently
> 15/15 identical.

---

## Quick start

The trained models live in **Git LFS**. A clone without LFS leaves 130-byte
pointer files behind and the app dies at the first prediction with a confusing
`UnpicklingError`.

```bash
git clone https://github.com/mohitbhatta2-ship-it/IDS.git
cd IDS
git lfs install && git lfs pull

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r webapp_django/requirements.txt

cd webapp_django
python manage.py migrate
python manage.py runserver
```

Then open <http://127.0.0.1:8000>. No environment variables are needed for local
development — it falls back to SQLite and a development secret key.

`scikit-learn==1.9.0` and `xgboost==3.3.0` are pinned because the saved models
must be unpickled by the versions that wrote them. Don't float them without
retraining.

### Try it with the sample data

| File | Rows | Labels | What it shows |
|---|---|:--:|---|
| `sample_data/test_sample_with_labels.csv` | 3,561 | yes | Balanced over all 15 classes — full evaluation with accuracy and macro F1 |
| `sample_data/test_sample_no_labels.csv` | 3,561 | no | The same rows with labels stripped — predictions only |
| `sample_data/foreign_dataset_ids2017_style.csv` | 3,561 | yes | Same data under CIC-IDS2017 column names — demonstrates the mapping layer |
| `sample_data/per_attack/*.csv` | 500* | yes | One file per traffic class (15), for demonstrating a single attack |
| `sample_data/per_attack/no_labels/*.csv` | 500* | no | The same rows without ground truth |

<sub>* 500 where the data allows — SQL Injection has only 87 rows in the test set.</sub>

Upload a **labelled** file to get Accuracy and Macro F1. The `no_labels/` copies
have identical filenames and produce identical predictions, but no scores —
there is no ground truth to score against.

---

## Notebooks

Run in order; each writes into `webapp_data/`.

| Notebook | What it does |
|---|---|
| `01_EDA_and_Data_Preparation` | Cleaning, feature selection down to 30 features, train/test split |
| `03_Data_Preparation_for_CTGAN` | Prepares the rare classes for synthetic augmentation |
| `04_Model_Training_Optuna` | Optuna tuning for the tree models |
| `05_Compare_All_Models` | Side-by-side comparison, writes `model_comparison.csv` |
| `06_MLP_Training_Optuna` | Optuna tuning for the neural network |
| `07_Data_Size_Scaling_Experiment` | The scaling study above, and reproduces 04/06's numbers as a validity check |

Training data: `webapp_data/Processed_Data/balanced_train_selected.parquet`
(281,295 × 31). Test: `test_selected.parquet` (200,498 × 31).

---

## Live capture (`Live/`)

Packet capture with Scapy, flow assembly, and a hand-written reimplementation of
the 30 CICFlowMeter features so live traffic can be scored with the same models.

**This module does not currently run.** It has hardcoded Windows paths, loads a
model that isn't in the repository, and has a tuple-unpacking bug that raises on
the first expired flow. `CHANGES.md` §3 lists each defect with its file and line.

The deeper risk is feature parity: the models were trained on CICFlowMeter
output, and if any of the 30 reimplemented definitions differs, the model gets
out-of-distribution input and its predictions are meaningless *even when the code
runs cleanly*. Four specific discrepancies are already known (`Fwd Header Len`,
`Fwd Seg Size Min`, `FLOW_TIMEOUT`, `Init Fwd Win Byts`) and are documented in
`CHANGES.md`. Treat any live result as unvalidated until a parity report exists.

---

## Tests

```bash
cd webapp_django && python manage.py test predictor    # 17 tests
```

They cover the column-mapping layer — alias resolution, leading spaces, rejecting
unrecognisable files, and confirming that a CIC-IDS2017-named file gives the same
answer as a natively-named one.

`Live/test_live_pipeline.py` and `Live/test_prediction.py` exist but depend on the
unfixed module above.

---

## Deployment

Deployed to Render as a Blueprint service (`render.yaml`), root directory
`webapp_django`, built by `build.sh` — which pulls the four runtime models from
LFS (~12.6 MB) and **fails the build** if any is still a pointer file, because a
failed build is far easier to debug than a 503 in production.

Blueprint-managed services don't auto-deploy on push, so
`.github/workflows/deploy.yml` calls Render's deploy hook on merges to `main`
that touch anything the live service actually serves.

Environment variables are documented in `webapp_django/.env.example`; the full
runbook, including the LFS fallback and the TLS ordering trap, is in
[`webapp_django/DEPLOY.md`](webapp_django/DEPLOY.md).

Two things to know about the free tier:

- **It sleeps.** The first request after a nap takes 30–60 s, plus model
  unpickling on the first prediction.
- **Prediction history does not survive a restart.** The default database is
  SQLite on Render's ephemeral filesystem, so every deploy or spin-down empties
  the History page. Mount a persistent disk or attach a managed database if that
  matters — populate it shortly before any demo otherwise.

---

## Known limitations

- `app.py` at the repo root is a broken 45-line Flask stub referencing five
  templates that don't exist. It is not the web app; `webapp_django/` is.
- `webapp_data/` and `c_filesnew/` largely duplicate each other.
- The `Live/` module is not usable as-is (above).
- No `LICENSE` file yet. The CSE-CIC-IDS2018 dataset has its own terms — see the
  [dataset page](https://www.unb.ca/cic/datasets/ids-2018.html) before
  redistributing any data derived from it.

`CHANGES.md` is the authoritative list, with file and line references.

---

## Dataset

Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani. *Toward Generating a
New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP
2018. Canadian Institute for Cybersecurity, University of New Brunswick.
