# Per-attack sample datasets

One CSV per traffic class. Upload any single file to `/batch/` and the result
should come back dominated by that one class — useful for demonstrating a
specific attack rather than the mixed bag in `../test_sample_with_labels.csv`.

Two sets of the same rows:

| Folder | Columns | Use |
|---|---|---|
| `per_attack/` | 30 features + `Label` | The batch page also reports accuracy, macro F1 and the per-class table, so the upload doubles as an evaluation run. |
| `per_attack/no_labels/` | 30 features only | What traffic of unknown origin looks like. Predictions only, no scoring. |

## The `Label` column does not affect predictions

Worth stating plainly, because a ground-truth column sitting in a prediction
input looks like leakage. It is not one. `predict_batch()` selects `df[FEATURES]`
— the fixed 30-name list, which does not contain `Label` — and only reads
`Label` afterwards, if present, to score what was already predicted.

`verify_per_attack_datasets.py` checks this on every run: it scores each file
against its `no_labels/` twin and requires byte-identical predicted classes and
confidences. Currently **15/15 identical**. Falsifying a label rather than
removing it changes nothing either.

The clearest evidence is `infilteration.csv`: every row says
`Label = Infilteration` and the model still calls 63.8% of them Benign. A
pipeline reading the answer could not get the answer wrong.

Rebuild with:

```bash
python sample_data/build_per_attack_datasets.py     # writes the CSVs
python sample_data/verify_per_attack_datasets.py    # scores them, writes VERIFICATION.csv
```

## Where the rows come from

Held-out rows from `test_selected.parquet` first — data no model ever trained
on, so a correct prediction on them is meaningful.

Six classes do not have 500 held-out rows to give. Those files are topped up
from `balanced_train_selected.parquet`. **Training rows are easier for the model
than unseen ones, so those six files score higher than the model's true
performance.** The split is in the table below and in `MANIFEST.csv`; the
"unseen recall" column is the number to trust.

Nothing is duplicated or synthesised. A file is short rather than padded, which
is why `sql_injection.csv` has 87 rows — 87 is all the real SQL Injection data
that exists across both splits.

CTGAN synthetic rows were deliberately not used: they were generated to augment
training, so scoring the model on them would be circular.

## What each file actually predicts

Scored with the default model (HistGradientBoosting). Full output in
`VERIFICATION.csv`.

| File | Rows | held-out / train | Predicted as its own class | Unseen recall | Notes |
|---|---:|---|---:|---:|---|
| `benign.csv` | 500 | 500 / 0 | 98.4% | 98.4% | |
| `bot.csv` | 500 | 500 / 0 | 100.0% | 100.0% | |
| `brute_force_web.csv` | 500 | 122 / 378 | 97.8% | 93.4% | |
| `brute_force_xss.csv` | 500 | 46 / 454 | 99.4% | 95.7% | mostly training rows |
| `ddos_attack_hoic.csv` | 500 | 500 / 0 | 100.0% | 100.0% | |
| `ddos_attack_loic_udp.csv` | 500 | 346 / 154 | 99.2% | 99.2% | |
| `ddos_attacks_loic_http.csv` | 500 | 500 / 0 | 100.0% | 100.0% | |
| `dos_attacks_goldeneye.csv` | 500 | 500 / 0 | 100.0% | 100.0% | |
| `dos_attacks_hulk.csv` | 500 | 500 / 0 | 100.0% | 100.0% | |
| `dos_attacks_slowhttptest.csv` | 500 | 246 / 254 | 73.6% | 53.7% | try the MLP — 95.9% |
| `dos_attacks_slowloris.csv` | 500 | 130 / 370 | 99.6% | 99.6% | |
| `ftp_bruteforce.csv` | 500 | 498 / 2 | 88.4% | 88.4% | 11.6% → SlowHTTPTest |
| `infilteration.csv` | 500 | 500 / 0 | **36.2%** | 37.5% | **predicts Benign — see below** |
| `sql_injection.csv` | 87 | 17 / 70 | 81.6% | 35.3% | 80% training rows |
| `ssh_bruteforce.csv` | 500 | 500 / 0 | 100.0% | 100.0% | |

14 of the 15 read as their own class. One does not.

## The three that are not clean

**Infilteration predicts Benign (63.8%), not Infilteration.** This is the model,
not the file. Recall on the full 2,049-row held-out pool is 37.5%, so the 500
sampled rows are representative rather than unlucky. No model helps — XGBoost
gets 22.4% and the MLP 20.5%, both worse. Infiltration traffic in
CSE-CIC-IDS2018 is an insider dropping a payload and behaving normally
afterwards; at flow-statistics level it genuinely resembles benign traffic.
The file is kept as an honest demonstration of the system's blind spot. If you
need it to "work" for a demo, the only way is to select the rows the model
already gets right, which is cherry-picking and would not mean anything.

**SlowHTTPTest and FTP-BruteForce trade places.** 26.4% of the SlowHTTPTest
file is called FTP-BruteForce, and 11.6% of the FTP-BruteForce file is called
SlowHTTPTest. Both attacks were captured on the same days against the same
targets and their flow statistics overlap heavily. Switching the model to the
MLP takes SlowHTTPTest from 53.7% to 95.9% — but the same switch drops
FTP-BruteForce from 88.4% to 28.9%, so there is no single model that gets both.

**SQL Injection flatters itself.** 81.6% on the file, but only 35.3% on the 17
genuinely unseen rows — the other 70 rows are ones the model trained on. With
87 examples in the entire dataset, no number from this file is stable.

## Per-class recall by model, on unseen rows only

Useful when picking a model for a demo:

| Class | HistGB | XGBoost | MLP |
|---|---:|---:|---:|
| Infilteration | **37.5%** | 22.4% | 20.5% |
| DoS attacks-SlowHTTPTest | 53.7% | 67.1% | **95.9%** |
| FTP-BruteForce | **88.4%** | 68.9% | 28.9% |
| SQL Injection | 35.3% | 29.4% | **58.8%** |
| Brute Force -Web | **93.4%** | 92.6% | 76.2% |
| Brute Force -XSS | **95.7%** | 93.5% | 41.3% |

HistGradientBoosting is the best default; the MLP is worth a try on the two
slow-connection classes.
