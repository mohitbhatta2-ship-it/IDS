# Sample datasets

Ready-to-upload files for the batch page at `/batch/`. All rows come from
`webapp_data/Processed_Data/test_selected.parquet` — the held-out test set the
models were never trained on.

| File | Rows | Label column | Purpose |
|---|---|---|---|
| `test_sample_with_labels.csv` | 3,561 | yes | Balanced across all 15 classes. Gives accuracy, macro F1 and the per-class table. |
| `test_sample_no_labels.csv` | 3,561 | no | The same rows with labels stripped — what an unknown dataset looks like. |
| `foreign_dataset_ids2017_style.csv` | 3,561 | yes | Same data with **CIC-IDS2017 column names** and the leading spaces the published CSVs carry. Demonstrates the column-mapping layer. |

Scores on the balanced sample are around accuracy 0.90 / macro F1 0.87. The
accuracy is lower than notebook 07's 0.9803 because ~300 rows per class heavily
over-represents the rare attacks; macro F1 is the comparable figure.

To regenerate, or to build a larger natural-distribution sample, read
`test_selected.parquet` and sample from it.
