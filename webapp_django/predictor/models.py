"""
No database models.

Prediction history used to live in a PredictionLog table. It is now an
append-only log file -- see predictor/history_log.py -- so the record of what
was classified is a plain JSONL file that can be read, copied, or charted
without the app, and does not vanish with the database.
"""
