"""
History moved out of the database and into an append-only log file
(predictor/history_log.py), so the table it used is dropped.
"""

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("predictor", "0001_initial"),
    ]

    operations = [
        migrations.DeleteModel(name="PredictionLog"),
    ]
