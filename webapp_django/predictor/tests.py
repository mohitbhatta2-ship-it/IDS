"""
Tests for the dataset column mapping.

Run with:
    python manage.py test predictor
"""

import pandas as pd
from django.test import TestCase

from . import ml
from .column_mapping import ALIASES, map_columns, normalise


class NormaliseTests(TestCase):
    def test_strips_the_leading_space_the_published_csvs_carry(self):
        self.assertEqual(normalise(" Flow Duration"), normalise("Flow Duration"))

    def test_is_case_and_separator_insensitive(self):
        self.assertEqual(normalise("Fwd_Packet-Length Max"), normalise("fwd packet length max"))

    def test_per_second_suffix_survives(self):
        self.assertNotEqual(normalise("Flow Pkts/s"), normalise("Flow Pkts"))


class MapColumnsTests(TestCase):
    def _frame(self, columns):
        return pd.DataFrame({c: [1.0, 2.0] for c in columns})

    def test_exact_names_map_to_themselves(self):
        _, report = map_columns(self._frame(ml.FEATURES), ml.FEATURES)
        self.assertEqual(report["matched"], 30)
        self.assertEqual(len(report["aliased"]), 0)

    def test_ids2017_names_are_recognised(self):
        renamed = [ALIASES.get(f, [f])[0] for f in ml.FEATURES]
        _, report = map_columns(self._frame(renamed), ml.FEATURES)
        self.assertEqual(report["matched"], 30, report["missing"])

    def test_leading_spaces_are_tolerated(self):
        _, report = map_columns(self._frame([" " + f for f in ml.FEATURES]), ml.FEATURES)
        self.assertEqual(report["matched"], 30)

    def test_label_column_is_found_under_an_alias(self):
        df = self._frame(ml.FEATURES)
        df["Attack"] = ["Benign", "Bot"]
        out, report = map_columns(df, ml.FEATURES)
        self.assertIn("Label", out.columns)
        self.assertEqual(report["label_renamed_from"], "Attack")

    def test_missing_columns_are_reported(self):
        subset = ml.FEATURES[:25]
        _, report = map_columns(self._frame(subset), ml.FEATURES)
        self.assertEqual(report["matched"], 25)
        self.assertEqual(len(report["missing"]), 5)

    def test_unrelated_columns_are_not_guessed(self):
        df = self._frame(["duration", "protocol_type", "service", "src_bytes"])
        _, report = map_columns(df, ml.FEATURES)
        self.assertEqual(report["matched"], 0)


class PredictBatchTests(TestCase):
    def _rows(self, columns, n=5):
        return pd.DataFrame({c: [float(i + 1) for i in range(n)] for c in columns})

    def test_rejects_a_dataset_with_nothing_recognisable(self):
        df = self._rows(["duration", "protocol_type", "service"])
        with self.assertRaises(ml.BatchError) as ctx:
            ml.predict_batch(df)
        self.assertIn("None of the 30", str(ctx.exception))

    def test_rejects_a_partial_dataset_and_names_every_missing_column(self):
        df = self._rows(ml.FEATURES[:25])
        with self.assertRaises(ml.BatchError) as ctx:
            ml.predict_batch(df)

        message = str(ctx.exception)
        self.assertIn("5 of the 30", message)
        for absent in ml.FEATURES[25:]:
            self.assertIn(absent, message, "the message must name every missing column")

    def test_rejects_a_dataset_missing_a_single_column(self):
        """One absent feature is enough. It used to be filled with a median."""
        df = self._rows([f for f in ml.FEATURES if f != "Fwd Seg Size Min"])
        with self.assertRaises(ml.BatchError) as ctx:
            ml.predict_batch(df)

        message = str(ctx.exception)
        self.assertIn("1 of the 30", message)
        self.assertIn("Fwd Seg Size Min", message)
        self.assertIn("is missing", message, "singular wording for one column")

    def test_a_complete_dataset_is_still_accepted(self):
        result = ml.predict_batch(self._rows(ml.FEATURES))
        self.assertEqual(result["rows"], 5)
        self.assertEqual(result["mapping"]["missing"], [])

    def test_foreign_naming_gives_the_same_answer_as_native(self):
        native = self._rows(ml.FEATURES)
        foreign = native.rename(columns={f: ALIASES.get(f, [f])[0] for f in ml.FEATURES})

        a = ml.predict_batch(native)
        b = ml.predict_batch(foreign)

        self.assertEqual(
            list(a["frame"]["Predicted Class"]),
            list(b["frame"]["Predicted Class"]),
            "renaming the columns changed the predictions",
        )

    def test_empty_file_is_rejected(self):
        with self.assertRaises(ml.BatchError):
            ml.predict_batch(pd.DataFrame())


class EvaluationTests(TestCase):
    """The Label path was not covered before, which let a shadowed import through."""

    def _labelled(self, n=8):
        import numpy as np
        rng = np.random.default_rng(0)
        df = pd.DataFrame({f: rng.normal(100, 20, n) for f in ml.FEATURES})
        df["Label"] = ["Benign", "Bot"] * (n // 2)
        return df

    def test_evaluation_runs_and_carries_family_on_each_class(self):
        result = ml.predict_batch(self._labelled())
        ev = result["evaluation"]
        self.assertIsNotNone(ev)
        self.assertIn("accuracy", ev)
        for row in ev["per_class"]:
            self.assertIn("family", row)
            self.assertIn(row["family"], {"benign", "ddos", "dos", "bruteforce", "web", "other"})

    def test_families_roll_up_to_the_row_count(self):
        result = ml.predict_batch(self._labelled())
        self.assertEqual(sum(f["count"] for f in result["families"]), result["rows"])

    def test_numeric_labels_are_mapped_to_names(self):
        df = self._labelled()
        df["Label"] = [0, 1] * (len(df) // 2)
        self.assertIsNotNone(ml.predict_batch(df)["evaluation"])
