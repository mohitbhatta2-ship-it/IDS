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

    def test_rejects_when_too_few_features_match(self):
        df = self._rows(ml.FEATURES[:10])
        with self.assertRaises(ml.BatchError) as ctx:
            ml.predict_batch(df)
        self.assertIn("too few", str(ctx.exception))

    def test_accepts_a_partial_dataset_and_flags_what_was_filled(self):
        df = self._rows(ml.FEATURES[:25])
        result = ml.predict_batch(df)
        self.assertEqual(result["rows"], 5)
        self.assertEqual(len(result["mapping"]["missing"]), 5)

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
