"""
Tests for the dataset column mapping, batch scoring, and the run history log.

Run with:
    python manage.py test predictor
"""

import tempfile
from pathlib import Path

import pandas as pd
from django.test import TestCase, override_settings

from . import history_log, ml
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

    def test_label_values_survive_a_spreadsheet_round_trip(self):
        """Trailing spaces and case changes must not silently disable scoring."""
        for mangled in ("Benign ", " benign", "BENIGN", "benign"):
            with self.subTest(label=mangled):
                df = self._labelled()
                df["Label"] = mangled
                self.assertIsNotNone(
                    ml.predict_batch(df)["evaluation"],
                    f"{mangled!r} should still be recognised as Benign",
                )

    def test_an_unreadable_label_column_is_reported_as_such(self):
        """Not the same thing as having no Label column, and must not look like it."""
        df = self._labelled()
        df["Label"] = "not-a-class-name"
        result = ml.predict_batch(df)

        self.assertIsNone(result["evaluation"])
        self.assertTrue(result["has_label_column"])
        self.assertIn("not-a-class-name", result["label_unusable"])

    def test_a_missing_label_column_is_distinguishable_from_an_unreadable_one(self):
        result = ml.predict_batch(self._labelled().drop(columns=["Label"]))
        self.assertIsNone(result["evaluation"])
        self.assertFalse(result["has_label_column"])
        self.assertIsNone(result["label_unusable"])


class HistoryLogTests(TestCase):
    """The run log is the whole of the history feature, so it is tested directly."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "nested" / "predictions.jsonl"
        patch = override_settings(HISTORY_LOG_PATH=self.path)
        patch.enable()
        self.addCleanup(patch.disable)

    def _batch(self, filename="a.csv", **extra):
        return history_log.record(
            kind=history_log.BATCH, model_key="hgb", model_name="HistGradientBoosting",
            source_filename=filename, row_count=10, attack_count=4, **extra,
        )

    def test_reading_a_log_that_does_not_exist_yet_is_empty_not_an_error(self):
        self.assertEqual(history_log.recent(), [])
        self.assertEqual(history_log.count(), 0)

    def test_a_run_survives_the_round_trip_through_the_file(self):
        self._batch(accuracy=0.91, macro_f1=0.88)

        (run,) = history_log.recent()
        self.assertEqual(run.source_filename, "a.csv")
        self.assertEqual(run.row_count, 10)
        self.assertAlmostEqual(run.accuracy, 0.91)
        self.assertTrue(run.was_evaluated)
        self.assertIsNotNone(run.at, "the timestamp must come back as a datetime")

    def test_the_directory_is_created_on_first_write(self):
        self._batch()
        self.assertTrue(self.path.exists())

    def test_newest_run_comes_back_first(self):
        self._batch("first.csv")
        self._batch("second.csv")
        self.assertEqual([r.source_filename for r in history_log.recent()], ["second.csv", "first.csv"])

    def test_a_zero_score_is_kept_rather_than_read_as_missing(self):
        """0.0 is a real result. Dropping it would blank the column instead."""
        self._batch(accuracy=0.0, macro_f1=0.0)
        (run,) = history_log.recent()
        self.assertTrue(run.was_evaluated)
        self.assertEqual(run.accuracy, 0.0)

    def test_a_truncated_final_line_costs_only_that_run(self):
        """A restart mid-write must not take the whole history page down."""
        self._batch("good.csv")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"kind": "batch", "source_filename": "cut-o')

        runs = history_log.recent()
        self.assertEqual([r.source_filename for r in runs], ["good.csv"])

    def test_an_unwritable_log_does_not_break_the_prediction(self):
        blocker = Path(self._dir.name) / "not-a-directory"
        blocker.write_text("")
        with override_settings(HISTORY_LOG_PATH=blocker / "predictions.jsonl"):
            run = history_log.record(kind=history_log.MANUAL, model_name="MLP", predicted_label="Bot")
        self.assertEqual(run.predicted_label, "Bot")

    def test_manual_runs_are_recognised_as_attacks_by_label(self):
        history_log.record(kind=history_log.MANUAL, model_name="MLP", predicted_label="Benign")
        history_log.record(kind=history_log.MANUAL, model_name="MLP", predicted_label="Bot")

        attack, benign = history_log.recent()
        self.assertTrue(attack.is_attack)
        self.assertFalse(benign.is_attack)

    def test_count_covers_runs_older_than_the_read_window(self):
        for i in range(5):
            self._batch(f"{i}.csv")
        self.assertEqual(len(history_log.recent(limit=2)), 2)
        self.assertEqual(history_log.count(), 5)

    def test_rotation_keeps_the_older_runs_readable(self):
        self._batch("old.csv")
        # Rotate by hand rather than writing 4 MB of runs.
        self.path.rename(self.path.with_name(self.path.name + ".1"))
        self._batch("new.csv")

        self.assertEqual(history_log.count(), 2)
        self.assertEqual([r.source_filename for r in history_log.recent()], ["new.csv", "old.csv"])

    def test_clearing_removes_the_log_and_its_rotated_predecessor(self):
        self._batch("old.csv")
        self.path.rename(self.path.with_name(self.path.name + ".1"))
        self._batch("new.csv")

        history_log.clear()
        self.assertEqual(history_log.count(), 0)

    def test_summary_rolls_the_runs_up(self):
        self._batch("a.csv", accuracy=0.90, macro_f1=0.8)
        self._batch("b.csv", accuracy=0.80, macro_f1=0.7)
        self._batch("c.csv")  # unlabelled, so it must not drag the mean down

        summary = history_log.summarise(history_log.recent())
        self.assertEqual(summary.runs, 3)
        self.assertEqual(summary.flows, 30)
        self.assertEqual(summary.attacks, 12)
        self.assertEqual(summary.scored_runs, 2)
        self.assertAlmostEqual(summary.mean_accuracy, 0.85)


class HistoryViewTests(TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        patch = override_settings(HISTORY_LOG_PATH=Path(self._dir.name) / "predictions.jsonl")
        patch.enable()
        self.addCleanup(patch.disable)

    def test_empty_log_renders_the_empty_state(self):
        response = self.client.get("/history/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Nothing yet")

    def test_a_recorded_run_appears_on_the_page(self):
        history_log.record(
            kind=history_log.BATCH, model_key="hgb", model_name="HistGradientBoosting",
            source_filename="traffic.csv", row_count=200, attack_count=25, accuracy=0.93, macro_f1=0.9,
        )
        response = self.client.get("/history/")
        self.assertContains(response, "traffic.csv")
        self.assertContains(response, "0.9300")

    def test_the_type_filter_narrows_the_table(self):
        history_log.record(kind=history_log.BATCH, model_name="HGB", source_filename="traffic.csv", row_count=2)
        history_log.record(kind=history_log.MANUAL, model_name="MLP", predicted_label="Bot", row_count=1)

        self.assertNotContains(self.client.get("/history/?kind=manual"), "traffic.csv")
        self.assertContains(self.client.get("/history/?kind=batch"), "traffic.csv")

    def test_the_log_can_be_downloaded_as_jsonl(self):
        history_log.record(kind=history_log.MANUAL, model_name="MLP", predicted_label="Bot")
        response = self.client.get("/history/download/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment", response["Content-Disposition"])
        self.assertIn("Bot", response.content.decode())

    def test_clearing_empties_the_page(self):
        history_log.record(kind=history_log.MANUAL, model_name="MLP", predicted_label="Bot")
        self.client.post("/history/clear/")
        self.assertContains(self.client.get("/history/"), "Nothing yet")
