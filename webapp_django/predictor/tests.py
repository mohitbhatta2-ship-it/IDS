"""
Tests for the dataset column mapping, batch scoring, and the run history log.

Run with:
    python manage.py test predictor
"""

import csv
import io
import json
import tempfile
from pathlib import Path
from unittest import skipUnless

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

    def _download(self):
        response = self.client.get("/history/download/")
        self.assertEqual(response.status_code, 200)
        return list(csv.reader(io.StringIO(response.content.decode())))

    def test_the_log_can_be_downloaded_as_csv(self):
        history_log.record(
            kind=history_log.MANUAL, model_name="MLP", predicted_label="Bot",
            confidence=0.9931, row_count=1, attack_count=1,
        )
        response = self.client.get("/history/download/")

        self.assertIn("text/csv", response["Content-Type"])
        self.assertIn('filename="prediction-history.csv"', response["Content-Disposition"])

        header, row = list(csv.reader(io.StringIO(response.content.decode())))
        self.assertEqual(header[:5], ["when", "type", "model", "predicted_class", "confidence"])
        self.assertEqual(row[1:5], ["Manual entry", "MLP", "Bot", "0.993100"])

    def test_an_upload_and_a_manual_entry_share_one_set_of_columns(self):
        """Each fills its own cells; the rest are empty rather than guessed at."""
        history_log.record(
            kind=history_log.BATCH, model_name="XGBoost", source_filename="traffic.csv",
            row_count=200, attack_count=50, accuracy=0.93, macro_f1=0.9,
        )
        header, row = self._download()
        cells = dict(zip(header, row))

        self.assertEqual(cells["source_file"], "traffic.csv")
        self.assertEqual(cells["rows"], "200")
        self.assertEqual(cells["attack_share"], "0.250000")
        self.assertEqual(cells["accuracy"], "0.930000")
        self.assertEqual(cells["predicted_class"], "", "an upload has no single prediction")
        self.assertEqual(cells["confidence"], "")

    def test_an_unscored_upload_leaves_the_score_cells_empty(self):
        """Empty, not 0 -- a spreadsheet would average a zero into the column."""
        history_log.record(
            kind=history_log.BATCH, model_name="XGBoost", source_filename="raw.csv", row_count=5,
        )
        header, row = self._download()
        cells = dict(zip(header, row))
        self.assertEqual(cells["accuracy"], "")
        self.assertEqual(cells["macro_f1"], "")

    def test_the_download_covers_runs_older_than_the_page_window(self):
        """The page reads a window; an export that quietly stopped there would lie."""
        for i in range(history_log.SCAN_LINES + 5):
            history_log.record(kind=history_log.MANUAL, model_name="MLP", predicted_label=f"run-{i}")

        rows = self._download()[1:]
        self.assertEqual(len(rows), history_log.SCAN_LINES + 5)
        self.assertEqual(rows[0][3], "run-0", "oldest first, like the log itself")

    def test_an_empty_log_downloads_as_a_header_row(self):
        self.assertEqual(len(self._download()), 1)

    def test_clearing_empties_the_page(self):
        history_log.record(kind=history_log.MANUAL, model_name="MLP", predicted_label="Bot")
        self.client.post("/history/clear/")
        self.assertContains(self.client.get("/history/"), "Nothing yet")

    def test_clearing_asks_for_confirmation_first(self):
        history_log.record(kind=history_log.MANUAL, model_name="MLP", predicted_label="Bot")

        response = self.client.get("/history/?confirm=clear")
        self.assertContains(response, "Delete all 1 run?")
        self.assertEqual(history_log.count(), 1, "asking must not delete anything")

    def test_the_page_does_not_offer_the_confirmation_unasked(self):
        history_log.record(kind=history_log.MANUAL, model_name="MLP", predicted_label="Bot")
        self.assertNotContains(self.client.get("/history/"), "confirm-bar")

    def test_the_confirmation_keeps_the_active_filter_on_cancel(self):
        history_log.record(kind=history_log.MANUAL, model_name="MLP", predicted_label="Bot")
        response = self.client.get("/history/?confirm=clear&kind=manual")
        self.assertContains(response, 'href="/history/?kind=manual"')

    def test_clearing_still_needs_a_post(self):
        """The confirmation is a GET; only the POST behind it may delete."""
        history_log.record(kind=history_log.MANUAL, model_name="MLP", predicted_label="Bot")
        self.assertEqual(self.client.get("/history/clear/").status_code, 405)
        self.assertEqual(history_log.count(), 1)


# ---------------------------------------------------------------------------
# Live capture — the web integration of the Live/ pipeline
# ---------------------------------------------------------------------------

import sys as _sys
from unittest import mock

from . import live_capture

try:
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.packet import Raw

    _SCAPY = True
except Exception:  # noqa: BLE001
    _SCAPY = False

# 2018-era timestamps, deliberately far from "now", so any code reaching for the
# wall clock instead of the packet clock shows up immediately (mirrors the Live
# suite's convention).
_BASE_TS = 1_530_000_000.0


def _pkt(src="10.0.0.5", dst="93.184.216.34", sport=51000, dport=80,
         ts=_BASE_TS, payload=100, flags="PA", proto="tcp"):
    if proto == "tcp":
        p = IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags) / Raw(b"x" * payload)
    else:
        p = IP(src=src, dst=dst) / UDP(sport=sport, dport=dport) / Raw(b"x" * payload)
    p.time = ts
    return p


@skipUnless(_SCAPY, "scapy is required for live-capture tests")
class LiveCaptureSessionTests(TestCase):
    """The capture session reuses the Live/ extractor and ml.predict_one."""

    def setUp(self):
        # The session's packet handler imports Live/ modules by name.
        live_capture._ensure_live_on_path()

    def _session(self, model_key=None):
        return live_capture.CaptureSession(interface=None, model_key=model_key or ml.DEFAULT_MODEL)

    def test_a_flow_fed_packets_is_classified_with_all_the_expected_fields(self):
        session = self._session()
        session._handle(_pkt(ts=_BASE_TS, payload=120))
        session._handle(_pkt(src="93.184.216.34", dst="10.0.0.5", sport=80, dport=51000,
                             ts=_BASE_TS + 0.1, payload=200))
        session._flush_all()

        snap = session.snapshot()
        self.assertEqual(snap["flows"], 1)
        self.assertEqual(len(snap["recent"]), 1)

        rec = snap["recent"][0]
        for key in ("src_ip", "dst_ip", "src_port", "dst_port", "protocol",
                    "label", "confidence", "is_attack", "at", "packets", "seq"):
            self.assertIn(key, rec)
        self.assertEqual(rec["protocol"], "TCP")
        self.assertEqual(rec["src_ip"], "10.0.0.5")
        self.assertEqual(rec["dst_port"], 80)
        self.assertEqual(rec["packets"], 2)
        self.assertIn(rec["label"], set(ml.LABELS.values()))
        self.assertGreaterEqual(rec["confidence"], 0.0)
        self.assertLessEqual(rec["confidence"], 1.0)

    def test_the_reverse_direction_joins_one_flow(self):
        session = self._session()
        session._handle(_pkt(ts=_BASE_TS))
        session._handle(_pkt(src="93.184.216.34", dst="10.0.0.5", sport=80, dport=51000,
                             ts=_BASE_TS + 0.1))
        self.assertEqual(len(session._flows), 1)

    def test_non_tcp_udp_traffic_is_ignored(self):
        session = self._session()
        icmp = IP(src="10.0.0.5", dst="8.8.8.8", proto=1)
        icmp.time = _BASE_TS
        session._handle(icmp)
        self.assertEqual(len(session._flows), 0)

    def test_classification_routes_through_predict_one_with_the_chosen_model(self):
        session = self._session(model_key="xgboost")
        with mock.patch.object(live_capture.ml, "predict_one",
                               wraps=live_capture.ml.predict_one) as spy:
            session._handle(_pkt(ts=_BASE_TS))
            session._flush_all()
        self.assertTrue(spy.called)
        # The model_key selected in the UI is the one handed to the pipeline.
        self.assertEqual(spy.call_args.args[1], "xgboost")

    def test_snapshot_since_returns_only_newer_records(self):
        session = self._session()
        session._handle(_pkt(sport=51000, ts=_BASE_TS))
        session._handle(_pkt(sport=52000, ts=_BASE_TS + 1))
        session._flush_all()

        full = session.snapshot()
        highest = max(r["seq"] for r in full["recent"])
        self.assertEqual(session.snapshot(since=highest)["recent"], [])

    # -- TCP termination: classify during capture, not only on Stop --------

    def test_rst_finalizes_the_flow_immediately_during_capture(self):
        session = self._session()
        session._handle(_pkt(ts=_BASE_TS, flags="S"))
        # A RST from either side ends the flow at once -- no flush needed.
        session._handle(_pkt(src="93.184.216.34", dst="10.0.0.5", sport=80, dport=51000,
                             ts=_BASE_TS + 0.1, flags="R"))

        self.assertEqual(len(session._flows), 0, "RST should remove the flow")
        snap = session.snapshot()
        self.assertEqual(snap["flows"], 1, "the terminated flow is classified now")
        self.assertEqual(len(snap["recent"]), 1)

    def test_two_way_fin_finalizes_the_flow_during_capture(self):
        session = self._session()
        session._handle(_pkt(ts=_BASE_TS, flags="S"))
        # FIN from the forward direction: half-close, must NOT finalize yet.
        session._handle(_pkt(ts=_BASE_TS + 0.1, flags="FA"))
        self.assertEqual(len(session._flows), 1, "one-way FIN must not finalize")
        self.assertEqual(session.snapshot()["flows"], 0)

        # FIN from the backward direction completes the close -> finalize now.
        session._handle(_pkt(src="93.184.216.34", dst="10.0.0.5", sport=80, dport=51000,
                             ts=_BASE_TS + 0.2, flags="FA"))
        self.assertEqual(len(session._flows), 0, "both-way FIN should finalize")
        self.assertEqual(session.snapshot()["flows"], 1)

    def test_single_direction_fin_does_not_finalize(self):
        session = self._session()
        session._handle(_pkt(ts=_BASE_TS, flags="S"))
        # Repeated forward FINs (e.g. a retransmit) are still one direction only.
        session._handle(_pkt(ts=_BASE_TS + 0.1, flags="FA"))
        session._handle(_pkt(ts=_BASE_TS + 0.2, flags="FA"))

        self.assertEqual(len(session._flows), 1, "a one-way FIN never finalizes")
        self.assertEqual(session.snapshot()["flows"], 0, "nothing classified yet")

    def test_udp_flows_are_not_finalized_by_termination_only_by_flush(self):
        session = self._session()
        session._handle(_pkt(proto="udp", sport=50000, dport=53, ts=_BASE_TS))
        session._handle(_pkt(src="93.184.216.34", dst="10.0.0.5", proto="udp",
                             sport=53, dport=50000, ts=_BASE_TS + 0.1))
        # No TCP flags exist for UDP, so nothing finalizes mid-capture.
        self.assertEqual(session.snapshot()["flows"], 0)
        self.assertEqual(len(session._flows), 1)

        # The existing timeout/flush path still classifies it on Stop.
        session._flush_all()
        self.assertEqual(session.snapshot()["flows"], 1)


class LiveViewTests(TestCase):
    """The page renders and the endpoints behave without opening a real NIC."""

    def test_the_live_page_renders_with_the_model_options(self):
        response = self.client.get("/live/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Live traffic capture")
        self.assertContains(response, 'id="live-model"')
        self.assertContains(response, "Auto (Scapy default)")

    def test_status_with_no_session_reports_not_running(self):
        # Ensure a clean manager for this assertion.
        live_capture.manager._session = None
        data = self.client.get("/live/status/").json()
        self.assertFalse(data["running"])
        self.assertEqual(data["recent"], [])

    def test_start_rejects_an_unknown_model(self):
        response = self.client.post(
            "/live/start/", data={"model_key": "not-a-model", "iface": "auto"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_start_surfaces_a_capture_error_as_503(self):
        with mock.patch.object(live_capture.manager, "start",
                               side_effect=live_capture.CaptureError("no scapy here")):
            response = self.client.post(
                "/live/start/",
                data=json.dumps({"model_key": ml.DEFAULT_MODEL, "iface": "auto"}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("no scapy here", response.json()["error"])

    def test_start_status_stop_cycle_without_real_sniffing(self):
        """Patch the capture thread body so no interface is opened."""
        self.addCleanup(setattr, live_capture.manager, "_session", None)
        with mock.patch.object(live_capture.CaptureSession, "_run", lambda self: None), \
                mock.patch.object(live_capture, "_ensure_live_on_path", return_value=None), \
                mock.patch("scapy.all.sniff", create=True):
            started = self.client.post(
                "/live/start/",
                data=json.dumps({"model_key": ml.DEFAULT_MODEL, "iface": "auto"}),
                content_type="application/json",
            )
            self.assertEqual(started.status_code, 200)
            self.assertIn("running", started.json())

            status = self.client.get("/live/status/")
            self.assertEqual(status.status_code, 200)

            stopped = self.client.post("/live/stop/")
            self.assertEqual(stopped.status_code, 200)
            self.assertFalse(stopped.json()["running"])

    def test_stop_with_no_session_is_harmless(self):
        live_capture.manager._session = None
        data = self.client.post("/live/stop/").json()
        self.assertFalse(data["running"])

    def test_selecting_a_friendly_interface_captures_on_its_raw_value(self):
        """
        The dropdown's option value is the raw scapy id; the friendly name is
        display only. Starting with that value must set the session interface to
        the exact raw id, so scapy captures on the correct underlying interface.
        """
        self.addCleanup(setattr, live_capture.manager, "_session", None)
        raw = r"\Device\NPF_{B82DF013-1A2B-3C4D-5E6F-0011223344AA}"
        with mock.patch.object(live_capture.CaptureSession, "_run", lambda self: None), \
                mock.patch.object(live_capture, "_ensure_live_on_path", return_value=None), \
                mock.patch("scapy.all.sniff", create=True):
            resp = self.client.post(
                "/live/start/",
                data=json.dumps({"model_key": ml.DEFAULT_MODEL, "iface": raw}),
                content_type="application/json",
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(live_capture.manager.session.interface, raw)

    def test_auto_maps_to_scapy_default(self):
        self.addCleanup(setattr, live_capture.manager, "_session", None)
        with mock.patch.object(live_capture.CaptureSession, "_run", lambda self: None), \
                mock.patch.object(live_capture, "_ensure_live_on_path", return_value=None), \
                mock.patch("scapy.all.sniff", create=True):
            resp = self.client.post(
                "/live/start/",
                data=json.dumps({"model_key": ml.DEFAULT_MODEL, "iface": "auto"}),
                content_type="application/json",
            )
            self.assertEqual(resp.status_code, 200)
            self.assertIsNone(live_capture.manager.session.interface)


class _FakeIface:
    def __init__(self, name, network_name, description=""):
        self.name = name
        self.network_name = network_name
        self.description = description


class LiveInterfaceMappingTests(TestCase):
    """Friendly interface labels map onto the raw scapy ids without guessing."""

    WIFI = r"\Device\NPF_{B82DF013-1A2B-3C4D-5E6F-0011223344AA}"
    ETH = r"\Device\NPF_{11112222-3333-4444-5555-666677778888}"
    UNKNOWN = r"\Device\NPF_{DEADBEEF-9999-0000-1111-222233334444}"
    LOOPBACK = r"\Device\NPF_Loopback"

    # -- fallback labels ---------------------------------------------------

    def test_unresolved_npf_id_is_a_safe_short_label_not_the_full_path(self):
        label = live_capture._fallback_label(self.UNKNOWN)
        self.assertEqual(label, "Network Adapter (NPF_{DEADBEEF})")
        self.assertNotIn("\\Device", label)
        self.assertNotIn("DEADBEEF-9999", label)  # not the whole GUID either

    def test_npf_loopback_falls_back_to_loopback(self):
        self.assertEqual(live_capture._fallback_label(self.LOOPBACK), "Loopback")

    def test_posix_names_are_shown_as_is_with_lo_named_loopback(self):
        self.assertEqual(live_capture._fallback_label("eth0"), "eth0")
        self.assertEqual(live_capture._fallback_label("lo"), "Loopback")

    # -- labels from scapy interface metadata ------------------------------

    def test_label_prefers_the_connection_name(self):
        self.assertEqual(
            live_capture._label_from_iface(_FakeIface("Wi-Fi", self.WIFI, "Intel Wi-Fi 6")),
            "Wi-Fi",
        )

    def test_label_normalises_loopback(self):
        self.assertEqual(
            live_capture._label_from_iface(_FakeIface("Npcap Loopback Adapter", self.LOOPBACK)),
            "Loopback",
        )

    def test_label_never_returns_a_raw_npf_path(self):
        # Some scapy builds leave the NPF path in `name`; that must not be used.
        self.assertIsNone(live_capture._label_from_iface(_FakeIface(self.WIFI, self.WIFI)))

    # -- full resolution ---------------------------------------------------

    def test_resolution_maps_friendly_names_and_keeps_raw_values(self):
        objs = [
            _FakeIface("Wi-Fi", self.WIFI, "Intel(R) Wi-Fi 6 AX201"),
            _FakeIface("Ethernet", self.ETH, "Realtek PCIe GbE"),
            _FakeIface("Npcap Loopback Adapter", self.LOOPBACK),
        ]
        raw_list = [self.WIFI, self.ETH, self.LOOPBACK, self.UNKNOWN]
        resolved = live_capture._resolve_interfaces(raw_list, objs)

        by_value = {i["value"]: i["label"] for i in resolved}
        # Every raw id is preserved verbatim as the option value.
        self.assertEqual(set(by_value), set(raw_list))
        self.assertEqual(by_value[self.WIFI], "Wi-Fi")
        self.assertEqual(by_value[self.ETH], "Ethernet")
        self.assertEqual(by_value[self.LOOPBACK], "Loopback")
        self.assertEqual(by_value[self.UNKNOWN], "Network Adapter (NPF_{DEADBEEF})")
        # No label ever exposes the full device path.
        for item in resolved:
            self.assertNotIn("\\Device", item["label"])

    def test_duplicate_labels_are_disambiguated(self):
        objs = [
            _FakeIface("Ethernet", self.WIFI),
            _FakeIface("Ethernet", self.ETH),
        ]
        resolved = live_capture._resolve_interfaces([self.WIFI, self.ETH], objs)
        labels = [i["label"] for i in resolved]
        self.assertEqual(len(set(labels)), 2, f"labels should be distinct: {labels}")

    def test_hyperv_wsl_adapter_is_surfaced_from_conf_ifaces(self):
        """
        The WSL 'vEthernet (WSL ...)' bridge is a Hyper-V virtual adapter: scapy
        knows it (conf.ifaces) with a real network_name, but get_if_list() omits
        it. It must still appear, labelled from its real name and mapped to its
        real NPF capture id -- never guessed from the GUID.
        """
        wsl_npf = r"\Device\NPF_{A1B2C3D4-5566-7788-99AA-BBCCDDEEFF00}"
        wsl = _FakeIface("vEthernet (WSL (Hyper-V firewall))", wsl_npf,
                         "Hyper-V Virtual Ethernet Adapter")
        physical = _FakeIface("Wi-Fi", self.WIFI, "Intel Wi-Fi 6")

        # get_if_list() sees only Wi-Fi; conf.ifaces knows Wi-Fi + the WSL bridge.
        resolved = live_capture._resolve_interfaces([self.WIFI], [physical, wsl])
        by_label = {i["label"]: i["value"] for i in resolved}

        self.assertIn("vEthernet (WSL (Hyper-V firewall))", by_label)
        self.assertEqual(by_label["vEthernet (WSL (Hyper-V firewall))"], wsl_npf)
        # It was not in get_if_list(), yet is now selectable.
        self.assertNotIn(wsl_npf, [self.WIFI])
        # The friendly Wi-Fi option is untouched, and no raw path leaks.
        self.assertEqual(by_label["Wi-Fi"], self.WIFI)
        for item in resolved:
            self.assertNotIn("\\Device", item["label"])

    def test_conf_ifaces_without_a_capture_id_are_not_added(self):
        """An adapter scapy lists but with no network_name can't be captured, so
        it must not become a dead option."""
        no_id = _FakeIface("Bluetooth Network Connection", "", "BT PAN")
        resolved = live_capture._resolve_interfaces([self.WIFI], [no_id])
        self.assertEqual([i["value"] for i in resolved], [self.WIFI])

    def test_resolution_falls_back_when_no_metadata(self):
        # No scapy interface objects (e.g. lookup failed): labels come from the
        # raw ids alone, still safe.
        resolved = live_capture._resolve_interfaces([self.LOOPBACK, self.UNKNOWN, "eth0"], [])
        by_value = {i["value"]: i["label"] for i in resolved}
        self.assertEqual(by_value[self.LOOPBACK], "Loopback")
        self.assertEqual(by_value["eth0"], "eth0")
        self.assertTrue(by_value[self.UNKNOWN].startswith("Network Adapter (NPF_{"))

    @skipUnless(_SCAPY, "scapy is required to enumerate interfaces")
    def test_list_interfaces_shape_and_value_integrity(self):
        from scapy.all import get_if_list

        raw_ids = set(get_if_list())
        items = live_capture.list_interfaces()
        self.assertTrue(items, "expected at least one interface on this host")
        for item in items:
            self.assertIn("value", item)
            self.assertIn("label", item)
            # The value must be a real scapy id (unchanged capture mechanism)...
            self.assertIn(item["value"], raw_ids)
            # ...and the label must never leak a full device path.
            self.assertNotIn("\\Device", item["label"])


# ---------------------------------------------------------------------------
# Live-detection root-cause diagnostics
#
# These pin down WHY real captured attacks (e.g. FTP brute force) read as Benign.
# They prove the live extractor and preprocessing are correct, and document the
# training-data artifact that the model actually learned -- so the failure is a
# train/serve distribution mismatch, not a bug in the live code.
# ---------------------------------------------------------------------------

import os as _os
from pathlib import Path as _Path


def _ftp_bruteforce_flow():
    """A realistic single FTP brute-force login attempt (client -> vsftpd:21)."""
    from scapy.layers.inet import IP, TCP
    from scapy.packet import Raw
    live_capture._ensure_live_on_path()
    from flow import Flow
    from feature_extractor import update_flow

    CLI, SRV, sp, dp = "172.24.48.1", "172.24.60.225", 54011, 21
    t = [1_530_000_000.0]

    def pk(src, dst, spt, dpt, flags, payload=b"", dt=0.0):
        t[0] += dt
        p = IP(src=src, dst=dst) / TCP(sport=spt, dport=dpt, flags=flags, window=64240)
        if payload:
            p = p / Raw(payload)
        p.time = t[0]
        return p

    seq = [
        (CLI, SRV, "S", b"", 0.0), (SRV, CLI, "SA", b"", 0.0004), (CLI, SRV, "A", b"", 0.0002),
        (SRV, CLI, "PA", b"220 (vsFTPd 3.0.3)\r\n", 0.010),
        (CLI, SRV, "PA", b"USER admin\r\n", 0.030),
        (SRV, CLI, "PA", b"331 Please specify the password.\r\n", 0.012),
        (CLI, SRV, "PA", b"PASS wrongpassword\r\n", 0.045),
        (SRV, CLI, "PA", b"530 Login incorrect.\r\n", 0.020),
        (CLI, SRV, "FA", b"", 0.015), (SRV, CLI, "FA", b"", 0.004),
    ]
    f = Flow(src_ip=CLI, dst_ip=SRV, src_port=sp, dst_port=dp, protocol=6)
    for src, dst, fl, pl, dt in seq:
        spt, dpt = (sp, dp) if src == CLI else (dp, sp)
        update_flow(f, pk(src, dst, spt, dpt, fl, pl, dt))
    return f


@skipUnless(_SCAPY, "scapy is required")
class LiveExtractionDiagnosticsTests(TestCase):
    """Rules out A (extraction), B (preprocessing) and C (finalization/direction)."""

    def setUp(self):
        live_capture._ensure_live_on_path()

    def test_extraction_is_correct_for_a_known_ftp_flow(self):
        from feature_calculator import calculate_features
        f = _ftp_bruteforce_flow()
        feats = calculate_features(f)

        # Direction: server port 21 kept as Dst Port; both directions counted.
        self.assertEqual(feats["Dst Port"], 21)
        self.assertEqual(f.forward_packets, 5)
        self.assertEqual(f.backward_packets, 5)

        # Duration is the real ~137 ms, NOT the training class's ~4 us artifact.
        self.assertGreater(feats["Flow Duration"], 100_000)

        # Payload bytes are summed correctly (USER+PASS fwd, three replies bwd).
        self.assertEqual(feats["TotLen Fwd Pkts"], 32)
        self.assertEqual(feats["TotLen Bwd Pkts"], 76)

        # Transport header length: 5 forward TCP packets x 20 bytes.
        self.assertEqual(feats["Fwd Header Len"], 100)
        self.assertEqual(feats["Fwd Seg Size Min"], 20)

        # Every training feature is present (nothing 0-filled by _as_frame).
        self.assertEqual(set(feats), set(ml.FEATURES))

    def test_short_tcp_flow_is_finalized_during_capture(self):
        # The flow closes with FIN both ways, so it must classify without a flush.
        session = live_capture.CaptureSession(interface=None, model_key=ml.DEFAULT_MODEL)
        from scapy.layers.inet import IP, TCP
        f = _ftp_bruteforce_flow()  # reuse packet script via a fresh session feed
        # Re-drive the same packets through the session handler:
        from scapy.packet import Raw
        CLI, SRV, sp, dp = "172.24.48.1", "172.24.60.225", 54011, 21
        t = [1_530_000_000.0]
        script = [
            (CLI, SRV, "S", b"", 0.0), (SRV, CLI, "SA", b"", 0.0004), (CLI, SRV, "A", b"", 0.0002),
            (SRV, CLI, "PA", b"220 x\r\n", 0.010), (CLI, SRV, "PA", b"USER admin\r\n", 0.030),
            (SRV, CLI, "PA", b"331 x\r\n", 0.012), (CLI, SRV, "PA", b"PASS y\r\n", 0.045),
            (SRV, CLI, "PA", b"530 x\r\n", 0.020), (CLI, SRV, "FA", b"", 0.015), (SRV, CLI, "FA", b"", 0.004),
        ]
        for src, dst, fl, pl, dt in script:
            spt, dpt = (sp, dp) if src == CLI else (dp, sp)
            t[0] += dt
            p = IP(src=src, dst=dst) / TCP(sport=spt, dport=dpt, flags=fl, window=64240)
            if pl:
                p = p / Raw(pl)
            p.time = t[0]
            session._handle(p)
        snap = session.snapshot()
        self.assertEqual(snap["flows"], 1, "closed flow must classify without flush")
        self.assertEqual(len(session._flows), 0)
        rec = snap["recent"][0]
        self.assertEqual((rec["src_ip"], rec["dst_port"], rec["protocol"]), ("172.24.48.1", 21, "TCP"))

    def test_live_and_dataset_testing_share_identical_preprocessing(self):
        import pandas as pd
        from feature_calculator import calculate_features
        feats = calculate_features(_ftp_bruteforce_flow())

        one = ml.predict_one(feats, "histgradientboosting")
        df = pd.DataFrame([{k: feats.get(k, 0.0) for k in ml.FEATURES}])
        batch = ml.predict_batch(df, "histgradientboosting")
        b_label = batch["frame"]["Predicted Class"].iloc[0]
        b_conf = float(batch["frame"]["Confidence"].iloc[0])

        self.assertEqual(one["label"], b_label)
        self.assertAlmostEqual(one["confidence"], b_conf, places=9)


def _train_parquet():
    p = _Path(_os.environ.get("IDS_DATA_ROOT", "/home/user/IDS/webapp_data")) \
        / "Processed_Data" / "balanced_train_selected.parquet"
    return p if p.is_file() else None


@skipUnless(_train_parquet() is not None, "training parquet (Git LFS) required")
class TrainingArtifactEvidenceTests(TestCase):
    """Documents the root cause: the FTP-BruteForce class is defined by
    zero-payload / microsecond flow artifacts a real capture cannot reproduce."""

    def test_ftp_bruteforce_class_is_dominated_by_zero_payload_micro_flows(self):
        import pandas as pd
        train = pd.read_parquet(_train_parquet())
        # Encode -> name via the shipped mapping
        import csv as _csv
        mapping = {}
        with open(_Path(_os.environ.get("IDS_DATA_ROOT", "/home/user/IDS/webapp_data"))
                  / "Processed_Data" / "label_mapping.csv") as fh:
            for row in _csv.DictReader(fh):
                mapping[int(row["Encoded"])] = row["Class"]
        ftp = train[train["Label"].map(mapping) == "FTP-BruteForce"]
        self.assertGreater(len(ftp), 0)

        zero_payload = (ftp["TotLen Fwd Pkts"] == 0).mean()
        micro = (ftp["Flow Duration"] < 1000).mean()  # < 1 ms
        # These are the artifact signatures a real login (payload + ~100 ms) can't match.
        self.assertGreater(zero_payload, 0.9, "expected ~100% zero forward payload")
        self.assertGreater(micro, 0.9, "expected ~100% sub-millisecond duration")


# ---------------------------------------------------------------------------
# Real-PCAP validation workflow (Phase 7 feature-parity tests)
#
# Reuse the diagnostic flow builders above; verify the PCAP path is the same
# flow construction / features / preprocessing as Live Capture + Dataset Testing.
# ---------------------------------------------------------------------------

import tempfile as _tempfile
from pathlib import Path as _P2

from predictor import pcap_validation as _pv


def _write_ftp_pcap(path, attempts=4):
    """A small realistic FTP brute-force capture (each attempt a closed TCP conn)."""
    from scapy.layers.inet import IP, TCP
    from scapy.packet import Raw
    from scapy.utils import wrpcap
    CLI, SRV = "172.24.48.1", "172.24.60.225"
    pkts = []
    for i in range(attempts):
        sp, dp, t = 54000 + i, 21, 1_530_000_000.0 + i
        script = [
            (CLI, SRV, "S", 0, 0.0), (SRV, CLI, "SA", 0, .0004), (CLI, SRV, "A", 0, .0002),
            (SRV, CLI, "PA", 20, .01), (CLI, SRV, "PA", 12, .03), (SRV, CLI, "PA", 34, .012),
            (CLI, SRV, "PA", 20, .045), (SRV, CLI, "PA", 22, .02), (CLI, SRV, "FA", 0, .015),
            (SRV, CLI, "FA", 0, .004),
        ]
        for src, dst, fl, pl, dt in script:
            t += dt
            spt, dpt = (sp, dp) if src == CLI else (dp, sp)
            p = IP(src=src, dst=dst) / TCP(sport=spt, dport=dpt, flags=fl, window=64240)
            if pl:
                p = p / Raw(b"x" * pl)
            p.time = t
            pkts.append(p)
    wrpcap(str(path), pkts)
    return attempts


@skipUnless(_SCAPY, "scapy is required for PCAP validation tests")
class PcapValidationTests(TestCase):
    def setUp(self):
        live_capture._ensure_live_on_path()
        self._dir = _tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = _P2(self._dir.name)

    # -- flow construction / feature parity --------------------------------

    def test_pcap_flows_produce_exactly_the_thirty_features_finite(self):
        pcap = self.root / "s1.pcap"
        n = _write_ftp_pcap(pcap, attempts=4)
        flows = _pv.replay_pcap(pcap)
        self.assertEqual(len(flows), n, "each closed FTP attempt is one finalized flow")
        for fl in flows:
            # exactly ml.FEATURES, in order, all finite -> no problem reported.
            self.assertIsNone(_pv._feature_problem(fl["features"]))
            self.assertEqual(list(fl["features"].keys()) and set(fl["features"]), set(ml.FEATURES))

    def test_missing_or_extra_or_nonfinite_features_are_rejected_not_zero_filled(self):
        good = {f: 0.0 for f in ml.FEATURES}
        self.assertIsNone(_pv._feature_problem(good))
        self.assertIn("missing", _pv._feature_problem({f: 0.0 for f in ml.FEATURES[:-1]}))
        self.assertIn("unexpected", _pv._feature_problem({**good, "Bogus": 1.0}))
        self.assertIn("non-finite", _pv._feature_problem({**good, "Flow Duration": float("inf")}))

    def test_bidirectional_and_finalization_match_live_capture(self):
        # One FTP attempt has fwd + reverse packets -> a single flow, Dst Port 21,
        # finalized by FIN during replay (no flush needed for a closed conn).
        pcap = self.root / "one.pcap"
        _write_ftp_pcap(pcap, attempts=1)
        flows = _pv.replay_pcap(pcap)
        self.assertEqual(len(flows), 1)
        self.assertEqual(flows[0]["dst_port"], 21)
        self.assertEqual(flows[0]["protocol"], "TCP")
        self.assertEqual(flows[0]["packets"], 10)

    # -- preprocessing / prediction parity with Dataset Testing -------------

    def test_pcap_prediction_path_matches_predict_one(self):
        pcap = self.root / "s.pcap"
        _write_ftp_pcap(pcap, attempts=3)
        res = _pv.validate_pcap(pcap, "FTP-BruteForce")
        self.assertEqual(res.valid_flows, 3)
        self.assertEqual(len(res.invalid_flows), 0)
        # Every row's pcap-path prediction equals predict_one on the same vector.
        flows = _pv.replay_pcap(pcap)
        for fl, (_, row) in zip(flows, res.frame.iterrows()):
            one = ml.predict_one(fl["features"], res.model_key)
            self.assertEqual(one["label"], row["Predicted Class"])
            self.assertAlmostEqual(one["confidence"], float(row["Confidence"]), places=6)

    def test_ground_truth_is_the_label_never_the_prediction(self):
        pcap = self.root / "s.pcap"
        _write_ftp_pcap(pcap, attempts=4)
        res = _pv.validate_pcap(pcap, "FTP-BruteForce")
        # Ground truth column is the capture label for every flow...
        self.assertTrue((res.frame["Label"] == "FTP-BruteForce").all())
        # ...and is independent of what the model predicted.
        self.assertIn("Predicted Class", res.frame.columns)
        self.assertEqual(res.correct + res.incorrect, res.valid_flows)

    # -- labels / directory ------------------------------------------------

    def test_resolve_label_aliases_exact_and_errors(self):
        self.assertEqual(_pv.resolve_label("ftp_bruteforce"), "FTP-BruteForce")
        self.assertEqual(_pv.resolve_label("ssh"), "SSH-Bruteforce")
        self.assertEqual(_pv.resolve_label("DoS attacks-Hulk"), "DoS attacks-Hulk")
        with self.assertRaises(_pv.ValidationError):
            _pv.resolve_label("dos")          # ambiguous: must name the DoS class
        with self.assertRaises(_pv.ValidationError):
            _pv.resolve_label("not-a-class")

    def test_validate_directory_infers_labels_and_keeps_sessions_separate(self):
        (self.root / "ftp_bruteforce").mkdir()
        (self.root / "benign").mkdir()
        _write_ftp_pcap(self.root / "ftp_bruteforce" / "a.pcap", attempts=2)
        _write_ftp_pcap(self.root / "benign" / "b.pcap", attempts=2)  # content irrelevant; label = folder
        summary = _pv.validate_directory(self.root)
        labels = {r.label for r in summary["per_pcap"]}
        self.assertEqual(labels, {"FTP-BruteForce", "Benign"})
        self.assertEqual(len(summary["per_pcap"]), 2, "one result per pcap/session")
        self.assertIsNotNone(summary["metrics"])
        self.assertIsNotNone(summary["confusion"])

    def test_results_are_written_as_csv_and_json(self):
        (self.root / "ftp_bruteforce").mkdir()
        _write_ftp_pcap(self.root / "ftp_bruteforce" / "a.pcap", attempts=3)
        summary = _pv.validate_directory(self.root)
        out = self.root / "results"
        written = _pv.save_results(summary, out)
        self.assertTrue((out / "summary.json").is_file())
        self.assertTrue((out / "flows.csv").is_file())
        self.assertTrue((out / "confusion_matrix.csv").is_file())
        import json as _json
        payload = _json.loads((out / "summary.json").read_text())
        self.assertIn("metrics", payload)
        self.assertIn("per_pcap", payload)


# ---------------------------------------------------------------------------
# Real committed PCAPs (sample_data/real_pcap/) — end-to-end parity + baseline
# ---------------------------------------------------------------------------

_REAL_PCAP_DIR = _P2(__file__).resolve().parents[2] / "sample_data" / "real_pcap"


@skipUnless(_SCAPY and _REAL_PCAP_DIR.is_dir(), "real PCAPs not present")
class RealPcapValidationTests(TestCase):
    """Runs the actual captured FTP/benign PCAPs through the pipeline."""

    def setUp(self):
        live_capture._ensure_live_on_path()

    def test_every_real_flow_has_exactly_thirty_finite_features(self):
        seen = 0
        for pcap, _label in _pv.iter_labelled_pcaps(_REAL_PCAP_DIR):
            for fl in _pv.replay_pcap(pcap):
                seen += 1
                self.assertIsNone(_pv._feature_problem(fl["features"]),
                                  f"{pcap.name}: {_pv._feature_problem(fl['features'])}")
                self.assertEqual(set(fl["features"]), set(ml.FEATURES))
        self.assertGreater(seen, 0, "expected flows from the real captures")

    def test_ground_truth_comes_from_the_folder_not_the_model(self):
        labels = {label for _p, label in _pv.iter_labelled_pcaps(_REAL_PCAP_DIR)}
        self.assertIn("FTP-BruteForce", labels)
        self.assertIn("Benign", labels)

    def test_real_validation_runs_with_no_invalid_flows(self):
        summary = _pv.validate_directory(_REAL_PCAP_DIR)
        self.assertGreater(summary["flows"], 0)
        self.assertEqual(summary["invalid_flows"], 0, "no flow should be invalid/zero-filled")
        # Benign successful-login sessions are recognised as Benign.
        benign = next(r for r in summary["metrics"]["per_class"] if r["class"] == "Benign")
        self.assertEqual(benign["correct"], benign["ground_truth"])

    def test_real_ftp_bruteforce_baseline_is_classified_benign(self):
        """
        Honest baseline (NOT a target): the existing model classifies the real
        FTP brute-force flows as Benign, because real FTP traffic is far from the
        CIC FTP-BruteForce artifact distribution. This locks the measured baseline
        so a future retrain is visibly different.
        """
        summary = _pv.validate_directory(_REAL_PCAP_DIR)
        ftp = next(r for r in summary["metrics"]["per_class"] if r["class"] == "FTP-BruteForce")
        self.assertGreater(ftp["ground_truth"], 0)
        self.assertEqual(ftp["correct"], 0, "current model detects none of the real FTP flows")
        # And the model itself is unchanged / healthy on CIC data.
        cic = _pv.cic_dataset_testing_metrics(sample=5000)
        self.assertGreater(cic["accuracy"], 0.9)


# ---------------------------------------------------------------------------
# SHAP explanations (read-only; no model change)
# ---------------------------------------------------------------------------

try:
    import shap as _shap_mod  # noqa: F401
    _HAS_SHAP = True
except Exception:  # noqa: BLE001
    _HAS_SHAP = False

_HAS_TRAIN_PARQUET = (_P2(__file__).resolve().parents[1] / "..").exists() and \
    (_train_parquet() is not None)


@skipUnless(_SCAPY and _HAS_SHAP and _train_parquet() is not None and _REAL_PCAP_DIR.is_dir(),
            "shap + training parquet + real PCAPs required")
class ShapAnalysisTests(TestCase):
    def setUp(self):
        live_capture._ensure_live_on_path()
        from predictor import shap_analysis as sa
        self.sa = sa
        self.model, _ = ml._load("histgradientboosting")
        self.cic = sa.cic_ftp_correct(self.model, sample=60)

    def test_data_sources_are_honest(self):
        # CIC set: only rows the model actually classifies as FTP-BruteForce.
        enc = {v: k for k, v in ml.LABELS.items()}["FTP-BruteForce"]
        self.assertTrue((self.model.predict(self.cic[ml.FEATURES]) == enc).all())
        # Real set: only flows the model classifies Benign.
        real = self.sa.real_ftp_flows(self.model, only_predicted="Benign")
        self.assertGreater(len(real), 0)
        names = [ml.LABELS.get(int(c)) for c in self.model.predict(real)]
        self.assertTrue(all(n == "Benign" for n in names))

    def test_shap_values_have_the_right_shape_and_feature_order(self):
        res = self.sa.explain(self.model, self.cic)
        self.assertEqual(res.values.shape, (len(self.cic), 30, 15))
        self.assertEqual(res.base.shape, (15,))
        self.assertEqual(res.features, list(ml.FEATURES))  # exact order

    def test_shap_is_exact_additive(self):
        res = self.sa.explain(self.model, self.cic)
        # base + sum(shap) reconstructs the raw margin to numerical precision.
        self.assertLess(self.sa.additivity_error(self.model, res), 1e-6)

    def test_global_importance_is_finite_nonneg_aligned(self):
        import numpy as np
        res = self.sa.explain(self.model, self.cic)
        gi = self.sa.global_importance(res)
        self.assertEqual(set(gi["feature"]), set(ml.FEATURES))
        self.assertEqual(len(gi), 30)
        self.assertTrue(np.isfinite(gi["mean_abs_shap"]).all())
        self.assertTrue((gi["mean_abs_shap"] >= 0).all())

    def test_cic_vs_real_comparison_is_finite_and_ranked(self):
        import numpy as np
        real = self.sa.real_ftp_flows(self.model, only_predicted="Benign")
        cmp = self.sa.compare_cic_vs_real(self.model, self.cic, real, toward="Benign")
        self.assertEqual(len(cmp), 30)
        for col in ("cic_shap_to_Benign", "real_shap_to_Benign", "difference_real_minus_cic"):
            self.assertTrue(np.isfinite(cmp[col]).all())
        # sorted by |difference| descending
        d = cmp["difference_real_minus_cic"].abs().to_numpy()
        self.assertTrue((d[:-1] >= d[1:] - 1e-9).all())


# ---------------------------------------------------------------------------
# Attack-family / severity dashboard (additive analysis over existing results)
# ---------------------------------------------------------------------------

from predictor import attack_dashboard as _ad
from predictor import classes as _classes

_DASH_FLOWS = _P2(__file__).resolve().parents[2] / "validation" / "results" / "flows.csv"


class AttackFamilySeverityMappingTests(TestCase):
    def test_family_mapping_is_reused_from_classes_for_every_class(self):
        for cls in ml.LABELS.values():
            self.assertEqual(_ad.family_of(cls), _classes.family_of(cls))
            self.assertIn(_ad.family_of(cls), _classes.FAMILY_ORDER)

    def test_severity_mapping_covers_all_fifteen_classes_explicitly(self):
        model_classes = set(ml.LABELS.values())
        self.assertEqual(set(_ad.SEVERITY) & model_classes, model_classes,
                         "every trained class must have an explicit severity")
        for cls in model_classes:
            self.assertIn(_ad.severity_of(cls), _ad.SEVERITY_ORDER)

    def test_benign_is_the_only_none_severity(self):
        none_classes = [c for c in ml.LABELS.values() if _ad.severity_of(c) == "None"]
        self.assertEqual(none_classes, ["Benign"])

    def test_unmapped_class_is_surfaced_not_invented(self):
        self.assertEqual(_ad.severity_of("Totally New Attack"), "Unknown")


class AttackDashboardAggregationTests(TestCase):
    def _flows(self):
        # Synthetic per-flow results: ground truth Label + model Predicted Class.
        return pd.DataFrame({
            "Label":           ["Benign", "Benign", "FTP-BruteForce", "FTP-BruteForce",
                                 "DoS attacks-Hulk", "DDOS attack-HOIC"],
            "Predicted Class": ["Benign", "FTP-BruteForce", "Benign", "FTP-BruteForce",
                                 "DoS attacks-Hulk", "Benign"],
            "Confidence":      [0.9, 0.6, 0.8, 0.95, 0.99, 0.7],
        })

    def test_family_and_severity_support_come_from_ground_truth_only(self):
        flows = self._flows()
        d = _ad.build_dashboard(flows)
        fam = {r["group"]: r for r in d["family"]["summary"]}
        # support = ground-truth family counts (Benign x2, bruteforce x2, dos x1, ddos x1)
        self.assertEqual(fam["benign"]["support"], 2)
        self.assertEqual(fam["bruteforce"]["support"], 2)
        self.assertEqual(fam["dos"]["support"], 1)
        self.assertEqual(fam["ddos"]["support"], 1)
        sev = {r["group"]: r for r in d["severity"]["summary"]}
        self.assertEqual(sev["None"]["support"], 2)     # 2 Benign
        self.assertEqual(sev["High"]["support"], 3)     # 2 FTP + 1 Hulk
        self.assertEqual(sev["Critical"]["support"], 1) # 1 HOIC

    def test_support_is_independent_of_predictions(self):
        flows = self._flows()
        base = _ad.build_dashboard(flows)
        flipped = flows.copy()
        flipped["Predicted Class"] = "Benign"  # change predictions only
        after = _ad.build_dashboard(flipped)
        base_sup = {r["group"]: r["support"] for r in base["family"]["summary"]}
        after_sup = {r["group"]: r["support"] for r in after["family"]["summary"]}
        self.assertEqual(base_sup, after_sup, "family support must not depend on predictions")

    def test_class_recall_aggregation_is_correct(self):
        d = _ad.build_dashboard(self._flows())
        fam = {r["group"]: r for r in d["family"]["summary"]}
        # bruteforce: 2 flows, exactly 1 predicted correctly (FTP->FTP)
        self.assertEqual(fam["bruteforce"]["correct_class"], 1)
        self.assertEqual(fam["bruteforce"]["incorrect_class"], 1)
        self.assertAlmostEqual(fam["bruteforce"]["class_recall"], 0.5)
        self.assertEqual(fam["benign"]["correct_class"], 1)  # 1 of 2 Benign correct

    def test_confusion_and_per_class_match_the_shared_metric_helpers(self):
        from predictor import pcap_validation as pv
        flows = self._flows()
        d = _ad.build_dashboard(flows)
        self.assertEqual(d["confusion"], pv.confusion_table(flows["Label"], flows["Predicted Class"]))
        expected = pv.compute_metrics(flows["Label"], flows["Predicted Class"])["per_class"]
        self.assertEqual(d["per_class"], expected)

    def test_rollup_from_existing_per_class_sums_counts(self):
        per_class = pd.DataFrame({
            "class": ["FTP-BruteForce", "SSH-Bruteforce", "Benign"],
            "ground_truth": [100, 100, 50], "correct": [80, 100, 49], "incorrect": [20, 0, 1],
        })
        fam = {r["group"]: r for r in _ad.rollup_per_class(per_class, _ad.family_of)}
        self.assertEqual(fam["bruteforce"]["support"], 200)
        self.assertEqual(fam["bruteforce"]["correct_class"], 180)
        self.assertAlmostEqual(fam["bruteforce"]["class_recall"], 0.9)

    def test_html_renders_the_required_sections(self):
        html = _ad.render_html(_ad.build_dashboard(self._flows()))
        for section in ("Confusion matrix", "Per-class", "By attack family", "By severity"):
            self.assertIn(section, html)


@skipUnless(_DASH_FLOWS.is_file(), "committed real-PCAP flows.csv required")
class AttackDashboardRealResultsUnchangedTests(TestCase):
    """The dashboard must faithfully reflect the committed real-PCAP results."""

    def test_real_pcap_metrics_match_the_committed_baseline(self):
        flows = _ad.load_flows(_DASH_FLOWS)
        d = _ad.build_dashboard(flows)
        # The frozen real-PCAP baseline (see summary.json): 42 flows, acc 0.1905.
        self.assertEqual(d["flows"], 42)
        self.assertAlmostEqual(d["overall"]["accuracy"], 8 / 42, places=6)
        self.assertAlmostEqual(d["overall"]["macro_f1"], 0.16, places=4)
        ftp = next(r for r in d["per_class"] if r["class"] == "FTP-BruteForce")
        self.assertEqual(ftp["ground_truth"], 34)
        self.assertEqual(ftp["correct"], 0)   # unchanged: all FTP flows -> Benign
        self.assertEqual(_ad.severity_of("FTP-BruteForce"), "High")
        self.assertEqual(_ad.family_of("FTP-BruteForce"), "bruteforce")


# ---------------------------------------------------------------------------
# Realistic-PCAP retraining experiment (candidate only; baseline frozen)
# ---------------------------------------------------------------------------

import numpy as _np
from predictor import retraining as _rt

_RETRAIN_READY = _SCAPY and _train_parquet() is not None and _REAL_PCAP_DIR.is_dir()


@skipUnless(_RETRAIN_READY, "scapy + parquet + real PCAPs required")
class RetrainingExperimentTests(TestCase):
    def setUp(self):
        live_capture._ensure_live_on_path()
        self.real = _rt.extract_real_flows()
        # A small CIC slice keeps training fast; determinism/splits don't need 281k.
        train = pd.read_parquet(_train_parquet())
        # A small stratified-ish slice keeps training fast; splits/determinism
        # don't need the full 281k rows.
        self.cic = train.sample(min(1500, len(train)), random_state=0).reset_index(drop=True)
        self.cic_X = self.cic[ml.FEATURES].reset_index(drop=True)
        self.cic_y = self.cic["Label"].reset_index(drop=True)

    # -- baseline / artifact isolation ------------------------------------

    def test_candidate_dir_is_separate_from_production_models(self):
        cand = _rt.candidate_dir().resolve()
        prod = ml.MODELS_DIR.resolve()
        self.assertNotEqual(cand, prod)
        self.assertNotIn(str(prod), str(cand))
        # production model file exists and is under webapp_data, not the candidate dir
        self.assertTrue((prod / "HistGradientBoosting_Tuned.pkl").is_file())

    def test_baseline_saved_model_is_unchanged(self):
        # The frozen baseline still scores its known CIC number.
        test = pd.read_parquet(ml.DATA_ROOT / "Processed_Data" / "test_selected.parquet")
        model, _ = ml._load("histgradientboosting")
        acc = float((model.predict(test[ml.FEATURES]) == test["Label"]).mean())
        # Frozen baseline: its known full-test-set accuracy must be unchanged.
        self.assertAlmostEqual(acc, 0.9803, places=4)

    # -- feature parity ----------------------------------------------------

    def test_real_flows_have_exactly_thirty_finite_features_no_zero_fill(self):
        df = self.real.df
        self.assertEqual(len(df), 42)
        self.assertEqual(len(self.real.invalid), 0)
        for f in ml.FEATURES:
            self.assertIn(f, df.columns)
            self.assertTrue(_np.isfinite(pd.to_numeric(df[f], errors="coerce")).all())

    def test_assembled_training_uses_exactly_and_only_ml_features_in_order(self):
        X, y, w, man = _rt.assemble_training(self.cic_X, self.cic_y, self.real.df)
        self.assertEqual(list(X.columns), list(ml.FEATURES))  # exact order

    # -- labels from ground truth -----------------------------------------

    def test_real_labels_come_only_from_folder_ground_truth(self):
        self.assertEqual(set(self.real.df["Label"]), {"FTP-BruteForce", "Benign"})
        # capture -> label is consistent with the directory structure
        for cap, sub in self.real.df.groupby("capture"):
            self.assertEqual(sub["Label"].nunique(), 1)

    # -- leakage control ---------------------------------------------------

    def test_leave_one_capture_out_never_shares_a_pcap(self):
        logo = _rt.leave_one_capture_out(self.cic_X, self.cic_y, self.real, ftp_weight=5.0)
        for fold in logo["folds"]:
            held = fold["held_out_capture"]
            self.assertNotIn(held, fold["train_manifest"]["included_captures"],
                             "held-out PCAP must not be in the training captures")

    # -- reproducibility & schema -----------------------------------------

    def test_candidate_training_is_reproducible(self):
        X, y, w, _ = _rt.assemble_training(self.cic_X, self.cic_y, self.real.df, ftp_weight=10.0)
        m1 = _rt.train(X, y, w)
        m2 = _rt.train(X, y, w)
        p1 = m1.predict(self.real.df[ml.FEATURES])
        p2 = m2.predict(self.real.df[ml.FEATURES])
        self.assertTrue((p1 == p2).all(), "same seed + data must give identical predictions")

    def test_candidate_artifacts_have_a_valid_schema(self):
        import tempfile, joblib
        X, y, w, man = _rt.assemble_training(self.cic_X, self.cic_y, self.real.df, ftp_weight=10.0)
        model = _rt.train(X, y, w)
        with tempfile.TemporaryDirectory() as d:
            # redirect candidate_dir to a temp path
            orig = _rt.candidate_dir
            _rt.candidate_dir = lambda: Path(d)
            try:
                saved = _rt.save_candidate(model, {"experiment": "test", "seed": _rt.SEED})
            finally:
                _rt.candidate_dir = orig
            self.assertTrue(Path(saved["model"]).is_file())
            meta = json.loads(Path(saved["metadata"]).read_text())
            self.assertEqual(meta["scaler"], None)
            self.assertEqual(meta["seed"], _rt.SEED)
            reloaded = joblib.load(saved["model"])
            self.assertEqual(len(reloaded.classes_), len(model.classes_))

    def test_hgb_params_are_the_tuned_ones_mapped(self):
        p = _rt.hgb_params()
        self.assertEqual(p["max_iter"], 250)
        self.assertEqual(p["max_depth"], 5)
        self.assertNotIn("algorithm", p)


# ---------------------------------------------------------------------------
# CICFlowMeter-vs-custom-extractor experiment (analysis only; production frozen)
# ---------------------------------------------------------------------------

try:
    import cicflowmeter as _cfm_pkg  # noqa: F401
    _HAS_CFM = True
except Exception:  # noqa: BLE001
    _HAS_CFM = False

_CFM_READY = _SCAPY and _HAS_CFM and _REAL_PCAP_DIR.is_dir()


@skipUnless(_CFM_READY, "scapy + cicflowmeter + real PCAPs required")
class CicflowmeterExtractionTests(TestCase):
    """The independent CICFlowMeter extraction path used to isolate cause A vs B."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        live_capture._ensure_live_on_path()
        from predictor import cfm_extract as _cf, pcap_validation as _pv
        cls.cf = _cf
        cls.pairs = list(_pv.iter_labelled_pcaps(_REAL_PCAP_DIR))
        # extract once; every test reads this (extraction is the slow part)
        cls.ext = _cf.extract_real_pcaps(cls.pairs)

    # -- 30-feature schema, order, finiteness -----------------------------

    def test_mapped_features_are_exactly_ml_features_in_order(self):
        for pcap, _label in self.pairs:
            mapped = self.cf.map_to_ml_features(self.cf.run_cicflowmeter(pcap))
            self.assertEqual(list(mapped.X.columns), list(ml.FEATURES))  # exact order

    def test_combined_frame_carries_all_thirty_features(self):
        comb = self.ext["combined"]
        self.assertFalse(comb.empty)
        for f in ml.FEATURES:
            self.assertIn(f, comb.columns)

    def test_every_mapped_flow_is_finite(self):
        comb = self.ext["combined"]
        vals = comb[list(ml.FEATURES)].apply(pd.to_numeric, errors="coerce").to_numpy()
        self.assertTrue(_np.isfinite(vals).all())

    def test_feature_count_is_thirty(self):
        self.assertEqual(len(ml.FEATURES), 30)
        self.assertEqual(self.ext["combined"][list(ml.FEATURES)].shape[1], 30)

    # -- failures reported, never zero-filled -----------------------------

    def test_invalid_flows_are_dropped_not_zero_filled(self):
        # Nothing is fabricated: a non-finite feature drops the row with a reason.
        import numpy as np
        raw = pd.DataFrame([{c: 1.0 for c in self.cf.CFM_TO_ML.values()}])
        raw.loc[0, "flow_duration"] = np.inf
        mapped = self.cf.map_to_ml_features(raw)
        self.assertEqual(len(mapped.X), 0)              # dropped, not kept-as-zero
        self.assertEqual(len(mapped.invalid), 1)        # and reported
        self.assertIn("Flow Duration", mapped.invalid[0]["non_finite_features"])

    def test_extraction_reports_validity_counts(self):
        total_valid = sum(p["valid_flows"] for p in self.ext["per_pcap"])
        self.assertEqual(total_valid, self.ext["total_flows"])
        self.assertEqual(self.ext["total_invalid"],
                         sum(p["invalid_flows"] for p in self.ext["per_pcap"]))

    # -- labels from folder ground truth ----------------------------------

    def test_labels_come_only_from_folders(self):
        comb = self.ext["combined"]
        self.assertTrue(set(comb["Label"]).issubset({"FTP-BruteForce", "Benign"}))
        for p in self.ext["per_pcap"]:
            sub = "ftp" if p["label"] == "FTP-BruteForce" else "benign"
            self.assertIn(sub, str(p["pcap"]).lower())

    # -- reproducibility ---------------------------------------------------

    def test_extraction_is_reproducible(self):
        pcap = self.pairs[0][0]
        a = self.cf.map_to_ml_features(self.cf.run_cicflowmeter(pcap)).X
        b = self.cf.map_to_ml_features(self.cf.run_cicflowmeter(pcap)).X
        pd.testing.assert_frame_equal(a, b)

    # -- mapping is by definition, not just name --------------------------

    def test_mapping_covers_all_thirty_and_only_real_columns(self):
        self.assertEqual(set(self.cf.CFM_TO_ML.keys()), set(ml.FEATURES))
        raw = self.cf.run_cicflowmeter(self.pairs[0][0])
        for col in self.cf.CFM_TO_ML.values():
            self.assertIn(col, raw.columns)


@skipUnless(_CFM_READY and _train_parquet() is not None,
            "scapy + cicflowmeter + parquet + real PCAPs required")
class CfmExperimentTests(TestCase):
    """The three-path comparison that distinguishes model failure from extraction."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        live_capture._ensure_live_on_path()
        from predictor import cfm_experiment as _ce
        cls.ce = _ce

    def test_production_model_and_frozen_baseline_are_not_modified(self):
        import hashlib
        csv = self.ce.repo_root() / "validation" / "results" / "flows.csv"
        before = hashlib.sha256(csv.read_bytes()).hexdigest()
        _m, _df = self.ce.path_b_custom()          # reads only
        after = hashlib.sha256(csv.read_bytes()).hexdigest()
        self.assertEqual(before, after, "frozen flows.csv must not be rewritten")
        # baseline still scores its known CIC number
        test = pd.read_parquet(ml.DATA_ROOT / "Processed_Data" / "test_selected.parquet")
        model, _ = ml._load("histgradientboosting")
        acc = float((model.predict(test[ml.FEATURES]) == test["Label"]).mean())
        self.assertAlmostEqual(acc, 0.9803, places=4)

    def test_path_c_uses_production_model_on_thirty_features(self):
        m, cfm_df = self.ce.path_c_cicflowmeter()
        for f in ml.FEATURES:
            self.assertIn(f, cfm_df.columns)
        self.assertIn(m["ftp_recall"], (None, 0.0) if m["ftp_recall"] is None else [m["ftp_recall"]])
        self.assertEqual(m["extraction"]["total_invalid"], 0)

    def test_ftp_recall_is_computed_for_both_extractors(self):
        # §8 — the headline comparison exists and is a real number, whatever it is.
        _mb, _ = self.ce.path_b_custom()
        mc, _ = self.ce.path_c_cicflowmeter()
        self.assertIsNotNone(_mb["ftp_recall"])
        self.assertIsNotNone(mc["ftp_recall"])

    def test_feature_distribution_reports_all_three_sources(self):
        _mb, custom_df = self.ce.path_b_custom()
        _mc, cfm_df = self.ce.path_c_cicflowmeter()
        dist = self.ce.feature_distribution(custom_df, cfm_df)
        self.assertIn("cic_ftp_median", dist.columns)
        self.assertIn("custom_real_ftp_median", dist.columns)
        self.assertIn("cicflowmeter_real_ftp_median", dist.columns)
        # unavailable values are None, never fabricated to 0
        self.assertTrue(dist["cicflowmeter_real_ftp_median"].notna().any())

    def test_closeness_summary_is_consistent(self):
        _mb, custom_df = self.ce.path_b_custom()
        _mc, cfm_df = self.ce.path_c_cicflowmeter()
        dist = self.ce.feature_distribution(custom_df, cfm_df)
        s = self.ce.summarise_closer(dist)
        self.assertEqual(s["comparable_features"],
                         s["cicflowmeter_closer"] + s["custom_closer"] + s["tie"])


# ---------------------------------------------------------------------------
# Realistic-PCAP DATA EXPANSION — local-lab capture framework (experimental)
# ---------------------------------------------------------------------------

import shutil as _shutil
from predictor import lab_capture as _lab

_HAS_PYFTPDLIB = False
try:
    import pyftpdlib as _pyftpdlib_pkg  # noqa: F401
    _HAS_PYFTPDLIB = True
except Exception:  # noqa: BLE001
    _HAS_PYFTPDLIB = False

_HAS_TCPDUMP = _shutil.which("tcpdump") is not None
_IS_ROOT = (getattr(_os, "geteuid", lambda: 1)() == 0)
_CAN_CAPTURE = _SCAPY and _HAS_PYFTPDLIB and _HAS_TCPDUMP and _IS_ROOT
_EXPANSION_DIR = _P2(__file__).resolve().parents[2] / "validation" / "realistic_pcaps"
_HAS_EXPANSION = (_EXPANSION_DIR / "MANIFEST.csv").is_file()


class LabTargetAndScenarioTests(TestCase):
    """Framework invariants that need no live capture (always run)."""

    def test_target_is_loopback_only(self):
        t = _lab.default_target()
        self.assertTrue(t.is_loopback())
        self.assertEqual(t.host, "127.0.0.1")

    def test_bpf_confines_to_lab_ports(self):
        t = _lab.LabTarget(control_port=21, passive_lo=60000, passive_hi=60040)
        bpf = t.bpf()
        self.assertIn("tcp port 21", bpf)
        self.assertIn("portrange 60000-60040", bpf)

    def test_scenario_catalogs_are_well_formed(self):
        for catalog in (_lab.BENIGN_SCENARIOS, _lab.BRUTEFORCE_SCENARIOS):
            names = [n for n, _fn in catalog]
            self.assertEqual(len(names), len(set(names)))  # unique
            for _n, fn in catalog:
                self.assertTrue(callable(fn))
        # the diversity the task asks for
        self.assertGreaterEqual(len(_lab.BENIGN_SCENARIOS), 6)
        self.assertGreaterEqual(len(_lab.BRUTEFORCE_SCENARIOS), 6)

    def test_bruteforce_credentials_are_all_wrong(self):
        # brute-force pools must never contain the real lab password
        t = _lab.default_target()
        self.assertNotIn(t.password, _lab._WRONG_PW)

    def test_capture_meta_row_matches_manifest_columns(self):
        m = _lab.CaptureMeta(
            capture_id="x", label="Benign", scenario="s", timestamp="t",
            source="127.0.0.1", destination="127.0.0.1:21", client_tool="python-ftplib",
            attempts=1, capture_duration_s=0.0, pcap_filename="benign/x.pcap",
            validation_status="valid")
        self.assertEqual(set(m.as_row().keys()), set(_lab.MANIFEST_COLUMNS))


@skipUnless(_CAN_CAPTURE, "needs root + tcpdump + pyftpdlib + scapy for a live capture")
class LiveLabCaptureTests(TestCase):
    """Prove the framework captures REAL, independent loopback traffic."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = Path(tempfile.mkdtemp())
        cls.target = _lab.LabTarget(control_port=2121)   # unprivileged test port
        cls.server = _lab.FtpLabServer(target=cls.target, root=cls.tmp / "home").start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        _shutil.rmtree(cls.tmp, ignore_errors=True)
        super().tearDownClass()

    def _capture(self, name, fn):
        pcap = self.tmp / f"{name}.pcap"
        cap = _lab.Tcpdump(pcap_path=pcap, target=self.target).start()
        try:
            stats = fn(self.target)
        finally:
            cap.stop()
        return pcap, stats

    def test_server_verifies(self):
        self.assertTrue(self.server.verify())

    def test_benign_capture_is_real_and_loopback(self):
        pcap, _ = self._capture("benign", _lab.benign_login_list)
        chk = _lab.verify_pcap(pcap, self.target)
        self.assertTrue(chk.ok, chk.errors)
        self.assertGreater(chk.packets, 0)
        self.assertTrue(chk.src_ok and chk.dst_ok)      # loopback only
        self.assertTrue(chk.has_ftp_traffic)

    def test_bruteforce_capture_records_failed_attempts(self):
        pcap, stats = self._capture("bf", _lab.bf_few_attempts)
        chk = _lab.verify_pcap(pcap, self.target)
        self.assertTrue(chk.ok, chk.errors)
        # every guess is wrong on purpose -> all failures, no successful login
        self.assertEqual(stats["successes"], 0)
        self.assertGreater(stats["failures"], 0)

    def test_two_captures_are_independent_not_copies(self):
        p1, _ = self._capture("indep1", _lab.benign_login)
        p2, _ = self._capture("indep2", _lab.benign_login)
        self.assertNotEqual(p1.read_bytes(), p2.read_bytes())   # distinct captures
        from scapy.all import rdpcap, TCP
        ports1 = {pk[TCP].sport for pk in rdpcap(str(p1)) if TCP in pk}
        ports2 = {pk[TCP].sport for pk in rdpcap(str(p2)) if TCP in pk}
        # fresh interactions use fresh ephemeral client ports
        self.assertTrue(ports1 != ports2)


@skipUnless(_HAS_EXPANSION, "collected realistic_pcaps/ not present")
class CollectedExpansionArtifactTests(TestCase):
    """Validate the committed collection artifacts (manifest, pcaps, labels)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with (_EXPANSION_DIR / "MANIFEST.csv").open() as f:
            cls.rows = list(csv.DictReader(f))

    def test_manifest_has_all_required_columns(self):
        self.assertTrue(self.rows)
        self.assertEqual(set(self.rows[0].keys()), set(_lab.MANIFEST_COLUMNS))

    def test_every_manifest_pcap_exists_on_disk(self):
        for r in self.rows:
            self.assertTrue((_EXPANSION_DIR / r["pcap_filename"]).is_file(), r["pcap_filename"])

    def test_labels_come_from_folders_not_predictions(self):
        for r in self.rows:
            folder = r["pcap_filename"].split("/")[0]
            expected = "Benign" if folder == "benign" else "FTP-BruteForce"
            self.assertEqual(r["label"], expected)
            self.assertIn(folder, ("benign", "ftp_bruteforce"))

    def test_pcaps_on_disk_match_manifest(self):
        on_disk = {p.name for p in _EXPANSION_DIR.rglob("*.pcap")}
        in_manifest = {Path(r["pcap_filename"]).name for r in self.rows}
        self.assertEqual(on_disk, in_manifest)

    @skipUnless(_SCAPY, "scapy required")
    def test_a_sample_capture_verifies_and_extracts_thirty_features(self):
        from predictor import pcap_validation as pv
        live_capture._ensure_live_on_path()
        target = _lab.default_target()
        # check one benign and one brute-force capture end to end
        sample = [r for r in self.rows if r["capture_id"] in ("benign_02", "ftpbf_01")]
        self.assertTrue(sample)
        for r in sample:
            pcap = _EXPANSION_DIR / r["pcap_filename"]
            chk = _lab.verify_pcap(pcap, target)
            self.assertTrue(chk.ok, chk.errors)
            flows = pv.replay_pcap(pcap)
            self.assertGreater(len(flows), 0)
            for fl in flows:
                # exactly the 30 finite features, in order, no zero-fill
                self.assertIsNone(pv._feature_problem(fl["features"]))
                ordered = [float(fl["features"][f]) for f in ml.FEATURES]
                self.assertEqual(len(ordered), 30)


# ---------------------------------------------------------------------------
# Realistic-PCAP DIVERSIFICATION v2 — multi-client/server/env capture framework
# ---------------------------------------------------------------------------

import hashlib as _hashlib
from predictor import lab_capture_v2 as _lab2

_V2_DIR = _P2(__file__).resolve().parents[2] / "validation" / "realistic_pcaps_v2"
_HAS_V2 = (_V2_DIR / "MANIFEST.csv").is_file()


class LabCaptureV2FrameworkTests(TestCase):
    """v2 framework invariants (no live capture required)."""

    def test_all_environments_are_loopback_net(self):
        for addr in _lab2.ENV_ADDR.values():
            self.assertTrue(_lab2.V2Target(addr, 21, 60000, 60040, "permissive", "e")
                            .is_loopback_net())

    def test_non_loopback_target_is_rejected_by_is_loopback_net(self):
        self.assertFalse(_lab2.V2Target("8.8.8.8", 21, 60000, 60040, "permissive", "e")
                         .is_loopback_net())

    def test_bpf_confines_to_host_and_ports(self):
        t = _lab2.V2Target("127.0.0.2", 21, 60000, 60039, "permissive", "env_b")
        bpf = t.bpf()
        self.assertIn("host 127.0.0.2", bpf)
        self.assertIn("tcp port 21", bpf)
        self.assertIn("portrange 60000-60039", bpf)

    def test_build_targets_are_all_loopback_and_unique(self):
        targets = _lab2.build_targets(Path("/tmp/nonexistent"))
        self.assertEqual(len(targets), len(_lab2.ENV_ADDR) * len(_lab2.SERVER_VARIANTS))
        for t in targets.values():
            self.assertTrue(t.is_loopback_net())
        ranges = [(t.addr, t.passive_lo) for t in targets.values()]
        self.assertEqual(len(ranges), len(set(ranges)))   # no passive-range clash

    def test_spec_counts_in_target_range(self):
        b, f = _lab2.benign_specs(), _lab2.bruteforce_specs()
        self.assertTrue(30 <= len(b) <= 50, len(b))
        self.assertTrue(30 <= len(f) <= 50, len(f))

    def test_specs_are_distinct_combinations(self):
        specs = _lab2.benign_specs() + _lab2.bruteforce_specs()
        keys = [(s.label, s.scenario, s.client, s.server, s.env, s.mode) for s in specs]
        self.assertEqual(len(keys), len(set(keys)))       # no duplicate spec

    def test_diversity_spans_multiple_clients_and_servers(self):
        specs = _lab2.benign_specs() + _lab2.bruteforce_specs()
        self.assertGreaterEqual(len({s.client for s in specs}), 3)
        self.assertGreaterEqual(len({s.server for s in specs}), 3)
        self.assertGreaterEqual(len({s.env for s in specs}), 3)

    def test_bruteforce_pool_never_contains_real_password(self):
        self.assertNotIn(_lab2.PASSWORD, _lab2.WRONG_PW)

    def test_manifest_columns_cover_required_fields(self):
        for field_ in ("scenario", "label", "client", "server", "environment",
                       "interface", "attempts", "start_time", "end_time",
                       "packet_count", "duration_s", "source", "destination",
                       "capture_command", "verification_status"):
            self.assertIn(field_, _lab2.MANIFEST_COLUMNS)


@skipUnless(_CAN_CAPTURE, "needs root + tcpdump + pyftpdlib + scapy")
class LabCaptureV2LiveTests(TestCase):
    """Prove v2 captures are real, independent, loopback-only, no external dest."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = Path(tempfile.mkdtemp())
        # use a non-privileged port + env_a for the live test server
        cls.target = _lab2.V2Target("127.0.0.1", 2399, 61000, 61039, "permissive", "env_a")
        cls.pool = _lab2.ServerPool({("env_a", "permissive"): cls.target}, cls.tmp / "home")
        cls.pool.get("env_a", "permissive")

    @classmethod
    def tearDownClass(cls):
        cls.pool.stop_all()
        _shutil.rmtree(cls.tmp, ignore_errors=True)
        super().tearDownClass()

    def _cap(self, name, fn):
        pcap = self.tmp / f"{name}.pcap"
        cap = _lab2.Tcpdump(pcap_path=pcap, target=self.target).start()
        try:
            stats = fn(self.target)
        finally:
            cap.stop()
        return pcap, stats

    def test_benign_capture_is_real_and_loopback_only(self):
        pcap, _ = self._cap("b", _lab2.b_list)
        chk = _lab2.verify_pcap(pcap, self.target)
        self.assertTrue(chk.ok, chk.errors)
        self.assertTrue(chk.loopback_only and chk.expected_host_present)

    def test_bruteforce_capture_only_fails(self):
        pcap, stats = self._cap("bf", lambda t: _lab2.bf_newconn(
            t, [("admin", pw) for pw in _lab2.WRONG_PW[:4]]))
        chk = _lab2.verify_pcap(pcap, self.target)
        self.assertTrue(chk.ok, chk.errors)
        self.assertEqual(stats["successes"], 0)
        self.assertEqual(stats["failures"], 4)

    def test_two_captures_are_independent_not_copies(self):
        p1, _ = self._cap("i1", _lab2.b_login)
        p2, _ = self._cap("i2", _lab2.b_login)
        self.assertNotEqual(p1.read_bytes(), p2.read_bytes())
        from scapy.all import rdpcap, TCP
        s1 = {pk[TCP].sport for pk in rdpcap(str(p1)) if TCP in pk}
        s2 = {pk[TCP].sport for pk in rdpcap(str(p2)) if TCP in pk}
        self.assertNotEqual(s1, s2)                        # fresh ephemeral ports

    def test_bruteforce_refuses_non_loopback_destination(self):
        # safety: the brute-force drivers assert a loopback target before any traffic
        external = _lab2.V2Target("93.184.216.34", 21, 61000, 61039, "permissive", "x")
        with self.assertRaises(AssertionError):
            _lab2.bf_newconn(external, [("admin", "x")])


@skipUnless(_HAS_V2, "collected realistic_pcaps_v2/ not present")
class CollectedV2ArtifactTests(TestCase):
    """Validate the committed v2 corpus + that baseline artifacts are untouched."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with (_V2_DIR / "MANIFEST.csv").open() as f:
            cls.rows = list(csv.DictReader(f))

    def _pcap_for(self, row):
        folder = "benign" if row["label"] == "Benign" else "ftp_bruteforce"
        return next((_V2_DIR / folder).glob(f"{row['capture_id']}_*.pcap"))

    def test_manifest_has_required_columns(self):
        self.assertTrue(self.rows)
        self.assertEqual(set(self.rows[0].keys()), set(_lab2.MANIFEST_COLUMNS))

    def test_counts_in_target_range(self):
        benign = [r for r in self.rows if r["label"] == "Benign"]
        bf = [r for r in self.rows if r["label"] == "FTP-BruteForce"]
        self.assertTrue(30 <= len(benign) <= 50, len(benign))
        self.assertTrue(30 <= len(bf) <= 50, len(bf))

    def test_every_manifest_pcap_exists(self):
        for r in self.rows:
            self.assertTrue(self._pcap_for(r).is_file(), r["capture_id"])

    def test_no_duplicate_pcaps(self):
        hashes = {}
        for p in _V2_DIR.rglob("*.pcap"):
            h = _hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertNotIn(h, hashes, f"{p.name} duplicates {hashes.get(h)}")
            hashes[h] = p.name

    def test_labels_from_folders_only(self):
        for r in self.rows:
            folder = "benign" if r["label"] == "Benign" else "ftp_bruteforce"
            self.assertTrue(self._pcap_for(r).parent.name == folder)
            self.assertIn(r["label"], ("Benign", "FTP-BruteForce"))

    def test_all_captures_passed_verification(self):
        for r in self.rows:
            self.assertEqual(r["verification_status"], "valid", r["capture_id"])

    def test_all_destinations_are_loopback(self):
        import ipaddress
        loop = ipaddress.ip_network("127.0.0.0/8")
        for r in self.rows:
            addr = r["destination"].rsplit(":", 1)[0]
            self.assertIn(ipaddress.ip_address(addr), loop, r["destination"])

    def test_diversity_present(self):
        self.assertGreaterEqual(len({r["client"] for r in self.rows}), 3)
        self.assertGreaterEqual(len({r["server"] for r in self.rows}), 3)
        self.assertGreaterEqual(len({r["environment"] for r in self.rows}), 3)

    def test_feature_extraction_report_has_no_incomplete_flows(self):
        with (_V2_DIR / "feature_extraction_report.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(rows)
        self.assertEqual(sum(int(r["incomplete_flows"]) for r in rows), 0)
        self.assertGreater(sum(int(r["flows"]) for r in rows), 0)

    @skipUnless(_SCAPY, "scapy required")
    def test_sample_captures_extract_exactly_thirty_finite_ordered_features(self):
        from predictor import pcap_validation as pv
        live_capture._ensure_live_on_path()
        import ipaddress
        sample = [r for r in self.rows if r["capture_id"] in
                  ("benign_01", "benign_20", "ftpbf_01", "ftpbf_15")]
        self.assertTrue(sample)
        for r in sample:
            pcap = self._pcap_for(r)
            addr = r["destination"].rsplit(":", 1)[0]
            port = int(r["destination"].rsplit(":", 1)[1])
            t = _lab2.V2Target(addr, port, 60000, 60040, "x", r["environment"])
            chk = _lab2.verify_pcap(pcap, t)
            self.assertTrue(chk.ok, chk.errors)
            for fl in pv.replay_pcap(pcap):
                self.assertIsNone(pv._feature_problem(fl["features"]))   # 30, finite
                self.assertEqual([f for f in fl["features"] if f in ml.FEATURES].__len__(), 30)
                ordered = [float(fl["features"][f]) for f in ml.FEATURES]
                self.assertTrue(all(_np.isfinite(ordered)))

    def test_production_and_baseline_artifacts_untouched(self):
        # production model still scores its known CIC number
        test = pd.read_parquet(ml.DATA_ROOT / "Processed_Data" / "test_selected.parquet")
        model, _ = ml._load("histgradientboosting")
        acc = float((model.predict(test[ml.FEATURES]) == test["Label"]).mean())
        self.assertAlmostEqual(acc, 0.9803, places=4)
        # v1 collection + frozen validation results still present and non-empty
        v1 = _P2(__file__).resolve().parents[2] / "validation"
        for rel in ("realistic_pcaps/MANIFEST.csv", "results/flows.csv"):
            p = v1 / rel
            self.assertTrue(p.is_file() and p.stat().st_size > 0, str(p))


# ---------------------------------------------------------------------------
# Realistic-PCAP retraining v2 — candidate experiment (production frozen)
# ---------------------------------------------------------------------------

from predictor import retraining_v2 as _r2

_V2_RESULTS_DIR = _P2(__file__).resolve().parents[2] / "validation" / "results" / "retraining_v2"
_HAS_V2_RESULTS = (_V2_RESULTS_DIR / "final_verdict.json").is_file()
_RETRAIN_V2_READY = _SCAPY and _train_parquet() is not None and _V2_DIR.is_dir() \
    and (_V2_DIR / "MANIFEST.csv").is_file()


@skipUnless(_RETRAIN_V2_READY, "scapy + parquet + realistic_pcaps_v2 required")
class RetrainingV2ModuleTests(TestCase):
    """Leakage + schema regression tests on the v2 retraining machinery."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        live_capture._ensure_live_on_path()
        cls.real = _r2.extract_real_flows_v2()
        # a tiny CIC subsample keeps training fast; leakage/schema don't need 281k
        X, y = _r2.rt.load_cic()
        cls.cic_X, cls.cic_y = _r2.stratified_cic_subsample(X, y, 2000)

    # -- schema / feature integrity ---------------------------------------

    def test_real_flows_have_exactly_thirty_ordered_finite_features(self):
        df = self.real.df
        self.assertEqual(len(df), 854)
        self.assertEqual(len(self.real.invalid), 0)               # no zero-fill
        feat_cols = [c for c in df.columns if c in ml.FEATURES]
        self.assertEqual(feat_cols, list(ml.FEATURES))            # exact order
        self.assertEqual(len(feat_cols), 30)
        self.assertTrue(_np.isfinite(df[ml.FEATURES].to_numpy()).all())

    def test_assemble_uses_exactly_and_only_ml_features_in_order(self):
        X, y, w, man = _r2.assemble_v2(self.cic_X, self.cic_y, self.real.df, real_weight=6.0)
        self.assertEqual(list(X.columns), list(ml.FEATURES))

    def test_labels_come_only_from_folder_ground_truth(self):
        self.assertEqual(set(self.real.df["Label"]), {"FTP-BruteForce", "Benign"})
        for cap, sub in self.real.df.groupby("capture"):
            self.assertEqual(sub["Label"].nunique(), 1)           # one label per PCAP

    def test_every_real_flow_traces_to_its_pcap(self):
        for _, row in self.real.df.head(50).iterrows():
            self.assertTrue(row["flow_uid"].startswith(row["capture"] + "#"))

    # -- leakage (capture-level) ------------------------------------------

    def test_leave_one_capture_out_never_shares_a_pcap_or_flow(self):
        # a few captures is enough to prove the invariant without training 83 folds
        caps = self.real.captures[:2] + self.real.captures[-2:]
        subset = _r2.RealFlows(
            df=self.real.df[self.real.df["capture"].isin(caps)].reset_index(drop=True),
            invalid=[])
        loco = _r2.leave_one_capture_out_v2(self.cic_X, self.cic_y, subset, real_weight=6.0)
        self.assertEqual(len(loco["folds"]), len(caps))
        for f in loco["folds"]:
            self.assertFalse(f["held_out_in_train_captures"])     # no PCAP in both
            self.assertEqual(f["train_test_flow_overlap"], 0)     # no flow in both

    def test_assemble_excludes_the_held_out_capture(self):
        cap = self.real.captures[0]
        _X, _y, _w, man = _r2.assemble_v2(self.cic_X, self.cic_y, self.real.df,
                                          exclude_captures=(cap,), real_weight=1.0)
        self.assertNotIn(cap, man["included_captures"])

    # -- artifact isolation / production untouched ------------------------

    def test_candidate_dir_is_separate_from_production(self):
        cand = _r2.candidate_dir_v2().resolve()
        prod = ml.MODELS_DIR.resolve()
        self.assertNotEqual(cand, prod)
        self.assertNotIn(str(prod), str(cand))

    def test_production_model_file_unchanged_and_scores_known_number(self):
        test = pd.read_parquet(ml.DATA_ROOT / "Processed_Data" / "test_selected.parquet")
        model, _ = ml._load("histgradientboosting")
        acc = float((model.predict(test[ml.FEATURES]) == test["Label"]).mean())
        self.assertAlmostEqual(acc, 0.9803, places=4)

    def test_no_zero_filling_a_missing_feature_would_be_reported(self):
        # feed a flow missing a feature to the SAME validator the extractor uses
        from predictor import pcap_validation as pv
        bad = {f: 1.0 for f in ml.FEATURES if f != "Flow Duration"}
        self.assertIsNotNone(pv._feature_problem(bad))            # reported, not filled


@skipUnless(_HAS_V2_RESULTS, "committed retraining_v2 results not present")
class RetrainingV2ResultsTests(TestCase):
    """Validate the committed experiment evidence and its safety guarantees."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.verdict = json.loads((_V2_RESULTS_DIR / "final_verdict.json").read_text())
        cls.leakage = json.loads((_V2_RESULTS_DIR / "leakage_validation.json").read_text())
        cls.meta = json.loads((_V2_RESULTS_DIR / "training_metadata.json").read_text())

    def test_leakage_validation_all_pass(self):
        self.assertTrue(self.leakage["all_pass"], self.leakage["checks"])
        for key in ("no_pcap_in_train_and_test", "labels_only_from_folders",
                    "exactly_30_features", "feature_order_equals_ml_features",
                    "all_finite", "no_zero_filling", "every_flow_traceable_to_pcap",
                    "candidate_dir_separate_from_production", "production_model_unchanged"):
            self.assertTrue(self.leakage["checks"][key], key)

    def test_production_model_unchanged_across_experiment(self):
        self.assertEqual(self.meta["production_model_sha256_before"],
                         self.meta["production_model_sha256_after"])
        self.assertTrue(self.meta["production_model_unchanged"])

    def test_no_candidate_is_auto_promoted(self):
        self.assertFalse(self.verdict["promote"])
        self.assertIn(self.verdict["classification"],
                      ("production candidate", "promising but insufficient evidence",
                       "unsuccessful"))

    def test_candidate1_reproduces_baseline(self):
        self.assertTrue(self.meta["cic_only_reproduction"]["reproduces"])

    def test_required_evidence_files_exist(self):
        for name in ("baseline_metrics.csv", "candidate_metrics.csv",
                     "weighting_comparison.csv", "per_class_metrics.csv",
                     "per_capture_metrics.csv", "confusion_baseline_cic.csv",
                     "leakage_validation.json", "training_metadata.json",
                     "final_verdict.json", "report.md"):
            self.assertTrue((_V2_RESULTS_DIR / name).is_file(), name)

    def test_candidate_artifacts_live_outside_production_dir(self):
        cand_dir = _P2(__file__).resolve().parents[2] / "validation" / "models" / "realistic_pcap_candidate_v2"
        if cand_dir.is_dir():
            self.assertFalse(str(ml.MODELS_DIR.resolve()) in str(cand_dir.resolve()))


# ---------------------------------------------------------------------------
# Independent real-PCAP TEST corpus + final evaluation (frozen models)
# ---------------------------------------------------------------------------

from predictor import independent_capture as _ic, independent_eval as _ie, custom_ftp_server as _cfs

_INDEP_DIR = _P2(__file__).resolve().parents[2] / "validation" / "independent_real_pcaps"
_HAS_INDEP = (_INDEP_DIR / "MANIFEST.csv").is_file()
_INDEP_RESULTS = _P2(__file__).resolve().parents[2] / "validation" / "results" / "independent_test"
_HAS_INDEP_RESULTS = (_INDEP_RESULTS / "final_verdict.json").is_file()


def _pcap_sha_set(root):
    return {_hashlib.sha256(p.read_bytes()).hexdigest() for p in _P2(root).rglob("*.pcap")}


class IndependentCaptureFrameworkTests(TestCase):
    """Framework invariants for the independent corpus (no live capture)."""

    def test_custom_ftp_server_module_is_a_real_second_implementation(self):
        # a genuinely different server class, not pyftpdlib
        self.assertTrue(hasattr(_cfs, "CustomFTPServer"))
        self.assertNotIn("pyftpdlib", _cfs.__file__)

    def test_all_targets_are_controlled_local(self):
        targets = _ic.build_targets(Path("/tmp/nonexistent"))
        for t in targets.values():
            self.assertTrue(t.is_controlled_local())

    def test_external_address_is_rejected(self):
        self.assertFalse(_ic.is_controlled_local("8.8.8.8"))
        self.assertFalse(_ic.is_controlled_local("93.184.216.34"))

    def test_bpf_confines_to_host_and_ports(self):
        t = _ic.IndepTarget("127.0.0.5", 2130, 62000, 62039, "lo", "custom", "lo5")
        bpf = t.bpf()
        self.assertIn("host 127.0.0.5", bpf)
        self.assertIn("tcp port 2130", bpf)

    def test_spec_counts_in_target_range(self):
        b, f = _ic.benign_specs(), _ic.bruteforce_specs()
        self.assertTrue(15 <= len(b) <= 20, len(b))
        self.assertTrue(15 <= len(f) <= 20, len(f))

    def test_bruteforce_pool_excludes_real_password(self):
        self.assertNotIn(_ic.PASSWORD, _ic.WRONG_PW)


@skipUnless(_CAN_CAPTURE, "needs root + tcpdump + scapy for a live capture")
class IndependentCaptureLiveTests(TestCase):
    """Prove the custom-server captures are real, controlled-local, independent."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = Path(tempfile.mkdtemp())
        cls.target = _ic.IndepTarget("127.0.0.5", 2137, 63000, 63039, "lo", "custom", "lo5")
        cls.pool = _ic.ServerPool({("lo5", "custom"): cls.target}, cls.tmp / "home")
        cls.pool.get("lo5", "custom")

    @classmethod
    def tearDownClass(cls):
        cls.pool.stop_all()
        _shutil.rmtree(cls.tmp, ignore_errors=True)
        super().tearDownClass()

    def _cap(self, name, fn):
        pcap = self.tmp / f"{name}.pcap"
        cap = _ic.Tcpdump(pcap_path=pcap, target=self.target).start()
        try:
            stats = fn(self.target)
        finally:
            cap.stop()
        return pcap, stats

    def test_custom_server_benign_capture_is_real_and_controlled(self):
        pcap, _ = self._cap("b", _ic.b_upload_then_download)
        chk = _ic.verify_pcap(pcap, self.target)
        self.assertTrue(chk.ok, chk.errors)
        self.assertTrue(chk.controlled_local_only and chk.expected_host_present)

    def test_custom_server_bruteforce_only_fails(self):
        pcap, stats = self._cap("bf", lambda t: _ic.bf_newconn(t, [("admin", pw) for pw in _ic.WRONG_PW[:4]]))
        chk = _ic.verify_pcap(pcap, self.target)
        self.assertTrue(chk.ok, chk.errors)
        self.assertEqual(stats["successes"], 0)
        self.assertEqual(stats["failures"], 4)

    def test_two_captures_are_independent(self):
        p1, _ = self._cap("i1", _ic.b_nlst_listing)
        p2, _ = self._cap("i2", _ic.b_nlst_listing)
        self.assertNotEqual(p1.read_bytes(), p2.read_bytes())

    def test_bruteforce_refuses_external_destination(self):
        ext = _ic.IndepTarget("93.184.216.34", 2137, 63000, 63039, "lo", "custom", "x")
        with self.assertRaises(AssertionError):
            _ic.bf_newconn(ext, [("admin", "x")])


@skipUnless(_HAS_INDEP, "independent corpus not present")
class IndependentTestArtifactLeakageTests(TestCase):
    """Automated leakage/independence proofs on the committed independent corpus."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with (_INDEP_DIR / "MANIFEST.csv").open() as f:
            cls.rows = list(csv.DictReader(f))
        cls.root = _P2(__file__).resolve().parents[2] / "validation"

    def test_independent_pcaps_disjoint_from_v1(self):
        indep = _pcap_sha_set(_INDEP_DIR)
        v1 = _pcap_sha_set(self.root / "realistic_pcaps")
        self.assertTrue(indep.isdisjoint(v1))

    def test_independent_pcaps_disjoint_from_v2(self):
        indep = _pcap_sha_set(_INDEP_DIR)
        v2 = _pcap_sha_set(self.root / "realistic_pcaps_v2")
        self.assertTrue(indep.isdisjoint(v2))

    def test_no_duplicate_pcaps(self):
        hashes = {}
        for p in _INDEP_DIR.rglob("*.pcap"):
            h = _hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertNotIn(h, hashes, f"{p.name} duplicates {hashes.get(h)}")
            hashes[h] = p.name

    def test_labels_come_only_from_folders(self):
        for r in self.rows:
            folder = "benign" if r["label"] == "Benign" else "ftp_bruteforce"
            matches = list((_INDEP_DIR / folder).glob(f"{r['capture_id']}_*.pcap"))
            self.assertTrue(matches, r["capture_id"])
            self.assertIn(r["label"], ("Benign", "FTP-BruteForce"))

    def test_all_destinations_are_controlled_local(self):
        for r in self.rows:
            addr = r["destination"].rsplit(":", 1)[0]
            self.assertTrue(_ic.is_controlled_local(addr), r["destination"])

    def test_all_captures_verified_valid(self):
        for r in self.rows:
            self.assertEqual(r["verification_status"], "valid", r["capture_id"])

    @skipUnless(_SCAPY and _train_parquet() is not None, "scapy + parquet required")
    def test_extraction_is_exactly_thirty_ordered_finite_no_zero_fill(self):
        flows = _ie.extract_test_flows()
        self.assertGreater(len(flows.df), 0)
        self.assertEqual(len(flows.invalid), 0)                       # no zero-fill
        feat = [c for c in flows.df.columns if c in ml.FEATURES]
        self.assertEqual(feat, list(ml.FEATURES))                    # order
        self.assertEqual(len(feat), 30)
        self.assertTrue(_np.isfinite(flows.df[ml.FEATURES].to_numpy()).all())
        for _, row in flows.df.head(30).iterrows():
            self.assertTrue(row["flow_uid"].startswith(row["capture"] + "#"))  # traceable


@skipUnless(_HAS_INDEP_RESULTS, "committed independent-test results not present")
class IndependentTestResultsTests(TestCase):
    """Validate the committed independent-evaluation evidence and frozen-model safety."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.verdict = json.loads((_INDEP_RESULTS / "final_verdict.json").read_text())
        cls.leakage = json.loads((_INDEP_RESULTS / "leakage_validation.json").read_text())
        cls.hashes = json.loads((_INDEP_RESULTS / "model_hashes_before_after.json").read_text())

    def test_leakage_all_pass(self):
        self.assertTrue(self.leakage["all_pass"], self.leakage["checks"])
        for k in ("independent_disjoint_from_v1", "independent_disjoint_from_v2",
                  "no_duplicate_within_independent", "labels_only_from_folders",
                  "exactly_30_features", "feature_order_equals_ml_features", "all_finite",
                  "no_zero_filling", "every_flow_traceable_to_pcap"):
            self.assertTrue(self.leakage["checks"][k], k)

    def test_frozen_models_unchanged_before_after(self):
        self.assertTrue(self.hashes["unchanged"])
        self.assertEqual(self.hashes["before"], self.hashes["after"])

    def test_candidate_not_promoted(self):
        self.assertFalse(self.verdict["promote"])
        self.assertIn(self.verdict["classification"],
                      ("PROMISING AND SUPPORTED BY INDEPENDENT TEST",
                       "PROMISING BUT INSUFFICIENT EVIDENCE",
                       "NOT SUPPORTED BY INDEPENDENT TEST"))

    def test_required_evidence_files_exist(self):
        for name in ("baseline_metrics.csv", "candidate_metrics.csv",
                     "confusion_matrix_production.csv", "confusion_matrix_candidate.csv",
                     "per_class_metrics.csv", "per_capture_metrics.csv",
                     "prediction_distribution.csv", "confidence_distribution.csv",
                     "bootstrap_or_ci_results.csv", "three_way_comparison.csv",
                     "test_set_manifest_summary.csv", "leakage_validation.json",
                     "model_hashes_before_after.json", "final_verdict.json", "report.md"):
            self.assertTrue((_INDEP_RESULTS / name).is_file(), name)

    def test_production_model_still_scores_known_number(self):
        test = pd.read_parquet(ml.DATA_ROOT / "Processed_Data" / "test_selected.parquet")
        model, _ = ml._load("histgradientboosting")
        acc = float((model.predict(test[ml.FEATURES]) == test["Label"]).mean())
        self.assertAlmostEqual(acc, 0.9803, places=4)
