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
