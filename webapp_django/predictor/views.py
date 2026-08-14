import csv
import json

from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.utils.timezone import localtime
from django.views.decorators.http import require_http_methods

from . import classes, history_log, live_capture, ml
from .forms import BatchUploadForm, ManualFlowForm


def _base_context(active: str) -> dict:
    return {
        "active": active,
        "models": ml.available_models(),
        "default_model": ml.DEFAULT_MODEL,
    }


# ---------------------------------------------------------------------------
# Landing page — a standing overview of the project
# ---------------------------------------------------------------------------


def home(request):
    """
    The operator dashboard: three feature entry points plus a live system
    overview drawn from real backend state -- the active model, whether capture
    is available/running, the run log's totals, and the most recent detection.

    Everything that reads the model registry works even when the .pkl files are
    not on disk, so the page still renders on a fresh clone. The history and
    capture look-ups are wrapped so a missing log or absent scapy degrades to a
    quiet default rather than a 500.
    """
    context = _base_context("home")

    default_spec = ml.MODEL_REGISTRY[ml.DEFAULT_MODEL]

    # History overview -- real totals from the append-only run log.
    try:
        recent_runs = history_log.recent()
        summary = history_log.summarise(recent_runs)
        history_total = history_log.count()
    except Exception:  # noqa: BLE001 - the log must never break the dashboard
        recent_runs, summary, history_total = [], None, 0

    last_run = recent_runs[0] if recent_runs else None

    # Live-capture status -- reflects the actual CaptureManager, if any.
    try:
        capture_supported = live_capture.capture_supported()
        live_session = live_capture.manager.session
        live_running = bool(live_session and live_session.running)
        live_snapshot = live_session.snapshot() if live_session else None
    except Exception:  # noqa: BLE001
        capture_supported, live_running, live_snapshot = False, False, None

    context.update(
        {
            "families": classes.legend(),
            "feature_count": len(ml.FEATURES),
            "group_count": len(ml.FEATURE_GROUPS),
            "class_count": len(ml.LABELS),
            "scorecards": [
                {
                    "key": key,
                    "acc_pct": 100.0 * spec["accuracy"],
                    "f1_pct": 100.0 * spec["macro_f1"],
                    "is_default": key == ml.DEFAULT_MODEL,
                    **spec,
                }
                for key, spec in ml.MODEL_REGISTRY.items()
            ],
            # System overview
            "active_model": default_spec,
            "model_count": len(ml.available_models()),
            "capture_supported": capture_supported,
            "live_running": live_running,
            "live_snapshot": live_snapshot,
            "history_total": history_total,
            "summary": summary,
            "last_run": last_run,
        }
    )
    return render(request, "predictor/home.html", context)


# ---------------------------------------------------------------------------
# Manual entry — one flow, 30 features
# ---------------------------------------------------------------------------


def _form_groups() -> list[dict]:
    """Feature groups with each field's stats attached, ready for the template."""
    groups = []
    for title, description, features in ml.FEATURE_GROUPS:
        groups.append(
            {
                "title": title,
                "description": description,
                "fields": [
                    {"name": f, "step": "1" if ml.FEATURE_STATS[f]["integer"] else "any", **ml.FEATURE_STATS[f]}
                    for f in features
                ],
            }
        )
    return groups


def manual(request):
    context = _base_context("manual")
    context.update(
        {
            "groups": _form_groups(),
            "presets": ml.PRESETS,
            "preset_names": sorted(ml.PRESETS.keys()),
            "families": classes.legend(),
        }
    )
    return render(request, "predictor/manual.html", context)


@require_http_methods(["POST"])
def api_predict(request):
    """
    Classify a single flow. Accepts JSON or form-encoded input and always
    answers JSON, so the page can update without a reload.
    """
    if request.content_type == "application/json":
        try:
            payload = json.loads(request.body or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"error": "Malformed JSON."}, status=400)
    else:
        payload = request.POST.dict()

    form = ManualFlowForm(payload)
    if not form.is_valid():
        return JsonResponse(
            {
                "error": "Some values are missing or not numeric.",
                "fields": {k: [str(e) for e in v] for k, v in form.errors.items()},
            },
            status=400,
        )

    model_key = form.cleaned_data["model_key"]
    values = form.feature_values()

    try:
        result = ml.predict_one(values, model_key)
    except (FileNotFoundError, ValueError) as exc:
        return JsonResponse({"error": str(exc)}, status=503)

    history_log.record(
        kind=history_log.MANUAL,
        model_key=result["model_key"],
        model_name=result["model"],
        predicted_label=result["label"],
        confidence=result["confidence"],
        features=values,
        # A manual entry is exactly one flow. Recording it means the Rows column
        # shows 1 rather than a dash that looks like missing data.
        row_count=1,
        attack_count=1 if result["is_attack"] else 0,
    )

    return JsonResponse(result)


# ---------------------------------------------------------------------------
# Batch — score a whole dataset
# ---------------------------------------------------------------------------


def batch(request):
    context = _base_context("batch")
    context["features"] = ml.FEATURES

    if request.method != "POST":
        context["form"] = BatchUploadForm()
        return render(request, "predictor/batch.html", context)

    form = BatchUploadForm(request.POST, request.FILES)
    context["form"] = form
    if not form.is_valid():
        return render(request, "predictor/batch.html", context)

    upload = form.cleaned_data["dataset"]
    model_key = form.cleaned_data["model_key"]

    try:
        frame = ml.read_upload(upload, upload.name)
        result = ml.predict_batch(frame, model_key)
    except (ml.BatchError, FileNotFoundError, ValueError) as exc:
        context["error"] = str(exc)
        return render(request, "predictor/batch.html", context)

    out_frame = result.pop("frame")

    evaluation = result.get("evaluation")
    history_log.record(
        kind=history_log.BATCH,
        model_key=result["model_key"],
        model_name=result["model"],
        source_filename=upload.name,
        row_count=result["rows"],
        attack_count=result["attack_count"],
        accuracy=evaluation["accuracy"] if evaluation else None,
        macro_f1=evaluation["macro_f1"] if evaluation else None,
    )

    preview_cols = ["Predicted Class", "Confidence"]
    if "Label" in out_frame.columns:
        preview_cols.insert(0, "Label")
    preview = out_frame[preview_cols + ml.FEATURES[:4]].head(25)

    context.update(
        {
            "result": result,
            "evaluation": evaluation,
            "filename": upload.name,
            "preview_columns": list(preview.columns),
            "preview_rows": list(preview.itertuples(index=False, name=None)),
        }
    )
    return render(request, "predictor/batch_result.html", context)


# ---------------------------------------------------------------------------
# Live capture — classify real traffic off a network interface
# ---------------------------------------------------------------------------


def live(request):
    """
    The live-capture console. Renders even where capture is impossible (no
    scapy, no privileges): the page then shows the interfaces it could find and
    reports any failure inline when Start is pressed, so the rest of the app is
    never affected.
    """
    context = _base_context("live")
    context.update(
        {
            "interfaces": live_capture.list_interfaces(),
            "capture_supported": live_capture.capture_supported(),
            "families": classes.legend(),
        }
    )
    return render(request, "predictor/live.html", context)


def _resolve_model_key(raw: str | None) -> str:
    """Same rule as the forms: fall back to the default, reject anything unknown."""
    key = (raw or "").strip() or ml.DEFAULT_MODEL
    if key not in ml.MODEL_REGISTRY:
        raise ValueError("Unknown model.")
    return key


@require_http_methods(["POST"])
def api_live_start(request):
    """Start a capture. Body: JSON or form with `iface` and `model_key`."""
    if request.content_type == "application/json":
        try:
            payload = json.loads(request.body or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"error": "Malformed JSON."}, status=400)
    else:
        payload = request.POST.dict()

    # "auto" (or blank) means "let scapy choose the interface".
    iface = (payload.get("iface") or "").strip()
    if iface.lower() in ("", "auto"):
        iface = None

    try:
        model_key = _resolve_model_key(payload.get("model_key"))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    try:
        session = live_capture.manager.start(iface, model_key)
    except live_capture.CaptureError as exc:
        return JsonResponse({"error": str(exc)}, status=503)

    return JsonResponse(session.snapshot())


@require_http_methods(["POST"])
def api_live_stop(request):
    """Stop the running capture, if any, and return its final snapshot."""
    session = live_capture.manager.stop()
    if session is None:
        return JsonResponse({"running": False, "recent": []})
    return JsonResponse(session.snapshot())


@require_http_methods(["GET"])
def api_live_status(request):
    """
    Poll endpoint. `?since=<seq>` returns only records newer than the client's
    highest seen sequence number, so the table can append rather than reload.
    """
    session = live_capture.manager.session
    if session is None:
        return JsonResponse({"running": False, "recent": []})

    try:
        since = int(request.GET.get("since", "0"))
    except (TypeError, ValueError):
        since = 0

    return JsonResponse(session.snapshot(since=since))


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def history(request):
    """
    Read the run log back. Everything shown is derived from the file, so the
    page is an honest view of what is on disk -- if the log is empty, the app
    genuinely has no record of a run.
    """
    context = _base_context("history")

    kind = request.GET.get("kind", "")
    if kind not in (history_log.MANUAL, history_log.BATCH):
        kind = ""

    window = history_log.recent()
    runs = [r for r in window if r.kind == kind] if kind else window
    total = history_log.count()

    context.update(
        {
            "runs": runs,
            "summary": history_log.summarise(runs),
            "total": total,
            "window": len(window),
            # True when the log is longer than the page reads, so the footnote
            # can say the totals cover recent runs rather than all of them.
            "truncated": total > len(window),
            "kind": kind,
            # Clearing is a two-step: this renders the confirmation bar rather
            # than letting one click delete the log.
            "confirming": request.GET.get("confirm") == "clear",
            "counts": {
                "manual": sum(1 for r in window if r.is_manual),
                "batch": sum(1 for r in window if not r.is_manual),
            },
            "log_name": history_log.log_path().name,
        }
    )
    return render(request, "predictor/history.html", context)


CSV_COLUMNS = [
    "when", "type", "model", "predicted_class", "confidence",
    "source_file", "rows", "attacks", "attack_share", "accuracy", "macro_f1",
]


def download_history(request):
    """
    Serve the whole log as CSV, so a run can be charted in Excel or read with
    pandas without unpacking JSON first. The columns are the history table's,
    flattened: a manual entry fills the prediction columns, an upload fills the
    file ones, and a cell the run has no answer for is left empty rather than
    guessed at. The 30 feature values a manual entry carries are left out --
    they would swamp the sheet, and the raw log on the server still has them.

    Oldest first, like the log itself, so the rows read as a timeline.
    """
    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = 'attachment; filename="prediction-history.csv"'

    writer = csv.writer(response)
    writer.writerow(CSV_COLUMNS)

    for run in history_log.all_runs():
        writer.writerow(
            [
                # Local time with its offset: unambiguous, and the same clock
                # the History page shows rather than UTC.
                localtime(run.at).isoformat(timespec="seconds") if run.at else "",
                run.kind_display,
                run.model_name,
                run.predicted_label,
                f"{run.confidence:.6f}" if run.confidence is not None else "",
                run.source_filename,
                run.row_count if run.row_count is not None else "",
                run.attack_count if run.attack_count is not None else "",
                f"{run.attack_share:.6f}" if run.row_count else "",
                f"{run.accuracy:.6f}" if run.accuracy is not None else "",
                f"{run.macro_f1:.6f}" if run.macro_f1 is not None else "",
            ]
        )

    return response


@require_http_methods(["POST"])
def clear_history(request):
    history_log.clear()
    return redirect("history")
