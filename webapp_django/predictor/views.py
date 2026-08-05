import json
import uuid
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from . import classes, ml
from .forms import BatchUploadForm, ManualFlowForm
from .models import PredictionLog

BATCH_OUTPUT_DIR = Path(settings.BASE_DIR) / "media" / "batch"


def _base_context(active: str) -> dict:
    return {
        "active": active,
        "models": ml.available_models(),
        "default_model": ml.DEFAULT_MODEL,
    }


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

    PredictionLog.objects.create(
        kind=PredictionLog.MANUAL,
        model_key=result["model_key"],
        model_name=result["model"],
        predicted_label=result["label"],
        confidence=result["confidence"],
        features=values,
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
    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    out_frame.to_csv(BATCH_OUTPUT_DIR / f"{token}.csv", index=False)

    evaluation = result.get("evaluation")
    PredictionLog.objects.create(
        kind=PredictionLog.BATCH,
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
            "download_token": token,
            "filename": upload.name,
            "preview_columns": list(preview.columns),
            "preview_rows": list(preview.itertuples(index=False, name=None)),
        }
    )
    return render(request, "predictor/batch_result.html", context)


def download_batch(request, token):
    if len(token) != 32 or not all(c in "0123456789abcdef" for c in token):
        raise Http404("Unknown download.")

    path = BATCH_OUTPUT_DIR / f"{token}.csv"
    if not path.is_file():
        raise Http404("That result is no longer available.")

    return FileResponse(path.open("rb"), as_attachment=True, filename="predictions.csv")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def history(request):
    context = _base_context("history")
    context["logs"] = PredictionLog.objects.all()[:100]
    context["total"] = PredictionLog.objects.count()
    return render(request, "predictor/history.html", context)


@require_http_methods(["POST"])
def clear_history(request):
    PredictionLog.objects.all().delete()
    return redirect("history")
