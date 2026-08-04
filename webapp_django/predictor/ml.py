"""
Model loading and prediction for the NIDS web app.

Models, the scaler and the feature list all live in the project's data bundle
(``webapp_data/`` by default). Everything is loaded lazily on first use and then
cached for the life of the process — the three models total roughly 12 MB and
take a couple of seconds to unpickle, which is fine once but not per request.
"""

from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

# ---------------------------------------------------------------------------
# Locating the data bundle
# ---------------------------------------------------------------------------

_META_PATH = Path(__file__).resolve().parent / "feature_meta.json"


def _locate_data_root() -> Path:
    """Find the folder holding Processed_Data/ and Results/Models/."""
    configured = getattr(settings, "IDS_DATA_ROOT", None)
    if configured:
        return Path(configured)

    here = Path(__file__).resolve()
    for parent in here.parents:
        for name in ("webapp_data", "c_filesnew"):
            candidate = parent / name
            if (candidate / "Results" / "Models").is_dir():
                return candidate

    raise FileNotFoundError(
        "Could not locate the data bundle. Set IDS_DATA_ROOT in settings.py to "
        "the folder containing Processed_Data/ and Results/Models/. If you have "
        "just cloned the repo, the model files are in Git LFS -- run "
        "`git lfs install && git lfs pull`."
    )


DATA_ROOT = _locate_data_root()
MODELS_DIR = DATA_ROOT / "Results" / "Models"


# ---------------------------------------------------------------------------
# Feature metadata (generated from the test set — see build notes in the README)
# ---------------------------------------------------------------------------

with open(_META_PATH) as _f:
    _META = json.load(_f)

FEATURES: list[str] = _META["features"]
FEATURE_STATS: dict = _META["stats"]
PRESETS: dict = _META["presets"]
LABELS: dict[int, str] = {int(k): v for k, v in _META["labels"].items()}

BENIGN_LABEL = "Benign"

# Grouping only affects how the manual-entry form is laid out. Thirty numeric
# inputs in one flat list is unusable; these are the natural clusters.
FEATURE_GROUPS: list[tuple[str, str, list[str]]] = [
    (
        "Flow identity & duration",
        "Which port the flow targets and how long it lasted.",
        ["Dst Port", "Flow Duration", "Init Fwd Win Byts", "Fwd Header Len"],
    ),
    (
        "Flow rates",
        "Packets per second across the whole flow and in the forward direction.",
        ["Flow Pkts/s", "Fwd Pkts/s"],
    ),
    (
        "Inter-arrival times",
        "Gaps between packets, in microseconds. Bursty attacks look very different here.",
        [
            "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
            "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Max", "Fwd IAT Min",
        ],
    ),
    (
        "Forward direction",
        "Bytes and packet sizes travelling from source to destination.",
        [
            "TotLen Fwd Pkts", "Subflow Fwd Byts", "Fwd Pkt Len Mean",
            "Fwd Pkt Len Max", "Fwd Pkt Len Std", "Fwd Seg Size Avg",
            "Fwd Seg Size Min",
        ],
    ),
    (
        "Backward direction",
        "The same measures for traffic coming back.",
        [
            "TotLen Bwd Pkts", "Subflow Bwd Byts", "Bwd Pkt Len Mean",
            "Bwd Pkt Len Max", "Bwd Seg Size Avg",
        ],
    ),
    (
        "Packet size distribution",
        "Aggregate packet-length statistics over the whole flow.",
        ["Pkt Len Mean", "Pkt Len Max", "Pkt Len Std", "Pkt Len Var", "Pkt Size Avg"],
    ),
]

# Sanity: the groups must cover the feature list exactly, or the form silently
# drops inputs the model needs.
_grouped = [f for _, _, fs in FEATURE_GROUPS for f in fs]
assert sorted(_grouped) == sorted(FEATURES), (
    f"FEATURE_GROUPS does not match FEATURES: "
    f"missing={set(FEATURES) - set(_grouped)}, extra={set(_grouped) - set(FEATURES)}"
)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Headline scores come from Results/Tables/model_comparison.csv (notebook 04/06)
# and were independently reproduced by notebook 07's Large run.

MODEL_REGISTRY: dict[str, dict] = {
    "histgradientboosting": {
        "name": "HistGradientBoosting",
        "file": "HistGradientBoosting_Tuned.pkl",
        "scaled": False,
        "accuracy": 0.9803,
        "macro_f1": 0.8627,
        "blurb": "Best macro F1 — strongest on the rare attack classes.",
    },
    "xgboost": {
        "name": "XGBoost",
        "file": "XGBoost_Tuned.pkl",
        "scaled": False,
        "accuracy": 0.9862,
        "macro_f1": 0.8421,
        "blurb": "Best raw accuracy, slightly weaker on rare classes.",
    },
    "mlp": {
        "name": "MLP (neural network)",
        "file": "MLP_Tuned.pkl",
        "scaled": True,
        "accuracy": 0.9837,
        "macro_f1": 0.7490,
        "blurb": "Deep-learning baseline. Needs feature scaling.",
    },
}

DEFAULT_MODEL = "histgradientboosting"

_cache: dict[str, object] = {}
_lock = threading.Lock()


def available_models() -> list[dict]:
    """Registry entries whose .pkl is actually present on disk."""
    out = []
    for key, spec in MODEL_REGISTRY.items():
        if (MODELS_DIR / spec["file"]).is_file():
            out.append({"key": key, **spec})
    return out


def _load(key: str):
    """Load (and cache) a model plus its scaler, if it needs one."""
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {key}")

    with _lock:
        if key in _cache:
            return _cache[key]

        spec = MODEL_REGISTRY[key]
        path = MODELS_DIR / spec["file"]
        if not path.is_file():
            raise FileNotFoundError(
                f"{spec['name']} is not available: {path} is missing. "
                "The .pkl files are stored in Git LFS -- run `git lfs pull`."
            )

        model = joblib.load(path)
        scaler = None
        if spec["scaled"]:
            scaler_path = MODELS_DIR / "MLP_StandardScaler.pkl"
            if not scaler_path.is_file():
                raise FileNotFoundError(
                    f"{spec['name']} needs MLP_StandardScaler.pkl, which is missing."
                )
            scaler = joblib.load(scaler_path)

        _cache[key] = (model, scaler)
        return _cache[key]


def warm_up(key: str = DEFAULT_MODEL) -> None:
    """Pre-load a model so the first real request isn't the one that pays for it."""
    try:
        _load(key)
    except Exception:  # noqa: BLE001 - warm-up must never break startup
        pass


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _as_frame(values: dict) -> pd.DataFrame:
    """Build a one-row frame with the columns in the exact training order."""
    return pd.DataFrame([{f: float(values.get(f, 0.0)) for f in FEATURES}])[FEATURES]


def _prepare(frame: pd.DataFrame, scaler) -> pd.DataFrame:
    if scaler is None:
        return frame
    return pd.DataFrame(scaler.transform(frame), columns=FEATURES, index=frame.index)


def _label_for(model, index: int) -> str:
    """Map a position in predict_proba output back to a human-readable class."""
    return LABELS.get(int(model.classes_[index]), f"Class {model.classes_[index]}")


def predict_one(values: dict, model_key: str = DEFAULT_MODEL, top_n: int = 3) -> dict:
    """Classify a single flow described by the 30 features."""
    model, scaler = _load(model_key)

    frame = _prepare(_as_frame(values), scaler)
    proba = model.predict_proba(frame)[0]

    order = np.argsort(proba)[::-1]
    ranked = [
        {"label": _label_for(model, i), "confidence": float(proba[i])}
        for i in order
        if proba[i] > 0
    ]
    top = ranked[:top_n] if ranked else [{"label": _label_for(model, order[0]), "confidence": 0.0}]

    verdict = top[0]
    return {
        "model": MODEL_REGISTRY[model_key]["name"],
        "model_key": model_key,
        "label": verdict["label"],
        "confidence": verdict["confidence"],
        "is_attack": verdict["label"] != BENIGN_LABEL,
        "top": top,
        "all": ranked,
    }


class BatchError(ValueError):
    """Raised when an uploaded file cannot be scored."""


def read_upload(file_obj, filename: str) -> pd.DataFrame:
    """Read an uploaded CSV or Parquet file into a DataFrame."""
    name = (filename or "").lower()
    try:
        if name.endswith(".parquet"):
            return pd.read_parquet(file_obj)
        return pd.read_csv(file_obj)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a message
        raise BatchError(f"Could not read the file: {exc}") from exc


def predict_batch(df: pd.DataFrame, model_key: str = DEFAULT_MODEL) -> dict:
    """
    Classify every row of an uploaded dataset.

    A ``Label`` column is optional. When present it is treated as ground truth
    and the result carries real accuracy / macro-F1 / per-class scores, so the
    upload doubles as an evaluation run rather than just a prediction run.
    """
    if df.empty:
        raise BatchError("The file has no rows.")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise BatchError(
            f"{len(missing)} of the 30 required feature columns are missing: "
            + ", ".join(missing[:6])
            + ("…" if len(missing) > 6 else "")
        )

    model, scaler = _load(model_key)

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    bad_rows = int(X.isna().any(axis=1).sum())
    X = X.fillna(0.0)

    frame = _prepare(X, scaler)
    proba = model.predict_proba(frame)
    best = np.argmax(proba, axis=1)

    predicted = np.array([_label_for(model, i) for i in best])
    confidence = proba[np.arange(len(best)), best]

    out = df.copy()
    out["Predicted Class"] = predicted
    out["Confidence"] = confidence

    counts = (
        pd.Series(predicted).value_counts().rename_axis("label").reset_index(name="count")
    )
    total = len(predicted)
    top_count = int(counts["count"].max()) if len(counts) else 1

    # bar_width is relative to the largest class so the chart uses its full
    # width; share_pct is the true share and is what the label shows.
    distribution = [
        {
            "label": r.label,
            "count": int(r.count),
            "share": int(r.count) / total,
            "share_pct": 100.0 * int(r.count) / total,
            "bar_width": max(100.0 * int(r.count) / top_count, 0.6),
            "is_attack": r.label != BENIGN_LABEL,
        }
        for r in counts.itertuples()
    ]

    attack_count = int((predicted != BENIGN_LABEL).sum())

    result = {
        "rows": total,
        "bad_rows": bad_rows,
        "model": MODEL_REGISTRY[model_key]["name"],
        "model_key": model_key,
        "attack_count": attack_count,
        "benign_count": total - attack_count,
        "attack_share_pct": 100.0 * attack_count / total,
        "distribution": distribution,
        "frame": out,
        "evaluation": None,
    }

    if "Label" in df.columns:
        result["evaluation"] = _evaluate(df["Label"], predicted)

    return result


def _evaluate(truth: pd.Series, predicted: np.ndarray) -> dict | None:
    """Score predictions against a ground-truth Label column, if it is usable."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        classification_report,
    )

    # Labels may be encoded (0-14) or already spelled out.
    truth = truth.copy()
    if pd.api.types.is_numeric_dtype(truth):
        truth = truth.map(LABELS)
    truth = truth.astype(str)

    if truth.isna().all() or not set(truth) & set(LABELS.values()):
        return None

    classes = sorted(set(truth) | set(predicted))
    report = classification_report(
        truth, predicted, labels=classes, output_dict=True, zero_division=0
    )

    per_class = [
        {
            "label": c,
            "f1": report[c]["f1-score"],
            "precision": report[c]["precision"],
            "recall": report[c]["recall"],
            "support": int(report[c]["support"]),
        }
        for c in classes
        if c in report and report[c]["support"] > 0
    ]
    per_class.sort(key=lambda r: r["support"], reverse=True)

    return {
        "accuracy": accuracy_score(truth, predicted),
        "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
        "weighted_f1": f1_score(truth, predicted, average="weighted", zero_division=0),
        "macro_precision": precision_score(truth, predicted, average="macro", zero_division=0),
        "macro_recall": recall_score(truth, predicted, average="macro", zero_division=0),
        "correct": int((truth.to_numpy() == predicted).sum()),
        "per_class": per_class,
    }
