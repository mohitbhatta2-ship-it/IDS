"""
Real-PCAP validation workflow.

    PCAP -> flow construction -> 30 features -> preprocessing -> model -> metrics

The whole point is to measure, honestly, how the *existing* saved model generalises
to realistically captured traffic. Nothing here retrains, tunes, thresholds, or
otherwise touches the model — it only feeds real captures through the exact same
path Live Capture and Dataset Testing already use.

Reuse, not reimplementation
---------------------------
* Flow construction is the Live Capture code verbatim: this module drives a
  :class:`predictor.live_capture.CaptureSession` with the packets read from the
  pcap (same ``sniff`` call, same bidirectional 5-tuple keying, same TCP
  FIN/RST finalisation, same idle timeout, same end-of-capture flush).
* The 30 features come from the single implementation, ``calculate_features``.
* Preprocessing + prediction + evaluation are the Dataset Testing function,
  ``ml.predict_batch`` — the pcap path literally converges into it, so parity is
  structural rather than asserted.

Ground truth comes only from the pcap's label (the attack session it was captured
from); the model's own prediction is never used as truth.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import live_capture, ml

# ---------------------------------------------------------------------------
# Label handling — folder / CLI name -> canonical model class (no guessing)
# ---------------------------------------------------------------------------

# Explicit aliases for the classes that can be reproduced safely on hardware we
# own. A "dos" folder on its own is intentionally NOT mapped: the model has four
# distinct DoS classes, so the caller must name the specific one.
_FOLDER_ALIASES = {
    "benign": "Benign",
    "ftp_bruteforce": "FTP-BruteForce", "ftp-bruteforce": "FTP-BruteForce", "ftp": "FTP-BruteForce",
    "ssh_bruteforce": "SSH-Bruteforce", "ssh-bruteforce": "SSH-Bruteforce", "ssh": "SSH-Bruteforce",
    "dos_hulk": "DoS attacks-Hulk", "hulk": "DoS attacks-Hulk",
    "dos_goldeneye": "DoS attacks-GoldenEye", "goldeneye": "DoS attacks-GoldenEye",
    "dos_slowloris": "DoS attacks-Slowloris", "slowloris": "DoS attacks-Slowloris",
    "dos_slowhttptest": "DoS attacks-SlowHTTPTest", "slowhttptest": "DoS attacks-SlowHTTPTest",
}


class ValidationError(ValueError):
    """Raised for an unusable label or an unreadable/empty capture."""


def resolve_label(name: str) -> str:
    """
    Map a folder/CLI name to a canonical model class, or raise a clear error.

    Resolution never guesses: it checks the explicit alias table, then an exact
    (normalised) match against the model's own class names, and otherwise fails
    loudly so a mislabelled capture cannot be scored as the wrong class.
    """
    raw = str(name).strip()
    norm = ml._normalise_label(raw)

    alias = {ml._normalise_label(k): v for k, v in _FOLDER_ALIASES.items()}
    if norm in alias:
        return alias[norm]

    canonical = {ml._normalise_label(v): v for v in ml.LABELS.values()}
    if norm in canonical:
        return canonical[norm]

    raise ValidationError(
        f"Cannot resolve ground-truth label {name!r} to a known class. "
        f"Use one of the aliases ({', '.join(sorted(_FOLDER_ALIASES))}) or an exact "
        f"class name ({', '.join(ml.LABELS.values())})."
    )


# ---------------------------------------------------------------------------
# PCAP -> flows: reuse the Live Capture session, collecting feature vectors
# ---------------------------------------------------------------------------


class _CollectingSession(live_capture.CaptureSession):
    """
    A CaptureSession whose only change is the sink: instead of pushing a compact
    record to the live-status buffer, it keeps the full 30-feature vector for
    every finalised flow. All flow construction and finalisation is inherited
    unchanged, so it is byte-for-byte the Live Capture behaviour.
    """

    def __init__(self, model_key: str):
        super().__init__(interface=None, model_key=model_key)
        self.collected: list[dict] = []

    def _classify(self, flow) -> None:  # overrides the live sink only
        from feature_calculator import calculate_features

        features = calculate_features(flow)
        self.collected.append(
            {
                "src_ip": flow.src_ip,
                "src_port": flow.src_port,
                "dst_ip": flow.dst_ip,
                "dst_port": flow.dst_port,
                "protocol": {6: "TCP", 17: "UDP"}.get(flow.protocol, str(flow.protocol)),
                "packets": flow.total_packets,
                "features": features,
            }
        )


def replay_pcap(path, model_key: str = ml.DEFAULT_MODEL) -> list[dict]:
    """
    Read a pcap and return one entry per finalised flow, each carrying its full
    30-feature vector. Uses the exact Live Capture flow engine via ``sniff``.
    """
    live_capture._ensure_live_on_path()
    from scapy.all import sniff

    path = Path(path)
    if not path.is_file():
        raise ValidationError(f"PCAP not found: {path}")

    session = _CollectingSession(model_key)
    try:
        sniff(offline=str(path), prn=session._handle, store=False)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller
        raise ValidationError(f"Could not read {path}: {exc}") from exc

    # Flows still open at end-of-capture are finalised, exactly as Stop does live.
    session._flush_all()
    return session.collected


# ---------------------------------------------------------------------------
# Feature validation — no silent zero-fill; explicit, finite, exactly 30
# ---------------------------------------------------------------------------


def _feature_problem(features: dict) -> str | None:
    """Return None if the vector is exactly the 30 finite FEATURES, else why not."""
    missing = [f for f in ml.FEATURES if f not in features]
    if missing:
        return f"missing features: {', '.join(missing)}"
    extra = [f for f in features if f not in ml.FEATURES]
    if extra:
        return f"unexpected features: {', '.join(map(str, extra))}"
    for f in ml.FEATURES:
        try:
            v = float(features[f])
        except (TypeError, ValueError):
            return f"non-numeric feature {f!r}={features[f]!r}"
        if not math.isfinite(v):
            return f"non-finite feature {f!r}={v!r}"
    return None


# ---------------------------------------------------------------------------
# One pcap -> prediction + metrics (converges into Dataset Testing)
# ---------------------------------------------------------------------------


@dataclass
class PcapResult:
    pcap: str
    label: str
    model_key: str
    total_flows: int = 0
    valid_flows: int = 0
    invalid_flows: list[dict] = field(default_factory=list)
    frame: pd.DataFrame | None = None      # per-flow: features + Label + Predicted Class + Confidence
    evaluation: dict | None = None         # from ml.predict_batch / _evaluate

    @property
    def correct(self) -> int:
        if self.frame is None or self.frame.empty:
            return 0
        return int((self.frame["Predicted Class"] == self.label).sum())

    @property
    def incorrect(self) -> int:
        return self.valid_flows - self.correct


def validate_pcap(path, label: str, model_key: str = ml.DEFAULT_MODEL) -> PcapResult:
    """
    Score one labelled pcap and return per-flow predictions + evaluation.

    Ground truth is ``label`` for every flow (the session the capture came from).
    Invalid/incomplete flows are reported, never silently zero-filled.
    """
    canonical = resolve_label(label)
    flows = replay_pcap(path, model_key)

    result = PcapResult(pcap=str(path), label=canonical, model_key=model_key,
                        total_flows=len(flows))

    valid_rows = []
    for fl in flows:
        problem = _feature_problem(fl["features"])
        if problem:
            result.invalid_flows.append({**{k: fl[k] for k in ("src_ip", "src_port", "dst_ip",
                                                                 "dst_port", "protocol")},
                                         "reason": problem})
            continue
        row = {f: float(fl["features"][f]) for f in ml.FEATURES}
        row.update(SrcIP=fl["src_ip"], SrcPort=fl["src_port"], DstIP=fl["dst_ip"],
                   DstPort=fl["dst_port"], Protocol=fl["protocol"], Packets=fl["packets"])
        valid_rows.append(row)

    result.valid_flows = len(valid_rows)
    if not valid_rows:
        return result

    # Converge into the Dataset Testing path: same preprocessing, model and
    # per-class evaluation. Ground truth rides along as the Label column.
    df = pd.DataFrame(valid_rows)
    df["Label"] = canonical
    batch = ml.predict_batch(df[ml.FEATURES + ["Label"]].copy(), model_key)

    # bad_rows must be 0 here -- we already excluded non-finite vectors, so this
    # asserts the "no silent zero-fill" guarantee held.
    if batch["bad_rows"]:
        raise ValidationError(
            f"{batch['bad_rows']} flow(s) reached preprocessing with non-numeric "
            "features despite validation; refusing to score silently."
        )

    scored = batch["frame"].reset_index(drop=True)
    meta = df[["SrcIP", "SrcPort", "DstIP", "DstPort", "Protocol", "Packets"]].reset_index(drop=True)
    result.frame = pd.concat([meta, scored], axis=1)
    result.evaluation = batch["evaluation"]
    return result


# ---------------------------------------------------------------------------
# Directory -> aggregate (sessions kept separate; ground truth per pcap)
# ---------------------------------------------------------------------------


def iter_labelled_pcaps(root):
    """
    Yield ``(pcap_path, canonical_label)`` for every capture under ``root``.

    Layout: ``root/<class-folder>/*.pcap``. The immediate folder name is the
    ground-truth label (resolved, never guessed). Sessions stay in separate
    files, so evaluation never mixes flows across captures for a split.
    """
    root = Path(root)
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name.startswith((".", "_")):
            continue  # ._demo etc. reserved for non-validation fixtures
        try:
            label = resolve_label(folder.name)
        except ValidationError:
            continue
        for pcap in sorted(list(folder.glob("*.pcap")) + list(folder.glob("*.pcapng"))):
            yield pcap, label


def validate_directory(root, model_key: str = ml.DEFAULT_MODEL) -> dict:
    """Validate every labelled pcap under ``root`` and aggregate the results."""
    per_pcap = []
    frames = []
    invalid_total = 0
    for pcap, label in iter_labelled_pcaps(root):
        res = validate_pcap(pcap, label, model_key)
        per_pcap.append(res)
        invalid_total += len(res.invalid_flows)
        if res.frame is not None and not res.frame.empty:
            frames.append(res.frame.assign(_pcap=str(pcap)))

    if not frames:
        return {"model_key": model_key, "per_pcap": per_pcap, "flows": 0, "metrics": None,
                "confusion": None, "invalid_flows": invalid_total, "combined": None}

    combined = pd.concat(frames, ignore_index=True)
    metrics = compute_metrics(combined["Label"], combined["Predicted Class"])
    confusion = confusion_table(combined["Label"], combined["Predicted Class"])
    return {
        "model_key": model_key,
        "per_pcap": per_pcap,
        "flows": len(combined),
        "metrics": metrics,
        "confusion": confusion,
        "invalid_flows": invalid_total,
        "combined": combined,
    }


# ---------------------------------------------------------------------------
# Metrics (reuse ml._evaluate; add confusion matrix + per-class counts)
# ---------------------------------------------------------------------------


def compute_metrics(truth, predicted) -> dict:
    """
    Full metric set for a set of scored flows. Delegates the scoring maths to the
    same ``ml._evaluate`` Dataset Testing uses, then adds per-class correct/
    incorrect counts.
    """
    truth = pd.Series(list(truth), dtype=str)
    predicted = np.array([str(p) for p in predicted])
    ev = ml._evaluate(truth, predicted) or {}

    labels = sorted(set(truth) | set(predicted))
    per_class = []
    truth_counts = truth.value_counts().to_dict()
    for c in labels:
        mask = (truth == c)
        support = int(mask.sum())
        correct = int((predicted[mask.to_numpy()] == c).sum()) if support else 0
        stats = next((r for r in ev.get("per_class", []) if r["label"] == c), None)
        per_class.append({
            "class": c,
            "ground_truth": support,
            "correct": correct,
            "incorrect": support - correct,
            "precision": stats["precision"] if stats else 0.0,
            "recall": stats["recall"] if stats else 0.0,
            "f1": stats["f1"] if stats else 0.0,
        })

    total = int(len(truth))
    correct = int((truth.to_numpy() == predicted).sum())
    return {
        "total_flows": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": ev.get("accuracy", correct / total if total else 0.0),
        "macro_precision": ev.get("macro_precision"),
        "macro_recall": ev.get("macro_recall"),
        "macro_f1": ev.get("macro_f1"),
        "weighted_f1": ev.get("weighted_f1"),
        "per_class": per_class,
    }


def confusion_table(truth, predicted) -> dict:
    """A JSON/CSV-friendly confusion matrix keyed by class name."""
    from sklearn.metrics import confusion_matrix

    truth = [str(t) for t in truth]
    predicted = [str(p) for p in predicted]
    labels = sorted(set(truth) | set(predicted))
    m = confusion_matrix(truth, predicted, labels=labels)
    return {"labels": labels, "matrix": m.tolist()}


# ---------------------------------------------------------------------------
# CIC vs real comparison + feature distributions
# ---------------------------------------------------------------------------


def cic_dataset_testing_metrics(model_key: str = ml.DEFAULT_MODEL, sample: int | None = 40000) -> dict | None:
    """
    Baseline: run the CIC held-out test set through the SAME Dataset Testing path.
    Returns None if the parquet is unavailable. Sampled for speed by default.
    """
    parquet = ml.DATA_ROOT / "Processed_Data" / "test_selected.parquet"
    if not parquet.is_file():
        return None
    df = pd.read_parquet(parquet)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=0)
    batch = ml.predict_batch(df.copy(), model_key)
    ev = batch["evaluation"] or {}
    return {"rows": int(batch["rows"]), "accuracy": ev.get("accuracy"),
            "macro_f1": ev.get("macro_f1"), "weighted_f1": ev.get("weighted_f1")}


# Features most worth comparing (the caller may add more; this is not exhaustive).
COMPARISON_FEATURES = [
    "Flow Duration", "Flow Pkts/s", "Fwd Pkts/s", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Mean", "Bwd Pkt Len Mean", "Pkt Len Mean", "Flow IAT Mean",
    "Fwd IAT Tot", "Init Fwd Win Byts", "Fwd Seg Size Min", "Fwd Header Len", "Dst Port",
]


def feature_distribution_comparison(combined: pd.DataFrame, features=None) -> pd.DataFrame:
    """
    For each validated class, compare real-pcap feature medians against the CIC
    training distribution. Shows, with actual numbers, where real traffic and the
    training data diverge. Returns an empty frame if the training parquet is absent.
    """
    features = features or COMPARISON_FEATURES
    parquet = ml.DATA_ROOT / "Processed_Data" / "balanced_train_selected.parquet"
    if not parquet.is_file() or combined is None or combined.empty:
        return pd.DataFrame()

    train = pd.read_parquet(parquet)
    enc2name = {int(k): v for k, v in ml.LABELS.items()}
    train_cls = train["Label"].map(enc2name)

    rows = []
    for cls in sorted(combined["Label"].unique()):
        real = combined[combined["Label"] == cls]
        cic = train[train_cls == cls]
        for feat in features:
            rows.append({
                "class": cls,
                "feature": feat,
                "real_median": float(pd.to_numeric(real[feat], errors="coerce").median()),
                "cic_median": (float(cic[feat].median()) if not cic.empty else float("nan")),
                "real_n": int(len(real)),
                "cic_n": int(len(cic)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Persisting results (CSV + JSON)
# ---------------------------------------------------------------------------


def save_results(summary: dict, out_dir) -> dict:
    """Write machine-readable outputs; return the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}

    combined = summary.get("combined")
    if combined is not None and not combined.empty:
        p = out / "flows.csv"
        combined.to_csv(p, index=False)
        written["flows_csv"] = str(p)

    if summary.get("metrics"):
        pc = pd.DataFrame(summary["metrics"]["per_class"])
        p = out / "per_class.csv"
        pc.to_csv(p, index=False)
        written["per_class_csv"] = str(p)

    if summary.get("confusion"):
        conf = summary["confusion"]
        cm = pd.DataFrame(conf["matrix"], index=conf["labels"], columns=conf["labels"])
        p = out / "confusion_matrix.csv"
        cm.to_csv(p)
        written["confusion_csv"] = str(p)

    dist = summary.get("distribution")
    if dist is not None and not dist.empty:
        p = out / "feature_distribution.csv"
        dist.to_csv(p, index=False)
        written["distribution_csv"] = str(p)

    # A single JSON with everything scalar/serialisable.
    payload = {
        "model_key": summary.get("model_key"),
        "flows": summary.get("flows"),
        "invalid_flows": summary.get("invalid_flows"),
        "metrics": summary.get("metrics"),
        "confusion": summary.get("confusion"),
        "cic_baseline": summary.get("cic_baseline"),
        "per_pcap": [
            {"pcap": r.pcap, "label": r.label, "total_flows": r.total_flows,
             "valid_flows": r.valid_flows, "invalid_flows": len(r.invalid_flows),
             "correct": r.correct, "incorrect": r.incorrect}
            for r in summary.get("per_pcap", [])
        ],
    }
    p = out / "summary.json"
    p.write_text(json.dumps(payload, indent=2, default=str))
    written["summary_json"] = str(p)
    return written
