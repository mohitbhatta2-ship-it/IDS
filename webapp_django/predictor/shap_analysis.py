"""
SHAP explanations for the existing saved model — read-only, no retraining.

Answers one question with evidence: why does the model classify real captured
FTP brute-force flows as Benign while classifying CIC-IDS2018 FTP-BruteForce
samples correctly? It does that by attributing each prediction to the 30
features with SHAP and comparing the attributions between the two populations.

The model (sklearn HistGradientBoostingClassifier) is supported by
``shap.TreeExplainer`` directly, so attributions are exact tree contributions in
the model's raw-margin (log-odds) space — additive with ``decision_function``.
Nothing here modifies, retrains, thresholds, or forces the model; SHAP only
reads it. ``shap`` is imported lazily so the web app never depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import ml, pcap_validation as pv

# The two classes this analysis contrasts.
BENIGN = "Benign"
FTP = "FTP-BruteForce"


def _require_shap():
    try:
        import shap  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "shap is not installed. It is an analysis-only dependency: "
            "`pip install shap`."
        ) from exc
    return shap


def _class_index(model, class_name: str) -> int:
    """Position of a class in the model's output / SHAP last axis."""
    name_to_encoded = {v: k for k, v in ml.LABELS.items()}
    encoded = name_to_encoded[class_name]
    return list(model.classes_).index(encoded)


# ---------------------------------------------------------------------------
# Data sources (reuse existing pipeline; never fabricate)
# ---------------------------------------------------------------------------


def cic_ftp_correct(model, sample: int = 300, split: str = "test") -> pd.DataFrame:
    """CIC FTP-BruteForce rows the model classifies correctly (in-distribution)."""
    fname = "test_selected.parquet" if split == "test" else "balanced_train_selected.parquet"
    parquet = ml.DATA_ROOT / "Processed_Data" / fname
    df = pd.read_parquet(parquet)
    enc = {v: k for k, v in ml.LABELS.items()}[FTP]
    ftp = df[df["Label"] == enc]
    X = ftp[ml.FEATURES]
    pred = model.predict(X)
    correct = X[pred == enc]
    if sample and len(correct) > sample:
        correct = correct.sample(sample, random_state=0)
    return correct.reset_index(drop=True)


def real_ftp_flows(model, only_predicted=BENIGN) -> pd.DataFrame:
    """
    Real FTP brute-force flows from the committed PCAPs, extracted by the exact
    live pipeline. By default returns only those the model calls Benign — the
    flows this analysis needs to explain.
    """
    root = ml.DATA_ROOT.parent  # repo root (…/IDS); webapp_data is DATA_ROOT
    pcap_dir = _find_real_ftp_dir()
    rows = []
    for pcap in sorted(pcap_dir.glob("*.pcap")) + sorted(pcap_dir.glob("*.pcapng")):
        for fl in pv.replay_pcap(pcap):
            problem = pv._feature_problem(fl["features"])
            if problem:
                continue
            rows.append({f: float(fl["features"][f]) for f in ml.FEATURES})
    X = pd.DataFrame(rows)[ml.FEATURES] if rows else pd.DataFrame(columns=ml.FEATURES)
    if not X.empty and only_predicted is not None:
        names = np.array([ml.LABELS.get(int(c), str(c)) for c in model.predict(X)])
        X = X[names == only_predicted]
    return X.reset_index(drop=True)


def _find_real_ftp_dir():
    from pathlib import Path
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "sample_data" / "real_pcap" / "ftp_bruteforce"
        if cand.is_dir():
            return cand
    raise pv.ValidationError("sample_data/real_pcap/ftp_bruteforce not found.")


# ---------------------------------------------------------------------------
# Explainer + attributions
# ---------------------------------------------------------------------------


@dataclass
class ShapResult:
    values: np.ndarray            # (n_samples, n_features, n_classes) raw-margin contributions
    base: np.ndarray              # (n_classes,) expected raw-margin per class
    features: list                # ml.FEATURES
    X: pd.DataFrame


def build_explainer(model):
    shap = _require_shap()
    return shap.TreeExplainer(model)


def explain(model, X: pd.DataFrame, explainer=None) -> ShapResult:
    """SHAP raw-margin contributions for X, shape (n, 30, 15)."""
    explainer = explainer or build_explainer(model)
    vals = np.asarray(explainer.shap_values(X))
    base = np.asarray(explainer.expected_value)
    return ShapResult(values=vals, base=base, features=list(ml.FEATURES), X=X.reset_index(drop=True))


def additivity_error(model, res: ShapResult) -> float:
    """
    Max |(base + sum(shap)) - decision_function| across samples/classes.
    A tiny value confirms the attributions are exact (a stability check).
    """
    margin = model.decision_function(res.X)
    margin = np.atleast_2d(margin)
    recon = res.base.reshape(1, -1) + res.values.sum(axis=1)
    return float(np.max(np.abs(recon - margin)))


# ---------------------------------------------------------------------------
# Global importance + per-class contribution comparison
# ---------------------------------------------------------------------------


def global_importance(res: ShapResult, class_name: str | None = None) -> pd.DataFrame:
    """
    Mean |SHAP| per feature. If ``class_name`` is given, importance toward that
    one class; otherwise averaged across all classes.
    """
    if class_name is None:
        imp = np.abs(res.values).mean(axis=(0, 2))
    else:
        # class index in the SHAP last axis follows model.classes_ order.
        from . import ml as _ml
        model, _ = _ml._load("histgradientboosting")
        ci = _class_index(model, class_name)
        imp = np.abs(res.values[:, :, ci]).mean(axis=0)
    return (pd.DataFrame({"feature": res.features, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def mean_contribution(res: ShapResult, class_name: str) -> pd.Series:
    """Mean signed SHAP contribution of each feature toward ``class_name``."""
    model, _ = ml._load("histgradientboosting")
    ci = _class_index(model, class_name)
    return pd.Series(res.values[:, :, ci].mean(axis=0), index=res.features)


def compare_cic_vs_real(model, cic_X: pd.DataFrame, real_X: pd.DataFrame,
                        toward: str = BENIGN) -> pd.DataFrame:
    """
    Per-feature mean SHAP contribution toward ``toward`` for CIC vs real flows,
    with the difference — the crux of "why real flows read Benign".
    """
    explainer = build_explainer(model)
    cic = explain(model, cic_X, explainer)
    real = explain(model, real_X, explainer)
    c = mean_contribution(cic, toward)
    r = mean_contribution(real, toward)
    out = pd.DataFrame({
        "feature": res_features(cic),
        f"cic_shap_to_{toward}": c.values,
        f"real_shap_to_{toward}": r.values,
    })
    out["difference_real_minus_cic"] = out[f"real_shap_to_{toward}"] - out[f"cic_shap_to_{toward}"]
    # Also carry the raw feature medians so the contribution is interpretable.
    out["cic_median"] = [float(cic_X[f].median()) for f in out["feature"]]
    out["real_median"] = [float(real_X[f].median()) for f in out["feature"]]
    return out.reindex(out["difference_real_minus_cic"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def res_features(res: ShapResult) -> list:
    return list(res.features)


def top_features_for_sample(res: ShapResult, i: int, class_name: str, k: int = 8) -> pd.DataFrame:
    """The k features contributing most (signed) toward ``class_name`` for sample i."""
    model, _ = ml._load("histgradientboosting")
    ci = _class_index(model, class_name)
    contrib = res.values[i, :, ci]
    df = pd.DataFrame({"feature": res.features, "value": res.X.iloc[i].values, "shap": contrib})
    return df.reindex(df["shap"].abs().sort_values(ascending=False).index).head(k).reset_index(drop=True)
