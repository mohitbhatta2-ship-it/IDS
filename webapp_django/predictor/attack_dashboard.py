"""
Attack-family / severity dashboard — additive analysis over EXISTING results.

Reads the validation outputs that are already on disk (per-flow predictions with
their PCAP/dataset ground-truth label) and rolls them up by attack *family* and
*severity*. It never retrains, never calls the model, never regenerates labels
or predictions, and never touches ml.py / live_capture.py / pcap_validation.py /
Dataset Testing.

Family mapping is REUSED from ``predictor.classes`` (the same 15-class → 6-family
table the app already uses). Severity has no existing source, so it is defined
here as an explicit, documented table (see ``SEVERITY`` / ``SEVERITY_RATIONALE``)
— it is assigned per class from domain knowledge, never inferred from model
predictions, and always taken from the ground-truth label of a flow.
"""

from __future__ import annotations

import pandas as pd

from . import classes, pcap_validation as pv

# ---------------------------------------------------------------------------
# Severity — explicit, documented, NOT inferred from predictions
# ---------------------------------------------------------------------------
# Ordered most-to-least urgent. "None" is reserved for Benign.

SEVERITY_ORDER = ["Critical", "High", "Medium", "Low", "None"]

# Every one of the 15 trained classes is assigned a severity explicitly. Ground
# truth (the flow's real label) decides a flow's severity; the model prediction
# never does. A class missing here resolves to "Unknown" (surfaced, not guessed).
SEVERITY: dict[str, str] = {
    "Benign": "None",

    # Distributed floods — highest volume / hardest to absorb.
    "DDOS attack-HOIC": "Critical",
    "DDOS attack-LOIC-UDP": "Critical",
    "DDoS attacks-LOIC-HTTP": "Critical",

    # Single-source denial of service.
    "DoS attacks-GoldenEye": "High",
    "DoS attacks-Hulk": "High",
    "DoS attacks-SlowHTTPTest": "High",
    "DoS attacks-Slowloris": "High",

    # Credential brute force — leads to account compromise.
    "FTP-BruteForce": "High",
    "SSH-Bruteforce": "High",

    # Web application attacks. SQL Injection can mean direct data exfiltration /
    # RCE, so it is rated Critical; the brute-force web probes are High.
    "Brute Force -Web": "High",
    "Brute Force -XSS": "High",
    "SQL Injection": "Critical",

    # Recon / stealthy. Infiltration is a realised internal compromise (Critical);
    # a bot beacon is lower urgency on its own (Medium).
    "Bot": "Medium",
    "Infilteration": "Critical",
}

SEVERITY_RATIONALE = {
    "Critical": "Immediate impact: distributed floods, SQL injection, or a realised "
                "internal compromise (Infiltration).",
    "High": "Service-affecting or credential-compromising: single-source DoS, SSH/FTP "
            "brute force, and web brute-force/XSS probes.",
    "Medium": "Suspicious but lower immediate urgency (e.g. bot beacon).",
    "Low": "Reserved; no trained class currently maps here.",
    "None": "Benign traffic — no action.",
}


def severity_of(class_name: str) -> str:
    """Severity for a class from the explicit table; 'Unknown' if unmapped."""
    return SEVERITY.get(class_name, "Unknown")


def family_of(class_name: str) -> str:
    """Reused from predictor.classes — no second family table."""
    return classes.family_of(class_name)


def family_display(class_name: str) -> str:
    return classes.family_name(class_name)


# ---------------------------------------------------------------------------
# Loading existing per-flow results (never regenerated here)
# ---------------------------------------------------------------------------


def load_flows(path) -> pd.DataFrame:
    """
    Read an existing per-flow results CSV (e.g. validation/results/flows.csv).

    Must contain a ground-truth ``Label`` and a ``Predicted Class``. Confidence
    is used when present. Nothing is fabricated: only these columns are read.
    """
    df = pd.read_csv(path)
    missing = [c for c in ("Label", "Predicted Class") if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s): {', '.join(missing)}")
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _group_summary(df: pd.DataFrame, key_fn) -> list[dict]:
    """
    Per-group detection summary keyed by the GROUND-TRUTH grouping (family or
    severity). Correctness is class-level (Predicted Class == Label); a
    'family/severity match' column shows predictions that at least land in the
    same group.
    """
    truth_group = df["Label"].map(key_fn)
    pred_group = df["Predicted Class"].map(key_fn)
    class_correct = df["Predicted Class"] == df["Label"]
    group_correct = pred_group == truth_group

    rows = []
    for g in sorted(truth_group.dropna().unique(), key=str):
        mask = truth_group == g
        support = int(mask.sum())
        cc = int(class_correct[mask].sum())
        gc = int(group_correct[mask].sum())
        rows.append({
            "group": g,
            "support": support,
            "correct_class": cc,
            "incorrect_class": support - cc,
            "class_recall": (cc / support) if support else 0.0,
            "same_group_predicted": gc,
            "group_recall": (gc / support) if support else 0.0,
        })
    return rows


def _grouped_metrics(df: pd.DataFrame, key_fn) -> dict:
    """Full precision/recall/F1 treating the family/severity as the target."""
    truth = df["Label"].map(key_fn)
    pred = df["Predicted Class"].map(key_fn)
    metrics = pv.compute_metrics(truth, pred)
    confusion = pv.confusion_table(truth, pred)
    return {"metrics": metrics, "confusion": confusion}


def _confidence_distribution(df: pd.DataFrame) -> dict | None:
    if "Confidence" not in df.columns:
        return None
    conf = pd.to_numeric(df["Confidence"], errors="coerce").dropna()
    if conf.empty:
        return None
    edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    labels = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    binned = pd.cut(conf, bins=edges, labels=labels, right=False)
    correct_mask = (df["Predicted Class"] == df["Label"]).reindex(conf.index)
    return {
        "overall_mean": float(conf.mean()),
        "overall_median": float(conf.median()),
        "mean_when_correct": float(conf[correct_mask].mean()) if correct_mask.any() else None,
        "mean_when_incorrect": float(conf[~correct_mask].mean()) if (~correct_mask).any() else None,
        "histogram": {str(k): int(v) for k, v in binned.value_counts().reindex(labels, fill_value=0).items()},
    }


def rollup_per_class(per_class: pd.DataFrame, key_fn) -> list[dict]:
    """
    Roll an EXISTING per-class metrics table (class, ground_truth, correct,
    incorrect) up to family or severity by summing counts. Reuses existing
    results — no predictions or labels are regenerated. Used for the
    cross-dataset view, which covers all 15 classes.
    """
    df = per_class.copy()
    df["group"] = df["class"].map(key_fn)
    rows = []
    for g, sub in df.groupby("group"):
        support = int(sub["ground_truth"].sum())
        correct = int(sub["correct"].sum())
        rows.append({
            "group": g,
            "classes": int(len(sub)),
            "support": support,
            "correct_class": correct,
            "incorrect_class": support - correct,
            "class_recall": (correct / support) if support else 0.0,
        })
    order = {**{k: i for i, k in enumerate(classes.FAMILY_ORDER)},
             **{k: i for i, k in enumerate(SEVERITY_ORDER)}}
    rows.sort(key=lambda r: order.get(r["group"], 99))
    return rows


def build_from_per_class(per_class: pd.DataFrame, source: str) -> dict:
    """A family/severity rollup dashboard from an existing per-class table."""
    return {
        "source": source,
        "classes": int(len(per_class)),
        "support": int(per_class["ground_truth"].sum()),
        "correct": int(per_class["correct"].sum()),
        "family": rollup_per_class(per_class, family_of),
        "severity": rollup_per_class(per_class, severity_of),
    }


def build_dashboard(flows: pd.DataFrame, cic_baseline: dict | None = None,
                    source: str = "real-pcap") -> dict:
    """Roll the existing per-flow results up into the dashboard structure."""
    overall = pv.compute_metrics(flows["Label"], flows["Predicted Class"])
    confusion = pv.confusion_table(flows["Label"], flows["Predicted Class"])

    dashboard = {
        "source": source,
        "flows": int(len(flows)),
        "overall": {
            "accuracy": overall["accuracy"],
            "macro_precision": overall["macro_precision"],
            "macro_recall": overall["macro_recall"],
            "macro_f1": overall["macro_f1"],
            "weighted_f1": overall["weighted_f1"],
            "correct": overall["correct"],
            "incorrect": overall["incorrect"],
        },
        "per_class": overall["per_class"],
        "confusion": confusion,

        "family": {
            "grouped_metrics": _grouped_metrics(flows, family_of),
            "summary": _group_summary(flows, family_of),
        },
        "severity": {
            "grouped_metrics": _grouped_metrics(flows, severity_of),
            "summary": _group_summary(flows, severity_of),
        },
        "confidence": _confidence_distribution(flows),
    }

    # Real-PCAP vs CIC comparison where the baseline is available.
    if cic_baseline:
        dashboard["comparison"] = {
            "cic": {"accuracy": cic_baseline.get("accuracy"),
                    "macro_f1": cic_baseline.get("macro_f1"),
                    "rows": cic_baseline.get("rows")},
            "real": {"accuracy": overall["accuracy"], "macro_f1": overall["macro_f1"],
                     "flows": int(len(flows))},
        }

    # Make the headline FTP-BruteForce failure explicit and easy to find.
    ftp = next((r for r in overall["per_class"] if r["class"] == "FTP-BruteForce"), None)
    if ftp:
        dashboard["highlight_ftp_bruteforce"] = {
            "family": family_display("FTP-BruteForce"),
            "severity": severity_of("FTP-BruteForce"),
            "support": ftp["ground_truth"],
            "correct": ftp["correct"],
            "recall": ftp["recall"],
            "note": "Real FTP brute-force flows are all predicted Benign (recall 0.0) "
                    "— the train/serve artifact mismatch, visible at family/severity level.",
        }
    return dashboard


# ---------------------------------------------------------------------------
# Persisting (CSV + JSON) and a self-contained HTML view
# ---------------------------------------------------------------------------


def save_dashboard(dashboard: dict, out_dir) -> dict:
    from pathlib import Path
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}

    pd.DataFrame(dashboard["per_class"]).to_csv(out / "per_class.csv", index=False)
    written["per_class_csv"] = str(out / "per_class.csv")

    pd.DataFrame(dashboard["family"]["summary"]).to_csv(out / "family_summary.csv", index=False)
    pd.DataFrame(dashboard["family"]["grouped_metrics"]["metrics"]["per_class"]).to_csv(
        out / "family_metrics.csv", index=False)
    pd.DataFrame(dashboard["severity"]["summary"]).to_csv(out / "severity_summary.csv", index=False)
    pd.DataFrame(dashboard["severity"]["grouped_metrics"]["metrics"]["per_class"]).to_csv(
        out / "severity_metrics.csv", index=False)

    conf = dashboard["confusion"]
    pd.DataFrame(conf["matrix"], index=conf["labels"], columns=conf["labels"]).to_csv(
        out / "confusion_matrix.csv")

    # The explicit family+severity mapping, so the ground-truth labelling is auditable.
    mapping = pd.DataFrame([
        {"class": c, "family": family_display(c), "family_key": family_of(c),
         "severity": severity_of(c)}
        for c in classes.CLASS_FAMILY
    ])
    mapping.to_csv(out / "class_family_severity_mapping.csv", index=False)
    written["mapping_csv"] = str(out / "class_family_severity_mapping.csv")

    (out / "dashboard.json").write_text(json.dumps(dashboard, indent=2, default=str))
    written["dashboard_json"] = str(out / "dashboard.json")

    html = render_html(dashboard)
    (out / "dashboard.html").write_text(html)
    written["dashboard_html"] = str(out / "dashboard.html")
    return written


def render_html(dashboard: dict) -> str:
    """A self-contained (no external assets) static HTML dashboard."""
    o = dashboard["overall"]

    def table(headers, rows):
        head = "".join(f"<th>{h}</th>" for h in headers)
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    per_class_rows = [
        [r["class"], family_display(r["class"]), severity_of(r["class"]), r["ground_truth"],
         r["correct"], r["incorrect"], f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}"]
        for r in dashboard["per_class"]
    ]
    fam_rows = [[r["group"], r["support"], r["correct_class"], r["incorrect_class"],
                 f"{r['class_recall']:.3f}"] for r in dashboard["family"]["summary"]]
    sev_rows = [[r["group"], r["support"], r["correct_class"], r["incorrect_class"],
                 f"{r['class_recall']:.3f}"] for r in dashboard["severity"]["summary"]]

    conf = dashboard["confusion"]
    conf_head = ["truth \\ pred"] + conf["labels"]
    conf_rows = [[lbl] + list(row) for lbl, row in zip(conf["labels"], conf["matrix"])]

    cmp_html = ""
    if dashboard.get("comparison"):
        c = dashboard["comparison"]
        cmp_html = ("<h2>Real-PCAP vs CIC-IDS2018</h2>" + table(
            ["Source", "Accuracy", "Macro-F1", "n"],
            [["CIC held-out", f"{c['cic']['accuracy']:.4f}", f"{c['cic']['macro_f1']:.4f}", c['cic']['rows']],
             ["Real capture", f"{c['real']['accuracy']:.4f}", f"{c['real']['macro_f1']:.4f}", c['real']['flows']]]))

    conf_dist_html = ""
    if dashboard.get("confidence"):
        cd = dashboard["confidence"]
        conf_dist_html = ("<h2>Confidence distribution</h2>"
                          f"<p>mean {cd['overall_mean']:.3f} · median {cd['overall_median']:.3f}</p>"
                          + table(["Bin", "Flows"], list(cd["histogram"].items())))

    ftp = dashboard.get("highlight_ftp_bruteforce")
    ftp_html = ""
    if ftp:
        ftp_html = (f"<div class='alert'><b>FTP-BruteForce (family: {ftp['family']}, "
                    f"severity: {ftp['severity']})</b> — support {ftp['support']}, correct "
                    f"{ftp['correct']}, recall {ftp['recall']:.3f}. {ftp['note']}</div>")

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Attack-family / severity dashboard</title>
<style>
 body{{font:14px system-ui,Segoe UI,sans-serif;margin:2rem;color:#111;background:#fafaf8}}
 h1{{font-size:1.4rem}} h2{{font-size:1.05rem;margin-top:1.6rem;border-bottom:1px solid #ddd;padding-bottom:.25rem}}
 table{{border-collapse:collapse;margin:.5rem 0;font-size:.85rem}}
 th,td{{border:1px solid #ddd;padding:.35rem .6rem;text-align:right}} th{{background:#f0f0ec;text-align:left}}
 td:first-child,th:first-child{{text-align:left}}
 .tiles{{display:flex;gap:1rem;flex-wrap:wrap;margin:.5rem 0}}
 .tile{{border:1px solid #ddd;border-radius:8px;padding:.7rem 1rem;background:#fff;min-width:120px}}
 .tile b{{display:block;font-size:1.5rem}} .tile span{{color:#666;font-size:.75rem;text-transform:uppercase}}
 .alert{{background:#fdecea;border:1px solid #e34948;border-radius:8px;padding:.7rem 1rem;margin:1rem 0}}
 .muted{{color:#666;font-size:.8rem}}
</style></head><body>
<h1>Attack-family / severity dashboard <span class=muted>({dashboard['source']}, {dashboard['flows']} flows)</span></h1>
{ftp_html}
<div class=tiles>
 <div class=tile><b>{o['accuracy']:.3f}</b><span>Accuracy</span></div>
 <div class=tile><b>{o['macro_precision']:.3f}</b><span>Macro Precision</span></div>
 <div class=tile><b>{o['macro_recall']:.3f}</b><span>Macro Recall</span></div>
 <div class=tile><b>{o['macro_f1']:.3f}</b><span>Macro F1</span></div>
 <div class=tile><b>{o['correct']}/{o['correct']+o['incorrect']}</b><span>Correct</span></div>
</div>
{cmp_html}
<h2>Confusion matrix</h2>{table(conf_head, conf_rows)}
<h2>Per-class</h2>{table(["Class","Family","Severity","Support","Correct","Incorrect","Precision","Recall","F1"], per_class_rows)}
<h2>By attack family (ground truth)</h2>{table(["Family","Support","Correct","Incorrect","Class recall"], fam_rows)}
<h2>By severity (ground truth)</h2>{table(["Severity","Support","Correct","Incorrect","Class recall"], sev_rows)}
{conf_dist_html}
<p class=muted>Family mapping reused from predictor.classes; severity from the explicit,
documented table in predictor.attack_dashboard. Ground truth comes only from the PCAP/dataset
labels in the existing results — no labels or predictions were regenerated.</p>
</body></html>"""
