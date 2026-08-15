# Attack-family / severity dashboard

An **additive, read-only** analysis layer over results that already exist. It
reuses the per-flow predictions and their PCAP/dataset **ground-truth labels**
from `validation/results/` and rolls them up by attack *family* and *severity*.
It never retrains, never calls the model, and never regenerates labels or
predictions — and it does not touch `ml.py`, `live_capture.py`,
`pcap_validation.py`, or Dataset Testing.

## Run it

```bash
# from webapp_django/  (uses validation/results/flows.csv + summary.json by default)
python manage.py attack_dashboard
# or explicitly:
python manage.py attack_dashboard \
    --flows ../validation/results/flows.csv \
    --summary ../validation/results/summary.json \
    --output ../validation/results/attack_dashboard
```

Prerequisite: the per-flow results must already exist (produced by
`validate_pcaps ... --output ../validation/results`). This command consumes them;
it never regenerates them.

## What it shows

- Overall **accuracy, macro precision, macro recall, macro F1**, correct/incorrect.
- **Confusion matrix** and **per-class** precision/recall/F1/support.
- **Attack-family** and **severity** grouping: flows/support, correct vs incorrect,
  and class-level recall per group.
- **Real-PCAP vs CIC-IDS2018** comparison (from the existing `cic_baseline`).
- **FTP-BruteForce real-PCAP failure** highlighted explicitly.
- **Confidence distribution** (histogram + mean/median, and mean when correct vs
  incorrect) when a `Confidence` column is present.
- A supplementary **all-15-class family/severity rollup** built from the existing
  cross-dataset per-class results.

Outputs under `validation/results/attack_dashboard/`: `dashboard.html`
(self-contained, no external assets), `dashboard.json`, `per_class.csv`,
`confusion_matrix.csv`, `family_summary.csv`, `family_metrics.csv`,
`severity_summary.csv`, `severity_metrics.csv`,
`class_family_severity_mapping.csv`, and the `cross_dataset_*` rollups.

## Mappings — explicit and documented

**Family** is **reused** from `predictor.classes` (the same 15-class → 6-family
table the app already uses); there is no second family table.

**Severity** has no existing source, so it is defined explicitly in
`predictor.attack_dashboard.SEVERITY`. It is assigned **per class from domain
knowledge, never inferred from model predictions**, and a flow's severity is
always taken from its **ground-truth label**. Ordering:
`Critical > High > Medium > Low > None`.

| Class | Family | Severity | Why |
|---|---|---|---|
| Benign | Benign | None | Normal traffic |
| DDOS attack-HOIC | DDoS | Critical | Distributed flood, highest volume |
| DDOS attack-LOIC-UDP | DDoS | Critical | Distributed flood |
| DDoS attacks-LOIC-HTTP | DDoS | Critical | Distributed flood |
| DoS attacks-GoldenEye | DoS | High | Single-source DoS |
| DoS attacks-Hulk | DoS | High | Single-source DoS |
| DoS attacks-SlowHTTPTest | DoS | High | Slow-connection DoS |
| DoS attacks-Slowloris | DoS | High | Slow-connection DoS |
| FTP-BruteForce | Brute force | High | Credential compromise |
| SSH-Bruteforce | Brute force | High | Credential compromise |
| Brute Force -Web | Web attack | High | Web brute-force probe |
| Brute Force -XSS | Web attack | High | Cross-site scripting probe |
| SQL Injection | Web attack | Critical | Direct data exfiltration / RCE |
| Bot | Recon / other | Medium | Bot beacon, lower immediate urgency |
| Infilteration | Recon / other | Critical | Realised internal compromise |

`Low` is reserved — no trained class maps there. Any class not in the table
resolves to `Unknown` (surfaced, never silently invented). The full mapping is
also written to `class_family_severity_mapping.csv` for audit.

## Headline finding (real-PCAP, existing results)

| | value |
|---|---|
| Flows | 42 (8 Benign, 34 FTP-BruteForce) |
| Accuracy / Macro-F1 | 0.190 / 0.160 |
| **Brute-force family recall (real PCAP)** | **0.000** |
| Brute-force family recall (CIC-shaped cross-dataset) | 0.943 |
| FTP-BruteForce confidence | model is **confidently wrong** — 20/34 flows at 0.9–1.0 |

The FTP-BruteForce failure is visible at family (Brute force) and severity (High)
level: every real high-severity brute-force flow is predicted Benign, at high
confidence — the train/serve artifact mismatch established by the SHAP and
cross-dataset analyses.
