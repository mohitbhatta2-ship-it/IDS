# CICFlowMeter vs custom-extractor experiment

**Analysis only — the production model, `ml.py`, `live_capture.py`,
`pcap_validation.py`, and the frozen real-PCAP baseline are never touched, and
nothing is retrained or merged.** All artifacts live under
`validation/results/cfm_experiment/`.

## The question

The frozen baseline showed real FTP-BruteForce PCAPs are all predicted **Benign**
(real-PCAP FTP recall **0.000**) while CIC held-out is healthy (0.980 / 0.863).
That failure has two candidate causes:

- **(A) model generalisation failure** — the model learned CIC-specific artifacts
  and does not transfer to real traffic; or
- **(B) feature-extraction mismatch** — *our* custom `Live/feature_calculator.py`
  computes the 30 features differently from the CICFlowMeter tool the CIC authors
  used, so the model is fed out-of-distribution inputs.

This experiment separates the two by re-extracting the **exact same PCAPs** with
an **independent CICFlowMeter** implementation and scoring them with the
**unchanged production model**.

## Why "same feature names" does not prove "same features"

CSE-CIC-IDS2018's CSVs were produced by the **Java CICFlowMeter V3**. Our
`Live` engine reimplements those 30 feature *names*, but a shared name does not
guarantee a shared *definition* — packet-length can be header-inclusive or
payload-only, flows can be segmented on different idle/activity timeouts, and
units can differ. So a column called `Fwd Pkt Len Max` can mean two different
things in two tools. The only way to test cause (B) is to run a **different**
CICFlowMeter and see whether the model's verdict changes.

## The extractor used, stated honestly

The Java CICFlowMeter V3 is not present here and its native (jnetpcap)
dependencies are impractical to build in this sandbox. This experiment therefore
uses the **Python `cicflowmeter` port (hieulw, 0.2.0)** — a genuine, widely used
CICFlowMeter reimplementation, but **not byte-identical to the Java original**.
This is an explicit, documented substitution; any residual gap between the port
and the Java tool is a stated caveat, not a hidden assumption.

`cfm_extract.run_cicflowmeter()` drives the port's own `FlowSession` feature code
directly from a filterless `PcapReader` (scapy 2.7 broke the port's
`AsyncSniffer(session=…)` packet delivery; only that plumbing is replaced — the
feature computation is the port's). Reproduce a single capture with:

```bash
# from webapp_django/
python -c "from predictor import cfm_extract as c; print(c.run_cicflowmeter('../sample_data/real_pcap/ftp_bruteforce/ftp_02.pcap').shape)"
```

## Feature mapping (verified by definition, not just name)

`cfm_extract.CFM_TO_ML` maps all **30 `ml.FEATURES`** to the port's snake_case
columns; `map_to_ml_features()` selects them **in exact `ml.FEATURES` order**,
verifies every value is finite, and **drops** (never zero-fills) any flow with a
missing/non-finite feature, recording the reason. Verified definition notes
(`cfm_extract.DEFINITION_NOTES`):

| Aspect | Finding |
|---|---|
| Units | Flow Duration / IAT are **microseconds** in both CIC and the port (an ~18 s flow reads ~18.7e6). No conversion applied. |
| Packet length | The port's `Fwd/Bwd Pkt Len` and `TotLen` **include L3/L4 headers**; Live/CIC use the L4 payload. Real definition difference — reported, not corrected. |
| Flow segmentation | The port cuts flows on an activity/idle timeout and omits the short post-FIN residual flows the Live engine emits, so the **flow count differs** (22 cicflowmeter flows vs 42 custom). |
| `Fwd Seg Size Min`, `Init Fwd Win Byts` | The SHAP-critical features. The port does **not** move them toward CIC (see results). |

Features requested for the §9 comparison that are **not** among the 30
(`Total Fwd/Bwd Packets`, `Flow Bytes/s`, `Fwd/Bwd Pkt Len Min`) are carried as
descriptive columns where a source has them and reported as `n/a` where it does
not — never fabricated.

## The three paths

| Path | Data | Extractor | Model |
|---|---|---|---|
| **A** | CIC-IDS2018 `test_selected.parquet` | (pre-extracted, Java V3) | production, unchanged |
| **B** | real PCAPs | **custom Live** (frozen `validation/results/flows.csv`, read-only) | production, unchanged |
| **C** | **same** real PCAPs | **cicflowmeter port** | production, unchanged |

## Results

| Path | n | Accuracy | Macro-F1 | FTP recall | Benign recall |
|---|---|---|---|---|---|
| A — CIC held-out | 200,498 | 0.9803 | 0.8627 | — | — |
| B — real / custom | 42 | 0.1905 | 0.1600 | **0.000** | 1.000 |
| C — real / cicflowmeter | 22 | 0.1818 | 0.1538 | **0.000** | 1.000 |

**Every real flow, under both extractors, is predicted Benign.** cicflowmeter
produces fewer flows (22 vs 42 — different segmentation) but the verdict is
identical: 0 of 18 cicflowmeter FTP flows are caught.

### §8 — FTP recall, custom vs CICFlowMeter

Custom **0.000**, CICFlowMeter **0.000**. No improvement.

### §9 / §10 — does CICFlowMeter move real FTP toward CIC?

Median real-FTP feature values (`feature_distribution_comparison.csv`):

| Feature | CIC FTP | custom | cicflowmeter | closer to CIC |
|---|---|---|---|---|
| Flow Duration (µs) | 4 | 1 | 3,277,591 | custom |
| Flow Pkts/s | 476,883 | 1,000,000 | 3.97 | cicflowmeter |
| Fwd Seg Size Min | **40** | 20 | 20 | tie |
| Init Fwd Win Byts | **26,883** | 8,134 | 65,535 | custom |
| Fwd Pkt Len Mean | 0 | 2.2 | 60.4 | custom |
| TotLen Fwd Pkts | 0 | 3 | 363 | custom |

Across the **12** features comparable in all three sources, cicflowmeter is
closer to CIC on **1**, the custom extractor on **9**, tie on **2**. Switching
extractor does **not** pull real FTP into the CIC region — on the SHAP-critical
features it is a **tie** (`Fwd Seg Size Min` = 20 for both, vs CIC's 40) or
**further away** (`Init Fwd Win Byts` = 65,535, further from CIC's 26,883 than
the custom 8,134). The CIC FTP signature (4 µs flows, 0-payload packets,
~477k pkts/s) is a capture artifact of that dataset; **neither** extractor
reproduces it from real multi-second, low-rate FTP brute-force traffic.

## Verdict — cause (A), model generalisation

Because an **independent CICFlowMeter** yields the **same 0.000 FTP recall** and
does **not** move the features toward CIC, the failure is **not** our custom
extractor. It is **model generalisation failure**: the model depends on
CIC-specific artifacts that real traffic does not contain, regardless of which
CICFlowMeter computes the features. Per the protocol, because cicflowmeter did
**not** substantially improve detection, **no retraining is performed in this
experiment** — the earlier retraining experiment
(`docs/realistic-pcap-retraining.md`) already explored the data-side fix and is
the appropriate place for that work.

## Limitations

- Only **5 real PCAPs** (3 FTP, 2 benign): 42 custom flows, 22 cicflowmeter
  flows. Exploratory, not statistically strong.
- The extractor is the **Python port, not Java CICFlowMeter V3**. A byte-exact
  Java run could differ in detail — but it would have to overturn a **complete**
  0.000-recall failure *and* a feature-closeness result that points the other
  way, which the residual port-vs-Java gap is very unlikely to do.
- cicflowmeter's header-inclusive packet lengths and different segmentation are
  genuine definition differences (documented above), reported rather than
  patched, so the two extractors are compared as-is.

## Reproduce

```bash
# from webapp_django/
python manage.py cfm_experiment --output ../validation/results/cfm_experiment
python manage.py test predictor.tests.CicflowmeterExtractionTests predictor.tests.CfmExperimentTests
```

## Evidence files (`validation/results/cfm_experiment/`)

`path_metrics.csv`, `per_class_{A_cic,B_custom,C_cicflowmeter}.csv`,
`confusion_{A_cic,B_custom,C_cicflowmeter}.csv`, `prediction_distribution.csv`,
`ftp_recall_comparison.csv`, `feature_distribution_comparison.csv`,
`extraction_per_pcap.csv`, `extraction_validity.csv`,
`experiment_metadata.json` (mapping, definition notes, verdict, sources,
versions).
