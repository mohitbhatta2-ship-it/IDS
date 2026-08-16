# Targeted benign real-PCAP collection

**Data collection only — no retraining, no model change, no threshold change.**
The production model, Candidate 2, `ml.py`, `live_capture.py`,
`pcap_validation.py`, and all existing validation results are untouched. This
corpus lives under `validation/targeted_benign_pcaps/` on branch
`claude/targeted-benign-pcap-collection`.

## Why this corpus

`final_robustness` showed Candidate 2's 28.2% benign false-positive rate is
**structured**: real benign traffic enters the model's CIC-artifact-based FTP
region, and the FPs concentrate in **active-mode**, **command-heavy**,
**transfer**, and **reconnect/multi-session** benign sessions — behaviours thin or
absent in the training corpora. This collection gathers **real** traffic of
exactly those kinds, so a *future* leakage-controlled experiment has material that
covers the FP-prone benign region. **Nothing is trained or tuned here.**

## What was collected

**41 fresh benign PCAPs**, 215 flows, **0 incomplete / 0 zero-filled**, 41 distinct
scenarios. Every capture is a fresh real network interaction (no synthetic
packets, no duplicated PCAPs, no injected sleeps, no fabricated features), on the
controlled local lab only.

### Diversity

- **Scenarios (41):** active-mode download/listing/multi-command/transfer/append/
  size-mdtm; command-heavy (MKD/RMD, RNFR/RNTO, DELE, APPE, SIZE, MDTM, NLST,
  STAT, multi-command, command-workout); transfers (download/upload/append/delete/
  mixed, binary); reconnect and multi-session (repeated, varied).
- **Servers (3 real configs):** a hand-written **raw-socket FTP server** (18) — a
  genuinely different implementation — plus **pyftpdlib permissive** (13) and
  **pyftpdlib bandwidth-throttled** (10, real slow/segmented transfers).
- **Mode:** 31 passive + **10 active** (active mode was the highest-FP mode).
- **Clients:** python-ftplib (35), curl (3), wget (3).
- **Environments:** new loopback addresses `127.0.0.8/.9/.10` + host-local
  `192.0.2.2`; new control ports `2230/2240/2250` (independent used 2130/2140/2150).
- **Connection patterns:** single (35), reconnect (2), repeated/multi-session (4).

### Slow/segmented transfers are real, not sleeps

The throttled pyftpdlib config (`ThrottledDTPHandler`, 8 KB/s) produces genuinely
slower, differently-segmented transfers — a real server behaviour, not an injected
delay.

## Data separation & integrity

- **Disjoint from v1, v2, and the independent 36-PCAP test set** by content hash
  (`leakage_validation.json` → `all_pass: true`); no duplicates; labels from the
  scenario folder only; all 41 captures verified.
- Extraction through the **existing unchanged `pcap_validation` pipeline**: exactly
  the 30 `ml.FEATURES`, correct order, all finite, no zero-fill, every flow
  traceable to its PCAP.
- The frozen production model + Candidate 2 were not loaded for any fitting
  decision and are unchanged (verified in tests and by the collection guard).

> **This corpus is NOT the independent test set, and must not be used to tune
> anything.** It is future training/collection material. Any model decision must
> still be validated on a *fresh* independent test set, per the robustness verdict.

## Feature-distribution diagnostic (read-only)

`feature_distribution.csv` compares the new targeted benign flows against the
independent benign and the independent **FP-benign** flows on the FP-driving
features. Observations (descriptive, not a tuning signal):

| Feature | targeted benign | indep benign | indep FP-benign |
|---|---|---|---|
| Fwd Seg Size Min | 32 | 32 | 32 |
| Flow Pkts/s | 78 643 | 366 438 | 849 525 |
| Flow Duration (µs) | 234 | 105 | 1.9 |
| Dst Port | 46 876 | 43 126 | 2140 |

`Fwd Seg Size Min` is 32 across all real benign (the value that overlaps real FTP).
The independent **FP-benign** flows are the *tiny, control-only* flows (~1.9 µs,
~850k pkts/s) that most resemble the CIC FTP artifact. The targeted corpus adds
broad command/transfer/active benign coverage and includes short control flows,
but **whether it fully populates that exact micro-flow FP region is a question for
a future independent re-test — it is not claimed here.**

## Reproduce

```bash
# from webapp_django/ (tcpdump needs root; controlled local lab only)
python manage.py collect_targeted_benign
python3 ../validation/targeted_benign_pcaps/verify_pcaps.py
```

## Tests

`predictor/tests.py` adds framework invariants (controlled-local targets, ports
disjoint from independent, ≥30 specs covering active/command/transfer/reconnect),
live-capture tests (active-mode benign is real + controlled; download-delete works
on any server; captures are independent), and artifact tests (manifest schema,
count, benign-from-folders, no duplicates, disjoint from v1/v2/independent,
controlled-local destinations, all verified, exactly-30 finite ordered features,
production/Candidate untouched).

## Limitations

- Loopback / host-local lab only (single host; same-host traffic stays on `lo`, so
  no separate physical interface).
- Two server implementations (custom raw-socket + pyftpdlib configs); no
  third-party FTP daemon; no FTPS/TLS.
- Benign-only by design (targets the benign FP problem); FTP-BruteForce is
  unchanged from prior corpora.
- Collection + validation only — **no retraining, and no claim that this data will
  fix the FPs** until a future independent evaluation says so.
