# Experimental live FTP-BruteForce detector

**Experimental, local-test only. Nothing is retrained and no production file is modified in a
breaking way. The production models under `webapp_data/Results/Models/` are never written to;
`ml.py` and `live_capture.py` are unchanged. The detector loads a *validated* candidate from
`validation/models/…` and is offered as an explicit, clearly-labelled extra model in the Live
console.** Branch `claude/ftp-live-detector`.

## What it does

The production live page classifies each finished flow with the 30 packet-flow features. That
feature set cannot separate real benign FTP from FTP-BruteForce (they overlap on the CIC
packet/timing statistics — see `docs/ftp-behavioral-robustness.md`). This detector wires the
best **validated** FTP candidate — the per-connection + cross-session model
(`validation/models/ftp-per-connection/candidate_pc_unweighted.pkl`, 67 features) — into the
live pipeline so real FTP traffic off an interface is scored with the behavioural signal the
production model lacks.

Pipeline (exactly as required):

```
packet capture → flow building → FTP behavioural extraction → per-connection features
→ preprocessing → candidate model → prediction
```

- **Flow building & the 30 packet features** reuse the verified `Live/` core
  (`feature_calculator.calculate_features`) — the *same* implementation the production live
  page and `pcap_validation.replay_pcap` use. The 30 features and their order are preserved:
  they are the first 30 columns of the candidate's 67 (`FEATURES_PC[:30] == ml.FEATURES`).
- **Behavioural / per-connection / cross-session features** (the other 37 columns) are
  computed by the *frozen, validated* extractors `ftp_behavioral`, `ftp_per_connection`,
  `ftp_cross_session`, from the cleartext FTP control traffic actually captured for a
  client↔server session.
- **No zero-fill, no fabrication.** When the application-layer features are unavailable — a
  non-FTP flow, or an **encrypted FTPS** control channel — they are left **NaN**. The
  candidate is a `HistGradientBoosting` model that consumes NaN natively, so the flow is still
  scored from its packet features alone, and the record is flagged
  `behavioral_available: false` with a reason (`no-ftp-control` / `encrypted-ftps`) and the
  event is logged (`predictor.ftp_live` logger).
- **No heuristics, thresholds or hard-coded attack rules.** The label comes only from the
  model's `predict_proba`.

## Architecture (new, isolated code)

| File | Role |
|---|---|
| `predictor/ftp_live.py` | The whole detector: candidate loader (separate from `ml._cache`), FTP control-channel parsing, per-session packet buffering, the 67-feature builder (NaN-preserving), the prediction, and `FtpLiveCaptureSession` (subclass of `live_capture.CaptureSession`) + its `FtpCaptureManager`. |
| `predictor/views.py` | Backward-compatible routing only: the Live console can now select the experimental detector; `start/stop/status` pick the right manager. Production model selection is unchanged. |
| `templates/predictor/live.html`, `static/predictor/js/live.js` | Two extra columns (Behavioural availability, Session summary) and an FTP-BruteForce counter, shown only when the detector is active. |

`FtpLiveCaptureSession` subclasses the verified `CaptureSession` and overrides only three
things: `_run` (warm the *candidate*, not a production-registry key), `_handle` (also buffer
FTP control packets before the flow core runs), and `_classify` (score the 67-feature vector
with the candidate instead of `ml.predict_one`). The flow table, TCP-termination logic and
timeout sweep are inherited unchanged.

### Model selection is explicit and safe

- The detector's key is `ftp_pc_candidate`. It is **deliberately not** in `ml.MODEL_REGISTRY`,
  so no production prediction path or model file is touched.
- The model is loaded from the repo's `validation/models/…` artifact via a private cache in
  `ftp_live.py`; the production `ml._cache` is never involved.
- The Live console lists it as *“FTP-BruteForce behavioural detector (experimental)”* with a
  note that it never overwrites the production models.

## What the Live Monitor shows

For each classified flow: **timestamp**, **source** (ip:port), **destination** (ip:port),
protocol, packet count, **prediction** (FTP-BruteForce / Benign / other class), **confidence**,
**behavioural availability**, and a **session summary** (login attempts, failed logins,
control connections for the client↔server FTP session). The stat bar adds a live
**FTP-BruteForce** counter alongside Packets / Flows / Attacks.

## Detection unavailable under FTPS (by design)

FTPS encrypts the control channel, so USER/PASS and response codes are not on the wire and the
behavioural features cannot be computed. The detector recognises the FTPS handshake (implicit
port 990, or an explicit `AUTH TLS/SSL`), marks the session, and records the flow with
`behavioral_available: false`, `behavioral_reason: "encrypted-ftps"`, logging it. It still
emits a prediction from the packet features (honest: it does not pretend to have the
behavioural signal).

## Limitations (honest)

- The per-connection candidate is the best **validated** FTP candidate but did not strictly
  meet the ≥0.90 promotion target on the last fresh corpus (see
  `docs/ftp-per-connection-detector.md`); it is experimental, not promoted.
- Behavioural features need cleartext FTP; FTPS is out of reach (above).
- A single capture runs at a time across the production and FTP detectors (one shared
  console). With multiple gunicorn workers each worker has its own manager — the same
  single-host caveat the production live page already documents.

## Tests

`predictor/tests.py` adds (all pass): feature schema / order / no-registry-leak
(`FtpLiveFeatureSchemaTests`), candidate load + NaN-row prediction (`FtpLivePredictionTests`),
missing-feature handling for non-FTP and FTPS (`FtpLiveMissingFeatureTests`), and end-to-end
replay of corpus pcaps through the live session — attacker → FTP-BruteForce, benign → Benign
(`FtpLiveEndToEndTests`).
