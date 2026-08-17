# Transparent FTP integration into the production Live Capture

**Integration only — no retraining, no new model, no frozen-artifact change. The production
models under `webapp_data/Results/Models/` are never written to; `ml.py` is unchanged; the FTP
candidate artifact is unchanged. The only production-code change is a minimal, backward-
compatible `session_factory` hook on `live_capture.CaptureManager` (default behaviour
identical).** Branch `claude/ftp-production-integration` (from `claude/ftp-live-detector`).

## Goal

Make FTP detection completely transparent. The user selects a **normal production model** in
the existing Live console (dropdown unchanged); the system routes traffic internally:

| Traffic | Pipeline | Model |
|---|---|---|
| Non-FTP | existing 30 packet features | **selected production model** (unchanged) |
| Cleartext FTP | 30 packet + behavioural + per-connection + cross-session (67) | **validated FTP detector** (`candidate_pc_unweighted.pkl`) |
| FTPS / encrypted | existing 30 packet features (behavioural **unavailable, not fabricated**) | **selected production model** |

The FTP model is never shown in the UI and is never in `ml.MODEL_REGISTRY`. The user never
selects an FTP model.

## Architecture

```
Live packet capture
      → Flow building (verified Live/ core)
      → Protocol/session identification (FTP control? FTPS? non-FTP)
            ├── non-FTP        → selected production model  (ml.predict_one, 30 features)
            ├── cleartext FTP  → validated FTP detector      (67 features, no zero-fill)
            └── FTPS/encrypted → production packet path       (behavioural unavailable)
      → Unified prediction record
      → Existing Live Monitor UI
```

`predictor/ftp_live.UnifiedLiveCaptureSession` subclasses the verified
`live_capture.CaptureSession` and overrides only:

- `_run` — warm **both** the selected production model (via the parent) and the FTP candidate;
  if the candidate can't load, degrade gracefully to production-only routing.
- `_handle` — additionally buffer cleartext FTP control packets per client↔server session.
- `_classify` — **route** each finished flow (below) and score it with the appropriate model.

On import the module installs itself as `live_capture.manager.session_factory`, so the
existing manager, endpoints, dashboard and console transparently gain FTP specialisation with
no view/API/UI redesign. Installation also happens in `PredictorConfig.ready()`.

### Routing logic (exact)

For each finished flow (in `UnifiedLiveCaptureSession._classify`):

1. `session_key, is_ftps = _session_for_flow(flow)` — look up the flow's TCP stream in the
   FTP stream table built during capture.
2. **FTPS** (`is_ftps`, i.e. implicit port 990 or an observed `AUTH TLS/SSL`) →
   `ml.predict_one(feat30, self.model_key)` (production), record flagged
   `behavioral_available=false`, `behavioral_reason="encrypted-ftps"`; no behavioural feature
   is computed or filled.
3. **Cleartext FTP** (recognised control session with usable cleartext) → build the 67-feature
   row (30 packet, order preserved; 37 app features from the frozen `ftp_behavioral` /
   `ftp_per_connection` / `ftp_cross_session` extractors, NaN where undefined) and score with
   the FTP candidate’s `predict_proba`.
4. **Recognised FTP but no usable cleartext yet** (e.g. a data connection) → production path,
   no fabrication.
5. **Non-FTP** → `ml.predict_one(feat30, self.model_key)`, byte-for-byte the pre-integration
   behaviour.

The router only *chooses the pipeline*. It never decides “attack” — the label always comes
from a model prediction (production model, or the FTP model’s `predict_proba`). No heuristics,
thresholds, or hard-coded attack rules were added.

## What the user sees

The Live console is unchanged: the same production model dropdown, the same 7-column table
(Time, Source, Destination, Proto, Pkts, Prediction, Confidence). For FTP-relevant rows only,
a small muted sub-line under the prediction shows behavioural availability and the login
attempt/failure/connection summary — “one NIDS”, not two systems.

## Guarantees / integrity

- **No retraining**; no training data added; no test data used for training.
- **Production model artifacts unchanged**; **FTP candidate artifact unchanged**; validation
  PCAPs and validation evidence unchanged (verified: no `.pkl` / `.pcap` / `.parquet` /
  `validation/…` file modified).
- **`ml.py` unchanged.** The only production-code change is the additive `session_factory`
  hook in `live_capture.py` (default `CaptureSession` → existing behaviour preserved; existing
  live tests that patch `CaptureSession._run` / `live_capture.manager` still pass).
- **30 packet features and their order preserved** (`FEATURES_PC[:30] == ml.FEATURES`).
- **No zero-fill / no fabrication**: unavailable app features are NaN; the HGB candidate
  consumes NaN natively.
- **FTP model isolated**: internal id `ftp_pc_candidate`, never in `ml.MODEL_REGISTRY`, never
  in the dropdown.
- **Offline / batch / manual prediction paths unchanged.**

## Live-session state isolation (loopback degeneracy)

Cross-session FTP features aggregate a source's connections. The source window is keyed by
`(client_ip, server_ip)`, and individual connections are tracked by their **5-tuple stream
identity**. On loopback `client_ip == server_ip == 127.0.0.1`, so *every* local FTP connection
shares one source-pair identity — a finished brute-force burst and a later clean login collapse
into the same window, and the clean login could inherit the burst's stale attack context.

Fix (session/context level only — no model or heuristic change): a source window **resets after
`FTP_WINDOW_IDLE_TIMEOUT` (30 s) of FTP inactivity**, measured on the capture clock. When a new
FTP control packet arrives after the source has been idle longer than that, the previous
window's buffered context is discarded and a fresh window begins, so a new FTP session **cannot
inherit stale attack history** from an earlier, temporally-separate burst. Contemporaneous
connections (a genuine multi-connection brute force reconnecting quickly) stay within one window
and still aggregate, preserving legitimate cross-session detection. A `window_resets` counter is
exposed in the status snapshot.

Note: within a *single active burst* window, a login from the same source is still evaluated
with that source's context (on loopback this is the only case where a same-IP login sees the
attack) — that is correct, since the source is actively attacking. Isolation applies once the
source goes idle and the context becomes stale. A very slow multi-connection brute force spaced
wider than the idle timeout will fragment into separate windows; per-connection detection of
packed attacks is unaffected by windowing.

## Limitations (honest)

The FTP candidate is the best **validated** FTP candidate but did **not** strictly meet the
≥0.90 promotion target on the last fresh corpus (see `docs/ftp-per-connection-detector.md`);
the later auth-forensics experiment found the residual mistype-vs-single-session case is not
reliably separable from traffic features (`docs/ftp-auth-forensics.md`). This integration wires
that validated detector in transparently; it does not claim production readiness beyond that
evidence. FTPS behavioural detection is out of reach by design.

## Tests

`predictor/tests.py` (`FtpIntegration*Tests`): production registry/dropdown unchanged and
factory installed (A/F); cleartext FTP routed to the FTP detector, schema/order, candidate
load (B); attacker corpus → FTP-BruteForce (C); benign FTP → Benign (D); FTPS/non-FTP →
production packet path with no fabrication (E). Existing production `LiveViewTests` remain
green.
