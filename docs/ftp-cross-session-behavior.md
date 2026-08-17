# Cross-session / source-level behaviour — resolving the benign-failed-login ambiguity

**Candidates only; production frozen. The production model, the current 45-feature
candidate, the C3 controlled candidate, and `ml.py` / `live_capture.py` /
`pcap_validation.py` / `ftp_behavioral.py` are byte-for-byte unchanged (verified
before == after). The 30 packet + 15 behavioural features are PRESERVED; cross-session
features are appended. Selection used CIC held-out + capture-level LOCO only; the frozen
vsFTPD corpus was evaluated exactly once. No promotion, no merge.** Branch
`claude/ftp-cross-session-behavior`.

## The question

Every single-session experiment hit the same wall: inside one session, "a legitimate user
failed a few times and left" is near-identical to "a short brute force". The controlled
`ftp_failed_logins` ablation confirmed feature *removal* cannot fix it without sacrificing
FTP detection. This experiment asks whether **cross-session / source-level** behaviour —
what a source does across *many* sessions over time — resolves the ambiguity, without
giving up the ~1.0 FTP recall.

## What was built

**13 label-free cross-session features** (`ftp_cross_session.py`), aggregated across the
FTP control connections (sessions) in a source's capture window: sessions-per-source,
failed attempts / failure-rate across sessions, unique usernames/passwords, credential
variation, mean time between sessions, per-session success/failure history, session rate,
source persistence, max fails in a session. Genuinely undefined values (e.g. time-between-
sessions with < 2 sessions) are **NaN**, never zero-filled. Feature set = 30 + 15 + 13 = **58**.

**A new training corpus** (`validation/cross_session_pcaps`): 28 fresh PCAPs (14 benign,
14 attacker), 328 flows, 0 incomplete, each capture a *source window with multiple
sessions* — benign sources make 1–4 sessions (succeed or give up once); attacker sources
persist across 6–12 sessions with sustained failure and credential sweeps. Loopback lab,
pyftpdlib + custom raw-socket servers (**vsftpd held out**), NEW addresses (127.0.0.30-32)
/ ports (2630/2640), hash-disjoint from every prior corpus **and** the frozen vsFTPD test.

Candidate trained on CIC + v1 + v2 + targeted + robust_train + benign_failed_login +
cross_session; selected (`candidate_cross_unweighted`) via CIC held-out + LOCO (macro-F1
0.9905). The vsFTPD test was **not** used for selection.

## FINAL vsFTPD test (cleartext, evaluated once)

| Model | FTP recall | Benign recall | FPR | FTP precision | macro-F1 | accuracy |
|---|---|---|---|---|---|---|
| production (30) | 0.000 | 0.984 | 0.016 | 0.000 | 0.209 | 0.458 |
| **C1 current (45)** | 1.000 | 0.852 | 0.148 | 0.886 | 0.930 | 0.931 |
| **C3 controlled (43)** | 1.000 | 0.869 | 0.131 | 0.897 | 0.938 | 0.938 |
| **cross-session (58)** | **1.000** | **0.934** | **0.066** | **0.946** | **0.969** | — |

Bootstrap 95% CIs (cross): FTP recall 1.000 [1.000, 1.000], benign recall 0.931 [0.761,
1.000], FPR 0.069 [0.000, 0.239]. The point estimate meets every target; the benign CI is
wide on this small corpus (lower bound 0.761).

## The critical test — does cross-session information resolve the ambiguity?

**Yes, substantially — and without sacrificing attack detection.**

| Family | C1 (45) | cross (58) |
|---|---|---|
| `mistype` (benign fails then succeeds) | 0.667 | **1.000** |
| `gave_up` (benign fails then leaves) | 0.000 | **0.200** |
| `eventual_success` (attacker succeeds) | 1.000 | **1.000** |
| every brute-force family | 1.000 | **1.000** |

Per **sessions-per-source** bucket (the source-level view):

| sessions/source | label | recall |
|---|---|---|
| 1 | Benign | **1.000** |
| 2–3 | Benign | 0.871 |
| 2–3 | FTP-BruteForce | **1.000** |
| 4–6 | FTP-BruteForce | **1.000** |
| 7+ | FTP-BruteForce | **1.000** |

Single-session benign is now perfectly classified, and — crucially — **attackers are still
caught at every session count** (2–3, 4–6, 7+ all 1.000). The feared failure mode ("few
sessions ⇒ benign lets low-session attackers through") did **not** occur. The residual
benign errors sit in the 2–3-session bucket (the pure `gave_up` reconnect case), which is
still genuinely hard: a give-up with no success looks like a short brute force even
across sessions.

## Why it works — the cross-session block carries the signal

**Feature ablation (same training data, cross block removed):**

| | FTP recall | Benign recall | FPR |
|---|---|---|---|
| with cross (58) | **1.000** | 0.934 | 0.066 |
| without cross (45, same data) | **0.600** | 0.918 | 0.082 |

Removing the cross-session block from the *same* training data collapses FTP recall to
0.600 — the cross-session features are exactly what let the model keep FTP recall at 1.000
while improving the benign side. **SHAP** confirms the mechanism: the top cross-session
feature is `ftpx_failure_rate_across_sessions` (4th overall; cross-block share 7.3%), i.e.
the model keys on the *rate of failure across a source's sessions*, not the raw count —
CIC artifacts still lead (44%), behavioural 4%, but the cross block is the marginal signal
that fixes the benign FPs. It is not a single-feature shortcut.

## Verdict — PROMISING — TARGET MET

`final_verdict.json`: `verdict = "PROMISING -- TARGET MET"`, `target_met = true`,
`ftp_recall_preserved = true`, `cic_no_regression = true`, `promote = false`.

On the frozen independent vsFTPD test the cross-session candidate meets all promotion
targets — FTP recall 1.000 (≥ 0.90), benign recall 0.934 (≥ 0.90), FPR 0.066 (≤ 0.10),
no CIC regression (accuracy 0.9803 = production) — **without sacrificing attack detection**
(FTP recall unchanged at 1.000; attackers caught at every session count). This is the
first candidate to reach the target. Promotion remains a **human decision** — see caveats.

### Caveats (read the success honestly)
- **Small test corpus / wide CI.** Benign recall point estimate 0.934 meets target but the
  95% CI lower bound is 0.761; more independent captures are needed to tighten it.
- **The `gave_up` case is improved (0.0 → 0.2), not solved.** A benign user who fails and
  leaves, with no successful session, is still near-indistinguishable from a short brute
  force — cross-session helps most when there is *eventual success* history to observe.
- **Session-count distribution shift.** The training attackers use one credential per
  connection (many sessions); the vsFTPD test attackers pack attempts per connection
  (fewer sessions). The candidate generalised across this shift here, but it is a real
  train/test difference to keep testing.
- Loopback / single-container lab; `cross_session` corpus is one source per capture.
- The frozen vsFTPD corpus was evaluated once, for reporting; never training/selection/tuning.
- No promotion, no threshold/heuristic change, no merge.

### Recommended next steps (before any promotion)
1. More independent, multi-server cross-session traffic to tighten the benign CI and stress
   the session-count shift.
2. Genuinely multi-session *sources over longer time windows* (the current lab compresses a
   "source window" into one capture); real deployments would aggregate a source over hours.
3. A decision on the residual `gave_up` boundary case (accept as suspicious, or add
   source-reputation history across days).

## Leakage / integrity (all verified)

Training/test source separation (loopback 127.0.0.30-32 vs the vsFTPD 10.77.0.x test);
content-hash disjointness from all eight prior corpora and the frozen test; no independent-
test session in training; labels from folders only; no zero-fill (undefined cross features
kept NaN); exact feature ordering preserved (30 packet, then 15 behavioural, then 13
cross); every flow traceable to its PCAP; production and all frozen models/PCAPs unchanged
(before == after).

## Reproduce

```bash
# from webapp_django/ (root needed for capture)
python manage.py collect_cross_session_pcaps        # multi-session-per-source TRAIN corpus
python manage.py ftp_cross_session_experiment --loco-caps 50
python manage.py test predictor.tests.FtpCrossSessionFeatureTests \
                      predictor.tests.CrossSessionCorpusTests \
                      predictor.tests.FtpCrossSessionResultsTests
```

## Files (`validation/results/ftp_cross_session/`)

`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`per_scenario_family_metrics.csv`, `per_source_session_metrics.csv`,
`per_scenario_metrics.csv`, `per_client_metrics.csv`, `per_server_metrics.csv`,
`per_environment_metrics.csv`, `bootstrap_cis.json`, `shap_top_features.csv`,
`ftps_encrypted_metrics.csv`, `confusion_*.csv`, `leakage_validation.json`,
`model_hashes_before_after.json`, `final_verdict.json`. Candidate models under
`validation/models/ftp-cross-session/`.
