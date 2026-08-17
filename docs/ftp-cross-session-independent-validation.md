# Final independent validation of the cross-session FTP detector

**All models frozen. The production model, Candidate 2, the 45-feature candidate, C3, the
cross-session candidate, and `ml.py` / `live_capture.py` / `pcap_validation.py` /
`ftp_behavioral.py` / `ftp_cross_session.py` are byte-for-byte unchanged (verified
before == after). This second independent corpus is TEST-ONLY — never training,
selection, tuning, or thresholding. No promotion, no merge.** Branch
`claude/ftp-cross-session-independent-validation`.

## Why this test

The cross-session candidate met the promotion target on the *first* vsFTPD independent
test (FTP 1.000 / benign 0.934 / FPR 0.066). That test's caveat flagged a train/test
**session-count distribution shift**: its attackers made many sessions, matching the
cross-session training. This experiment builds a *second*, genuinely-different corpus that
deliberately includes **single-session attacks** to find out whether the cross-session
benefit is real or an artifact of session-count matching.

## What makes this corpus genuinely different

| Axis | First independent test | This (second) test |
|---|---|---|
| Server | vsFTPd | **pure-ftpd** (new; escalating anti-bruteforce delay) |
| Client | lftp | **ncftp** (new) |
| Network | netns, 10.77.0.x, ports 2121/2131 | **netns `ivlab2`, 10.88.0.x, ports 2222/2323** |
| Scenarios | `independent_ftp_lab` | **fresh `independent_ftp_lab2`, single- vs multi-session structure** |

**25 fresh PCAPs (14 benign, 11 attacker; 3 FTPS), 98 flows, 0 incomplete**, hash-disjoint
from all nine prior corpora. Includes **single-session attacks** (all attempts in one
connection) and multi-session attacks (6–8 reconnects), plus benign single- and
multi-session use.

## Results — the cross-session improvement did NOT hold

### FINAL cleartext evaluation (evaluated once)

| Model | FTP recall | Benign recall | FPR | FTP precision | macro-F1 |
|---|---|---|---|---|---|
| production (30) | 0.000 | 0.978 | 0.022 | 0.000 | 0.217 |
| Candidate 2 (30) | 0.000 | 0.756 | 0.244 | 0.000 | 0.272 |
| C1 current (45) | 0.674 | 0.978 | 0.022 | 0.969 | 0.821 |
| **C3 controlled (43)** | **0.891** | 0.978 | 0.022 | 0.976 | 0.934 |
| **cross-session (58)** | **0.804** | **1.000** | **0.000** | 1.000 | 0.900 |

Bootstrap 95% CIs (cross): FTP recall 0.789 [0.500, 0.978], benign recall 1.000 [1.000,
1.000], FPR 0.000 [0.000, 0.000]. CIC held-out: no regression (accuracy 0.9803).

The cross-session candidate has a **perfect benign side** (recall 1.000, FPR 0.000) but
its **FTP recall falls to 0.804 — below the 0.90 target**, and *below C3's 0.891*. On this
genuinely-independent corpus the cross-session block does **not** improve robust detection;
it trades attack recall for benign precision.

### The failure mode — over-reliance on session count

Cross-session candidate attack recall, broken down by how many sessions the source made:

| sessions per source | attacker flows | cross recall |
|---|---|---|
| 1 (single-session) | 2 | **0.000** |
| 2–3 | 3 | **0.000** |
| 4–6 | 33 | 0.879 |
| 7+ | 8 | **1.000** |

The model catches high-session attackers (7+ = 1.000) but **misses low-session attacks**
(1–3 sessions = 0.000). It learned "few sessions ⇒ benign" from a training corpus where
attackers always made many sessions — so a genuine **single-session brute force** (many
attempts packed into one connection) is classified benign. This is exactly the caveat the
first test flagged, now confirmed on independent data.

(Single-session low-volume attacks are hard for C3 too — both miss the 2 literal
single-connection captures — but the cross-session model **additionally** misses the 2–3
session attacks that C3 catches, which is why cross 0.804 < C3 0.891.)

### SHAP — not a single-feature shortcut, but the cross block hurts here

Largest single feature `Dst Port` at 0.179 (no single-feature shortcut); cross-session
block share 7.1% (top: `ftpx_failure_rate_across_sessions`); CIC artifacts dominate. The
cross block genuinely contributes — but on this corpus its session-count signal *reduces*
attack recall on low-session attacks.

### FTPS / TLS
Every model: FTP recall 0.000, benign recall 1.000 — the behavioural + cross-session
features are unavailable under encryption, so cleartext-FTP detection does not extend to
FTPS (expected; documented, not hidden).

## Verdict — NOT EFFECTIVE

`final_verdict.json`: `verdict = "NOT EFFECTIVE"`, `promote = false`. On the second
genuinely-independent corpus the cross-session candidate does **not** meet the promotion
target (FTP recall 0.804 < 0.90), because it sacrifices attack detection on
single/low-session attacks. The improvement seen on the first vsFTPD test **did not
generalise** — it depended on the attacker session-count structure matching the
cross-session training corpus.

### What this test proves and does not prove
- **Proves:** the cross-session detector's benefit is **not robust** to session-count
  structure. On a different server/client/network with single-session attacks it misses
  them; the session-per-source signal is a train-corpus artifact, not a general attack
  discriminator (an attacker can pack a brute force into one connection).
- **Does not prove** anything about a different OS/host or the public internet; and note
  that pure-ftpd's failure delay *bounds* single-session brute force here, so a real
  attacker could sustain longer single-session attacks than we could collect — likely
  making the cross-session model's blind spot *worse*, not better.

### Recommended next experiment
1. **Do not promote the cross-session candidate.** Session-per-source count is not a
   robust attack feature. The most robust candidate on this corpus was **C3** (FTP 0.891,
   benign 0.978) — the controlled feature set that kept failure-*context* features without
   the raw failed-login count.
2. If cross-session signal is pursued, it must be combined with **per-connection** attack
   evidence (attempt rate *within* a session, not just session count) so single-session
   brute force is still caught, and trained on a corpus where attackers span the full
   single→multi-session spectrum.
3. Re-run this independent validation after any such change — the single-session attack
   family is the decisive test.

## Leakage / integrity (all verified)

New corpus hash-disjoint from all nine prior corpora; source/session separation
(pure-ftpd 10.88.0.x vs the first test's vsftpd 10.77.0.x and loopback training); labels
from folders only; no zero-fill (undefined cross features NaN); exact feature ordering
preserved (30 packet, 15 behavioural, 13 cross); every flow traceable to its PCAP;
production and all frozen models + prior corpora unchanged (before == after); the corpus
was evaluated once, for reporting only.

## Reproduce

```bash
# from webapp_django/ (root needed for capture)
python manage.py collect_independent_ftp_validation2      # pure-ftpd/ncftp/10.88.0.x corpus
python manage.py ftp_cross_session_final_validation       # evaluate 5 frozen models, SHAP, ablation
python manage.py test predictor.tests.IndependentFtpLab2Tests \
                      predictor.tests.IndependentFtp2CorpusTests \
                      predictor.tests.CrossSessionIndependentValidationResultsTests
```

## Files (`validation/results/ftp_cross_session_independent_validation/`)

`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `per_scenario_family_metrics.csv`,
`per_session_structure_metrics.csv`, `per_scenario_metrics.csv`, `per_client_metrics.csv`,
`bootstrap_cis.json`, `shap_top_features.csv`, `ftps_encrypted_metrics.csv`,
`confusion_*.csv`, `leakage_validation.json`, `model_hashes_before_after.json`,
`final_verdict.json`.
