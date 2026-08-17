# Per-connection + cross-session FTP detector

**Candidates only; production frozen. The production model, Candidate 2, C3, the
cross-session model, and `ml.py` / `live_capture.py` / `pcap_validation.py` /
`ftp_behavioral.py` are byte-for-byte unchanged (verified before == after). The 30 packet
features and their order are preserved; per-connection and cross-session features are
appended. Selection used CIC held-out + capture-level LOCO only; the fresh proftpd corpus
was evaluated exactly once. No promotion, no merge.** Branch
`claude/ftp-per-connection-detector`.

## The problem being fixed

The cross-session detector failed its 2nd independent test (pure-ftpd) by missing
**single-session** attacks: it had learned "few sessions ⇒ benign", so a brute force
packed into one connection was called benign. This experiment adds **per-connection**
brute-force features so an attack is detectable regardless of how many connections the
attacker uses, keeping cross-session features as *supporting context*, and trains on a
corpus that explicitly spans the whole single↔multi-session spectrum.

## What was built (label-free, no zero-fill, no rules)

`ftp_per_connection.py` — 11 per-connection features aggregated across a capture's control
connections: max/mean **attempts per connection**, max **failures per connection**, max
**attempt rate** and min inter-attempt gap *within* a connection, max distinct
passwords/usernames per connection, connection durations, max fail-ratio per connection,
fraction of connections with a failure. Undefined values are NaN, never zero-filled. No
"X attempts = attack" rule — the model learns the behaviour.

Feature set (candidate): **C3 (43) + per-connection (11) + cross-session (13) = 67**.

`per_connection_pcaps` — a training corpus (32 PCAPs, 84 flows) covering **single-session
packed** attacks (5–12 attempts in ONE connection), 2–3 / 4–6 / 7+ session attacks,
reconnecting/slow/eventual-success, and benign mistypes/give-ups/normal. Custom raw-socket
server (allows packed attempts), NEW addresses (127.0.0.40-42) / ports (2730/2740),
hash-disjoint from every prior corpus and every independent test.

## The fresh independent test (proftpd — a 5th distinct server)

`independent_ftp_validation3_pcaps` — 22 PCAPs (12 benign, 10 attacker), 72 flows, all
valid, **hash-disjoint from all 11 prior corpora**. **proftpd** (ProFTPD 1.3.8b, never used
anywhere), **ncftp** client, new netns/subnet **10.99.0.0/24** / port 2424, fresh scenario
code with single-session (packed) and multi-session attacks. Evaluated ONCE.

## Results

| Model | FTP recall | Benign recall | FPR | macro-F1 |
|---|---|---|---|---|
| production (30) | 0.000 | 0.977 | 0.023 | 0.261 |
| Candidate 2 (30) | 0.000 | 0.773 | 0.227 | 0.321 |
| C3 (43) | 0.857 | 0.977 | 0.023 | 0.925 |
| cross-session (58) | 0.821 | 1.000 | 0.000 | 0.924 |
| **candidate_pc (67)** | **0.893** | **1.000** | **0.000** | **0.955** |
| ablation: no cross (54) | 0.821 | 1.000 | 0.000 | 0.924 |
| ablation: no per-connection (56) | 0.857 | 1.000 | 0.000 | 0.940 |

Bootstrap 95% CIs (candidate_pc): FTP recall 0.885 [0.714, 1.000], benign recall 1.000
[1.000, 1.000], FPR 0.000 [0.000, 0.000]. CIC held-out: no regression (candidate 0.892 FTP
/ 0.986 benign / 0.9803 accuracy).

`candidate_pc` is the **best model** — highest FTP recall with a perfect benign side.

### Attack recall by sessions-per-source (the decisive test)

| sessions/source | n | candidate_pc | cross | C3 | no-cross | no-pc |
|---|---|---|---|---|---|---|
| 1 | 5 | 0.800 | 0.800 | 1.000 | 0.800 | 0.800 |
| 2–3 | 6 | **0.667** | 0.333 | 0.333 | 0.333 | 0.500 |
| 4–6 | 17 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

By scenario family, candidate_pc catches **single_packed 1.000 (4/4)**, cred_sweep 1.000,
slow 1.000, multi_session 0.667, **eventual_success 0.857 (6/7)** — the one miss is the
attacker who **fails then succeeds in a single session**.

## SHAP + ablation — the goals were achieved

- **Session count is NOT the main signal.** `ftpx_sessions_per_source` is SHAP **rank 61
  (0.0% share)** — the cross model's crutch is gone. `session_count_is_main_signal = false`.
- **No single-feature shortcut.** Largest feature `Dst Port` at 0.186 (< 0.5). The top
  application-layer feature is `ftpx_failure_rate_across_sessions` (failure *rate*, not a
  count or session count).
- **Packed single-session attacks are detected** (single_packed 1.000), so the specific
  failure mode of the cross model is fixed.
- Ablation nuance: removing the per-connection block does **not** drop single-session
  recall, and removing cross does not either — because the cross block already carries
  failure-volume features (`ftpx_max_fails_in_session`, `ftpx_failure_rate_across_sessions`)
  that also fire on packed single-session attacks. The two blocks are partly redundant for
  packed-attack detection; the per-connection block's clearest marginal gain is on the 2–3
  session attacks (0.667 vs 0.500 without it) and overall FTP recall.

## Verdict — PROMISING — NEEDS MORE DATA (target not strictly met; do not tune)

`final_verdict.json`: `verdict = "PROMISING -- NEEDS MORE DATA"`, `promote = false`.

The approach **achieves its purpose**: it breaks the session-count shortcut (SHAP rank 61),
detects packed single-session attacks (single_packed 1.000), introduces **no** new
single-feature shortcut, keeps a perfect benign side (1.000 / 0.000), and is the best model
(FTP 0.893 vs cross 0.821 / C3 0.857). But it does **not** strictly meet the promotion
target on this small fresh corpus:

- FTP recall **0.893 < 0.90** (CI [0.714, 1.000] does include 0.90);
- single-session attack recall **0.800 < 0.90**.

Both shortfalls trace to **one** genuinely-ambiguous case: an **attacker who fails a few
times and then succeeds within a single session** is, on the wire, identical to a benign
user who mistyped and then logged in. No label-free feature resolves that — it is the same
intent-not-on-the-wire limit seen throughout this line of work, now in single-session form.
Per the brief, we **stop here and report honestly rather than tune on the independent
test**.

### What this proves / does not prove
- **Proves:** per-connection features remove the cross model's session-count over-reliance
  and restore single-session *packed* attack detection, without a new shortcut and without
  hurting benign — validated on a genuinely different server/client/network.
- **Does not prove** the strict ≥0.90 target is reached; and note proftpd's
  `MaxLoginAttempts` bounds single-session length, so a real attacker could pack more (which
  would *help* per-connection detection, but the eventual-success-in-one-session case would
  remain).

### Recommended next steps (no tuning on this corpus)
1. A larger, multi-server fresh corpus to tighten the FTP-recall CI (lower bound 0.714 here
   is small-sample noise) and confirm ≥0.90.
2. An explicit product decision on **attacker-eventual-success**: treat "several failures
   then success" as suspicious (accepting some benign-mistype cost), or add
   cross-*session*/source-reputation history over longer windows to see the attacker's other
   sessions.
3. Keep per-connection + cross-session together (they are complementary and neither is a
   shortcut), and re-run this independent validation after any change.

## Integrity / leakage (all verified)

Test corpus hash-disjoint from all 11 prior corpora and the training set; source/session
separation (proftpd 10.99.0.x vs loopback training and the earlier 10.77/10.88 tests);
labels from folders only; no zero-fill (undefined per-connection/cross features NaN); 30
packet features preserved in order; every flow traceable to its PCAP; production and all
frozen models unchanged (before == after); selection used CIC + LOCO only; the fresh corpus
was evaluated once.

## Reproduce

```bash
# from webapp_django/ (root needed for capture)
python manage.py collect_per_connection_pcaps            # packed-single-session + spectrum TRAIN corpus
python manage.py collect_independent_ftp_validation3     # fresh proftpd test corpus
python manage.py ftp_per_connection_experiment --loco-caps 50
python manage.py test predictor.tests.FtpPerConnectionFeatureTests \
                      predictor.tests.PerConnectionCorpusTests \
                      predictor.tests.IndependentFtp3CorpusTests \
                      predictor.tests.FtpPerConnectionResultsTests
```

## Files (`validation/results/ftp_per_connection/`)

`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`per_scenario_family_metrics.csv`, `per_session_structure_metrics.csv`, `bootstrap_cis.json`,
`shap_top_features.csv`, `confusion_*.csv`, `leakage_validation.json`,
`model_hashes_before_after.json`, `final_verdict.json`. Candidate + ablation models under
`validation/models/ftp-per-connection/`.
