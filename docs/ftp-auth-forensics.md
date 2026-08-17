# Auth-forensics FTP detector — the mistype-vs-single-session-attack case

**Candidates only; production frozen. The production model, Candidate 2, C3, the
cross-session model, the per-connection candidate, and `ml.py` / `live_capture.py` /
`pcap_validation.py` / `ftp_behavioral.py` / `ftp_cross_session.py` are byte-for-byte
unchanged (verified before == after). The 30 packet features and their order are preserved;
per-connection, cross-session and auth-forensics features are appended. Selection used CIC
held-out + capture-level LOCO only; the fresh pure-ftpd corpus was evaluated exactly once.
No promotion, no merge.** Branch `claude/ftp-auth-forensics`.

## The problem being tested (the last remaining failure case)

Every prior detector could not separate the single hardest pair:

- a **benign user who mistypes** their password and then logs in, vs
- an **attacker who fails a few times and then succeeds** in a single session.

Failure count and session structure are *identical* for the two, so the behavioural,
cross-session and per-connection features all treat them the same. The hypothesis for this
final experiment was that *what the failed passwords look like* would separate them: a human
mistypes (attempts **edit-distance-close** to the correct password — a dropped/added/
transposed character, wrong case), while an attacker guesses a **dictionary**
(edit-distance-far, unrelated strings).

## What was built (label-free, no zero-fill, no rules)

`ftp_auth_forensics.py` — 10 authentication-forensics features from the cleartext control
channel: distinct failed passwords; distinct passwords before the first success;
min / mean **Levenshtein distance** from failed attempts to the successful password;
edit-distance between consecutive attempts; password length spread; password reuse ratio;
inter-attempt timing and its regularity. A genuinely undefined measurement (e.g. the
distance to the successful password when there was no success) is **NaN, never zero-filled**.
No "edit distance > k = attack" rule — the idea was to let the model learn it.
`ftp_failed_logins` is deliberately **not** in the feature set (C3 drops it), so the model
cannot lean on failure count.

Feature set (candidate): **per-connection candidate (67) + auth-forensics (10) = 77**.

`auth_forensics_pcaps` — a TRAIN corpus (28 PCAPs, 46 flows): benign typo-then-success /
typo-give-up / alternate-password vs single-session dictionary brute force (incl.
dictionary-then-success). Custom raw-socket server, NEW addresses (127.0.0.50-52) / ports
(2830/2840), hash-disjoint from every prior corpus and every independent test. On this
corpus the crux feature does separate the two: benign mistype `min_editdist_fail_to_success`
median **1.0** vs attacker dictionary-then-success median **6.0**.

## The fresh independent test (pure-ftpd — evaluated once)

`independent_ftp_validation4_pcaps` — 17 PCAPs (10 benign, 7 attacker), 34 flows, all valid,
**hash-disjoint from all 13 prior corpora**. **pure-ftpd** server, new netns/subnet
**10.111.0.0/24** / port 2525, fresh typo-vs-dictionary scenario code. The decisive cases:
benign `mistype` (typo-then-success, n=4 flows) and attacker `dict_success`
(dictionary-then-success, n=3 flows).

## Results — the forensic features did not help, and slightly hurt

| Model | FTP recall | Benign recall | FPR | macro-F1 |
|---|---|---|---|---|
| production (30) | 0.000 | 0.913 | 0.087 | 0.255 |
| Candidate 2 (30) | 0.000 | 0.783 | 0.217 | 0.346 |
| C3 (43) | 1.000 | 0.957 | 0.043 | 0.967 |
| cross-session (58) | 1.000 | 1.000 | 0.000 | 1.000 |
| per-connection (67) | 1.000 | 0.957 | 0.043 | 0.967 |
| **candidate_af (77)** | **0.909** | **1.000** | **0.000** | 0.966 |
| ablation: no forensic (67) | 1.000 | 1.000 | 0.000 | 1.000 |
| ablation: no edit-distance (74) | 1.000 | 1.000 | 0.000 | 1.000 |

Bootstrap 95% CIs (candidate_af): FTP recall 0.898 [0.600, 1.000], benign 1.000 [1.000,
1.000], FPR 0.000. CIC held-out: **no regression** (candidate 0.876 FTP / 0.986 benign /
0.9803 accuracy — identical accuracy to production).

### The decisive case: benign typo vs attacker dictionary-then-success

| scenario family (label) | candidate_af (77) | per-connection (67) | no-edit-distance (74) |
|---|---|---|---|
| mistype / typo-then-success (Benign) | 1.000 | 1.000 | 1.000 |
| dict_success / dict-then-success (FTP) | **0.667** | 1.000 | 1.000 |

- The forensic features **do not improve** the decisive case: `forensic_improves_decisive_case = false`.
- Edit-distance features **do not carry** any separation: `editdistance_carries_separation = false`
  — removing them (no-edit-distance ablation) changes nothing (still 1.000 / 1.000).
- Adding the 10 forensic features **degraded** attacker recall from 1.000 (no-forensic
  ablation, same data and 67 features) to 0.909, and dictionary-then-success recall from
  1.000 to 0.667 — the extra mostly-NaN/low-signal columns perturbed the tree fit and cost
  one attacker capture.

## SHAP — the model ignored the edit-distance signal

- **Forensic block SHAP share 0.6%**; the three **edit-distance features together 0.02%**
  (top forensic feature `ftpaf_mean_interattempt_s`, not an edit-distance one). The signal
  that separated the two families on the *training* corpus did not survive as useful model
  weight — the trees split on the usual CIC packet artifacts (`Dst Port` 0.174, etc.).
- `ftpx_sessions_per_source` SHAP **rank 73, share 0.0** — session count is not the signal.
- `ftp_failed_logins` is **not in the feature set** (excluded by design) — the model cannot
  lean on failure count.
- Largest single feature `Dst Port` at 0.174 (< 0.5): no single-feature shortcut,
  `session_count_is_main_signal = false`, `single_feature_shortcut = false`.

## Verdict — NOT EFFECTIVE — DISTINCTION NOT RELIABLY OBSERVABLE

`final_verdict.json`: `verdict = "NOT EFFECTIVE -- DISTINCTION NOT RELIABLY OBSERVABLE"`,
`promote = false`.

The auth-forensics edit-distance hypothesis **did not pan out**. Although failed-password
edit distance cleanly separates typo from dictionary on the *training* corpus, the trained
model assigns it ~0 importance and it provides **no** benefit on the fresh test — in fact the
forensic candidate is *worse* on the attacker side than the identical model without those
features. The two events (benign mistype-then-success and attacker
dictionary-fail-then-success) are, on the wire, the same shape: identical failure counts,
identical single-session structure, and once the attacker's guess happens to land the
successful login is indistinguishable from a human who finally typed it right. Per the brief,
we **stop and report the negative result** rather than tune further.

### What this proves / does not prove
- **Proves:** the edit-distance-of-failed-passwords idea does not add a usable, generalising
  signal — the model ignores it (0.02% SHAP) and it slightly hurts. Do not pursue more
  edit-distance / password-forensics feature tuning for this case.
- **Does not prove** that the two are *never* separable: on this small corpus the existing
  behavioural models (cross-session, and the no-forensic 67-feature model) happened to label
  the decisive cases perfectly (1.000 / 1.000) — but on **n=3** attacker and **n=4** benign
  decisive flows, that reflects incidental behavioural differences (timing / attempt counts)
  on tiny samples, not a robust intent signal. The fundamental ambiguity stands.

### Recommendation — handle at the product level, not with more traffic features
The intent that distinguishes "mistyped then logged in" from "guessed then broke in" is not
reliably on the FTP wire. The appropriate handling is **outside the traffic classifier**:

1. **Post-login step-up verification** for a session that logged in after failures — MFA, or
   device / IP reputation on the successful login.
2. **Server-side rate-limiting and account lockout** (pure-ftpd already delays/locks after
   failures) so an attacker cannot reach the eventual-success case in one session.
3. **Alert on the successful login's context** (new/geo-anomalous source, off-hours, first
   time from this IP) rather than the failure pattern alone.
4. Keep the existing per-connection + cross-session detector for the cases it already handles
   (packed and multi-session brute force); do not add the auth-forensics features to it.

## Integrity / leakage (all verified)

Test corpus hash-disjoint from all 13 prior corpora and the training set; source/session
separation (pure-ftpd 10.111.0.x vs loopback training and the earlier independent tests);
labels from folders only; no zero-fill (undefined forensic measurements NaN); 30 packet
features preserved in order; `ftp_failed_logins` excluded from the feature set; every flow
traceable to its PCAP; production and all frozen models unchanged (before == after);
selection used CIC + LOCO only; the fresh corpus was evaluated once.

## Reproduce

```bash
# from webapp_django/ (root needed for capture)
python manage.py collect_auth_forensics_pcaps            # typo-vs-dictionary TRAIN corpus
python manage.py collect_independent_ftp_validation4     # fresh pure-ftpd test corpus
python manage.py ftp_auth_forensics_experiment --loco-caps 50
python manage.py test predictor.tests.FtpAuthForensicsFeatureTests \
                      predictor.tests.AuthForensicsCorpusTests \
                      predictor.tests.IndependentFtp4CorpusTests \
                      predictor.tests.FtpAuthForensicsResultsTests
```

## Files (`validation/results/ftp_auth_forensics/`)

`final_test_metrics.csv`, `cic_heldout_metrics.csv`, `loco_pooled_metrics.json`,
`per_scenario_family_metrics.csv`, `bootstrap_cis.json`, `shap_top_features.csv`,
`confusion_*.csv`, `leakage_validation.json`, `model_hashes_before_after.json`,
`final_verdict.json`. Candidate + ablation models under
`validation/models/ftp-auth-forensics/`.
