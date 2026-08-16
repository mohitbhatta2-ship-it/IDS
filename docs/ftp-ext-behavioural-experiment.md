# Extended-behavioural feature experiment — separating benign failed logins from brute force

**Read-only w.r.t. production. The production model, Candidate 2, the 45-feature robust
candidate, and `ml.py` / `live_capture.py` / `pcap_validation.py` / `ftp_behavioral.py`
are byte-for-byte unchanged (verified before == after). The existing 30 packet features
and 15 behavioural features are PRESERVED; this experiment only ADDS features. The
frozen independent vsFTPD corpus is used for final evaluation only — never for training,
selection, or tuning. No promotion, no merge.** Branch
`claude/final-independent-ftp-validation`.

## Question

The independent vsFTPD validation left one weakness: the behavioural model catches all
FTP brute force but false-positives on benign failed-login sessions (benign recall
0.852, FPR 0.148; `gave_up` 0/5, `mistype` 4/12). Can additional **label-free**
application/network features distinguish (1) benign mistype/give-up, (2) genuine brute
force, and (3) attacker-eventually-succeeds?

## What was built (all label-free, no zero-fill, no prediction-as-feature)

`ftp_behavioral_ext.py` — 12 new features computed directly from the PCAP:

- **timing**: `ftpx_interattempt_mean_s`, `_min_s`, `_cv` (NaN when < 2 attempts);
- **credential variation**: `ftpx_distinct_passwords`, `ftpx_distinct_usernames`,
  `ftpx_password_reuse_ratio` (NaN when no PASS);
- **session/pacing**: `ftpx_attempts_per_conn_max`, `ftpx_session_duration_s`,
  `ftpx_ended_with_quit`;
- **activity after auth**: `ftpx_post_auth_commands`, `ftpx_post_auth_data_transfers`;
- **ordering**: `ftpx_fail_run_before_success`.

Genuinely undefined measurements are **NaN** (HGB handles them), never zero-filled. CIC
flows (no PCAP) have all 15+12 application-layer features NaN. Feature set = 30 + 15 +
12 = **57**.

## Step 1 — separability analysis (approved corpora only)

On v1/v2/targeted/robust_train, grouping rows by scenario metadata (never as a feature):

- **benign-fail vs brute** is separable mostly by *existing* counts (`ftp_user_commands`
  AUC 0.99, `ftp_failed_logins` 0.97); new features add secondary signal
  (`ftpx_fail_run_before_success` 0.87, `ftpx_post_auth_commands` 0.87).
- **benign-mistype vs attacker-eventual-success** is well separated by new features:
  `ftpx_ended_with_quit` (1.00), `ftpx_distinct_passwords` (0.98),
  `ftpx_fail_run_before_success` (0.98).
- **Critically**, inter-attempt **timing was 0 everywhere** in the existing corpora
  (attempts back-to-back), so the timing features were dormant. This motivated a new
  training corpus with realistic pacing.

## Step 2 — new training-only corpus (realistic benign failed logins)

`validation/benign_failed_login_pcaps`: **24 fresh benign PCAPs, 93 flows, 0
incomplete**, with human think-time pacing (inter-attempt 0.5–2.3 s), password REUSE
(reuse ratio up to 0.75), small attempt counts, graceful QUIT, and real post-success
activity. Loopback lab, pyftpdlib + custom raw-socket servers (**vsftpd deliberately
held out** to keep the vsFTPD test independent), NEW addresses (127.0.0.20-22) / ports
(2530/2540). Hash-disjoint from every prior corpus **and** the frozen vsFTPD test.

## Step 3 — train + select (CIC held-out + LOCO only) + final vsFTPD test

Candidates (57 features) trained on CIC + v1 + v2 + targeted + robust_train +
benign_failed_login, unweighted/balanced. Selected `candidate_ext_unweighted` via CIC
held-out (no regression, acc 0.9805) + capture-level LOCO (macro-F1 0.9957) — the vsFTPD
test was **not** consulted for selection.

### Final vsFTPD test (cleartext)

| Model | FTP recall | Benign recall | FPR | macro-F1 | accuracy |
|---|---|---|---|---|---|
| production (30) | 0.000 | 0.984 | 0.016 | 0.209 | 0.458 |
| Candidate 2 (30) | 0.086 | 0.852 | 0.148 | 0.364 | 0.443 |
| robust45 (current, 30+15) | 1.000 | 0.852 | 0.148 | 0.930 | 0.931 |
| **candidate_ext (30+15+12)** | **1.000** | **0.836** | **0.164** | **0.922** | **0.924** |

Bootstrap 95% CIs (ext): FTP recall 1.000 [1.000, 1.000], benign recall 0.829 [0.604,
1.000]. Per family: `gave_up` **0/5** (unchanged), `mistype` **0.583** (no better).

## Verdict — TARGET NOT MET

`final_verdict.json`: `verdict = "TARGET NOT MET"`, `target_met = false`,
`promote = false`. The promotion target (FTP ≥ 0.90, benign ≥ 0.90, FPR ≤ 0.10, no CIC
regression, no single-feature dependence, holds on the frozen test) is not met: benign
recall 0.836 < 0.90 and FPR 0.164 > 0.10 on the vsFTPD test — no improvement over the
current candidate.

## Why it did not work — three concrete, measured reasons

1. **The model ignored the new features.** SHAP on the vsFTPD test: the 12 EXT features
   contribute only **1.3%** of importance; CIC artifacts still dominate (43%), then
   `ftp_failed_logins` (4.2%). On the training data the *existing* count features already
   separate benign-fail from brute, so HGB never had to learn the EXT dynamics — the hard
   cases (`gave_up` vs a short brute force) are too rare in training to reshape the model.

2. **The discriminating dynamics don't fire for the failing case.** The features that
   genuinely separate benign from attack — post-auth activity, eventual-success ordering —
   are **zero for `gave_up`** (a benign user who never authenticates has no post-auth
   activity, exactly like a failed brute force). And the pacing signal learned from the
   loopback training corpus (human think-time) does not transfer to the vsFTPD test, where
   both benign and brute timing are dominated by the *server's* failure-delay, not the
   client.

3. **Single-feature dependence (drop-one ablation on the vsFTPD test).** Removing
   `ftp_user_commands` collapses FTP recall 1.000 → **0.671**; the model leans on that one
   behavioural count — which itself fails the "no single-feature dependence" criterion.

## The remaining limitation (stated plainly)

**A benign user who fails to authenticate and leaves (`gave_up`) produces control- and
network-level traffic that is near-identical to a short failed brute force.** They have
the same failed-login counts, the same lack of success, the same lack of post-auth
activity; the only true difference is *intent*, which is not on the wire. No label-free
feature fully separates them. Timing helps only when the benign side is human-paced AND
the attacker is not — which is not the case once a real server's failure-delay dominates.

## A concrete lead for the next experiment (hypothesis only — seen on the test, not acted on)

The drop-one ablation showed that removing **`ftp_failed_logins`** *improves* the vsFTPD
benign recall to **0.918** and FPR to **0.082** (FTP recall stays 1.000) — i.e. the
false positives are being driven by the raw failed-login count. This is a **test-set
observation** and was NOT used to select or tune anything here. It suggests a properly
controlled follow-up:

1. Train a candidate that **excludes or strongly down-weights `ftp_failed_logins`** (and
   `ftp_user_commands`), selecting only on CIC/LOCO, and re-test on the frozen vsFTPD
   corpus — does benign recall reach ≥ 0.90 without losing FTP recall?
2. Collect benign failed-login traffic from **multiple real servers** (vsftpd, proftpd,
   pure-ftpd) so pacing/activity signals generalise beyond one server's timing.
3. Add **source-reputation / cross-session** features (repeated connections from one
   source over time) — the one dimension that genuinely separates a returning legitimate
   user from a brute-forcer, which single-session PCAP features cannot.
4. If none of these close the gap, adopt an explicit **product policy** that "many failed
   logins with no success" is treated as suspicious regardless of intent, accepting
   `gave_up` as a labelled boundary case.

## Integrity / leakage (all verified)

Production, Candidate 2, the 45-feature robust candidate, and
`ml.py`/`live_capture.py`/`pcap_validation.py`/`ftp_behavioral.py` unchanged
(before == after). 57 features (30 packet in original order + 15 behavioural preserved +
12 new); no zero-fill; CIC application-layer features NaN; labels from folders only; the
new benign corpus is hash-disjoint from all prior corpora and the frozen vsFTPD test;
selection used CIC + LOCO only. No independent-test PCAP entered training.

## Reproduce

```bash
# from webapp_django/ (root needed for the capture step)
python manage.py ftp_ext_separability                  # separability analysis (approved corpora)
python manage.py collect_benign_failed_login_pcaps     # realistic benign failed-login TRAIN corpus
python manage.py ftp_ext_experiment --loco-caps 60     # train, CIC/LOCO select, vsFTPD test, SHAP, drop-one
python manage.py test predictor.tests.FtpExtFeatureTests \
                      predictor.tests.BenignFailedLoginCorpusTests \
                      predictor.tests.FtpExtExperimentResultsTests
```

## Files

- Separability: `validation/results/ftp_ext_separability/`
- New corpus: `validation/benign_failed_login_pcaps/`
- Experiment: `validation/results/ftp_ext_experiment/` (`final_test_metrics.csv`,
  `drop_one_ablation.csv`, `shap_top_features.csv`, `bootstrap_cis.json`,
  `per_scenario_family_metrics.csv`, `leakage_validation.json`,
  `model_hashes_before_after.json`, `final_verdict.json`, `report.md`)
- Candidate models: `validation/models/ftp-behavioral-ext/` (never `webapp_data/`)
