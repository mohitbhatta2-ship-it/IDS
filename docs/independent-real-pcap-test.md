# Independent real-PCAP test — final evaluation of Candidate 2

**These test PCAPs were NEVER used for training, retraining, fine-tuning, sample
weighting, threshold selection, hyperparameter selection, feature selection,
candidate selection, or SHAP-based tuning.** They were collected fresh, after
Candidate 2 was already frozen, purely to evaluate it. Candidate 2 was **not
modified** after seeing these results, and is **not promoted**. Both models are
frozen; hashes were recorded before and verified unchanged after.

## What was frozen

- **Production model** `HistGradientBoosting_Tuned.pkl` (sha256 `8fb91333…`).
- **Candidate 2** = the best *unweighted* CIC+real model from retraining-v2,
  `validation/models/realistic_pcap_candidate_v2/candidate2_none.pkl`
  (sha256 `e18383c5…`), identified from `retraining_v2/final_verdict.json`
  (`best_candidate = cic+real_none`).

Both were verified byte-for-byte unchanged before == after
(`model_hashes_before_after.json`), along with the v2 evidence and the v1/v2
datasets.

## The independent test corpus

36 fresh PCAPs (**18 Benign, 18 FTP-BruteForce**, 420 flows, 0 incomplete /
0 zero-filled), 36 distinct scenarios. Genuinely NEW diversity vs v1/v2:

- **A different server implementation:** a hand-written **raw-socket FTP server**
  (`custom_ftp_server.py`) — not pyftpdlib — used for 22 captures, alongside
  pyftpdlib (permissive / rate-limited) for the rest.
- **FTP commands not exercised in v2:** MKD/RMD, RNFR/RNTO, DELE, APPE, SIZE,
  MDTM, NLST, STAT, active-mode data, upload-then-download.
- **New addresses/ports:** loopback `127.0.0.5/.6` and the host's own address
  `192.0.2.2`; control ports `2130/2140/2150` (v2 used `21/2121/2221`).
- **Clients:** python-ftplib, curl, wget, raw-socket.

Every capture is a fresh network interaction; content-hash disjoint from v1 and
v2 (proven), no duplicates. Ground truth from the folder only.

> **Honest limitation.** This is a single-host container: Linux routes same-host
> traffic to any local address (including `192.0.2.2`) over the loopback device,
> so no genuinely separate *physical interface* is exercised — the address varies,
> the interface does not. Only two server implementations are available (custom +
> pyftpdlib); there is no third-party FTP daemon and no FTPS/TLS. Diversity is real
> but the lab is still loopback-only.

## Independence / leakage — all checks pass

`leakage_validation.json` → `all_pass: true`: independent PCAPs disjoint from v1
and v2 (by content hash), no duplicates, labels from folders only, exactly 30
ordered finite `ml.FEATURES`, no zero-fill, every flow traceable to its PCAP,
extraction via the existing `pcap_validation` pipeline (unchanged).

## Results (independent test set, n = 420 flows / 36 captures)

| Model | FTP recall | FTP precision | Benign recall | Benign FP rate | macro-F1 | accuracy |
|---|---|---|---|---|---|---|
| **production** | **0.000** | 0.000 | 0.974 | 0.026 | 0.123 | 0.181 |
| **Candidate 2** | **0.754** | 0.921 | 0.718 | 0.282 | 0.672 | 0.748 |

**Capture-level bootstrap 95% CIs** (36 captures, 2000 resamples):

| Model | FTP recall | Benign recall | Accuracy |
|---|---|---|---|
| production | 0.000 [0.000, 0.000] | 0.973 [0.925, 1.000] | 0.181 |
| Candidate 2 | 0.751 [0.595, 0.898] | 0.718 [0.646, 0.789] | 0.748 |

## Generalization test — explicit answers

1. **Does Candidate 2 still beat production on completely unseen real FTP?**
   Yes — FTP recall **0.754 vs 0.000**. The gain is real on independent traffic.
2. **Does FTP recall remain high?** Partially. It **drops from 0.928 (v2 LOCO,
   in-distribution) to 0.754** on genuinely different traffic — still far above
   production, but meaningfully lower, with a wide CI lower bound (0.595).
3. **Is the Benign false-positive rate acceptable?** **No.** Benign FP rate rises
   to **0.282** (Benign recall 0.718) vs production's 0.026 — ~28% of benign
   traffic misclassified as attack. That is too high for deployment.
4. **Consistent across captures?** Mostly — 16/18 FTP captures have recall ≥ 0.5
   (per-capture consistency 0.89), but recall is uneven.
5. **Where does Candidate 2 fail?** FTP recall is weakest on `many_attempts`
   (0.44), `different_usernames` (0.45), and several ~0.50 captures
   (`reconnecting_bruteforce`, `slow_defended_server`, `different_passwords`,
   `fast_custom_active`). Benign FPs are spread across the NEW command-sequence
   scenarios (`size_mdtm_stat`, `multi_command_session`, `active_mode_download`,
   `reconnect_session`, `wget_download`) — i.e. behaviours not represented in
   training push benign flows over the line.
6. **Different client/server/environment?** The custom raw-socket server (a
   genuinely different implementation) is where much of the FTP recall lands, so
   the improvement is not purely a pyftpdlib artifact — but the higher benign-FP
   cost also shows up there.
7. **Does the result support promotion?** **No.**

## Confidence analysis

Candidate 2's benign false positives are not merely low-confidence noise — see
`confidence_distribution.csv` for confidently-wrong counts at ≥0.90. Production is
confidently correct on benign (few benign FPs) but **confidently wrong on FTP**
(recall 0.000). Candidate 2 trades that for FTP detection at the cost of confident
benign false positives. The decision threshold was **not** changed.

## SHAP (diagnosis only, frozen models)

- Production global top-5: Dst Port, Init Fwd Win Byts, Fwd Seg Size Min,
  Flow IAT Min, Fwd Pkts/s.
- Candidate 2 global top-5: **Fwd Seg Size Min, Init Fwd Win Byts**, Dst Port,
  Fwd IAT Min, Fwd Pkts/s.
- On the independent FTP test flows, Candidate 2's top features are again
  `Fwd Seg Size Min`, Dst Port, `Init Fwd Win Byts`.
- **Candidate 2 remains dependent on the CIC-specific artifact features**
  (`candidate_still_artifact_dependent = true`). It did not learn a
  fundamentally different representation; it extended the artifact-based mapping.

## Three-way comparison (do not conflate the data roles)

| Source | Data role | Prod FTP rec | Cand FTP rec | Prod Benign rec | Cand Benign rec | Cand macro-F1 |
|---|---|---|---|---|---|---|
| A · CIC held-out | training/validation dist. | — | — | — | — | 0.865 |
| B · v2 realistic LOCO | prior LOCO eval | 0.000 | 0.928 | 0.880 | 0.792 | 0.853 |
| **C · independent test** | **completely independent** | **0.000** | **0.754** | **0.974** | **0.718** | **0.672** |

The candidate's real-FTP advantage **shrinks** from the in-distribution LOCO (B)
to the independent test (C), and its benign recall degrades further — exactly the
optimism-correction an independent test is designed to expose. The independent
results were **not** fed back into training or model selection.

## Statistical honesty

Small sample: **36 captures / 420 flows**, loopback lab, 2 classes. The bootstrap
CIs are wide (FTP recall [0.595, 0.898]). **No statistical-significance claim is
made.** Point estimates with capture-level 95% CIs are reported; that is the
strongest claim the data supports.

## FINAL VERDICT — PROMISING BUT INSUFFICIENT EVIDENCE

Candidate 2 delivers a genuine, independently-confirmed FTP-recall improvement
over production (0.754 vs 0.000) that partially survives a change of server
implementation. But it is **not promoted**, because:

- Benign false-positive rate **0.282** is too high for deployment.
- FTP recall **degrades** from the in-distribution 0.928 to 0.754, with a wide CI
  lower bound (0.595).
- SHAP shows it **still leans on CIC-specific artifacts**, so robustness to other
  tools/OSes/networks is unproven.
- The corpus is small and loopback-only (single host, two server implementations).

`final_verdict.json` → `promote: false`. **A candidate is not promoted merely
because FTP recall is higher.** Production readiness would need a larger, more
diverse real corpus (multiple real hosts, tools, OSes) and a benign FP rate driven
well down, evaluated on a fresh independent test.

## Reproduce

```bash
# from webapp_django/  (tcpdump needs root; controlled local lab only)
python manage.py collect_independent_pcaps
python3 ../validation/independent_real_pcaps/verify_pcaps.py
python manage.py independent_test
python manage.py test predictor.tests.IndependentTestArtifactLeakageTests \
                      predictor.tests.IndependentTestResultsTests
```

## Evidence (`validation/results/independent_test/`)

`baseline_metrics.csv`, `candidate_metrics.csv`, `confusion_matrix_production.csv`,
`confusion_matrix_candidate.csv`, `per_class_metrics.csv`, `per_capture_metrics.csv`,
`prediction_distribution.csv`, `confidence_distribution.csv`,
`bootstrap_or_ci_results.csv`, `shap_comparison.csv`, `three_way_comparison.csv`,
`test_set_manifest_summary.csv`, `leakage_validation.json`,
`model_hashes_before_after.json`, `final_verdict.json`, `report.md`.
