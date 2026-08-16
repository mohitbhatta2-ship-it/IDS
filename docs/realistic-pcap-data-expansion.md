# Realistic-PCAP data expansion (local-lab capture)

**Experimental data-collection phase. No retraining. The production model,
`ml.py`, `live_capture.py`, the PCAP-validation results, and the realistic-PCAP
retraining results are all untouched.** Everything here lives under
`validation/realistic_pcaps/` on branch `claude/realistic-pcap-data-expansion`.

## Goal

Grow the realistic-PCAP set (the frozen baseline had only **5 captures / 42
flows**) with **real, controlled** FTP captures, so a future
leakage-controlled retraining experiment has more, more diverse held-out
material. This phase only **collects and validates** data.

## What "real" means here (and what was NOT done)

- **Not fabricated / not synthesised.** Packets are recorded by real `tcpdump`
  from a real TCP conversation between a real FTP client and a real FTP server.
- **Not duplicated / not relabelled / not edited.** Each capture is a fresh
  `tcpdump` process over fresh FTP connections; no PCAP is copied, replayed, or
  modified after capture.
- **Labels from the scenario, never the model.** A capture is `Benign` or
  `FTP-BruteForce` because of *which scenario generated it* (its folder), not
  because of any prediction. The model is never consulted during collection.

## The lab, identified before any traffic

| | |
|---|---|
| Target | **`127.0.0.1:21`** (loopback only) |
| Server | `pyftpdlib` (real FTP server; permissive `max_login_attempts` so the single-connection brute-force pattern is coherent) |
| Client | `python-ftplib` (real control + data connections) |
| Capture | `tcpdump -i lo`, BPF-filtered to `tcp port 21 or tcp portrange 60000-60040` |

All traffic stays on the loopback interface. The framework **refuses to run**
against any non-loopback host, and brute force asserts the target is loopback
before sending a single guess. Nothing external is ever contacted.

## Pipeline (per capture)

1. Start + **verify** the FTP lab target (real control connection + login).
2. Start `tcpdump` on `lo` (wait for "listening", settle) — capture lab ports only.
3. Drive the scenario's **real** FTP session(s).
4. Stop `tcpdump` cleanly (drain, SIGTERM, flush).
5. **Verify** the PCAP: exists, readable, has packets, carries FTP traffic on
   port 21, is loopback-only TCP, has a valid finite duration.
6. Record metadata (below) and append to the manifest.

Each capture is therefore an **independent new network interaction**.

## Captures collected

**16 captures — 8 benign, 8 brute-force — all passing verification. 203 flows
(48 benign, 155 brute-force), 0 incomplete/zero-filled.**

Benign diversity: successful login; login + directory listing; login + normal
commands (PWD/SYST/TYPE/NOOP); file **download** (safe local test file); file
**upload**; multiple sequential sessions; a long idle-held session (~3.6 s); a
short session.

Brute-force diversity: slow failed logins (~0.8 s apart), fast failed logins,
different usernames, different password sets (incl. right-user/wrong-password),
many attempts (24), few attempts (3), single-connection-many-attempts, and
multi-burst. Attempts per capture range **3–24**; capture durations **9.6–72.6 s**
(the long tails are pyftpdlib's real anti-brute-force response delays — genuine
defended-server timing, not injected sleeps).

Every guessed credential is wrong by construction, so every brute-force capture
is real *failed* logins (0 successful logins), while benign captures authenticate
with the real lab user.

## Metadata & manifest

Per capture (`metadata/<id>.json`, and the key fields in `MANIFEST.csv`):
`capture_id, label, scenario, timestamp, source, destination, client_tool,
attempts, capture_duration_s, pcap_filename, validation_status` — plus richer
detail (packet count, per-scenario stats, full verification record) in the JSON.

## Feature extraction & validity (existing pipeline, unchanged)

Every capture was replayed through **`pcap_validation.replay_pcap`** — the same
validated Live-Capture flow engine used by the frozen baseline, **with no
changes**. Each flow is checked for **exactly the 30 `ml.FEATURES`, in order, all
finite**; a flow missing any feature would be **reported as incomplete, never
zero-filled**. Result: **203/203 flows valid, 0 incomplete**. Per-capture counts
are in `feature_extraction_report.csv`. **The model was not run to produce or
check labels.**

## Layout

```
validation/realistic_pcaps/
├── benign/                 8 *.pcap
├── ftp_bruteforce/         8 *.pcap
├── metadata/               16 *.json
├── MANIFEST.csv            one row per capture
├── feature_extraction_report.csv
├── collection_report.md / .json
└── verify_pcaps.py         standalone re-verifier (exit!=0 on any failure)
```

## Reproduce

```bash
# from webapp_django/ (tcpdump needs root; loopback only)
python manage.py collect_realistic_pcaps
# independent re-check of every capture (no Django, no model):
python3 ../validation/realistic_pcaps/verify_pcaps.py
```

Re-running performs **new real captures** (fresh network interactions); it does
not reuse or copy existing PCAPs.

## Tests

`predictor/tests.py` adds: framework invariants (loopback-only target, BPF
confinement, scenario catalogs, brute-force creds never valid, manifest schema);
**live-capture** tests (server verifies; a benign capture is real & loopback; a
brute-force capture records only failed logins; two captures are independent, not
copies — different ephemeral ports); and **artifact** tests (manifest columns,
every PCAP exists, labels come from folders, disk matches manifest, a sampled
capture verifies and extracts exactly 30 finite in-order features).

## Limitations

- **Loopback lab only** — a single host, OS and FTP stack (`pyftpdlib`) with a
  single client (`ftplib`). This is an *expansion* of the tiny baseline, not a
  diverse real-world corpus; it remains exploratory. A production-grade set would
  need multiple tools, OSes, and networks.
- Loopback timing/MSS differ from a real LAN; the anti-brute-force delays are
  pyftpdlib-specific.

## Next step (not done here)

Feed these captures into a **larger leave-one-capture-out retraining
experiment** (as in `docs/realistic-pcap-retraining.md`). **Stopped before any
retraining, per the task.**
