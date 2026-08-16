# Realistic-PCAP diversification (v2)

**Experimental data collection only. No retraining. The production model, `ml.py`,
`live_capture.py`, `pcap_validation.py`, the frozen validation results, and the v1
realistic-PCAP collection are all untouched.** Everything here lives under
`validation/realistic_pcaps_v2/` on branch `claude/realistic-pcap-diversification`.

## Goal

Grow the realistic-PCAP corpus into something **substantially more diverse** than
v1 (which was 16 captures, single client / single server / single address), so a
future leakage-controlled retraining experiment has varied held-out material.
This phase only **collects and validates**.

## Diversity from genuine behaviour — not sleeps

The task forbids injecting `sleep` to fake variety. Every axis here is a real
behavioural difference:

| Axis | Values | Why it is genuine |
|---|---|---|
| **Client** | `python-ftplib`, `curl`, `wget`, raw-socket | Different implementations → different command sequencing, batching and timing on the wire |
| **Server** | pyftpdlib `ratelimited` / `permissive` / `throttled` | Real config variants: `auth_failed_timeout=3s` (a defended server that answers bad logins slowly), no rate limit, and a real `ThrottledDTPHandler` bandwidth cap |
| **Mode** | passive / active | A real FTP protocol difference (PASV vs PORT data channel) |
| **Environment** | `127.0.0.1`, `127.0.0.2`, `127.0.0.3` | Genuinely different source/destination addresses, all inside `127.0.0.0/8` |
| **Pattern** | new-conn-per-attempt, single-conn-many, multi-burst, reconnect | Real TCP/session behaviour |
| **Credentials / attempts** | varied username & password sets, 2–36 attempts | Real guessing behaviour |

**Slow vs fast brute-force pacing is a property of the real server** (its
`auth_failed_timeout` response delay), never a client `time.sleep`. The collector
injects no artificial per-packet or per-attempt sleeps.

> One honest caveat: there is only **one FTP server package** available here
> (pyftpdlib), run in three real behavioural variants — not three different FTP
> daemons. A TLS/FTPS variant was intended but **pyOpenSSL is unavailable** in
> this environment, so it is omitted rather than faked.

## Safety — loopback lab only

The lab target is a pyftpdlib server this process starts and controls, bound to
loopback addresses only. The collector **refuses** any target outside
`127.0.0.0/8`, and every brute-force driver `assert`s a loopback target before
sending a single guess. The per-capture `tcpdump` BPF is
`host <addr> and (tcp port <ctrl> or tcp port 20 or tcp portrange <passive>)`, so
only that endpoint's traffic is recorded. Verification independently confirms
every packet is loopback-only. **No external host is ever contacted.**

## Integrity

Each capture is an **independent new network interaction** — a fresh `tcpdump`
over fresh FTP connections. No PCAP is fabricated, synthesised, copied, replayed,
relabelled, or edited after capture. A test hashes every PCAP and asserts there
are **no duplicates**. Ground-truth labels come from the scenario/capture
directory, never from a model prediction.

## Per-capture record (`metadata/<id>.json`, key fields in `MANIFEST.csv`)

`scenario, label, client, server, environment, interface, mode, attempts,
start_time, end_time, packet_count, duration_s, source, destination,
capture_command, verification_status` — plus per-scenario stats and the full
verification record in the JSON.

## Feature extraction & validity (existing pipeline, unchanged)

Every capture is replayed through **`pcap_validation.replay_pcap`** — the same
validated Live-Capture engine as the frozen baseline, **with no changes**. Each
flow is checked for **exactly the 30 `ml.FEATURES`, in order, all finite**; a flow
missing any feature is **reported incomplete, never zero-filled**. The model is
never run to produce or check labels. Counts are in
`feature_extraction_report.csv`.

## Layout

```
validation/realistic_pcaps_v2/
├── benign/                 *.pcap
├── ftp_bruteforce/         *.pcap
├── metadata/               *.json  (one per capture)
├── MANIFEST.csv
├── feature_extraction_report.csv
├── collection_report.md / .json
└── verify_pcaps.py         standalone re-verifier (exit!=0 on any failure)
```

## Reproduce

```bash
# from webapp_django/ (tcpdump needs root; loopback only)
python manage.py collect_realistic_pcaps_v2
# independent re-check (no Django, no model):
python3 ../validation/realistic_pcaps_v2/verify_pcaps.py
```

Re-running performs **new real captures**; it never reuses or copies existing
PCAPs.

## Tests (`predictor/tests.py`)

Framework invariants (loopback-only environments, non-loopback rejected, BPF
confinement, spec counts 30–50, specs are distinct combinations, ≥3
clients/servers/envs, brute-force pool excludes the real password); **live**
tests (benign capture is real + loopback-only; brute-force capture only fails;
two captures are independent, not copies; brute force refuses a non-loopback
destination); **artifact** tests (manifest schema, counts in range, every PCAP
exists, **no duplicate PCAPs**, labels-from-folders, all verified valid, all
destinations loopback, diversity present, extraction report has 0 incomplete,
sampled captures extract exactly 30 finite in-order features, **production &
baseline artifacts untouched**).

## Limitations

- **Loopback lab only** — addresses vary but it is one host; not real multi-host
  network traffic.
- **One server package** (pyftpdlib) with real config variants; no second FTP
  daemon, and no FTPS/TLS (dependency unavailable).
- Still an *expansion* for exploratory retraining, not a production-grade,
  real-world-diverse corpus.

## Next step (not done here)

Feed the combined v1 + v2 corpus into a **larger leave-one-capture-out retraining
experiment** (as in `docs/realistic-pcap-retraining.md`). **Stopped before any
retraining, per the task.**
