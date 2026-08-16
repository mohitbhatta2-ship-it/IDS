# Realistic-PCAP data expansion — collection report

**Experimental data collection only. No retraining. Ground-truth labels come from
the capture scenario, never from model predictions.**

## Target (identified before any traffic)

- local lab FTP target 127.0.0.1:21 (passive 60000-60040) on iface lo
- Loopback only: **True** — all traffic stayed inside the local lab.
- Server: `pyftpdlib`  ·  client: `python-ftplib`  ·  capture: `tcpdump -i lo`

## Totals

| Metric | Value |
|---|---|
| PCAP count | **16** (8 benign, 8 FTP-bruteforce) |
| Captures passing verification | 16 / 16 |
| Total flows extracted (existing pipeline) | **203** |
| Incomplete flows (reported, not zero-filled) | 0 |
| Model features per flow | 30 (exact order preserved) |

## Benign captures

| capture_id | scenario | attempts | packets | duration_s | flows | incomplete | validation |
|---|---|---|---|---|---|---|---|
| benign_01 | successful_login | 1 | 15 | 0.00 | 2 | 0 | valid |
| benign_02 | login_then_listing | 1 | 31 | 0.00 | 4 | 0 | valid |
| benign_03 | login_then_commands | 1 | 25 | 0.00 | 2 | 0 | valid |
| benign_04 | file_download | 1 | 31 | 0.00 | 4 | 0 | valid |
| benign_05 | file_upload | 1 | 31 | 0.00 | 4 | 0 | valid |
| benign_06 | multiple_sessions | 4 | 126 | 0.46 | 16 | 0 | valid |
| benign_07 | long_session | 1 | 123 | 3.61 | 14 | 0 | valid |
| benign_08 | short_session | 1 | 18 | 0.00 | 2 | 0 | valid |

## FTP-BruteForce captures

| capture_id | scenario | attempts | packets | duration_s | flows | incomplete | validation |
|---|---|---|---|---|---|---|---|
| ftpbf_01 | slow_failed_logins | 6 | 103 | 22.02 | 12 | 0 | valid |
| ftpbf_02 | fast_failed_logins | 12 | 198 | 33.31 | 23 | 0 | valid |
| ftpbf_03 | different_usernames | 8 | 136 | 24.74 | 16 | 0 | valid |
| ftpbf_04 | different_passwords | 12 | 205 | 37.16 | 24 | 0 | valid |
| ftpbf_05 | many_attempts | 24 | 408 | 72.58 | 48 | 0 | valid |
| ftpbf_06 | few_attempts | 3 | 51 | 9.61 | 6 | 0 | valid |
| ftpbf_07 | single_connection_many_attempts | 8 | 75 | 24.83 | 2 | 0 | valid |
| ftpbf_08 | multi_burst | 12 | 205 | 37.21 | 24 | 0 | valid |

## Capture diversity

- **Benign:** successful_login, login_then_listing, login_then_commands, file_download, file_upload, multiple_sessions, long_session, short_session.
- **Brute force:** slow_failed_logins, fast_failed_logins, different_usernames, different_passwords, many_attempts, few_attempts, single_connection_many_attempts, multi_burst — varying speed, username
  and password sets, attempt counts, and connection/session patterns.

## Feature extraction & validity (existing pipeline, unchanged)

Every capture was replayed through `pcap_validation.replay_pcap` (the validated
Live-Capture flow engine) with no changes. Each flow is checked for **exactly the
30 `ml.FEATURES`, in order, all finite**; any flow missing
a feature is **reported as incomplete, never zero-filled**. Per-capture counts are
in `feature_extraction_report.csv`.

## Independence & integrity

Each capture is an independent new network interaction — a fresh `tcpdump` plus
fresh FTP connections — never a copy, replay, or relabel of an existing PCAP. No
packets were edited after capture.

## Limitations

- Loopback lab only: a single host / OS / FTP stack, not diverse real-world
  traffic. This expands the 5-PCAP baseline but remains exploratory.
- One server (`pyftpdlib`) and one client (`ftplib`).

## Files

`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `collection_report.json`, this report.
