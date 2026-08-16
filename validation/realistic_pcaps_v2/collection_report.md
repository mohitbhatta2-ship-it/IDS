# Realistic-PCAP diversification (v2) — collection report

**Experimental data collection only. No retraining. Ground-truth labels come from
the scenario/capture directory, never from model predictions.** All traffic stayed
inside the local loopback lab (127.0.0.0/8); no external host was contacted.

## Totals

| Metric | Value |
|---|---|
| PCAP count | **83** (42 benign, 41 FTP-bruteforce) |
| Verified valid | 83 / 83 |
| Total flows (existing pipeline) | **854** |
| Incomplete flows (reported, not zero-filled) | 0 |
| Model features per flow | 30 (exact order preserved) |

## Diversity

- **Clients:** {"python-ftplib": 53, "curl": 17, "wget": 4, "raw-socket": 9}
- **Servers (pyftpdlib variants):** {"pyftpdlib:ratelimited": 17, "pyftpdlib:permissive": 45, "pyftpdlib:throttled": 21}
- **Environments (loopback addresses):** {"env_a (127.0.0.1)": 28, "env_b (127.0.0.2)": 28, "env_c (127.0.0.3)": 27}
- **Mode:** {"passive": 57, "active": 26}
- **Distinct scenarios:** 51

Diversity is genuine: it comes from different client implementations, real server
behaviour (rate-limited vs permissive vs bandwidth-throttled), passive/active
protocol mode, connection reuse patterns, credential sets, attempt counts, and
capture address — **not** from injected sleeps. Slow vs fast brute-force pacing is
a property of the real server's response timing (`auth_failed_timeout`), not a
`time.sleep`.

## How genuineness is guaranteed

Each capture is an independent new network interaction — a fresh `tcpdump` over
fresh FTP connections against a server this process starts and controls. No PCAP
is copied, replayed, relabelled, or edited after capture. The verifier confirms
every packet is loopback-only (127.0.0.0/8), TCP, and on the expected FTP port.

## Benign captures

| id | scenario | client | server | env | mode | attempts | pkts | dur_s | flows | incomplete | verify |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign_01 | successful_login | python-ftplib | pyftpdlib:ratelimited | env_a (127.0.0.1) | passive | 1 | 15 | 0.00 | 2 | 0 | valid |
| benign_02 | successful_login | python-ftplib | pyftpdlib:permissive | env_b (127.0.0.2) | active | 1 | 15 | 0.00 | 2 | 0 | valid |
| benign_03 | successful_login | python-ftplib | pyftpdlib:throttled | env_c (127.0.0.3) | passive | 1 | 15 | 0.00 | 2 | 0 | valid |
| benign_04 | failed_then_successful_login | python-ftplib | pyftpdlib:ratelimited | env_a (127.0.0.1) | active | 2 | 29 | 3.01 | 4 | 0 | valid |
| benign_05 | failed_then_successful_login | python-ftplib | pyftpdlib:permissive | env_b (127.0.0.2) | passive | 2 | 28 | 0.00 | 4 | 0 | valid |
| benign_06 | failed_then_successful_login | python-ftplib | pyftpdlib:throttled | env_c (127.0.0.3) | active | 2 | 28 | 0.00 | 4 | 0 | valid |
| benign_07 | directory_listing | python-ftplib | pyftpdlib:ratelimited | env_a (127.0.0.1) | passive | 1 | 31 | 0.00 | 4 | 0 | valid |
| benign_08 | directory_listing | python-ftplib | pyftpdlib:permissive | env_b (127.0.0.2) | active | 1 | 31 | 0.00 | 4 | 0 | valid |
| benign_09 | directory_listing | python-ftplib | pyftpdlib:throttled | env_c (127.0.0.3) | passive | 1 | 31 | 0.00 | 4 | 0 | valid |
| benign_10 | download | python-ftplib | pyftpdlib:ratelimited | env_a (127.0.0.1) | active | 1 | 31 | 0.00 | 4 | 0 | valid |
| benign_11 | download | python-ftplib | pyftpdlib:permissive | env_b (127.0.0.2) | passive | 1 | 31 | 0.00 | 4 | 0 | valid |
| benign_12 | download | python-ftplib | pyftpdlib:throttled | env_c (127.0.0.3) | active | 1 | 31 | 0.00 | 4 | 0 | valid |
| benign_13 | upload | python-ftplib | pyftpdlib:ratelimited | env_a (127.0.0.1) | passive | 1 | 31 | 0.00 | 4 | 0 | valid |
| benign_14 | upload | python-ftplib | pyftpdlib:permissive | env_b (127.0.0.2) | active | 1 | 31 | 0.00 | 4 | 0 | valid |
| benign_15 | upload | python-ftplib | pyftpdlib:throttled | env_c (127.0.0.3) | passive | 1 | 32 | 0.00 | 4 | 0 | valid |
| benign_16 | multiple_sessions | python-ftplib | pyftpdlib:ratelimited | env_a (127.0.0.1) | active | 3 | 94 | 0.01 | 12 | 0 | valid |
| benign_17 | multiple_sessions | python-ftplib | pyftpdlib:permissive | env_b (127.0.0.2) | passive | 3 | 95 | 0.01 | 12 | 0 | valid |
| benign_18 | multiple_sessions | python-ftplib | pyftpdlib:throttled | env_c (127.0.0.3) | active | 3 | 96 | 0.01 | 12 | 0 | valid |
| benign_19 | reconnect | python-ftplib | pyftpdlib:ratelimited | env_a (127.0.0.1) | passive | 2 | 48 | 0.00 | 6 | 0 | valid |
| benign_20 | reconnect | python-ftplib | pyftpdlib:permissive | env_b (127.0.0.2) | active | 2 | 48 | 0.00 | 6 | 0 | valid |
| benign_21 | reconnect | python-ftplib | pyftpdlib:throttled | env_c (127.0.0.3) | passive | 2 | 46 | 0.00 | 6 | 0 | valid |
| benign_22 | interactive_session | python-ftplib | pyftpdlib:ratelimited | env_a (127.0.0.1) | active | 1 | 62 | 0.00 | 6 | 0 | valid |
| benign_23 | interactive_session | python-ftplib | pyftpdlib:permissive | env_b (127.0.0.2) | passive | 1 | 61 | 0.00 | 6 | 0 | valid |
| benign_24 | interactive_session | python-ftplib | pyftpdlib:throttled | env_c (127.0.0.3) | active | 1 | 62 | 0.00 | 6 | 0 | valid |
| benign_25 | short_session | python-ftplib | pyftpdlib:ratelimited | env_a (127.0.0.1) | passive | 1 | 15 | 0.00 | 2 | 0 | valid |
| benign_26 | short_session | python-ftplib | pyftpdlib:permissive | env_b (127.0.0.2) | active | 1 | 16 | 0.00 | 2 | 0 | valid |
| benign_27 | short_session | python-ftplib | pyftpdlib:throttled | env_c (127.0.0.3) | passive | 1 | 15 | 0.00 | 2 | 0 | valid |
| benign_28 | download | curl | pyftpdlib:permissive | env_a (127.0.0.1) | passive | 1 | 35 | 0.00 | 4 | 0 | valid |
| benign_29 | download | curl | pyftpdlib:throttled | env_b (127.0.0.2) | passive | 1 | 35 | 0.00 | 4 | 0 | valid |
| benign_30 | download | curl | pyftpdlib:permissive | env_c (127.0.0.3) | active | 1 | 35 | 0.00 | 4 | 0 | valid |
| benign_31 | download | curl | pyftpdlib:throttled | env_a (127.0.0.1) | active | 1 | 35 | 0.00 | 4 | 0 | valid |
| benign_32 | directory_listing | curl | pyftpdlib:permissive | env_b (127.0.0.2) | passive | 1 | 34 | 0.00 | 4 | 0 | valid |
| benign_33 | directory_listing | curl | pyftpdlib:throttled | env_c (127.0.0.3) | passive | 1 | 33 | 0.00 | 4 | 0 | valid |
| benign_34 | directory_listing | curl | pyftpdlib:permissive | env_a (127.0.0.1) | active | 1 | 33 | 0.00 | 4 | 0 | valid |
| benign_35 | directory_listing | curl | pyftpdlib:throttled | env_b (127.0.0.2) | active | 1 | 33 | 0.00 | 4 | 0 | valid |
| benign_36 | download | wget | pyftpdlib:permissive | env_c (127.0.0.3) | passive | 1 | 35 | 0.00 | 4 | 0 | valid |
| benign_37 | download | wget | pyftpdlib:throttled | env_a (127.0.0.1) | passive | 1 | 35 | 0.00 | 4 | 0 | valid |
| benign_38 | directory_listing | wget | pyftpdlib:permissive | env_b (127.0.0.2) | passive | 1 | 33 | 0.00 | 4 | 0 | valid |
| benign_39 | directory_listing | wget | pyftpdlib:throttled | env_c (127.0.0.3) | passive | 1 | 33 | 0.00 | 4 | 0 | valid |
| benign_40 | raw_socket_login | raw-socket | pyftpdlib:permissive | env_a (127.0.0.1) | passive | 1 | 18 | 0.00 | 1 | 0 | valid |
| benign_41 | raw_socket_login | raw-socket | pyftpdlib:permissive | env_b (127.0.0.2) | passive | 1 | 18 | 0.00 | 1 | 0 | valid |
| benign_42 | raw_socket_login | raw-socket | pyftpdlib:permissive | env_c (127.0.0.3) | passive | 1 | 18 | 0.00 | 1 | 0 | valid |

## FTP-BruteForce captures

| id | scenario | client | server | env | mode | attempts | pkts | dur_s | flows | incomplete | verify |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ftpbf_01 | slow_pacing_defended_server | python-ftplib | pyftpdlib:ratelimited | env_a (127.0.0.1) | passive | 3 | 52 | 9.01 | 6 | 0 | valid |
| ftpbf_02 | slow_pacing_defended_server_active | python-ftplib | pyftpdlib:ratelimited | env_b (127.0.0.2) | active | 3 | 51 | 9.01 | 6 | 0 | valid |
| ftpbf_03 | medium_pacing_defended_server | python-ftplib | pyftpdlib:ratelimited | env_c (127.0.0.3) | passive | 5 | 87 | 15.02 | 10 | 0 | valid |
| ftpbf_04 | fast_pacing_permissive_server | python-ftplib | pyftpdlib:permissive | env_a (127.0.0.1) | passive | 16 | 251 | 0.01 | 32 | 0 | valid |
| ftpbf_05 | fast_pacing_permissive_active | python-ftplib | pyftpdlib:permissive | env_b (127.0.0.2) | active | 16 | 252 | 0.01 | 32 | 0 | valid |
| ftpbf_06 | different_usernames | python-ftplib | pyftpdlib:permissive | env_c (127.0.0.3) | passive | 10 | 154 | 0.01 | 20 | 0 | valid |
| ftpbf_07 | different_passwords_valid_user | python-ftplib | pyftpdlib:permissive | env_a (127.0.0.1) | passive | 16 | 247 | 0.01 | 32 | 0 | valid |
| ftpbf_08 | few_attempts | python-ftplib | pyftpdlib:ratelimited | env_b (127.0.0.2) | passive | 3 | 52 | 9.01 | 6 | 0 | valid |
| ftpbf_09 | many_attempts | python-ftplib | pyftpdlib:permissive | env_c (127.0.0.3) | passive | 24 | 379 | 0.01 | 48 | 0 | valid |
| ftpbf_10 | single_connection_many | python-ftplib | pyftpdlib:permissive | env_a (127.0.0.1) | passive | 16 | 76 | 0.00 | 2 | 0 | valid |
| ftpbf_11 | single_connection_active | python-ftplib | pyftpdlib:permissive | env_b (127.0.0.2) | active | 8 | 43 | 0.00 | 2 | 0 | valid |
| ftpbf_12 | multi_burst | python-ftplib | pyftpdlib:permissive | env_c (127.0.0.3) | passive | 16 | 254 | 0.01 | 32 | 0 | valid |
| ftpbf_13 | multi_burst_large | python-ftplib | pyftpdlib:permissive | env_a (127.0.0.1) | passive | 25 | 399 | 0.01 | 50 | 0 | valid |
| ftpbf_14 | repeated_connections | python-ftplib | pyftpdlib:permissive | env_b (127.0.0.2) | passive | 8 | 124 | 0.00 | 16 | 0 | valid |
| ftpbf_15 | curl_fast | curl | pyftpdlib:permissive | env_c (127.0.0.3) | passive | 8 | 104 | 0.04 | 16 | 0 | valid |
| ftpbf_16 | curl_active | curl | pyftpdlib:permissive | env_a (127.0.0.1) | active | 6 | 78 | 0.03 | 12 | 0 | valid |
| ftpbf_17 | curl_different_usernames | curl | pyftpdlib:permissive | env_b (127.0.0.2) | passive | 8 | 104 | 0.04 | 16 | 0 | valid |
| ftpbf_18 | curl_defended_server | curl | pyftpdlib:ratelimited | env_c (127.0.0.3) | passive | 3 | 42 | 9.02 | 6 | 0 | valid |
| ftpbf_19 | raw_pipelined_fast | raw-socket | pyftpdlib:permissive | env_a (127.0.0.1) | passive | 16 | 74 | 0.00 | 1 | 0 | valid |
| ftpbf_20 | raw_pipelined_usernames | raw-socket | pyftpdlib:permissive | env_b (127.0.0.2) | passive | 10 | 50 | 0.00 | 1 | 0 | valid |
| ftpbf_21 | raw_pipelined_valid_user | raw-socket | pyftpdlib:permissive | env_c (127.0.0.3) | passive | 16 | 74 | 0.00 | 1 | 0 | valid |
| ftpbf_22 | fast_pacing_throttled_server | python-ftplib | pyftpdlib:throttled | env_a (127.0.0.1) | passive | 10 | 154 | 0.00 | 20 | 0 | valid |
| ftpbf_23 | different_usernames_valid_pw_shape | python-ftplib | pyftpdlib:permissive | env_b (127.0.0.2) | active | 10 | 155 | 0.00 | 20 | 0 | valid |
| ftpbf_24 | many_attempts_mixed | python-ftplib | pyftpdlib:permissive | env_c (127.0.0.3) | passive | 25 | 394 | 0.01 | 50 | 0 | valid |
| ftpbf_25 | single_connection_usernames | python-ftplib | pyftpdlib:permissive | env_a (127.0.0.1) | passive | 10 | 51 | 0.00 | 2 | 0 | valid |
| ftpbf_26 | curl_many | curl | pyftpdlib:permissive | env_b (127.0.0.2) | passive | 12 | 156 | 0.06 | 24 | 0 | valid |
| ftpbf_27 | raw_pipelined_many | raw-socket | pyftpdlib:permissive | env_c (127.0.0.3) | passive | 20 | 90 | 0.00 | 1 | 0 | valid |
| ftpbf_28 | multi_burst_defended | python-ftplib | pyftpdlib:ratelimited | env_a (127.0.0.1) | passive | 4 | 69 | 12.02 | 8 | 0 | valid |
| ftpbf_29 | few_attempts_curl | curl | pyftpdlib:ratelimited | env_b (127.0.0.2) | active | 2 | 28 | 6.01 | 4 | 0 | valid |
| ftpbf_30 | throttled_server_usernames | python-ftplib | pyftpdlib:throttled | env_c (127.0.0.3) | passive | 8 | 126 | 0.00 | 16 | 0 | valid |
| ftpbf_31 | throttled_server_passwords | python-ftplib | pyftpdlib:throttled | env_a (127.0.0.1) | active | 8 | 126 | 0.00 | 16 | 0 | valid |
| ftpbf_32 | throttled_single_connection | python-ftplib | pyftpdlib:throttled | env_b (127.0.0.2) | passive | 10 | 51 | 0.00 | 2 | 0 | valid |
| ftpbf_33 | curl_throttled_server | curl | pyftpdlib:throttled | env_c (127.0.0.3) | passive | 6 | 78 | 0.03 | 12 | 0 | valid |
| ftpbf_34 | raw_pipelined_throttled | raw-socket | pyftpdlib:throttled | env_a (127.0.0.1) | passive | 10 | 50 | 0.00 | 1 | 0 | valid |
| ftpbf_35 | valid_user_wrong_pw_curl | curl | pyftpdlib:permissive | env_b (127.0.0.2) | passive | 8 | 104 | 0.04 | 16 | 0 | valid |
| ftpbf_36 | valid_user_wrong_pw_active | python-ftplib | pyftpdlib:permissive | env_c (127.0.0.3) | active | 10 | 151 | 0.01 | 20 | 0 | valid |
| ftpbf_37 | mixed_users_passwords_large | python-ftplib | pyftpdlib:permissive | env_a (127.0.0.1) | passive | 36 | 567 | 0.01 | 72 | 0 | valid |
| ftpbf_38 | multi_burst_curl | curl | pyftpdlib:permissive | env_b (127.0.0.2) | active | 9 | 117 | 0.05 | 18 | 0 | valid |
| ftpbf_39 | single_connection_defended | python-ftplib | pyftpdlib:ratelimited | env_c (127.0.0.3) | passive | 3 | 26 | 9.01 | 1 | 0 | valid |
| ftpbf_40 | raw_valid_user_active_env | raw-socket | pyftpdlib:permissive | env_a (127.0.0.1) | passive | 6 | 34 | 0.00 | 1 | 0 | valid |
| ftpbf_41 | fast_pacing_many_users | python-ftplib | pyftpdlib:permissive | env_b (127.0.0.2) | passive | 10 | 153 | 0.00 | 20 | 0 | valid |

## Feature extraction & validity (existing pipeline, unchanged)

Every capture was replayed through `pcap_validation.replay_pcap` (the validated
Live-Capture engine, unchanged). Each flow is checked for **exactly the
30 `ml.FEATURES`, in order, all finite**; a flow missing a
feature is **reported incomplete, never zero-filled**. See
`feature_extraction_report.csv`. The model was never used to produce or check
labels.

## Limitations

- Loopback lab only (single host); addresses vary but the network is local.
- One server package (pyftpdlib) with real config variants; no second FTP daemon.
- No FTPS/TLS (dependency unavailable).
- Still exploratory relative to real-world traffic diversity.

## Files

`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `collection_report.json`, `verify_pcaps.py`.
