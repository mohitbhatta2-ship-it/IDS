# Independent real-PCAP TEST corpus — collection report

**TEST-ONLY corpus for the final independent evaluation of the frozen Candidate 2.
These PCAPs were NEVER used for training, retraining, tuning, sample weighting,
threshold selection, hyperparameter/feature/candidate selection, or SHAP tuning.**
Ground-truth labels come only from the scenario/capture directory. All traffic
stayed inside the controlled local lab; no external host was contacted.

## Totals

| Metric | Value |
|---|---|
| PCAP count | **36** (18 benign, 18 FTP-bruteforce) |
| Verified valid | 36 / 36 |
| Total flows (existing pipeline) | **420** |
| Incomplete flows (reported, not zero-filled) | 0 |
| Model features per flow | 30 (exact order) |

## Diversity (genuine — new relative to v2)

- **Clients:** {"python-ftplib": 28, "curl": 4, "wget": 2, "raw-socket": 2}
- **Servers:** {"custom(custom-raw-socket)": 22, "permissive(pyftpdlib)": 11, "ratelimited(pyftpdlib)": 3} — includes a
  hand-written raw-socket FTP server, a genuinely different implementation.
- **Environments:** {"lo5 (127.0.0.5)": 12, "lo6 (127.0.0.6)": 12, "hostip (192.0.2.2)": 12}
- **Connection patterns:** {"single": 16, "repeated": 1, "reconnect": 1, "new_conn": 12, "single_conn": 4, "multi_burst": 2}
- **Distinct scenarios:** 36
- **New command sequences vs v2:** MKD/RMD, RNFR/RNTO, DELE, APPE, SIZE, MDTM,
  NLST, STAT, active-mode data, upload-then-download.

## Benign captures

| id | scenario | client | server | env | pattern | attempts | pkts | dur_s | flows | incomplete | verify |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign_01 | mkd_rmd_session | python-ftplib | custom(custom-raw-socket) | lo5 (127.0.0.5) | single | 1 | 19 | 0.00 | 2 | 0 | valid |
| benign_02 | upload_rename | python-ftplib | custom(custom-raw-socket) | lo6 (127.0.0.6) | single | 1 | 36 | 0.04 | 4 | 0 | valid |
| benign_03 | size_mdtm_stat | python-ftplib | custom(custom-raw-socket) | hostip (192.0.2.2) | single | 1 | 21 | 0.00 | 2 | 0 | valid |
| benign_04 | append_upload | python-ftplib | custom(custom-raw-socket) | lo5 (127.0.0.5) | single | 1 | 49 | 0.09 | 6 | 0 | valid |
| benign_05 | download_then_delete | python-ftplib | custom(custom-raw-socket) | lo6 (127.0.0.6) | single | 1 | 34 | 0.04 | 4 | 0 | valid |
| benign_06 | nlst_listing | python-ftplib | custom(custom-raw-socket) | hostip (192.0.2.2) | single | 1 | 38 | 0.04 | 4 | 0 | valid |
| benign_07 | multi_command_session | python-ftplib | custom(custom-raw-socket) | lo5 (127.0.0.5) | single | 1 | 31 | 0.00 | 2 | 0 | valid |
| benign_08 | active_mode_download | python-ftplib | custom(custom-raw-socket) | lo6 (127.0.0.6) | single | 1 | 24 | 0.04 | 2 | 0 | valid |
| benign_09 | binary_download | python-ftplib | custom(custom-raw-socket) | hostip (192.0.2.2) | single | 1 | 32 | 0.04 | 4 | 0 | valid |
| benign_10 | upload_then_download | python-ftplib | custom(custom-raw-socket) | lo5 (127.0.0.5) | single | 1 | 49 | 0.09 | 6 | 0 | valid |
| benign_11 | repeated_sessions | python-ftplib | permissive(pyftpdlib) | lo6 (127.0.0.6) | repeated | 3 | 94 | 0.01 | 12 | 0 | valid |
| benign_12 | reconnect_session | python-ftplib | permissive(pyftpdlib) | hostip (192.0.2.2) | reconnect | 2 | 47 | 0.00 | 6 | 0 | valid |
| benign_13 | multi_command_pyftpdlib | python-ftplib | permissive(pyftpdlib) | lo5 (127.0.0.5) | single | 1 | 34 | 0.00 | 2 | 0 | valid |
| benign_14 | curl_download_custom | curl | custom(custom-raw-socket) | lo6 (127.0.0.6) | single | 1 | 38 | 0.04 | 4 | 0 | valid |
| benign_15 | curl_download_pyftpdlib | curl | permissive(pyftpdlib) | hostip (192.0.2.2) | single | 1 | 35 | 0.00 | 4 | 0 | valid |
| benign_16 | wget_download_custom | wget | custom(custom-raw-socket) | lo5 (127.0.0.5) | single | 1 | 36 | 0.05 | 4 | 0 | valid |
| benign_17 | wget_download_pyftpdlib | wget | permissive(pyftpdlib) | lo6 (127.0.0.6) | single | 1 | 35 | 0.00 | 4 | 0 | valid |
| benign_18 | upload_then_download_pyftpdlib | python-ftplib | permissive(pyftpdlib) | hostip (192.0.2.2) | single | 1 | 47 | 0.00 | 6 | 0 | valid |

## FTP-BruteForce captures

| id | scenario | client | server | env | pattern | attempts | pkts | dur_s | flows | incomplete | verify |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ftpbf_01 | slow_defended_server | python-ftplib | ratelimited(pyftpdlib) | lo5 (127.0.0.5) | new_conn | 3 | 51 | 9.01 | 6 | 0 | valid |
| ftpbf_02 | medium_defended_server | python-ftplib | ratelimited(pyftpdlib) | lo6 (127.0.0.6) | new_conn | 5 | 86 | 15.02 | 10 | 0 | valid |
| ftpbf_03 | fast_custom_server | python-ftplib | custom(custom-raw-socket) | hostip (192.0.2.2) | new_conn | 16 | 241 | 0.01 | 32 | 0 | valid |
| ftpbf_04 | fast_custom_active | python-ftplib | custom(custom-raw-socket) | lo5 (127.0.0.5) | new_conn | 8 | 120 | 0.01 | 16 | 0 | valid |
| ftpbf_05 | single_connection_custom | python-ftplib | custom(custom-raw-socket) | lo6 (127.0.0.6) | single_conn | 16 | 75 | 0.00 | 2 | 0 | valid |
| ftpbf_06 | single_connection_pyftpdlib | python-ftplib | permissive(pyftpdlib) | hostip (192.0.2.2) | single_conn | 10 | 51 | 0.00 | 2 | 0 | valid |
| ftpbf_07 | reconnecting_bruteforce | python-ftplib | custom(custom-raw-socket) | lo5 (127.0.0.5) | new_conn | 10 | 151 | 0.01 | 20 | 0 | valid |
| ftpbf_08 | multi_burst | python-ftplib | custom(custom-raw-socket) | lo6 (127.0.0.6) | multi_burst | 16 | 240 | 0.01 | 32 | 0 | valid |
| ftpbf_09 | multi_burst_large | python-ftplib | permissive(pyftpdlib) | hostip (192.0.2.2) | multi_burst | 25 | 388 | 0.01 | 50 | 0 | valid |
| ftpbf_10 | different_usernames | python-ftplib | custom(custom-raw-socket) | lo5 (127.0.0.5) | new_conn | 10 | 150 | 0.01 | 20 | 0 | valid |
| ftpbf_11 | different_passwords | python-ftplib | custom(custom-raw-socket) | lo6 (127.0.0.6) | new_conn | 16 | 242 | 0.01 | 32 | 0 | valid |
| ftpbf_12 | valid_user_wrong_password | python-ftplib | permissive(pyftpdlib) | hostip (192.0.2.2) | new_conn | 16 | 240 | 0.01 | 32 | 0 | valid |
| ftpbf_13 | few_attempts | python-ftplib | ratelimited(pyftpdlib) | lo5 (127.0.0.5) | new_conn | 3 | 52 | 9.01 | 6 | 0 | valid |
| ftpbf_14 | many_attempts | python-ftplib | custom(custom-raw-socket) | lo6 (127.0.0.6) | new_conn | 24 | 360 | 0.02 | 48 | 0 | valid |
| ftpbf_15 | curl_bruteforce | curl | custom(custom-raw-socket) | hostip (192.0.2.2) | new_conn | 8 | 104 | 0.04 | 16 | 0 | valid |
| ftpbf_16 | curl_bruteforce_usernames | curl | permissive(pyftpdlib) | lo5 (127.0.0.5) | new_conn | 8 | 104 | 0.04 | 16 | 0 | valid |
| ftpbf_17 | raw_socket_pipelined | raw-socket | custom(custom-raw-socket) | lo6 (127.0.0.6) | single_conn | 16 | 74 | 0.00 | 1 | 0 | valid |
| ftpbf_18 | raw_socket_usernames | raw-socket | permissive(pyftpdlib) | hostip (192.0.2.2) | single_conn | 10 | 50 | 0.00 | 1 | 0 | valid |

## Honest limitations

- Single-host container: same-host traffic always traverses lo, so no genuinely separate physical interface is exercised (address varies, not iface).
- Two server implementations (custom raw-socket + pyftpdlib); no third-party daemon.
- No FTPS/TLS (pyOpenSSL unavailable).
- Loopback lab; not real multi-host network traffic.

## Files

`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `collection_report.json`, `verify_pcaps.py`.
