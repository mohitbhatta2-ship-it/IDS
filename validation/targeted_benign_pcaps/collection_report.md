# Targeted benign real-PCAP corpus — collection report

**Collected to address the STRUCTURED benign false positives diagnosed in
`final_robustness` (real benign traffic entering the model's CIC-artifact-based FTP
region). No retraining, no model change, no threshold change.** Ground-truth labels
come only from the scenario folder. All traffic stayed inside the controlled local
lab; no external host was contacted. This corpus is content-hash **disjoint from
v1, v2, and the independent 36-PCAP test set**.

## Totals

| Metric | Value |
|---|---|
| Benign PCAPs | **41** |
| Verified valid | 41 / 41 |
| Total flows (existing pipeline) | **215** |
| Incomplete flows (reported, not zero-filled) | 0 |
| Model features per flow | 30 (exact order) |

## Diversity

- **Scenarios:** 41 distinct
- **Clients:** {"python-ftplib": 35, "curl": 3, "wget": 3}
- **Servers:** {"custom(custom-raw-socket)": 18, "permissive(pyftpdlib)": 13, "throttled(pyftpdlib)": 10}
- **Environments:** {"te8 (127.0.0.8)": 11, "te9 (127.0.0.9)": 10, "te10 (127.0.0.10)": 10, "hostip2 (192.0.2.2)": 10}
- **Mode:** {"active": 10, "passive": 31}
- **Connection patterns:** {"single": 35, "reconnect": 2, "repeated": 4}

Directly targets the FP categories: active-mode benign, command-heavy sessions
(MKD/RMD/RNFR/RNTO/DELE/APPE/SIZE/MDTM/NLST/STAT, multi-command, command-workout),
transfers (download/upload/append/delete/mixed), reconnect / multi-session.

## Leakage / disjointness — all pass: True

Content-hash disjoint from v1 (16), v2 (83), and the
independent set (36); no duplicates; labels from folders
only; all captures verified. See `leakage_validation.json`.

## Captures

| id | scenario | client | server | env | mode | pattern | pkts | dur_s | flows | incomplete | verify |
|---|---|---|---|---|---|---|---|---|---|---|---|
| benign_01 | active_download | python-ftplib | custom(custom-raw-socket) | te8 (127.0.0.8) | active | single | 24 | 0.04 | 2 | 0 | valid |
| benign_02 | active_download_pyftpdlib | python-ftplib | permissive(pyftpdlib) | te9 (127.0.0.9) | active | single | 23 | 0.00 | 2 | 0 | valid |
| benign_03 | active_download_throttled | python-ftplib | throttled(pyftpdlib) | te10 (127.0.0.10) | active | single | 23 | 0.00 | 2 | 0 | valid |
| benign_04 | active_listing | python-ftplib | custom(custom-raw-socket) | hostip2 (192.0.2.2) | active | single | 24 | 0.04 | 2 | 0 | valid |
| benign_05 | active_multi_command | python-ftplib | permissive(pyftpdlib) | te8 (127.0.0.8) | active | single | 34 | 0.00 | 2 | 0 | valid |
| benign_06 | active_upload_then_download | python-ftplib | custom(custom-raw-socket) | te9 (127.0.0.9) | active | single | 33 | 0.09 | 2 | 0 | valid |
| benign_07 | active_mixed_transfer | python-ftplib | throttled(pyftpdlib) | te10 (127.0.0.10) | active | single | 48 | 0.00 | 2 | 0 | valid |
| benign_08 | active_command_workout | python-ftplib | permissive(pyftpdlib) | hostip2 (192.0.2.2) | active | single | 64 | 0.01 | 2 | 0 | valid |
| benign_09 | mkd_rmd_session | python-ftplib | custom(custom-raw-socket) | te8 (127.0.0.8) | passive | single | 19 | 0.00 | 2 | 0 | valid |
| benign_10 | upload_rename | python-ftplib | permissive(pyftpdlib) | te9 (127.0.0.9) | passive | single | 35 | 0.00 | 4 | 0 | valid |
| benign_11 | size_mdtm_stat | python-ftplib | throttled(pyftpdlib) | te10 (127.0.0.10) | passive | single | 24 | 0.00 | 2 | 0 | valid |
| benign_12 | multi_command_session | python-ftplib | custom(custom-raw-socket) | hostip2 (192.0.2.2) | passive | single | 32 | 0.00 | 2 | 0 | valid |
| benign_13 | nlst_listing | python-ftplib | permissive(pyftpdlib) | te8 (127.0.0.8) | passive | single | 32 | 0.00 | 4 | 0 | valid |
| benign_14 | command_workout | python-ftplib | custom(custom-raw-socket) | te9 (127.0.0.9) | passive | single | 90 | 0.13 | 8 | 0 | valid |
| benign_15 | command_workout_pyftpdlib | python-ftplib | permissive(pyftpdlib) | te10 (127.0.0.10) | passive | single | 89 | 0.01 | 8 | 0 | valid |
| benign_16 | command_workout_throttled | python-ftplib | throttled(pyftpdlib) | hostip2 (192.0.2.2) | passive | single | 88 | 0.00 | 8 | 0 | valid |
| benign_17 | append_upload | python-ftplib | custom(custom-raw-socket) | te8 (127.0.0.8) | passive | single | 49 | 0.09 | 6 | 0 | valid |
| benign_18 | append_upload_throttled | python-ftplib | throttled(pyftpdlib) | te9 (127.0.0.9) | passive | single | 47 | 0.00 | 6 | 0 | valid |
| benign_19 | download_delete | python-ftplib | permissive(pyftpdlib) | te10 (127.0.0.10) | passive | single | 50 | 0.00 | 6 | 0 | valid |
| benign_20 | upload_then_download | python-ftplib | custom(custom-raw-socket) | hostip2 (192.0.2.2) | passive | single | 49 | 0.09 | 6 | 0 | valid |
| benign_21 | upload_then_download_throttled | python-ftplib | throttled(pyftpdlib) | te8 (127.0.0.8) | passive | single | 48 | 0.00 | 6 | 0 | valid |
| benign_22 | binary_download | python-ftplib | permissive(pyftpdlib) | te9 (127.0.0.9) | passive | single | 31 | 0.00 | 4 | 0 | valid |
| benign_23 | mixed_transfer | python-ftplib | custom(custom-raw-socket) | te10 (127.0.0.10) | passive | single | 83 | 0.18 | 10 | 0 | valid |
| benign_24 | mixed_transfer_throttled | python-ftplib | throttled(pyftpdlib) | hostip2 (192.0.2.2) | passive | single | 80 | 0.00 | 10 | 0 | valid |
| benign_25 | curl_download_custom | curl | custom(custom-raw-socket) | te8 (127.0.0.8) | passive | single | 39 | 0.04 | 4 | 0 | valid |
| benign_26 | curl_download_pyftpdlib | curl | permissive(pyftpdlib) | te9 (127.0.0.9) | passive | single | 35 | 0.00 | 4 | 0 | valid |
| benign_27 | curl_download_throttled | curl | throttled(pyftpdlib) | te10 (127.0.0.10) | passive | single | 35 | 0.00 | 4 | 0 | valid |
| benign_28 | wget_download_custom | wget | custom(custom-raw-socket) | hostip2 (192.0.2.2) | passive | single | 36 | 0.04 | 4 | 0 | valid |
| benign_29 | wget_download_pyftpdlib | wget | permissive(pyftpdlib) | te8 (127.0.0.8) | passive | single | 35 | 0.00 | 4 | 0 | valid |
| benign_30 | wget_download_throttled | wget | throttled(pyftpdlib) | te9 (127.0.0.9) | passive | single | 35 | 0.00 | 4 | 0 | valid |
| benign_31 | reconnect_session | python-ftplib | custom(custom-raw-socket) | te10 (127.0.0.10) | passive | reconnect | 51 | 0.04 | 6 | 0 | valid |
| benign_32 | reconnect_pyftpdlib | python-ftplib | permissive(pyftpdlib) | hostip2 (192.0.2.2) | passive | reconnect | 47 | 0.00 | 6 | 0 | valid |
| benign_33 | repeated_sessions | python-ftplib | custom(custom-raw-socket) | te8 (127.0.0.8) | passive | repeated | 120 | 0.13 | 12 | 0 | valid |
| benign_34 | repeated_sessions_pyftpdlib | python-ftplib | permissive(pyftpdlib) | te9 (127.0.0.9) | passive | repeated | 94 | 0.00 | 12 | 0 | valid |
| benign_35 | multi_session_varied | python-ftplib | custom(custom-raw-socket) | te10 (127.0.0.10) | passive | repeated | 142 | 0.13 | 15 | 0 | valid |
| benign_36 | multi_session_varied_throttled | python-ftplib | throttled(pyftpdlib) | hostip2 (192.0.2.2) | passive | repeated | 125 | 0.00 | 16 | 0 | valid |
| benign_37 | upload_rename_custom | python-ftplib | custom(custom-raw-socket) | te8 (127.0.0.8) | passive | single | 36 | 0.04 | 4 | 0 | valid |
| benign_38 | size_mdtm_custom | python-ftplib | custom(custom-raw-socket) | te9 (127.0.0.9) | passive | single | 21 | 0.00 | 2 | 0 | valid |
| benign_39 | download_delete_custom | python-ftplib | custom(custom-raw-socket) | te10 (127.0.0.10) | passive | single | 51 | 0.09 | 6 | 0 | valid |
| benign_40 | active_append | python-ftplib | custom(custom-raw-socket) | hostip2 (192.0.2.2) | active | single | 33 | 0.09 | 2 | 0 | valid |
| benign_41 | active_size_mdtm | python-ftplib | permissive(pyftpdlib) | te8 (127.0.0.8) | active | single | 24 | 0.00 | 2 | 0 | valid |

## Feature-distribution diagnostic

`feature_distribution.csv` compares the new targeted benign flows against the
independent benign and the independent FP-benign flows on the FP-driving features
(`Fwd Seg Size Min`, `Init Fwd Win Byts`, `Dst Port`, `Flow Pkts/s`, ...). This is
a read-only diagnostic to show the new corpus populates the FP-prone benign region;
it is NOT used to tune anything.

## Limitations

- Loopback / host-local lab only (single host; same-host traffic stays on lo).
- Two server implementations (custom raw-socket + pyftpdlib configs); no third-party daemon; no FTPS/TLS.
- Benign-only corpus by design (targets the benign FP problem).
- For FUTURE evaluation/retraining decisions only after a fresh independent test; not used to tune anything here.

## Files

`MANIFEST.csv`, `benign/*.pcap`, `metadata/*.json`, `feature_extraction_report.csv`,
`feature_distribution.csv`, `leakage_validation.json`, `collection_report.json`,
`verify_pcaps.py`.
