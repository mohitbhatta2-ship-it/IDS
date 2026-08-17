# Per-connection training corpus - collection report

**TRAIN-only. Attacks across the full session-structure spectrum incl. SINGLE-SESSION
PACKED brute force, plus benign mistypes/give-ups/normal. Loopback lab; NEW addresses
(127.0.0.40-42) / ports (2730/2740); hash-disjoint from all prior corpora and every
independent test (leakage all_pass=True).**

## Totals
- PCAPs: **32** (16 benign, 16 FTP-bruteforce); verified 32/32
- Flows: 84 (incomplete: 0)
- Families: {"single_packed": 6, "eventual_success": 2, "multi_session": 6, "reconnecting": 1, "slow": 1, "clean": 3, "mistype": 4, "gave_up": 3, "reconnect": 2, "repeated": 3, "multi_user": 1}

## Captures (attempts col = sessions per source)
| id | scenario | family | label | sessions | server | pkts | verify |
|---|---|---|---|---|---|---|---|
| ftpbf_01 | single_packed_5 | single_packed | FTP-BruteForce | 1 | custom(custom-raw-socket) | 30 | valid |
| ftpbf_02 | single_packed_8 | single_packed | FTP-BruteForce | 1 | custom(custom-raw-socket) | 42 | valid |
| ftpbf_03 | single_packed_12 | single_packed | FTP-BruteForce | 1 | custom(custom-raw-socket) | 58 | valid |
| ftpbf_04 | single_packed_sweep_8 | single_packed | FTP-BruteForce | 1 | custom(custom-raw-socket) | 42 | valid |
| ftpbf_05 | fast_packed_10 | single_packed | FTP-BruteForce | 1 | custom(custom-raw-socket) | 50 | valid |
| ftpbf_06 | single_packed_eventual_6 | eventual_success | FTP-BruteForce | 1 | custom(custom-raw-socket) | 38 | valid |
| ftpbf_07 | two_three_session_3 | multi_session | FTP-BruteForce | 3 | custom(custom-raw-socket) | 54 | valid |
| ftpbf_08 | two_three_session_2 | multi_session | FTP-BruteForce | 2 | custom(custom-raw-socket) | 36 | valid |
| ftpbf_09 | four_six_session_5 | multi_session | FTP-BruteForce | 5 | custom(custom-raw-socket) | 70 | valid |
| ftpbf_10 | four_six_session_6 | multi_session | FTP-BruteForce | 6 | custom(custom-raw-socket) | 84 | valid |
| ftpbf_11 | seven_plus_session_9 | multi_session | FTP-BruteForce | 9 | custom(custom-raw-socket) | 126 | valid |
| ftpbf_12 | seven_plus_session_11 | multi_session | FTP-BruteForce | 11 | custom(custom-raw-socket) | 154 | valid |
| ftpbf_13 | reconnecting_6 | reconnecting | FTP-BruteForce | 6 | custom(custom-raw-socket) | 84 | valid |
| ftpbf_14 | slow_4 | slow | FTP-BruteForce | 4 | custom(custom-raw-socket) | 56 | valid |
| ftpbf_15 | multi_eventual_5 | eventual_success | FTP-BruteForce | 6 | custom(custom-raw-socket) | 84 | valid |
| ftpbf_16 | single_packed_6b | single_packed | FTP-BruteForce | 1 | custom(custom-raw-socket) | 34 | valid |
| benign_01 | clean_1 | clean | Benign | 1 | custom(custom-raw-socket) | 14 | valid |
| benign_02 | clean_2 | clean | Benign | 1 | custom(custom-raw-socket) | 14 | valid |
| benign_03 | mistype_success_1 | mistype | Benign | 1 | custom(custom-raw-socket) | 18 | valid |
| benign_04 | mistype_success_2 | mistype | Benign | 1 | custom(custom-raw-socket) | 22 | valid |
| benign_05 | mistype_success_3 | mistype | Benign | 1 | custom(custom-raw-socket) | 26 | valid |
| benign_06 | giveup_2 | gave_up | Benign | 1 | custom(custom-raw-socket) | 18 | valid |
| benign_07 | giveup_3 | gave_up | Benign | 1 | custom(custom-raw-socket) | 22 | valid |
| benign_08 | reconnect_success_1 | reconnect | Benign | 2 | custom(custom-raw-socket) | 28 | valid |
| benign_09 | reconnect_success_2 | reconnect | Benign | 2 | custom(custom-raw-socket) | 28 | valid |
| benign_10 | repeated_normal_2 | repeated | Benign | 2 | custom(custom-raw-socket) | 28 | valid |
| benign_11 | repeated_normal_3 | repeated | Benign | 3 | custom(custom-raw-socket) | 42 | valid |
| benign_12 | repeated_normal_4 | repeated | Benign | 4 | custom(custom-raw-socket) | 56 | valid |
| benign_13 | two_users | multi_user | Benign | 2 | custom(custom-raw-socket) | 28 | valid |
| benign_14 | clean_3 | clean | Benign | 1 | custom(custom-raw-socket) | 14 | valid |
| benign_15 | mistype_success_1b | mistype | Benign | 1 | custom(custom-raw-socket) | 18 | valid |
| benign_16 | giveup_2b | gave_up | Benign | 1 | custom(custom-raw-socket) | 18 | valid |

## Files
`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `leakage_validation.json`, `collection_report.json`.
