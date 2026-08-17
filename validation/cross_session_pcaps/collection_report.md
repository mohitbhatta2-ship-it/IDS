# Cross-session / source-level corpus - collection report

**TRAIN-only. Each capture is one source's window with MULTIPLE FTP sessions, so
cross-session features can aggregate. Loopback lab; NEW addresses (127.0.0.30-32) / ports
(2630/2640); hash-disjoint from all prior corpora and the frozen vsFTPD test (leakage
all_pass=True).**

## Totals
- PCAPs: **28** (14 benign, 14 FTP-bruteforce); verified 28/28
- Flows: 328 (incomplete: 0)
- Families: {"mistype": 4, "gave_up": 3, "reconnect": 2, "repeated": 4, "multi_user": 1, "repeated_fails": 4, "cred_sweep": 4, "slow_brute": 2, "eventual_success": 2, "bursts": 2}

## Captures (attempts column = sessions per source)
| id | scenario | family | label | sessions | server | pkts | verify |
|---|---|---|---|---|---|---|---|
| benign_01 | mistype_then_success_1 | mistype | Benign | 1 | custom(custom-raw-socket) | 38 | valid |
| benign_02 | mistype_then_success_2 | mistype | Benign | 1 | permissive(pyftpdlib) | 39 | valid |
| benign_03 | mistype_then_success_3 | mistype | Benign | 1 | custom(custom-raw-socket) | 46 | valid |
| benign_04 | giveup_single_2 | gave_up | Benign | 1 | custom(custom-raw-socket) | 19 | valid |
| benign_05 | giveup_single_3 | gave_up | Benign | 1 | permissive(pyftpdlib) | 23 | valid |
| benign_06 | giveup_single_4 | gave_up | Benign | 1 | custom(custom-raw-socket) | 27 | valid |
| benign_07 | reconnect_after_mistake | reconnect | Benign | 2 | custom(custom-raw-socket) | 53 | valid |
| benign_08 | reconnect_after_mistake_2 | reconnect | Benign | 2 | permissive(pyftpdlib) | 51 | valid |
| benign_09 | repeated_normal_2 | repeated | Benign | 2 | custom(custom-raw-socket) | 68 | valid |
| benign_10 | repeated_normal_3 | repeated | Benign | 3 | permissive(pyftpdlib) | 93 | valid |
| benign_11 | repeated_normal_4 | repeated | Benign | 4 | custom(custom-raw-socket) | 136 | valid |
| benign_12 | two_users_normal | multi_user | Benign | 2 | permissive(pyftpdlib) | 63 | valid |
| benign_13 | mixed_success_mistype | mistype | Benign | 2 | custom(custom-raw-socket) | 72 | valid |
| benign_14 | repeated_normal_3b | repeated | Benign | 3 | permissive(pyftpdlib) | 95 | valid |
| ftpbf_01 | repeated_fails_6 | repeated_fails | FTP-BruteForce | 6 | custom(custom-raw-socket) | 94 | valid |
| ftpbf_02 | repeated_fails_8 | repeated_fails | FTP-BruteForce | 8 | permissive(pyftpdlib) | 121 | valid |
| ftpbf_03 | repeated_fails_10 | repeated_fails | FTP-BruteForce | 10 | custom(custom-raw-socket) | 156 | valid |
| ftpbf_04 | credential_sweep_8 | cred_sweep | FTP-BruteForce | 8 | custom(custom-raw-socket) | 124 | valid |
| ftpbf_05 | credential_sweep_12 | cred_sweep | FTP-BruteForce | 12 | permissive(pyftpdlib) | 180 | valid |
| ftpbf_06 | slow_brute_6 | slow_brute | FTP-BruteForce | 6 | custom(custom-raw-socket) | 93 | valid |
| ftpbf_07 | slow_brute_8 | slow_brute | FTP-BruteForce | 8 | permissive(pyftpdlib) | 120 | valid |
| ftpbf_08 | eventual_success_8 | eventual_success | FTP-BruteForce | 9 | custom(custom-raw-socket) | 142 | valid |
| ftpbf_09 | eventual_success_6 | eventual_success | FTP-BruteForce | 7 | permissive(pyftpdlib) | 105 | valid |
| ftpbf_10 | bursts_3x3 | bursts | FTP-BruteForce | 9 | custom(custom-raw-socket) | 140 | valid |
| ftpbf_11 | bursts_4x2 | bursts | FTP-BruteForce | 8 | permissive(pyftpdlib) | 121 | valid |
| ftpbf_12 | multiuser_persistent_9 | cred_sweep | FTP-BruteForce | 9 | custom(custom-raw-socket) | 177 | valid |
| ftpbf_13 | repeated_fails_7 | repeated_fails | FTP-BruteForce | 7 | permissive(pyftpdlib) | 105 | valid |
| ftpbf_14 | credential_sweep_10 | cred_sweep | FTP-BruteForce | 10 | custom(custom-raw-socket) | 159 | valid |

## Files
`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `leakage_validation.json`, `collection_report.json`.
