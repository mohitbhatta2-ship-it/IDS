# Robustness stress-test corpus - collection report

**TEST-ONLY. Messy real FTP traffic to stress-test the behavioural model.** Labels
from the scenario folder only. Controlled local lab; hash-disjoint from v1/v2/
independent/targeted (leakage all_pass=True).

## Totals
- PCAPs: **34** (20 benign, 14 FTP-bruteforce); verified 34/34
- Flows: 170 (incomplete: 0)

## Scenario families (the stress axes)
{
"mistype": 5,
"gave_up": 2,
"multi_user": 2,
"typo_user": 1,
"activity": 7,
"reconnect": 1,
"incomplete": 2,
"all_fail": 8,
"eventual_success": 6
}

Adversarial families: benign **mistype** (fails then succeeds), benign **gave_up**
(fails then quits -> looks like brute force), **eventual_success** (attack guesses a
correct password -> looks partially benign), **incomplete** (no auth evidence),
**multi_user**.

## Captures
| id | scenario | family | label | client | server | mode | pkts | verify |
|---|---|---|---|---|---|---|---|---|
| benign_01 | one_wrong_then_success | mistype | Benign | python-ftplib | custom(custom-raw-socket) | passive | 36 | valid |
| benign_02 | two_wrong_then_success | mistype | Benign | python-ftplib | permissive(pyftpdlib) | passive | 17 | valid |
| benign_03 | three_wrong_then_success | mistype | Benign | python-ftplib | custom(custom-raw-socket) | passive | 40 | valid |
| benign_04 | two_wrong_then_success_active | mistype | Benign | python-ftplib | custom(custom-raw-socket) | active | 28 | valid |
| benign_05 | failed_then_disconnect | gave_up | Benign | python-ftplib | custom(custom-raw-socket) | passive | 17 | valid |
| benign_06 | failed_then_disconnect_3 | gave_up | Benign | python-ftplib | permissive(pyftpdlib) | passive | 19 | valid |
| benign_07 | multi_user_session | multi_user | Benign | python-ftplib | custom(custom-raw-socket) | passive | 102 | valid |
| benign_08 | multi_user_session_pyftpd | multi_user | Benign | python-ftplib | permissive(pyftpdlib) | passive | 95 | valid |
| benign_09 | wrong_user_then_correct | typo_user | Benign | python-ftplib | custom(custom-raw-socket) | passive | 38 | valid |
| benign_10 | mixed_activity | activity | Benign | python-ftplib | custom(custom-raw-socket) | passive | 78 | valid |
| benign_11 | mixed_activity_pyftpd | activity | Benign | python-ftplib | permissive(pyftpdlib) | passive | 72 | valid |
| benign_12 | transfer_session | activity | Benign | python-ftplib | custom(custom-raw-socket) | passive | 49 | valid |
| benign_13 | command_heavy | activity | Benign | python-ftplib | custom(custom-raw-socket) | passive | 27 | valid |
| benign_14 | reconnect_session | reconnect | Benign | python-ftplib | permissive(pyftpdlib) | passive | 63 | valid |
| benign_15 | active_success | activity | Benign | python-ftplib | custom(custom-raw-socket) | active | 24 | valid |
| benign_16 | interrupted_no_auth | incomplete | Benign | python-ftplib | custom(custom-raw-socket) | passive | 11 | valid |
| benign_17 | interrupted_no_auth_pyftpd | incomplete | Benign | python-ftplib | permissive(pyftpdlib) | passive | 11 | valid |
| benign_18 | curl_benign | activity | Benign | curl | custom(custom-raw-socket) | passive | 38 | valid |
| benign_19 | wget_benign | activity | Benign | wget | permissive(pyftpdlib) | passive | 35 | valid |
| benign_20 | one_wrong_then_success_pyftpd | mistype | Benign | python-ftplib | permissive(pyftpdlib) | passive | 15 | valid |
| ftpbf_01 | all_fail | all_fail | FTP-BruteForce | python-ftplib | custom(custom-raw-socket) | passive | 43 | valid |
| ftpbf_02 | all_fail_pyftpd | all_fail | FTP-BruteForce | python-ftplib | permissive(pyftpdlib) | passive | 44 | valid |
| ftpbf_03 | eventual_success | eventual_success | FTP-BruteForce | python-ftplib | custom(custom-raw-socket) | passive | 39 | valid |
| ftpbf_04 | eventual_success_pyftpd | eventual_success | FTP-BruteForce | python-ftplib | permissive(pyftpdlib) | passive | 39 | valid |
| ftpbf_05 | eventual_success_active | eventual_success | FTP-BruteForce | python-ftplib | custom(custom-raw-socket) | active | 31 | valid |
| ftpbf_06 | eventual_success_reconnect | eventual_success | FTP-BruteForce | python-ftplib | custom(custom-raw-socket) | passive | 105 | valid |
| ftpbf_07 | multiuser_eventual_success | eventual_success | FTP-BruteForce | python-ftplib | permissive(pyftpdlib) | passive | 77 | valid |
| ftpbf_08 | different_users_fail | all_fail | FTP-BruteForce | python-ftplib | custom(custom-raw-socket) | passive | 92 | valid |
| ftpbf_09 | single_conn_many | all_fail | FTP-BruteForce | python-ftplib | custom(custom-raw-socket) | passive | 51 | valid |
| ftpbf_10 | single_conn_many_pyftpd | all_fail | FTP-BruteForce | python-ftplib | permissive(pyftpdlib) | passive | 51 | valid |
| ftpbf_11 | reconnecting_fail | all_fail | FTP-BruteForce | python-ftplib | permissive(pyftpdlib) | passive | 92 | valid |
| ftpbf_12 | curl_bruteforce | all_fail | FTP-BruteForce | curl | custom(custom-raw-socket) | passive | 78 | valid |
| ftpbf_13 | raw_bruteforce | all_fail | FTP-BruteForce | raw-socket | custom(custom-raw-socket) | passive | 41 | valid |
| ftpbf_14 | eventual_success_raw | eventual_success | FTP-BruteForce | raw-socket | permissive(pyftpdlib) | passive | 35 | valid |

## Files
`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `leakage_validation.json`, `collection_report.json`.
