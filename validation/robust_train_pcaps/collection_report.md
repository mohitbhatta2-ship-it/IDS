# Messy TRAINING corpus - collection report

**For TRAINING the behavioural candidate.** Messy real FTP traffic with **overlapping
failed-login counts across both classes** so the model cannot use raw failed-login
count as a label shortcut. Labels from the scenario folder only. Controlled local
lab; NEW addresses (127.0.0.14-16)/ports (2430/2440); hash-disjoint from v1/v2/
independent/targeted **and the frozen 34-PCAP TEST corpus** (leakage all_pass=True).

## Totals
- PCAPs: **48** (30 benign, 18 FTP-bruteforce); verified 48/48
- Flows: 224 (incomplete: 0)

## Scenario families
{
"clean": 4,
"mistype": 9,
"gave_up": 5,
"multi_user": 2,
"typo_user": 2,
"activity": 6,
"reconnect": 1,
"incomplete": 1,
"all_fail": 10,
"eventual_success": 8
}

Benign families include **mistype** (fails then succeeds), **gave_up** (fails then
quits), **multi_user**, **typo_user**, **activity**, **reconnect**, **incomplete**.
Brute-force families include **all_fail** (1/2/4/8 failures) and **eventual_success**
(attacker eventually guesses right). The 1..N failed-login counts overlap the benign
mistype/gave-up sessions on purpose.

## Captures
| id | scenario | family | label | client | server | mode | pkts | verify |
|---|---|---|---|---|---|---|---|---|
| benign_01 | clean_success | clean | Benign | python-ftplib | custom(custom-raw-socket) | passive | 78 | valid |
| benign_02 | clean_success_pyftpd | clean | Benign | python-ftplib | permissive(pyftpdlib) | passive | 47 | valid |
| benign_03 | clean_success_alice | clean | Benign | python-ftplib | custom(custom-raw-socket) | passive | 36 | valid |
| benign_04 | clean_success_bob | clean | Benign | python-ftplib | permissive(pyftpdlib) | passive | 31 | valid |
| benign_05 | one_wrong_then_success | mistype | Benign | python-ftplib | custom(custom-raw-socket) | passive | 40 | valid |
| benign_06 | two_wrong_then_success | mistype | Benign | python-ftplib | permissive(pyftpdlib) | passive | 17 | valid |
| benign_07 | three_wrong_then_success | mistype | Benign | python-ftplib | custom(custom-raw-socket) | passive | 44 | valid |
| benign_08 | one_wrong_then_success_alice | mistype | Benign | python-ftplib | permissive(pyftpdlib) | passive | 17 | valid |
| benign_09 | two_wrong_then_success_bob | mistype | Benign | python-ftplib | custom(custom-raw-socket) | passive | 42 | valid |
| benign_10 | two_wrong_then_success_active | mistype | Benign | python-ftplib | custom(custom-raw-socket) | active | 28 | valid |
| benign_11 | three_wrong_then_success_pyftpd | mistype | Benign | python-ftplib | permissive(pyftpdlib) | passive | 19 | valid |
| benign_12 | failed_then_disconnect_1 | gave_up | Benign | python-ftplib | custom(custom-raw-socket) | passive | 15 | valid |
| benign_13 | failed_then_disconnect_2 | gave_up | Benign | python-ftplib | permissive(pyftpdlib) | passive | 17 | valid |
| benign_14 | failed_then_disconnect_3 | gave_up | Benign | python-ftplib | custom(custom-raw-socket) | passive | 19 | valid |
| benign_15 | gave_up_alice_2 | gave_up | Benign | python-ftplib | permissive(pyftpdlib) | passive | 17 | valid |
| benign_16 | gave_up_bob_3 | gave_up | Benign | python-ftplib | custom(custom-raw-socket) | passive | 19 | valid |
| benign_17 | fail_reconnect_success | mistype | Benign | python-ftplib | permissive(pyftpdlib) | passive | 46 | valid |
| benign_18 | fail_reconnect_success_3 | mistype | Benign | python-ftplib | custom(custom-raw-socket) | passive | 55 | valid |
| benign_19 | multi_user_session | multi_user | Benign | python-ftplib | custom(custom-raw-socket) | passive | 114 | valid |
| benign_20 | multi_user_session_pyftpd | multi_user | Benign | python-ftplib | permissive(pyftpdlib) | passive | 93 | valid |
| benign_21 | wrong_user_then_correct | typo_user | Benign | python-ftplib | custom(custom-raw-socket) | passive | 42 | valid |
| benign_22 | wrong_user_then_correct_pyftpd | typo_user | Benign | python-ftplib | permissive(pyftpdlib) | passive | 35 | valid |
| benign_23 | mixed_activity | activity | Benign | python-ftplib | permissive(pyftpdlib) | passive | 71 | valid |
| benign_24 | transfer_session | activity | Benign | python-ftplib | custom(custom-raw-socket) | passive | 49 | valid |
| benign_25 | command_heavy | activity | Benign | python-ftplib | permissive(pyftpdlib) | passive | 30 | valid |
| benign_26 | reconnect_session | reconnect | Benign | python-ftplib | custom(custom-raw-socket) | passive | 76 | valid |
| benign_27 | active_success | activity | Benign | python-ftplib | permissive(pyftpdlib) | active | 24 | valid |
| benign_28 | interrupted_no_auth | incomplete | Benign | python-ftplib | custom(custom-raw-socket) | passive | 11 | valid |
| benign_29 | curl_benign | activity | Benign | curl | custom(custom-raw-socket) | passive | 39 | valid |
| benign_30 | wget_benign | activity | Benign | wget | permissive(pyftpdlib) | passive | 35 | valid |
| ftpbf_01 | all_fail_4 | all_fail | FTP-BruteForce | python-ftplib | custom(custom-raw-socket) | passive | 27 | valid |
| ftpbf_02 | all_fail_8 | all_fail | FTP-BruteForce | python-ftplib | permissive(pyftpdlib) | passive | 43 | valid |
| ftpbf_03 | all_fail_2 | all_fail | FTP-BruteForce | python-ftplib | custom(custom-raw-socket) | passive | 19 | valid |
| ftpbf_04 | all_fail_1 | all_fail | FTP-BruteForce | python-ftplib | permissive(pyftpdlib) | passive | 16 | valid |
| ftpbf_05 | eventual_success_6 | eventual_success | FTP-BruteForce | python-ftplib | custom(custom-raw-socket) | passive | 39 | valid |
| ftpbf_06 | eventual_success_3 | eventual_success | FTP-BruteForce | python-ftplib | permissive(pyftpdlib) | passive | 27 | valid |
| ftpbf_07 | eventual_success_2 | eventual_success | FTP-BruteForce | python-ftplib | custom(custom-raw-socket) | passive | 23 | valid |
| ftpbf_08 | eventual_success_active | eventual_success | FTP-BruteForce | python-ftplib | permissive(pyftpdlib) | active | 31 | valid |
| ftpbf_09 | eventual_success_reconnect | eventual_success | FTP-BruteForce | python-ftplib | custom(custom-raw-socket) | passive | 105 | valid |
| ftpbf_10 | eventual_success_reconnect_3 | eventual_success | FTP-BruteForce | python-ftplib | permissive(pyftpdlib) | passive | 60 | valid |
| ftpbf_11 | multiuser_eventual_success | eventual_success | FTP-BruteForce | python-ftplib | custom(custom-raw-socket) | passive | 75 | valid |
| ftpbf_12 | different_users_fail | all_fail | FTP-BruteForce | python-ftplib | permissive(pyftpdlib) | passive | 91 | valid |
| ftpbf_13 | different_users_fail_custom | all_fail | FTP-BruteForce | python-ftplib | custom(custom-raw-socket) | passive | 90 | valid |
| ftpbf_14 | single_conn_many | all_fail | FTP-BruteForce | python-ftplib | permissive(pyftpdlib) | passive | 51 | valid |
| ftpbf_15 | single_conn_many_custom | all_fail | FTP-BruteForce | python-ftplib | custom(custom-raw-socket) | passive | 35 | valid |
| ftpbf_16 | curl_bruteforce | all_fail | FTP-BruteForce | curl | custom(custom-raw-socket) | passive | 78 | valid |
| ftpbf_17 | raw_bruteforce | all_fail | FTP-BruteForce | raw-socket | permissive(pyftpdlib) | passive | 41 | valid |
| ftpbf_18 | eventual_success_raw | eventual_success | FTP-BruteForce | raw-socket | custom(custom-raw-socket) | passive | 35 | valid |

## Files
`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `leakage_validation.json`, `collection_report.json`.
