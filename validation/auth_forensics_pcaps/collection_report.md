# Auth-forensics training corpus - collection report

**TRAIN-only. Benign typo-then-success / typo-give-up / alternate-password vs single-session
dictionary brute force (incl. dictionary-then-success). Loopback lab; NEW addresses
(127.0.0.50-52) / ports (2830/2840); hash-disjoint from all prior corpora and every
independent test (leakage all_pass=True).**

## Totals
- PCAPs: **28** (16 benign, 12 FTP-bruteforce); verified 28/28
- Flows: 46 (incomplete: 0)
- Families: {"mistype": 5, "gave_up": 4, "alt_pw": 2, "clean": 2, "repeated": 2, "reconnect": 1, "single_dict": 6, "dict_success": 4, "multi_dict": 2}

## Captures
| id | scenario | family | label | pkts | verify |
|---|---|---|---|---|---|
| benign_01 | typo_then_success_1 | mistype | Benign | 22 | valid |
| benign_02 | typo_then_success_2 | mistype | Benign | 26 | valid |
| benign_03 | typo_then_success_3 | mistype | Benign | 32 | valid |
| benign_04 | typo_then_success_2b | mistype | Benign | 26 | valid |
| benign_05 | typo_give_up_2 | gave_up | Benign | 11 | valid |
| benign_06 | typo_give_up_3 | gave_up | Benign | 26 | valid |
| benign_07 | typo_give_up_4 | gave_up | Benign | 32 | valid |
| benign_08 | alt_then_success_1 | alt_pw | Benign | 20 | valid |
| benign_09 | alt_then_success_2 | alt_pw | Benign | 26 | valid |
| benign_10 | clean_1 | clean | Benign | 14 | valid |
| benign_11 | clean_2 | clean | Benign | 14 | valid |
| benign_12 | repeated_2 | repeated | Benign | 28 | valid |
| benign_13 | repeated_3 | repeated | Benign | 44 | valid |
| benign_14 | reconnect_success | reconnect | Benign | 28 | valid |
| benign_15 | typo_then_success_1b | mistype | Benign | 20 | valid |
| benign_16 | typo_give_up_2b | gave_up | Benign | 20 | valid |
| ftpbf_01 | dict_fail_6 | single_dict | FTP-BruteForce | 34 | valid |
| ftpbf_02 | dict_fail_10 | single_dict | FTP-BruteForce | 50 | valid |
| ftpbf_03 | dict_fail_14 | single_dict | FTP-BruteForce | 66 | valid |
| ftpbf_04 | dict_then_success_6 | dict_success | FTP-BruteForce | 38 | valid |
| ftpbf_05 | dict_then_success_10 | dict_success | FTP-BruteForce | 54 | valid |
| ftpbf_06 | dict_then_success_4 | dict_success | FTP-BruteForce | 30 | valid |
| ftpbf_07 | dict_sweep_8 | single_dict | FTP-BruteForce | 42 | valid |
| ftpbf_08 | dict_sweep_12 | single_dict | FTP-BruteForce | 58 | valid |
| ftpbf_09 | dict_multi_5 | multi_dict | FTP-BruteForce | 70 | valid |
| ftpbf_10 | dict_multi_7 | multi_dict | FTP-BruteForce | 98 | valid |
| ftpbf_11 | dict_fail_8 | single_dict | FTP-BruteForce | 42 | valid |
| ftpbf_12 | dict_then_success_8 | dict_success | FTP-BruteForce | 46 | valid |

## Files
`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `leakage_validation.json`, `collection_report.json`.
