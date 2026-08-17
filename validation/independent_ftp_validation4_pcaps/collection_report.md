# Fourth independent FTP validation corpus - collection report

**TEST-ONLY. Benign typo-then-success / typo-give-up vs single-session dictionary brute
force (incl. dictionary-then-success). pure-ftpd server, NEW netns/subnet 10.111.0.0/24 /
port 2525; hash-disjoint from all prior corpora and every earlier independent test
(leakage all_pass=True).**

## Totals
- PCAPs: **17** (10 benign, 7 FTP-bruteforce); verified 17/17
- Flows: 34 (incomplete: 0)
- Families: {"clean": 2, "repeated": 1, "mistype": 4, "gave_up": 3, "single_dict": 3, "dict_success": 3, "multi_dict": 1}

## Captures
| id | scenario | family | label | structure | pkts | verify |
|---|---|---|---|---|---|---|
| benign_01 | clean_login | clean | Benign | single_session | 41 | valid |
| benign_02 | clean_login_2 | clean | Benign | single_session | 41 | valid |
| benign_03 | repeated_2 | repeated | Benign | multi_session | 82 | valid |
| benign_04 | typo_then_success_1 | mistype | Benign | single_session | 24 | valid |
| benign_05 | typo_then_success_2 | mistype | Benign | single_session | 32 | valid |
| benign_06 | typo_then_success_3 | mistype | Benign | single_session | 38 | valid |
| benign_07 | typo_then_success_2b | mistype | Benign | single_session | 30 | valid |
| benign_08 | typo_give_up_2 | gave_up | Benign | single_session | 24 | valid |
| benign_09 | typo_give_up_3 | gave_up | Benign | single_session | 26 | valid |
| benign_10 | typo_give_up_4 | gave_up | Benign | single_session | 40 | valid |
| ftpbf_01 | dict_fail_6 | single_dict | FTP-BruteForce | single_session | 40 | valid |
| ftpbf_02 | dict_fail_10 | single_dict | FTP-BruteForce | single_session | 44 | valid |
| ftpbf_03 | dict_then_success_5 | dict_success | FTP-BruteForce | single_session | 46 | valid |
| ftpbf_04 | dict_then_success_8 | dict_success | FTP-BruteForce | single_session | 44 | valid |
| ftpbf_05 | dict_then_success_4 | dict_success | FTP-BruteForce | single_session | 40 | valid |
| ftpbf_06 | dict_sweep_6 | single_dict | FTP-BruteForce | single_session | 45 | valid |
| ftpbf_07 | dict_multi_5 | multi_dict | FTP-BruteForce | multi_session | 80 | valid |

## Files
`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `leakage_validation.json`, `collection_report.json`.
