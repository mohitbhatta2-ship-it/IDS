# Third independent FTP validation corpus - collection report

**TEST-ONLY. Genuinely different stack: real proftpd server, ncftp client, non-loopback
private network (veth + ip netns 10.99.0.0/24), fresh scenario code with single-session
(packed) vs multi-session attacks. Labels from folders; hash-disjoint from all prior
corpora and the two earlier independent tests (leakage all_pass=True).**

## Totals
- PCAPs: **22** (12 benign, 10 FTP-bruteforce); verified 22/22
- Flows: 72 (incomplete: 0)
- By session structure: {"single_session": 14, "multi_session": 8}

## Captures (sessions = sessions per source)
| id | scenario | family | label | structure | client | sessions | pkts | verify |
|---|---|---|---|---|---|---|---|---|
| benign_01 | clean_login | clean | Benign | single_session | ncftp | 1 | 63 | valid |
| benign_02 | clean_login_2 | clean | Benign | single_session | ncftp | 1 | 61 | valid |
| benign_03 | transfer | activity | Benign | single_session | ncftp | 1 | 85 | valid |
| benign_04 | curl_get | activity | Benign | single_session | curl | 1 | 37 | valid |
| benign_05 | mistype_success_1 | mistype | Benign | single_session | raw-socket | 1 | 18 | valid |
| benign_06 | mistype_success_2 | mistype | Benign | single_session | raw-socket | 1 | 22 | valid |
| benign_07 | mistype_success_3 | mistype | Benign | single_session | raw-socket | 1 | 26 | valid |
| benign_08 | giveup_2 | gave_up | Benign | single_session | raw-socket | 1 | 18 | valid |
| benign_09 | giveup_3 | gave_up | Benign | single_session | raw-socket | 1 | 22 | valid |
| benign_10 | reconnect_success | reconnect | Benign | multi_session | raw-socket | 2 | 28 | valid |
| benign_11 | repeated_normal_2 | repeated | Benign | multi_session | ncftp | 2 | 122 | valid |
| benign_12 | repeated_normal_3 | repeated | Benign | multi_session | ncftp | 3 | 178 | valid |
| ftpbf_01 | single_packed_5 | single_packed | FTP-BruteForce | single_session | raw-socket | 1 | 30 | valid |
| ftpbf_02 | single_packed_8 | single_packed | FTP-BruteForce | single_session | raw-socket | 1 | 42 | valid |
| ftpbf_03 | single_packed_12 | single_packed | FTP-BruteForce | single_session | raw-socket | 1 | 45 | valid |
| ftpbf_04 | single_packed_sweep_8 | single_packed | FTP-BruteForce | single_session | raw-socket | 1 | 42 | valid |
| ftpbf_05 | single_packed_eventual_6 | eventual_success | FTP-BruteForce | single_session | raw-socket | 1 | 21 | valid |
| ftpbf_06 | multi_session_6 | multi_session | FTP-BruteForce | multi_session | raw-socket | 6 | 31 | valid |
| ftpbf_07 | multi_session_9 | multi_session | FTP-BruteForce | multi_session | raw-socket | 9 | 31 | valid |
| ftpbf_08 | credential_sweep_6 | cred_sweep | FTP-BruteForce | multi_session | raw-socket | 6 | 73 | valid |
| ftpbf_09 | slow_5 | slow | FTP-BruteForce | multi_session | raw-socket | 5 | 70 | valid |
| ftpbf_10 | multi_eventual_5 | eventual_success | FTP-BruteForce | multi_session | raw-socket | 6 | 73 | valid |

## Files
`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `leakage_validation.json`, `collection_report.json`.
