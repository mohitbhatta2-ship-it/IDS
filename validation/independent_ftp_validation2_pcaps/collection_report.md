# Second independent FTP validation corpus - collection report

**TEST-ONLY. Genuinely different stack: real pure-ftpd server, ncftp client, non-loopback
private network (veth + ip netns 10.88.0.0/24), plus FTPS. Fresh scenario code with
single- vs multi-session structure. Labels from folders; hash-disjoint from all prior
corpora and the first vsFTPD test (leakage all_pass=True).**

## Totals
- PCAPs: **25** (14 benign, 11 FTP-bruteforce; 3 FTPS); verified 25/25
- Flows: 98 (incomplete: 0)
- By session structure: {"single_session": 12, "multi_session": 13}

## Captures (sessions = sessions per source)
| id | scenario | family | label | structure | client | ftps | sessions | pkts | verify |
|---|---|---|---|---|---|---|---|---|---|
| benign_01 | clean_login | clean | Benign | single_session | ncftp | no | 1 | 41 | valid |
| benign_02 | clean_login_2 | clean | Benign | single_session | ncftp | no | 1 | 41 | valid |
| benign_03 | transfer | activity | Benign | single_session | ncftp | no | 1 | 58 | valid |
| benign_04 | curl_get | activity | Benign | single_session | curl | no | 1 | 39 | valid |
| benign_05 | mistype_success_1 | mistype | Benign | single_session | raw-socket | no | 1 | 22 | valid |
| benign_06 | mistype_success_2 | mistype | Benign | single_session | raw-socket | no | 1 | 28 | valid |
| benign_07 | giveup_2 | gave_up | Benign | single_session | raw-socket | no | 1 | 22 | valid |
| benign_08 | giveup_3 | gave_up | Benign | single_session | raw-socket | no | 1 | 22 | valid |
| benign_09 | reconnect_after_mistake | reconnect | Benign | multi_session | raw-socket | no | 2 | 32 | valid |
| benign_10 | repeated_normal_2 | repeated | Benign | multi_session | ncftp | no | 2 | 82 | valid |
| benign_11 | repeated_normal_3 | repeated | Benign | multi_session | ncftp | no | 3 | 121 | valid |
| benign_12 | multi_user | multi_user | Benign | multi_session | raw-socket | no | 2 | 32 | valid |
| ftpbf_01 | single_session_brute_3 | single_session_brute | FTP-BruteForce | single_session | raw-socket | no | 1 | 28 | valid |
| ftpbf_02 | single_session_brute_4 | single_session_brute | FTP-BruteForce | single_session | raw-socket | no | 1 | 34 | valid |
| ftpbf_03 | single_session_curl | single_session_brute | FTP-BruteForce | multi_session | curl | no | 3 | 42 | valid |
| ftpbf_04 | multi_session_brute_6 | multi_session_brute | FTP-BruteForce | multi_session | raw-socket | no | 6 | 96 | valid |
| ftpbf_05 | multi_session_brute_8 | multi_session_brute | FTP-BruteForce | multi_session | raw-socket | no | 8 | 128 | valid |
| ftpbf_06 | credential_sweep_6 | cred_sweep | FTP-BruteForce | multi_session | raw-socket | no | 6 | 91 | valid |
| ftpbf_07 | slow_brute_5 | slow_brute | FTP-BruteForce | multi_session | raw-socket | no | 5 | 80 | valid |
| ftpbf_08 | eventual_success_5 | eventual_success | FTP-BruteForce | multi_session | raw-socket | no | 6 | 96 | valid |
| ftpbf_09 | eventual_success_4 | eventual_success | FTP-BruteForce | multi_session | raw-socket | no | 5 | 58 | valid |
| ftpbf_10 | reconnect_bursts | bursts | FTP-BruteForce | multi_session | raw-socket | no | 6 | 96 | valid |
| benign_13 | ftps_login | tls_benign | Benign | single_session | curl-ftps | yes | 1 | 61 | valid |
| benign_14 | ftps_login_2 | tls_benign | Benign | single_session | curl-ftps | yes | 1 | 63 | valid |
| ftpbf_11 | ftps_brute_3 | tls_brute | FTP-BruteForce | multi_session | curl-ftps | yes | 3 | 76 | valid |

## Files
`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `leakage_validation.json`, `collection_report.json`.
