# Independent FTP validation corpus - collection report

**TEST-ONLY. Genuinely different stack: real vsFTPd server, lftp client, non-loopback
private network (veth + ip netns 10.77.0.0/24), plus separate FTPS/TLS.** Fresh
scenario code; labels from the scenario folder only; hash-disjoint from v1/v2/
independent/targeted/robustness/robust_train (leakage all_pass=True).

## Totals
- PCAPs: **28** (16 benign, 12 FTP-bruteforce; 4 FTPS/encrypted); verified 28/28
- Flows: 162 (incomplete: 0)

## What makes this independent
- **Server**: vsFTPd (real production daemon) - never used in any prior corpus (which used pyftpdlib + a bespoke raw-socket server).
- **Client**: lftp (new), plus curl / python-ftplib / raw-socket.
- **Network**: a real veth pair to an isolated `ip netns`; traffic traverses `veth-ivh` (non-loopback), addresses in 10.77.0.0/24.
- **FTPS/TLS**: separate encrypted captures - the cleartext behavioural features are genuinely unavailable.

## Captures
| id | scenario | family | label | client | server | ftps | pkts | verify |
|---|---|---|---|---|---|---|---|---|
| benign_01 | clean_login_ls | clean | Benign | lftp | vsftpd(plaintext) | no | 45 | valid |
| benign_02 | one_wrong_then_success | mistype | Benign | python-ftplib | vsftpd(plaintext) | no | 48 | valid |
| benign_03 | two_wrong_then_success | mistype | Benign | python-ftplib | vsftpd(plaintext) | no | 16 | valid |
| benign_04 | three_wrong_then_success | mistype | Benign | python-ftplib | vsftpd(plaintext) | no | 57 | valid |
| benign_05 | several_wrong_giveup_3 | gave_up | Benign | raw-socket | vsftpd(plaintext) | no | 26 | valid |
| benign_06 | several_wrong_giveup_5 | gave_up | Benign | raw-socket | vsftpd(plaintext) | no | 49 | valid |
| benign_07 | file_transfer | activity | Benign | lftp | vsftpd(plaintext) | no | 75 | valid |
| benign_08 | command_heavy | activity | Benign | python-ftplib | vsftpd(plaintext) | no | 58 | valid |
| benign_09 | reconnect_session | reconnect | Benign | lftp | vsftpd(plaintext) | no | 92 | valid |
| benign_10 | curl_get | activity | Benign | curl | vsftpd(plaintext) | no | 35 | valid |
| benign_11 | multi_user | multi_user | Benign | python-ftplib | vsftpd(plaintext) | no | 62 | valid |
| benign_12 | active_transfer | activity | Benign | python-ftplib | vsftpd(plaintext) | no | 23 | valid |
| benign_13 | clean_login_ls_2 | clean | Benign | lftp | vsftpd(plaintext) | no | 42 | valid |
| benign_14 | file_transfer_2 | activity | Benign | lftp | vsftpd(plaintext) | no | 73 | valid |
| ftpbf_01 | slow_brute_6 | slow_brute | FTP-BruteForce | raw-socket | vsftpd(plaintext) | no | 64 | valid |
| ftpbf_02 | slow_brute_8 | slow_brute | FTP-BruteForce | raw-socket | vsftpd(plaintext) | no | 91 | valid |
| ftpbf_03 | fast_brute_12 | fast_brute | FTP-BruteForce | python-ftplib | vsftpd(plaintext) | no | 104 | valid |
| ftpbf_04 | fast_brute_16 | fast_brute | FTP-BruteForce | python-ftplib | vsftpd(plaintext) | no | 140 | valid |
| ftpbf_05 | different_usernames | user_enum | FTP-BruteForce | raw-socket | vsftpd(plaintext) | no | 68 | valid |
| ftpbf_06 | eventual_success_6 | eventual_success | FTP-BruteForce | python-ftplib | vsftpd(plaintext) | no | 48 | valid |
| ftpbf_07 | eventual_success_10 | eventual_success | FTP-BruteForce | python-ftplib | vsftpd(plaintext) | no | 99 | valid |
| ftpbf_08 | multi_conn_brute_8 | multi_conn | FTP-BruteForce | lftp | vsftpd(plaintext) | no | 248 | valid |
| ftpbf_09 | multi_conn_brute_6 | multi_conn | FTP-BruteForce | lftp | vsftpd(plaintext) | no | 188 | valid |
| ftpbf_10 | curl_brute | fast_brute | FTP-BruteForce | curl | vsftpd(plaintext) | no | 80 | valid |
| benign_15 | ftps_login_get | tls_benign | Benign | curl-ftps | vsftpd(tls) | yes | 59 | valid |
| benign_16 | ftps_login_get_2 | tls_benign | Benign | curl-ftps | vsftpd(tls) | yes | 60 | valid |
| ftpbf_11 | ftps_brute_5 | tls_brute | FTP-BruteForce | curl-ftps | vsftpd(tls) | yes | 131 | valid |
| ftpbf_12 | ftps_brute_7 | tls_brute | FTP-BruteForce | curl-ftps | vsftpd(tls) | yes | 182 | valid |

## Files
`MANIFEST.csv`, `benign/*.pcap`, `ftp_bruteforce/*.pcap`, `metadata/*.json`,
`feature_extraction_report.csv`, `leakage_validation.json`, `collection_report.json`.
