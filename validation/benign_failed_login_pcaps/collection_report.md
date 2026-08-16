# Realistic benign failed-login corpus - collection report

**TRAIN-only. Benign FTP traffic with realistic *dynamics* (think-time pacing, password
REUSE, small attempt counts, graceful QUIT, post-success activity) to teach the model
"human who mis-typed" vs "brute force". Loopback lab; NEW addresses (127.0.0.20-22) /
ports (2530/2540); hash-disjoint from all prior corpora and the frozen independent
vsFTPD test (leakage all_pass=True).**

## Totals
- PCAPs: **24** (all Benign); verified 24/24
- Flows: 93 (incomplete: 0)
- Families: {"mistype": 9, "gave_up": 7, "typo_user": 2, "reconnect": 3, "activity": 2, "multi_user": 1}

## Captures
| id | scenario | family | server | mode | pkts | dur(s) | verify |
|---|---|---|---|---|---|---|---|
| benign_01 | mistype_slow_1 | mistype | custom(custom-raw-socket) | passive | 56 | 3.6993 | valid |
| benign_02 | mistype_slow_2 | mistype | permissive(pyftpdlib) | passive | 21 | 5.5884 | valid |
| benign_03 | mistype_slow_3 | mistype | custom(custom-raw-socket) | passive | 62 | 6.755 | valid |
| benign_04 | mistype_slow_2b | mistype | permissive(pyftpdlib) | passive | 21 | 4.6471 | valid |
| benign_05 | mistype_slow_active | mistype | custom(custom-raw-socket) | active | 41 | 3.9231 | valid |
| benign_06 | reuse_password_2 | mistype | custom(custom-raw-socket) | passive | 42 | 3.7955 | valid |
| benign_07 | reuse_password_3 | mistype | permissive(pyftpdlib) | passive | 24 | 7.935 | valid |
| benign_08 | reuse_password_4 | mistype | custom(custom-raw-socket) | passive | 48 | 6.5593 | valid |
| benign_09 | giveup_graceful_1 | gave_up | permissive(pyftpdlib) | passive | 18 | 2.8682 | valid |
| benign_10 | giveup_graceful_2 | gave_up | custom(custom-raw-socket) | passive | 21 | 2.7277 | valid |
| benign_11 | giveup_graceful_3 | gave_up | permissive(pyftpdlib) | passive | 25 | 5.6811 | valid |
| benign_12 | giveup_reuse_2 | gave_up | custom(custom-raw-socket) | passive | 15 | 2.7417 | valid |
| benign_13 | giveup_reuse_3 | gave_up | permissive(pyftpdlib) | passive | 25 | 5.8781 | valid |
| benign_14 | giveup_reuse_4 | gave_up | custom(custom-raw-socket) | passive | 21 | 4.2051 | valid |
| benign_15 | typo_user_fix | typo_user | custom(custom-raw-socket) | passive | 12 | 0.7217 | valid |
| benign_16 | typo_user_fix_2 | typo_user | permissive(pyftpdlib) | passive | 38 | 3.0459 | valid |
| benign_17 | reconnect_success_2 | reconnect | custom(custom-raw-socket) | passive | 69 | 5.007 | valid |
| benign_18 | reconnect_success_3 | reconnect | permissive(pyftpdlib) | passive | 69 | 4.8294 | valid |
| benign_19 | success_then_activity | activity | custom(custom-raw-socket) | passive | 83 | 3.5274 | valid |
| benign_20 | success_then_activity_2 | activity | permissive(pyftpdlib) | passive | 18 | 2.6204 | valid |
| benign_21 | multi_user_mistype | multi_user | permissive(pyftpdlib) | passive | 77 | 5.5548 | valid |
| benign_22 | mistype_slow_1b | mistype | custom(custom-raw-socket) | passive | 58 | 3.0512 | valid |
| benign_23 | giveup_graceful_2b | gave_up | custom(custom-raw-socket) | passive | 21 | 5.0241 | valid |
| benign_24 | reconnect_success_2b | reconnect | permissive(pyftpdlib) | passive | 66 | 5.4211 | valid |

## Files
`MANIFEST.csv`, `benign/*.pcap`, `metadata/*.json`, `feature_extraction_report.csv`,
`leakage_validation.json`, `collection_report.json`.
