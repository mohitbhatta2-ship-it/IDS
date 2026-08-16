# Extended-behavioural separability analysis

**Approved training corpora only (v1/v2/targeted/robust_train). Scenario metadata is
used only to GROUP rows; no feature is built from a label or a prediction. The frozen
independent vsFTPD corpus is NOT used here.**

Groups (from scenario metadata + measured auth outcome): benign_fail
(19), brute
(59), attacker_success
(8), benign_clean
(102).

## Single-feature separability (direction-agnostic AUC)

| new | feature | AUC benign-fail vs brute | AUC benign-fail vs attacker-success | benign-fail med | brute med | attacker-succ med |
|---|---|---|---|---|---|---|
|  | `ftp_user_commands` | 0.986 | 1.000 | 1.000 | 8.000 | 5.000 |
|  | `ftp_failed_logins` | 0.972 | 0.961 | 1.000 | 8.000 | 4.000 |
|  | `ftp_error_responses_5xx` | 0.948 | 0.928 | 2.000 | 8.000 | 4.000 |
|  | `ftp_total_commands` | 0.938 | 0.947 | 5.000 | 21.000 | 12.500 |
|  | `ftp_login_attempts` | 0.927 | 0.980 | 2.000 | 8.000 | 5.000 |
|  | `ftp_successful_logins` | 0.867 | 1.000 | 1.000 | 0.000 | 1.000 |
|  | `ftp_has_successful_auth` | 0.867 | 1.000 | 1.000 | 0.000 | 1.000 |
| NEW | `ftpx_fail_run_before_success` | 0.867 | 0.980 | 1.000 | 0.000 | 4.000 |
| NEW | `ftpx_post_auth_commands` | 0.867 | 0.579 | 1.000 | 0.000 | 1.000 |
|  | `ftp_reconnects` | 0.813 | 0.809 | 0.000 | 5.000 | 0.000 |
|  | `ftp_control_connections` | 0.813 | 0.809 | 1.000 | 6.000 | 1.000 |
| NEW | `ftpx_post_auth_data_transfers` | 0.785 | 0.579 | 0.000 | 0.000 | 0.000 |

## Key finding

New EXT features most separating benign-fail from brute: ftpx_fail_run_before_success(AUC 0.87), ftpx_post_auth_commands(AUC 0.87), ftpx_post_auth_data_transfers(AUC 0.79), ftpx_interattempt_min_s(AUC 0.78), ftpx_interattempt_mean_s(AUC 0.77). Inter-attempt TIMING is NOT usable on the existing corpora (attempts were back-to-back), so a realistic-pacing benign failed-login training corpus is needed to activate it.

- Best NEW features (benign-fail vs brute): ftpx_fail_run_before_success (0.87), ftpx_post_auth_commands (0.87), ftpx_post_auth_data_transfers (0.79), ftpx_interattempt_min_s (0.78), ftpx_interattempt_mean_s (0.77)
- Timing usable in current corpora: **False** ->
  recommend collecting realistic-pacing benign failed-login data: **True**

## Files
`separability_auc.csv`, `per_group_feature_stats.csv`, `separability_summary.json`.
