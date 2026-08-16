"""
MESSY *training* corpus capture (experimental; production models frozen).

Collects a NEW, deliberately MESSY real FTP corpus intended for **training** the
45-feature behavioural candidate so it stops treating "any failed login -> attack"
as a shortcut. It contains the ambiguous families the earlier CLEAN corpora never
had, with **overlapping failed-login counts across the two classes**:

  * benign users who mistype 1-3 passwords then log in successfully;
  * benign users who fail 1-3 times then give up (never authenticate);
  * benign users who fail, disconnect, reconnect and succeed;
  * attackers who fail a few / many times and give up;
  * attackers who fail then EVENTUALLY guess a correct password;
  * different users / passwords, different clients, passive + active mode,
    command-heavy and transfer sessions.

So both classes carry sessions with 1..N failed logins -- the model must learn the
behavioural context (sustained failure, no-eventual-success, attempt rate) rather
than the raw failed-login count.

This corpus is DISJOINT from the frozen 34-PCAP messy TEST corpus and every earlier
corpus: NEW addresses (127.0.0.14-16 / eth0) and NEW ports (2430/2440). Ground truth
is the scenario folder, never a prediction. It reuses the scenario functions,
servers and verifier from ``robustness_capture`` unchanged; nothing here touches
ml.py / live_capture.py / pcap_validation.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from ftplib import error_perm, error_temp
from pathlib import Path

from . import independent_capture as ic, robustness_capture as rc
from .independent_capture import host_eth0_ip, is_controlled_local

# Reuse the same user/password universe so rc's scenario functions work unchanged.
USER, PASSWORD = rc.USER, rc.PASSWORD
EXTRA_USERS = rc.EXTRA_USERS
ALL_USERS = rc.ALL_USERS
WRONG_PW = rc.WRONG_PW

# NEW ports (disjoint from the test corpus's 2330/2340).
TRAIN_PORT = {"custom": 2430, "permissive": 2440}


def train_environments() -> dict[str, tuple[str, str]]:
    """NEW addresses (127.0.0.14-16), disjoint from the test corpus (127.0.0.11-13)."""
    envs = {"rt14": ("127.0.0.14", "lo"), "rt15": ("127.0.0.15", "lo"),
            "rt16": ("127.0.0.16", "lo")}
    eth0 = host_eth0_ip()
    if eth0:
        envs["rthost"] = (eth0, "lo")
    return envs


def build_targets(root: Path) -> dict:
    """RobTarget matrix on NEW addresses/ports; passive ranges disjoint from rc."""
    targets, idx = {}, 0
    for env_key, (addr, iface) in train_environments().items():
        for server in TRAIN_PORT:
            plo = 62000 + idx * 40           # distinct base from rc (64000)
            targets[(env_key, server)] = rc.RobTarget(
                addr=addr, control_port=TRAIN_PORT[server], passive_lo=plo,
                passive_hi=plo + 39, iface=iface, server=server, env_key=env_key)
            idx += 1
    return targets


# ---------------------------------------------------------------------------
# A few user-diverse wrappers (different user/password) on top of rc primitives.
# ---------------------------------------------------------------------------


def b_success_as(t, user, pw, passive=True):
    """Benign clean login as an arbitrary valid user, with light activity."""
    f = rc._connect(t, passive)
    rc._try_pass(f, user, pw)
    f.retrlines("LIST", lambda _l: None)
    try:
        f.quit()
    except Exception:  # noqa: BLE001
        f.close()
    return {"attempts": 1, "successes": 1, "user": user, "label_reason": "benign_success_user"}


def b_mistype_as(t, user, pw, passive=True, n=1):
    """Benign user (arbitrary) mistypes n passwords then logs in."""
    f = rc._connect(t, passive)
    fails = 0
    f.sendcmd(f"USER {user}")
    for i in range(n):
        try:
            f.sendcmd(f"PASS {WRONG_PW[i]}")
        except (error_perm, error_temp):
            fails += 1
    try:
        f.sendcmd(f"PASS {pw}")
        f.retrlines("LIST", lambda _l: None)
    except (error_perm, error_temp):
        pass
    try:
        f.quit()
    except Exception:  # noqa: BLE001
        f.close()
    return {"attempts": n + 1, "successes": 1, "failed_before_success": fails,
            "user": user, "label_reason": "benign_mistype_user"}


def b_gaveup_as(t, user, passive=True, n=2):
    """Benign user (arbitrary) fails n times then gives up (never authenticates)."""
    f = rc._connect(t, passive)
    f.sendcmd(f"USER {user}")
    fails = 0
    for i in range(n):
        try:
            f.sendcmd(f"PASS {WRONG_PW[i]}")
        except (error_perm, error_temp):
            fails += 1
    try:
        f.sendcmd("QUIT")
    except Exception:  # noqa: BLE001
        pass
    try:
        f.close()
    except Exception:  # noqa: BLE001
        pass
    return {"attempts": n, "successes": 0, "failures": fails, "user": user,
            "label_reason": "benign_gave_up_user"}


def b_fail_reconnect_success(t, passive=True, n=2):
    """Benign: fail n times, disconnect, reconnect on a fresh connection and succeed."""
    f = rc._connect(t, passive)
    f.sendcmd(f"USER {USER}")
    fails = 0
    for i in range(n):
        try:
            f.sendcmd(f"PASS {WRONG_PW[i]}")
        except (error_perm, error_temp):
            fails += 1
    try:
        f.close()
    except Exception:  # noqa: BLE001
        pass
    g = rc._connect(t, passive)
    rc._try_pass(g, USER, PASSWORD)
    g.retrlines("LIST", lambda _l: None)
    try:
        g.quit()
    except Exception:  # noqa: BLE001
        g.close()
    return {"attempts": n + 1, "successes": 1, "failures": fails,
            "label_reason": "benign_fail_reconnect_success"}


# ---------------------------------------------------------------------------
# Training scenario matrix (messy benign + messy brute force, overlapping fails)
# ---------------------------------------------------------------------------


@dataclass
class Spec(rc.Spec):
    pass


def specs() -> list[rc.Spec]:
    envs = list(train_environments())

    def e(i):
        return envs[i % len(envs)]

    plan = [
        # ---------------- messy BENIGN (many with 1..3 failed logins) -----------
        ("Benign", "clean_success", "clean", "python-ftplib", "custom", "passive", lambda t: rc.b_mixed_activity(t)),
        ("Benign", "clean_success_pyftpd", "clean", "python-ftplib", "permissive", "passive", lambda t: rc.b_transfer(t)),
        ("Benign", "clean_success_alice", "clean", "python-ftplib", "custom", "passive", lambda t: b_success_as(t, "alice", "alicepw")),
        ("Benign", "clean_success_bob", "clean", "python-ftplib", "permissive", "passive", lambda t: b_success_as(t, "bob", "bobpw")),
        ("Benign", "one_wrong_then_success", "mistype", "python-ftplib", "custom", "passive", lambda t: rc.b_n_wrong_then_success(t, n=1)),
        ("Benign", "two_wrong_then_success", "mistype", "python-ftplib", "permissive", "passive", lambda t: rc.b_n_wrong_then_success(t, n=2)),
        ("Benign", "three_wrong_then_success", "mistype", "python-ftplib", "custom", "passive", lambda t: rc.b_n_wrong_then_success(t, n=3)),
        ("Benign", "one_wrong_then_success_alice", "mistype", "python-ftplib", "permissive", "passive", lambda t: b_mistype_as(t, "alice", "alicepw", n=1)),
        ("Benign", "two_wrong_then_success_bob", "mistype", "python-ftplib", "custom", "passive", lambda t: b_mistype_as(t, "bob", "bobpw", n=2)),
        ("Benign", "two_wrong_then_success_active", "mistype", "python-ftplib", "custom", "active", lambda t: rc.b_n_wrong_then_success(t, passive=False, n=2)),
        ("Benign", "three_wrong_then_success_pyftpd", "mistype", "python-ftplib", "permissive", "passive", lambda t: rc.b_n_wrong_then_success(t, n=3)),
        ("Benign", "failed_then_disconnect_1", "gave_up", "python-ftplib", "custom", "passive", lambda t: rc.b_failed_then_disconnect(t, n=1)),
        ("Benign", "failed_then_disconnect_2", "gave_up", "python-ftplib", "permissive", "passive", lambda t: rc.b_failed_then_disconnect(t, n=2)),
        ("Benign", "failed_then_disconnect_3", "gave_up", "python-ftplib", "custom", "passive", lambda t: rc.b_failed_then_disconnect(t, n=3)),
        ("Benign", "gave_up_alice_2", "gave_up", "python-ftplib", "permissive", "passive", lambda t: b_gaveup_as(t, "alice", n=2)),
        ("Benign", "gave_up_bob_3", "gave_up", "python-ftplib", "custom", "passive", lambda t: b_gaveup_as(t, "bob", n=3)),
        ("Benign", "fail_reconnect_success", "mistype", "python-ftplib", "permissive", "passive", lambda t: b_fail_reconnect_success(t, n=2)),
        ("Benign", "fail_reconnect_success_3", "mistype", "python-ftplib", "custom", "passive", lambda t: b_fail_reconnect_success(t, n=3)),
        ("Benign", "multi_user_session", "multi_user", "python-ftplib", "custom", "passive", rc.b_multi_user),
        ("Benign", "multi_user_session_pyftpd", "multi_user", "python-ftplib", "permissive", "passive", rc.b_multi_user),
        ("Benign", "wrong_user_then_correct", "typo_user", "python-ftplib", "custom", "passive", rc.b_wrong_user_then_correct),
        ("Benign", "wrong_user_then_correct_pyftpd", "typo_user", "python-ftplib", "permissive", "passive", rc.b_wrong_user_then_correct),
        ("Benign", "mixed_activity", "activity", "python-ftplib", "permissive", "passive", rc.b_mixed_activity),
        ("Benign", "transfer_session", "activity", "python-ftplib", "custom", "passive", rc.b_transfer),
        ("Benign", "command_heavy", "activity", "python-ftplib", "permissive", "passive", rc.b_command_heavy),
        ("Benign", "reconnect_session", "reconnect", "python-ftplib", "custom", "passive", rc.b_reconnect),
        ("Benign", "active_success", "activity", "python-ftplib", "permissive", "active", rc.b_active_success),
        ("Benign", "interrupted_no_auth", "incomplete", "python-ftplib", "custom", "passive", rc.b_interrupted),
        ("Benign", "curl_benign", "activity", "curl", "custom", "passive", rc.b_curl_benign),
        ("Benign", "wget_benign", "activity", "wget", "permissive", "passive", rc.b_wget_benign),
        # ---------------- messy BRUTE FORCE (many with the SAME 1..N fails) ------
        ("FTP-BruteForce", "all_fail_4", "all_fail", "python-ftplib", "custom", "passive", lambda t: rc.bf_all_fail(t, n=4)),
        ("FTP-BruteForce", "all_fail_8", "all_fail", "python-ftplib", "permissive", "passive", lambda t: rc.bf_all_fail(t, n=8)),
        ("FTP-BruteForce", "all_fail_2", "all_fail", "python-ftplib", "custom", "passive", lambda t: rc.bf_all_fail(t, n=2)),
        ("FTP-BruteForce", "all_fail_1", "all_fail", "python-ftplib", "permissive", "passive", lambda t: rc.bf_all_fail(t, n=1)),
        ("FTP-BruteForce", "eventual_success_6", "eventual_success", "python-ftplib", "custom", "passive", lambda t: rc.bf_eventual_success(t, n_wrong=6)),
        ("FTP-BruteForce", "eventual_success_3", "eventual_success", "python-ftplib", "permissive", "passive", lambda t: rc.bf_eventual_success(t, n_wrong=3)),
        ("FTP-BruteForce", "eventual_success_2", "eventual_success", "python-ftplib", "custom", "passive", lambda t: rc.bf_eventual_success(t, n_wrong=2)),
        ("FTP-BruteForce", "eventual_success_active", "eventual_success", "python-ftplib", "permissive", "active", lambda t: rc.bf_eventual_success(t, passive=False, n_wrong=4)),
        ("FTP-BruteForce", "eventual_success_reconnect", "eventual_success", "python-ftplib", "custom", "passive", lambda t: rc.bf_eventual_success_newconn(t, n_wrong=6)),
        ("FTP-BruteForce", "eventual_success_reconnect_3", "eventual_success", "python-ftplib", "permissive", "passive", lambda t: rc.bf_eventual_success_newconn(t, n_wrong=3)),
        ("FTP-BruteForce", "multiuser_eventual_success", "eventual_success", "python-ftplib", "custom", "passive", rc.bf_multiuser_eventual),
        ("FTP-BruteForce", "different_users_fail", "all_fail", "python-ftplib", "permissive", "passive", rc.bf_different_users_fail),
        ("FTP-BruteForce", "different_users_fail_custom", "all_fail", "python-ftplib", "custom", "passive", rc.bf_different_users_fail),
        ("FTP-BruteForce", "single_conn_many", "all_fail", "python-ftplib", "permissive", "passive", lambda t: rc.bf_single_conn_many(t, n=10)),
        ("FTP-BruteForce", "single_conn_many_custom", "all_fail", "python-ftplib", "custom", "passive", lambda t: rc.bf_single_conn_many(t, n=6)),
        ("FTP-BruteForce", "curl_bruteforce", "all_fail", "curl", "custom", "passive", rc.bf_curl),
        ("FTP-BruteForce", "raw_bruteforce", "all_fail", "raw-socket", "permissive", "passive", rc.bf_raw),
        ("FTP-BruteForce", "eventual_success_raw", "eventual_success", "raw-socket", "custom", "passive", lambda t: rc.bf_eventual_success(t, n_wrong=5)),
    ]
    return [rc.Spec(sc, lab, fam, cl, srv, e(i), mode, fn)
            for i, (lab, sc, fam, cl, srv, mode, fn) in enumerate(plan)]
