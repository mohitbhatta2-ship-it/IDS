"""
Realistic benign FAILED-LOGIN training corpus (experimental; models frozen; TRAIN-only).

The independent vsFTPD test showed the behavioural model false-positives on benign
sessions that fail to authenticate. Separability analysis showed the discriminating
*dynamics* (human pacing between attempts, password REUSE, few attempts, graceful QUIT,
and real activity after an eventual success) are absent from the earlier corpora, whose
failed logins were all back-to-back. This module collects fresh benign traffic that
contains those dynamics, so a model can learn "a human who mis-typed" vs "a brute force".

Fresh scenario code with realistic pacing (think-time between attempts, retries of the
SAME password, small attempt counts, post-success browsing/transfers). Loopback lab,
NEW addresses (127.0.0.20-22) and ports (2530/2540) -- disjoint from every prior corpus
AND from the frozen independent vsFTPD test. Reuses the pyftpdlib + custom raw-socket
servers (NOT vsftpd, to keep the vsFTPD test genuinely held-out). Ground truth is the
scenario folder, never a prediction. Nothing here changes ml.py / live_capture.py /
pcap_validation.py.
"""

from __future__ import annotations

import io
import random
import time
from dataclasses import dataclass
from ftplib import error_perm, error_temp
from pathlib import Path

from . import independent_capture as ic, robustness_capture as rc
from .independent_capture import host_eth0_ip, is_controlled_local

USER, PASSWORD = rc.USER, rc.PASSWORD
EXTRA_USERS = rc.EXTRA_USERS
ALL_USERS = rc.ALL_USERS
WRONG_PW = rc.WRONG_PW

TRAIN_PORT = {"custom": 2530, "permissive": 2540}
_RNG = random.Random(20260816)          # reproducible think-times


def train_environments() -> dict[str, tuple[str, str]]:
    envs = {"bf20": ("127.0.0.20", "lo"), "bf21": ("127.0.0.21", "lo"), "bf22": ("127.0.0.22", "lo")}
    eth0 = host_eth0_ip()
    if eth0:
        envs["bfhost"] = (eth0, "lo")
    return envs


def build_targets(root: Path) -> dict:
    targets, idx = {}, 0
    for env_key, (addr, iface) in train_environments().items():
        for server in TRAIN_PORT:
            plo = 61000 + idx * 40
            targets[(env_key, server)] = rc.RobTarget(
                addr=addr, control_port=TRAIN_PORT[server], passive_lo=plo, passive_hi=plo + 39,
                iface=iface, server=server, env_key=env_key)
            idx += 1
    return targets


def _think(lo=0.4, hi=2.5):
    time.sleep(_RNG.uniform(lo, hi))


# ---------------------------------------------------------------------------
# Realistic benign failed-login scenarios (label Benign) -- fresh, human-paced
# ---------------------------------------------------------------------------


def bfl_mistype_slow(t, passive=True, n=2):
    """Human mistypes n DISTINCT passwords with think-time, then logs in and works."""
    f = rc._connect(t, passive)
    fails = 0
    f.sendcmd(f"USER {USER}")
    for i in range(n):
        _think()
        try:
            f.sendcmd(f"PASS {WRONG_PW[i]}")
        except (error_perm, error_temp):
            fails += 1
    _think()
    f.sendcmd(f"PASS {PASSWORD}")            # eventually correct
    f.retrlines("LIST", lambda _l: None)     # post-auth activity
    buf = []
    try:
        f.retrbinary(f"RETR {ic.SAFE_FILE}", buf.append)
    except Exception:  # noqa: BLE001
        pass
    f.quit()
    return {"attempts": n + 1, "successes": 1, "failures": fails, "label_reason": "benign_mistype_slow"}


def bfl_reuse_password(t, passive=True, n=3):
    """Human retypes the SAME wrong password a few times (reuse), then the correct one."""
    f = rc._connect(t, passive)
    fails = 0
    f.sendcmd(f"USER {USER}")
    for _ in range(n):
        _think()
        try:
            f.sendcmd(f"PASS {WRONG_PW[0]}")     # same wrong password every time
        except (error_perm, error_temp):
            fails += 1
    _think()
    f.sendcmd(f"PASS {PASSWORD}")
    f.retrlines("LIST", lambda _l: None)
    f.quit()
    return {"attempts": n + 1, "successes": 1, "failures": fails, "label_reason": "benign_reuse_password"}


def bfl_giveup_graceful(t, passive=True, n=2):
    """Human fails a couple of times with think-time then gives up cleanly (QUIT)."""
    f = rc._connect(t, passive)
    fails = 0
    f.sendcmd(f"USER {USER}")
    for i in range(n):
        _think()
        try:
            f.sendcmd(f"PASS {WRONG_PW[i]}")
        except (error_perm, error_temp):
            fails += 1
    _think()
    try:
        f.sendcmd("QUIT")
    except Exception:  # noqa: BLE001
        pass
    try:
        f.close()
    except Exception:  # noqa: BLE001
        pass
    return {"attempts": n, "successes": 0, "failures": fails, "label_reason": "benign_giveup_graceful"}


def bfl_giveup_reuse(t, passive=True, n=3):
    """Human retries the same wrong password, then gives up (reuse + give-up)."""
    f = rc._connect(t, passive)
    fails = 0
    f.sendcmd(f"USER {USER}")
    for _ in range(n):
        _think()
        try:
            f.sendcmd(f"PASS {WRONG_PW[1]}")
        except (error_perm, error_temp):
            fails += 1
    _think()
    try:
        f.sendcmd("QUIT")
    except Exception:  # noqa: BLE001
        pass
    try:
        f.close()
    except Exception:  # noqa: BLE001
        pass
    return {"attempts": n, "successes": 0, "failures": fails, "label_reason": "benign_giveup_reuse"}


def bfl_typo_user_then_fix(t, passive=True):
    """Human types the wrong USERNAME, fails, then the correct user/pass, then works."""
    f = rc._connect(t, passive)
    try:
        f.sendcmd("USER labuse"); _think(); f.sendcmd(f"PASS {PASSWORD}")   # wrong username
    except (error_perm, error_temp):
        pass
    _think()
    rc._try_pass(f, USER, PASSWORD)
    f.retrlines("LIST", lambda _l: None)
    f.quit()
    return {"attempts": 2, "successes": 1, "label_reason": "benign_typo_user_slow"}


def bfl_reconnect_then_success(t, passive=True, n=2):
    """Fail a couple of times, disconnect, come back later, log in and work."""
    f = rc._connect(t, passive)
    fails = 0
    f.sendcmd(f"USER {USER}")
    for i in range(n):
        _think()
        try:
            f.sendcmd(f"PASS {WRONG_PW[i]}")
        except (error_perm, error_temp):
            fails += 1
    try:
        f.close()
    except Exception:  # noqa: BLE001
        pass
    _think(1.0, 3.0)                          # user comes back after a pause
    g = rc._connect(t, passive)
    rc._try_pass(g, USER, PASSWORD)
    g.retrlines("LIST", lambda _l: None)
    buf = []
    try:
        g.retrbinary(f"RETR {ic.SAFE_FILE}", buf.append)
    except Exception:  # noqa: BLE001
        pass
    g.quit()
    return {"attempts": n + 1, "successes": 1, "failures": fails, "label_reason": "benign_reconnect_success"}


def bfl_success_then_heavy_activity(t, passive=True):
    """One mistype, then success followed by a realistic working session."""
    f = rc._connect(t, passive)
    f.sendcmd(f"USER {USER}"); _think()
    try:
        f.sendcmd(f"PASS {WRONG_PW[0]}")
    except (error_perm, error_temp):
        pass
    _think()
    f.sendcmd(f"PASS {PASSWORD}")
    for c in ("PWD", "SYST", "TYPE I", "NOOP"):
        try:
            f.voidcmd(c) if c in ("TYPE I", "NOOP") else f.sendcmd(c)
        except Exception:  # noqa: BLE001
            pass
    buf = []
    try:
        f.retrbinary("RETR data.bin", buf.append)
        f.storbinary("STOR up_bfl.txt", io.BytesIO(b"work\n" * 40))
        f.retrlines("LIST", lambda _l: None)
    except Exception:  # noqa: BLE001
        pass
    f.quit()
    return {"attempts": 2, "successes": 1, "label_reason": "benign_mistype_then_activity"}


def bfl_multi_user_mistype(t, passive=True):
    """Two different valid users each mistype once then log in (human pacing)."""
    ok = 0
    for u, pw in (("alice", "alicepw"), ("bob", "bobpw")):
        f = rc._connect(t, passive)
        f.sendcmd(f"USER {u}"); _think()
        try:
            f.sendcmd(f"PASS {WRONG_PW[2]}")
        except (error_perm, error_temp):
            pass
        _think()
        if rc._try_pass(f, u, pw):
            ok += 1
            f.retrlines("LIST", lambda _l: None)
        try:
            f.quit()
        except Exception:  # noqa: BLE001
            f.close()
    return {"attempts": 4, "successes": ok, "distinct_users": 2, "label_reason": "benign_multiuser_mistype"}


# ---------------------------------------------------------------------------
# Scenario matrix (benign failed-login realism, varied pacing/commands)
# ---------------------------------------------------------------------------


def specs() -> list[rc.Spec]:
    envs = list(train_environments())

    def e(i):
        return envs[i % len(envs)]

    plan = [
        ("Benign", "mistype_slow_1", "mistype", "python-ftplib", "custom", "passive", lambda t: bfl_mistype_slow(t, n=1)),
        ("Benign", "mistype_slow_2", "mistype", "python-ftplib", "permissive", "passive", lambda t: bfl_mistype_slow(t, n=2)),
        ("Benign", "mistype_slow_3", "mistype", "python-ftplib", "custom", "passive", lambda t: bfl_mistype_slow(t, n=3)),
        ("Benign", "mistype_slow_2b", "mistype", "python-ftplib", "permissive", "passive", lambda t: bfl_mistype_slow(t, n=2)),
        ("Benign", "mistype_slow_active", "mistype", "python-ftplib", "custom", "active", lambda t: bfl_mistype_slow(t, passive=False, n=2)),
        ("Benign", "reuse_password_2", "mistype", "python-ftplib", "custom", "passive", lambda t: bfl_reuse_password(t, n=2)),
        ("Benign", "reuse_password_3", "mistype", "python-ftplib", "permissive", "passive", lambda t: bfl_reuse_password(t, n=3)),
        ("Benign", "reuse_password_4", "mistype", "python-ftplib", "custom", "passive", lambda t: bfl_reuse_password(t, n=4)),
        ("Benign", "giveup_graceful_1", "gave_up", "python-ftplib", "permissive", "passive", lambda t: bfl_giveup_graceful(t, n=1)),
        ("Benign", "giveup_graceful_2", "gave_up", "python-ftplib", "custom", "passive", lambda t: bfl_giveup_graceful(t, n=2)),
        ("Benign", "giveup_graceful_3", "gave_up", "python-ftplib", "permissive", "passive", lambda t: bfl_giveup_graceful(t, n=3)),
        ("Benign", "giveup_reuse_2", "gave_up", "python-ftplib", "custom", "passive", lambda t: bfl_giveup_reuse(t, n=2)),
        ("Benign", "giveup_reuse_3", "gave_up", "python-ftplib", "permissive", "passive", lambda t: bfl_giveup_reuse(t, n=3)),
        ("Benign", "giveup_reuse_4", "gave_up", "python-ftplib", "custom", "passive", lambda t: bfl_giveup_reuse(t, n=4)),
        ("Benign", "typo_user_fix", "typo_user", "python-ftplib", "custom", "passive", bfl_typo_user_then_fix),
        ("Benign", "typo_user_fix_2", "typo_user", "python-ftplib", "permissive", "passive", bfl_typo_user_then_fix),
        ("Benign", "reconnect_success_2", "reconnect", "python-ftplib", "custom", "passive", lambda t: bfl_reconnect_then_success(t, n=2)),
        ("Benign", "reconnect_success_3", "reconnect", "python-ftplib", "permissive", "passive", lambda t: bfl_reconnect_then_success(t, n=3)),
        ("Benign", "success_then_activity", "activity", "python-ftplib", "custom", "passive", bfl_success_then_heavy_activity),
        ("Benign", "success_then_activity_2", "activity", "python-ftplib", "permissive", "passive", bfl_success_then_heavy_activity),
        ("Benign", "multi_user_mistype", "multi_user", "python-ftplib", "permissive", "passive", bfl_multi_user_mistype),
        ("Benign", "mistype_slow_1b", "mistype", "python-ftplib", "custom", "passive", lambda t: bfl_mistype_slow(t, n=1)),
        ("Benign", "giveup_graceful_2b", "gave_up", "python-ftplib", "custom", "passive", lambda t: bfl_giveup_graceful(t, n=2)),
        ("Benign", "reconnect_success_2b", "reconnect", "python-ftplib", "permissive", "passive", lambda t: bfl_reconnect_then_success(t, n=2)),
    ]
    return [rc.Spec(sc, lab, fam, cl, srv, e(i), mode, fn)
            for i, (lab, sc, fam, cl, srv, mode, fn) in enumerate(plan)]
