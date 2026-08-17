"""
Per-connection training corpus (experimental; models frozen; TRAIN-only).

Collects attacks across the FULL session-structure spectrum -- crucially including
SINGLE-SESSION PACKED brute force (many attempts in ONE connection) that the cross-session
corpus lacked -- so a model can learn per-connection brute-force behaviour and detect
attacks regardless of how many connections the attacker uses. Also benign mistypes,
give-ups and normal repeated sessions.

Fresh scenario code. Loopback lab, custom raw-socket server (allows many attempts per
connection -> packed single-session attacks) + pyftpdlib, NEW addresses (127.0.0.40-42) /
ports (2730/2740), hash-disjoint from every prior corpus AND every independent test.
Labels from the scenario folder. Nothing here changes ml.py / live_capture.py /
pcap_validation.py / ftp_behavioral.py.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path

from . import robustness_capture as rc
from .independent_capture import host_eth0_ip, is_controlled_local

USER, PASSWORD = rc.USER, rc.PASSWORD
WRONG_PW = rc.WRONG_PW + ["hunter2", "welcome", "abc123", "iloveyou", "sunshine", "princess"]
WRONG_USERS = ["admin", "root", "oracle", "ftp", "guest", "test", "www", "mysql", "user", "ubuntu"]
TRAIN_PORT = {"custom": 2730, "permissive": 2740}


def train_environments() -> dict[str, tuple[str, str]]:
    # loopback-only, three addresses; the custom raw-socket server handles every scenario
    # (incl. packed single-session attacks). pyftpdlib is not used here -- raw-socket packed
    # attacks against it are unreliable -- so we keep a single robust server implementation.
    return {"pc40": ("127.0.0.40", "lo"), "pc41": ("127.0.0.41", "lo"), "pc42": ("127.0.0.42", "lo")}


def build_targets(root: Path) -> dict:
    targets, idx = {}, 0
    for env_key, (addr, iface) in train_environments().items():
        plo = 59000 + idx * 40
        targets[(env_key, "custom")] = rc.RobTarget(
            addr=addr, control_port=TRAIN_PORT["custom"], passive_lo=plo, passive_hi=plo + 39,
            iface=iface, server="custom", env_key=env_key)
        idx += 1
    return targets


def _raw_conn(t, creds, timeout=10.0):
    """One connection, send all (user,pw); return (ok, fails). Custom server allows many attempts."""
    ok = fail = 0
    try:
        s = socket.create_connection((t.addr, t.control_port), timeout=timeout); s.settimeout(timeout); s.recv(4096)
    except OSError:
        return 0, 0
    for user, pw in creds:
        try:
            s.sendall(f"USER {user}\r\n".encode()); s.recv(4096)
            s.sendall(f"PASS {pw}\r\n".encode()); resp = s.recv(4096)
            if resp[:3] == b"230":
                ok += 1
            else:
                fail += 1
        except OSError:
            break
    try:
        s.sendall(b"QUIT\r\n"); s.close()
    except OSError:
        pass
    return ok, fail


# ---------------------------------------------------------------------------
# ATTACKS across the full session-structure spectrum
# ---------------------------------------------------------------------------


def at_single_packed(t, n=6):
    """SINGLE-SESSION PACKED: one connection, n failed attempts (the key training case)."""
    _o, f = _raw_conn(t, [(USER, WRONG_PW[i % len(WRONG_PW)]) for i in range(n)])
    return {"sessions": 1, "failures": f, "label_reason": "attack_single_packed"}


def at_single_packed_sweep(t, n=8):
    """Single connection sweeping usernames+passwords (packed credential sweep)."""
    _o, f = _raw_conn(t, [(WRONG_USERS[i % len(WRONG_USERS)], WRONG_PW[i % len(WRONG_PW)]) for i in range(n)])
    return {"sessions": 1, "failures": f, "label_reason": "attack_single_packed_sweep"}


def at_single_packed_eventual(t, n=6):
    """Single connection: n fails then the correct password (packed eventual success)."""
    creds = [(USER, WRONG_PW[i % len(WRONG_PW)]) for i in range(n)] + [(USER, PASSWORD)]
    o, f = _raw_conn(t, creds)
    return {"sessions": 1, "successes": o, "failures": f, "label_reason": "attack_single_packed_eventual"}


def at_two_three_session(t, sessions=3, per=2):
    total_f = 0
    for i in range(sessions):
        _o, f = _raw_conn(t, [(USER, WRONG_PW[(i * per + j) % len(WRONG_PW)]) for j in range(per)]); total_f += f
    return {"sessions": sessions, "failures": total_f, "label_reason": "attack_2_3_session"}


def at_four_six_session(t, sessions=5):
    total_f = 0
    for i in range(sessions):
        _o, f = _raw_conn(t, [(USER, WRONG_PW[i % len(WRONG_PW)])]); total_f += f
    return {"sessions": sessions, "failures": total_f, "label_reason": "attack_4_6_session"}


def at_seven_plus_session(t, sessions=9):
    total_f = 0
    for i in range(sessions):
        _o, f = _raw_conn(t, [(USER, WRONG_PW[i % len(WRONG_PW)])]); total_f += f
    return {"sessions": sessions, "failures": total_f, "label_reason": "attack_7plus_session"}


def at_reconnecting(t, sessions=6):
    total_f = 0
    for i in range(sessions):
        _o, f = _raw_conn(t, [(WRONG_USERS[i % len(WRONG_USERS)], WRONG_PW[i % len(WRONG_PW)])]); total_f += f
    return {"sessions": sessions, "failures": total_f, "label_reason": "attack_reconnecting"}


def at_slow(t, sessions=4, delay=0.6):
    total_f = 0
    for i in range(sessions):
        _o, f = _raw_conn(t, [(USER, WRONG_PW[i % len(WRONG_PW)])]); total_f += f
        time.sleep(delay)
    return {"sessions": sessions, "failures": total_f, "label_reason": "attack_slow"}


def at_multi_eventual(t, n_wrong=5):
    total_f = 0
    for i in range(n_wrong):
        _o, f = _raw_conn(t, [(USER, WRONG_PW[i % len(WRONG_PW)])]); total_f += f
    o, _f = _raw_conn(t, [(USER, PASSWORD)])
    return {"sessions": n_wrong + 1, "successes": o, "failures": total_f, "label_reason": "attack_multi_eventual"}


def at_fast_packed(t, n=10):
    """Fast packed single-session: many attempts back-to-back in one connection."""
    _o, f = _raw_conn(t, [(USER, WRONG_PW[i % len(WRONG_PW)]) for i in range(n)])
    return {"sessions": 1, "failures": f, "label_reason": "attack_fast_packed"}


# ---------------------------------------------------------------------------
# BENIGN
# ---------------------------------------------------------------------------


def bn_clean(t):
    o, _f = _raw_conn(t, [(USER, PASSWORD)]); return {"sessions": 1, "successes": o, "label_reason": "benign_clean"}


def bn_mistype_success(t, n=2):
    creds = [(USER, WRONG_PW[i]) for i in range(n)] + [(USER, PASSWORD)]
    o, f = _raw_conn(t, creds)
    return {"sessions": 1, "successes": o, "failures": f, "label_reason": "benign_mistype_success"}


def bn_giveup(t, n=2):
    _o, f = _raw_conn(t, [(USER, WRONG_PW[i]) for i in range(n)])
    return {"sessions": 1, "failures": f, "label_reason": "benign_giveup"}


def bn_reconnect_success(t):
    _o1, f1 = _raw_conn(t, [(USER, WRONG_PW[0])])
    o2, _f2 = _raw_conn(t, [(USER, PASSWORD)])
    return {"sessions": 2, "successes": o2, "failures": f1, "label_reason": "benign_reconnect_success"}


def bn_repeated_normal(t, n=3):
    ok = 0
    for _ in range(n):
        o, _f = _raw_conn(t, [(USER, PASSWORD)]); ok += o
        time.sleep(0.3)
    return {"sessions": n, "successes": ok, "label_reason": "benign_repeated_normal"}


def bn_two_users(t):
    ok = 0
    for u, pw in (("alice", "alicepw"), ("bob", "bobpw")):
        o, _f = _raw_conn(t, [(u, pw)]); ok += o
    return {"sessions": 2, "successes": ok, "label_reason": "benign_two_users"}


# ---------------------------------------------------------------------------
# Scenario matrix (custom server allows packed single-session attacks)
# ---------------------------------------------------------------------------


def specs() -> list[rc.Spec]:
    envs = list(train_environments())

    def e(i):
        return envs[i % len(envs)]

    plan = [
        # ---- attacks: full session-structure spectrum (custom server for packed) ----
        ("FTP-BruteForce", "single_packed_5", "single_packed", "raw-socket", "custom", "passive", lambda t: at_single_packed(t, 5)),
        ("FTP-BruteForce", "single_packed_8", "single_packed", "raw-socket", "custom", "passive", lambda t: at_single_packed(t, 8)),
        ("FTP-BruteForce", "single_packed_12", "single_packed", "raw-socket", "custom", "passive", lambda t: at_single_packed(t, 12)),
        ("FTP-BruteForce", "single_packed_sweep_8", "single_packed", "raw-socket", "custom", "passive", lambda t: at_single_packed_sweep(t, 8)),
        ("FTP-BruteForce", "fast_packed_10", "single_packed", "raw-socket", "custom", "passive", lambda t: at_fast_packed(t, 10)),
        ("FTP-BruteForce", "single_packed_eventual_6", "eventual_success", "raw-socket", "custom", "passive", lambda t: at_single_packed_eventual(t, 6)),
        ("FTP-BruteForce", "two_three_session_3", "multi_session", "raw-socket", "custom", "passive", lambda t: at_two_three_session(t, 3, 2)),
        ("FTP-BruteForce", "two_three_session_2", "multi_session", "raw-socket", "custom", "passive", lambda t: at_two_three_session(t, 2, 2)),
        ("FTP-BruteForce", "four_six_session_5", "multi_session", "raw-socket", "custom", "passive", lambda t: at_four_six_session(t, 5)),
        ("FTP-BruteForce", "four_six_session_6", "multi_session", "raw-socket", "custom", "passive", lambda t: at_four_six_session(t, 6)),
        ("FTP-BruteForce", "seven_plus_session_9", "multi_session", "raw-socket", "custom", "passive", lambda t: at_seven_plus_session(t, 9)),
        ("FTP-BruteForce", "seven_plus_session_11", "multi_session", "raw-socket", "custom", "passive", lambda t: at_seven_plus_session(t, 11)),
        ("FTP-BruteForce", "reconnecting_6", "reconnecting", "raw-socket", "custom", "passive", lambda t: at_reconnecting(t, 6)),
        ("FTP-BruteForce", "slow_4", "slow", "raw-socket", "custom", "passive", lambda t: at_slow(t, 4)),
        ("FTP-BruteForce", "multi_eventual_5", "eventual_success", "raw-socket", "custom", "passive", lambda t: at_multi_eventual(t, 5)),
        ("FTP-BruteForce", "single_packed_6b", "single_packed", "raw-socket", "custom", "passive", lambda t: at_single_packed(t, 6)),
        # ---- benign ----
        ("Benign", "clean_1", "clean", "raw-socket", "custom", "passive", bn_clean),
        ("Benign", "clean_2", "clean", "raw-socket", "custom", "passive", bn_clean),
        ("Benign", "mistype_success_1", "mistype", "raw-socket", "custom", "passive", lambda t: bn_mistype_success(t, 1)),
        ("Benign", "mistype_success_2", "mistype", "raw-socket", "custom", "passive", lambda t: bn_mistype_success(t, 2)),
        ("Benign", "mistype_success_3", "mistype", "raw-socket", "custom", "passive", lambda t: bn_mistype_success(t, 3)),
        ("Benign", "giveup_2", "gave_up", "raw-socket", "custom", "passive", lambda t: bn_giveup(t, 2)),
        ("Benign", "giveup_3", "gave_up", "raw-socket", "custom", "passive", lambda t: bn_giveup(t, 3)),
        ("Benign", "reconnect_success_1", "reconnect", "raw-socket", "custom", "passive", bn_reconnect_success),
        ("Benign", "reconnect_success_2", "reconnect", "raw-socket", "custom", "passive", bn_reconnect_success),
        ("Benign", "repeated_normal_2", "repeated", "raw-socket", "custom", "passive", lambda t: bn_repeated_normal(t, 2)),
        ("Benign", "repeated_normal_3", "repeated", "raw-socket", "custom", "passive", lambda t: bn_repeated_normal(t, 3)),
        ("Benign", "repeated_normal_4", "repeated", "raw-socket", "custom", "passive", lambda t: bn_repeated_normal(t, 4)),
        ("Benign", "two_users", "multi_user", "raw-socket", "custom", "passive", bn_two_users),
        ("Benign", "clean_3", "clean", "raw-socket", "custom", "passive", bn_clean),
        ("Benign", "mistype_success_1b", "mistype", "raw-socket", "custom", "passive", lambda t: bn_mistype_success(t, 1)),
        ("Benign", "giveup_2b", "gave_up", "raw-socket", "custom", "passive", lambda t: bn_giveup(t, 2)),
    ]
    return [rc.Spec(sc, lab, fam, cl, srv, e(i), mode, fn)
            for i, (lab, sc, fam, cl, srv, mode, fn) in enumerate(plan)]
