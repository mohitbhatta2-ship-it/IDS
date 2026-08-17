"""
Cross-session / source-level training corpus (experimental; models frozen; TRAIN-only).

Each capture is one SOURCE's window containing MULTIPLE FTP sessions (control
connections), so cross-session features have something to aggregate. Benign sources make
few sessions and either succeed or give up once; attacker sources persist across many
sessions with sustained failure and credential variation. Ground truth is the scenario
folder, never a prediction.

Fresh scenario code with realistic multi-session patterns and think-time between sessions.
Loopback lab, pyftpdlib + custom raw-socket servers (vsftpd deliberately held out to keep
the frozen vsFTPD test independent), NEW addresses (127.0.0.30-32) and ports (2630/2640) --
disjoint from every prior corpus and the frozen vsFTPD test. Nothing here changes ml.py /
live_capture.py / pcap_validation.py / ftp_behavioral.py.
"""

from __future__ import annotations

import random
import socket
import time
from ftplib import error_perm, error_temp
from pathlib import Path

from . import robustness_capture as rc
from .independent_capture import host_eth0_ip, is_controlled_local

USER, PASSWORD = rc.USER, rc.PASSWORD
ALL_USERS = rc.ALL_USERS
WRONG_PW = rc.WRONG_PW
WRONG_USERS = ["admin", "root", "oracle", "ftp", "guest", "test", "www", "mysql", "user", "ftpuser"]

TRAIN_PORT = {"custom": 2630, "permissive": 2640}
_RNG = random.Random(20260817)


def train_environments() -> dict[str, tuple[str, str]]:
    envs = {"cs30": ("127.0.0.30", "lo"), "cs31": ("127.0.0.31", "lo"), "cs32": ("127.0.0.32", "lo")}
    eth0 = host_eth0_ip()
    if eth0:
        envs["cshost"] = (eth0, "lo")
    return envs


def build_targets(root: Path) -> dict:
    targets, idx = {}, 0
    for env_key, (addr, iface) in train_environments().items():
        for server in TRAIN_PORT:
            plo = 60000 + idx * 40
            targets[(env_key, server)] = rc.RobTarget(
                addr=addr, control_port=TRAIN_PORT[server], passive_lo=plo, passive_hi=plo + 39,
                iface=iface, server=server, env_key=env_key)
            idx += 1
    return targets


def _think(lo=0.3, hi=1.5):
    time.sleep(_RNG.uniform(lo, hi))


def _one_session(t, passive, creds, do_activity=False):
    """Open one control connection, try creds (list of (user,pw)); return (ok, fails)."""
    f = rc._connect(t, passive); ok = fails = 0
    for user, pw in creds:
        try:
            f.sendcmd(f"USER {user}"); f.sendcmd(f"PASS {pw}")
            ok += 1
        except (error_perm, error_temp):
            fails += 1
        except (EOFError, OSError):
            break
    if ok and do_activity:
        try:
            f.retrlines("LIST", lambda _l: None)
        except Exception:  # noqa: BLE001
            pass
    try:
        f.quit()
    except Exception:  # noqa: BLE001
        try:
            f.close()
        except Exception:  # noqa: BLE001
            pass
    return ok, fails


def _raw_session(t, creds):
    ok = fails = 0
    try:
        s = socket.create_connection((t.addr, t.control_port), timeout=6); s.recv(4096)
    except OSError:
        return 0, 0
    for user, pw in creds:
        try:
            s.sendall(f"USER {user}\r\n".encode()); s.recv(4096)
            s.sendall(f"PASS {pw}\r\n".encode()); resp = s.recv(4096)
            if resp[:3] == b"230":
                ok += 1
            else:
                fails += 1
        except OSError:
            break
    try:
        s.sendall(b"QUIT\r\n"); s.recv(4096); s.close()
    except OSError:
        pass
    return ok, fails


# ---------------------------------------------------------------------------
# BENIGN source windows (few sessions; success or a single give-up)
# ---------------------------------------------------------------------------


def bcs_mistype_then_success(t, passive=True, n=2):
    """One source, one session: mistype n then succeed + activity."""
    creds = [(USER, WRONG_PW[i]) for i in range(n)] + [(USER, PASSWORD)]
    ok, fails = _one_session(t, passive, creds, do_activity=True)
    return {"sessions": 1, "successes": ok, "failures": fails, "label_reason": "benign_mistype_then_success"}


def bcs_giveup_single(t, passive=True, n=3):
    """One source, one session: fail n times, give up (the ambiguous benign case)."""
    creds = [(USER, WRONG_PW[i]) for i in range(n)]
    ok, fails = _raw_session(t, creds)
    return {"sessions": 1, "successes": ok, "failures": fails, "label_reason": "benign_giveup_single"}


def bcs_reconnect_after_mistake(t, passive=True):
    """Two sessions: first fails and quits, user comes back and logs in."""
    _o1, f1 = _one_session(t, passive, [(USER, WRONG_PW[0]), (USER, WRONG_PW[1])])
    _think(0.8, 2.0)
    o2, f2 = _one_session(t, passive, [(USER, PASSWORD)], do_activity=True)
    return {"sessions": 2, "successes": o2, "failures": f1 + f2, "label_reason": "benign_reconnect_after_mistake"}


def bcs_repeated_normal(t, passive=True, n=3):
    """A returning legitimate user: n successful sessions spaced out (no failures)."""
    ok = 0
    for _ in range(n):
        o, _f = _one_session(t, passive, [(USER, PASSWORD)], do_activity=True)
        ok += o
        _think(0.6, 1.8)
    return {"sessions": n, "successes": ok, "failures": 0, "label_reason": "benign_repeated_normal"}


def bcs_two_users_normal(t, passive=True):
    """Two different valid users on the same source, both succeed (shared workstation)."""
    ok = 0
    for u, pw in (("alice", "alicepw"), ("bob", "bobpw")):
        o, _f = _one_session(t, passive, [(u, pw)], do_activity=True); ok += o
        _think()
    return {"sessions": 2, "successes": ok, "failures": 0, "label_reason": "benign_two_users_normal"}


def bcs_mixed_success_then_mistype(t, passive=True):
    """Session 1 clean success; session 2 one mistype then success."""
    o1, _f1 = _one_session(t, passive, [(USER, PASSWORD)], do_activity=True)
    _think()
    o2, f2 = _one_session(t, passive, [(USER, WRONG_PW[0]), (USER, PASSWORD)], do_activity=True)
    return {"sessions": 2, "successes": o1 + o2, "failures": f2, "label_reason": "benign_mixed"}


# ---------------------------------------------------------------------------
# ATTACKER source windows (many sessions; sustained failure / persistence)
# ---------------------------------------------------------------------------


def acs_repeated_fails(t, passive=True, n_sessions=8):
    """One source hammering across many sessions: same user, different passwords, all fail."""
    total_f = 0
    for i in range(n_sessions):
        _o, f = _raw_session(t, [(USER, WRONG_PW[i % len(WRONG_PW)])])
        total_f += f
    return {"sessions": n_sessions, "successes": 0, "failures": total_f, "label_reason": "attack_repeated_fails"}


def acs_credential_sweep(t, passive=True, n_sessions=10):
    """Many sessions sweeping different usernames AND passwords."""
    total_f = 0
    for i in range(n_sessions):
        u = WRONG_USERS[i % len(WRONG_USERS)]; pw = WRONG_PW[i % len(WRONG_PW)]
        _o, f = _raw_session(t, [(u, pw)]); total_f += f
    return {"sessions": n_sessions, "successes": 0, "failures": total_f, "label_reason": "attack_credential_sweep"}


def acs_slow_brute(t, passive=True, n_sessions=6, delay=0.6):
    """Slow brute: sessions spread over time with delays between them."""
    total_f = 0
    for i in range(n_sessions):
        _o, f = _raw_session(t, [(USER, WRONG_PW[i % len(WRONG_PW)])]); total_f += f
        time.sleep(delay)
    return {"sessions": n_sessions, "successes": 0, "failures": total_f, "label_reason": "attack_slow_brute"}


def acs_eventual_success(t, passive=True, n_wrong=8):
    """Many failing sessions, then a final session that guesses the correct password."""
    total_f = 0
    for i in range(n_wrong):
        _o, f = _raw_session(t, [(USER, WRONG_PW[i % len(WRONG_PW)])]); total_f += f
    o, _f = _raw_session(t, [(USER, PASSWORD)])
    return {"sessions": n_wrong + 1, "successes": o, "failures": total_f, "label_reason": "attack_eventual_success"}


def acs_bursts(t, passive=True, bursts=3, per_burst=3):
    """Bursts of reconnections: several bursts, each several failed sessions."""
    total_f = 0
    for b in range(bursts):
        for i in range(per_burst):
            _o, f = _raw_session(t, [(USER, WRONG_PW[(b * per_burst + i) % len(WRONG_PW)])]); total_f += f
        _think(0.5, 1.2)
    return {"sessions": bursts * per_burst, "successes": 0, "failures": total_f, "label_reason": "attack_bursts"}


def acs_multiuser_persistent(t, passive=True, n_sessions=9):
    """Persistent attacker cycling usernames with a couple of passwords each."""
    total_f = 0
    for i in range(n_sessions):
        u = WRONG_USERS[i % len(WRONG_USERS)]
        _o, f = _raw_session(t, [(u, WRONG_PW[i % len(WRONG_PW)]), (u, WRONG_PW[(i + 1) % len(WRONG_PW)])])
        total_f += f
    return {"sessions": n_sessions, "successes": 0, "failures": total_f, "label_reason": "attack_multiuser_persistent"}


# ---------------------------------------------------------------------------
# Scenario matrix
# ---------------------------------------------------------------------------


def specs() -> list[rc.Spec]:
    envs = list(train_environments())

    def e(i):
        return envs[i % len(envs)]

    plan = [
        # ---- benign source windows ----
        ("Benign", "mistype_then_success_1", "mistype", "python-ftplib", "custom", "passive", lambda t: bcs_mistype_then_success(t, n=1)),
        ("Benign", "mistype_then_success_2", "mistype", "python-ftplib", "permissive", "passive", lambda t: bcs_mistype_then_success(t, n=2)),
        ("Benign", "mistype_then_success_3", "mistype", "python-ftplib", "custom", "passive", lambda t: bcs_mistype_then_success(t, n=3)),
        ("Benign", "giveup_single_2", "gave_up", "raw-socket", "custom", "passive", lambda t: bcs_giveup_single(t, n=2)),
        ("Benign", "giveup_single_3", "gave_up", "raw-socket", "permissive", "passive", lambda t: bcs_giveup_single(t, n=3)),
        ("Benign", "giveup_single_4", "gave_up", "raw-socket", "custom", "passive", lambda t: bcs_giveup_single(t, n=4)),
        ("Benign", "reconnect_after_mistake", "reconnect", "python-ftplib", "custom", "passive", bcs_reconnect_after_mistake),
        ("Benign", "reconnect_after_mistake_2", "reconnect", "python-ftplib", "permissive", "passive", bcs_reconnect_after_mistake),
        ("Benign", "repeated_normal_2", "repeated", "python-ftplib", "custom", "passive", lambda t: bcs_repeated_normal(t, n=2)),
        ("Benign", "repeated_normal_3", "repeated", "python-ftplib", "permissive", "passive", lambda t: bcs_repeated_normal(t, n=3)),
        ("Benign", "repeated_normal_4", "repeated", "python-ftplib", "custom", "passive", lambda t: bcs_repeated_normal(t, n=4)),
        ("Benign", "two_users_normal", "multi_user", "python-ftplib", "permissive", "passive", bcs_two_users_normal),
        ("Benign", "mixed_success_mistype", "mistype", "python-ftplib", "custom", "passive", bcs_mixed_success_then_mistype),
        ("Benign", "repeated_normal_3b", "repeated", "python-ftplib", "permissive", "passive", lambda t: bcs_repeated_normal(t, n=3)),
        # ---- attacker source windows ----
        ("FTP-BruteForce", "repeated_fails_6", "repeated_fails", "raw-socket", "custom", "passive", lambda t: acs_repeated_fails(t, n_sessions=6)),
        ("FTP-BruteForce", "repeated_fails_8", "repeated_fails", "raw-socket", "permissive", "passive", lambda t: acs_repeated_fails(t, n_sessions=8)),
        ("FTP-BruteForce", "repeated_fails_10", "repeated_fails", "raw-socket", "custom", "passive", lambda t: acs_repeated_fails(t, n_sessions=10)),
        ("FTP-BruteForce", "credential_sweep_8", "cred_sweep", "raw-socket", "custom", "passive", lambda t: acs_credential_sweep(t, n_sessions=8)),
        ("FTP-BruteForce", "credential_sweep_12", "cred_sweep", "raw-socket", "permissive", "passive", lambda t: acs_credential_sweep(t, n_sessions=12)),
        ("FTP-BruteForce", "slow_brute_6", "slow_brute", "raw-socket", "custom", "passive", lambda t: acs_slow_brute(t, n_sessions=6, delay=0.5)),
        ("FTP-BruteForce", "slow_brute_8", "slow_brute", "raw-socket", "permissive", "passive", lambda t: acs_slow_brute(t, n_sessions=8, delay=0.4)),
        ("FTP-BruteForce", "eventual_success_8", "eventual_success", "raw-socket", "custom", "passive", lambda t: acs_eventual_success(t, n_wrong=8)),
        ("FTP-BruteForce", "eventual_success_6", "eventual_success", "raw-socket", "permissive", "passive", lambda t: acs_eventual_success(t, n_wrong=6)),
        ("FTP-BruteForce", "bursts_3x3", "bursts", "raw-socket", "custom", "passive", lambda t: acs_bursts(t, bursts=3, per_burst=3)),
        ("FTP-BruteForce", "bursts_4x2", "bursts", "raw-socket", "permissive", "passive", lambda t: acs_bursts(t, bursts=4, per_burst=2)),
        ("FTP-BruteForce", "multiuser_persistent_9", "cred_sweep", "raw-socket", "custom", "passive", lambda t: acs_multiuser_persistent(t, n_sessions=9)),
        ("FTP-BruteForce", "repeated_fails_7", "repeated_fails", "raw-socket", "permissive", "passive", lambda t: acs_repeated_fails(t, n_sessions=7)),
        ("FTP-BruteForce", "credential_sweep_10", "cred_sweep", "raw-socket", "custom", "passive", lambda t: acs_credential_sweep(t, n_sessions=10)),
    ]
    return [rc.Spec(sc, lab, fam, cl, srv, e(i), mode, fn)
            for i, (lab, sc, fam, cl, srv, mode, fn) in enumerate(plan)]
