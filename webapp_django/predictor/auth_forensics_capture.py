"""
Authentication-forensics training corpus (experimental; models frozen; TRAIN-only).

The distinction under test: benign **typo-then-success / typo-then-give-up** (attempts are
edit-distance-close to the real password, human paced) vs attacker **single-session
dictionary brute force / dictionary-then-success** (unrelated guesses, fast). Also the hard
benign boundary case: a user who tries a couple of *alternate real* passwords (edit-far but
few) before succeeding.

Realistic typos are generated from the correct password (drop / add / transpose / case /
adjacent-key). Loopback lab, custom raw-socket server (allows packed attempts and arbitrary
passwords), NEW addresses (127.0.0.50-52) / ports (2830/2840), hash-disjoint from every
prior corpus and every independent test. Ground truth from the scenario folder. Nothing
here changes ml.py / live_capture.py / pcap_validation.py / ftp_behavioral.py.
"""

from __future__ import annotations

import random
import socket
import time
from pathlib import Path

from . import robustness_capture as rc
from .independent_capture import is_controlled_local

USER, PASSWORD = rc.USER, rc.PASSWORD          # labuser / labpass (custom server)
DICT_PW = ["123456", "password", "admin", "qwerty", "letmein", "root", "test123", "secret",
           "111111", "dragon", "monkey", "shadow", "welcome", "hunter2", "abc123", "iloveyou"]
ALT_PW = ["labpass1", "mypassword", "Passw0rd", "changeme"]   # plausible non-typo alternates (edit-far, few)
TRAIN_PORT = {"custom": 2830, "permissive": 2840}
_RNG = random.Random(20260818)


def typos_of(pw: str, n: int) -> list[str]:
    """Generate n realistic single-edit typos of pw (drop / add / transpose / case / adjacent)."""
    adj = {"a": "s", "s": "a", "l": "k", "p": "o", "b": "v", "1": "2", "2": "1", "3": "4"}
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < 200:
        tries += 1
        i = _RNG.randrange(len(pw))
        kind = _RNG.choice(("drop", "add", "transpose", "case", "adjacent"))
        if kind == "drop" and len(pw) > 1:
            t = pw[:i] + pw[i + 1:]
        elif kind == "add":
            t = pw[:i] + _RNG.choice("abcdefghijklmnopqrstuvwxyz0123456789") + pw[i:]
        elif kind == "transpose" and i < len(pw) - 1:
            t = pw[:i] + pw[i + 1] + pw[i] + pw[i + 2:]
        elif kind == "case":
            t = pw[:i] + (pw[i].upper() if pw[i].islower() else pw[i].lower()) + pw[i + 1:]
        else:
            c = pw[i].lower()
            t = pw[:i] + adj.get(c, c) + pw[i + 1:]
        if t != pw and t not in seen:
            seen.add(t); out.append(t)
    return out or [pw + "x"]


def train_environments() -> dict[str, tuple[str, str]]:
    return {"af50": ("127.0.0.50", "lo"), "af51": ("127.0.0.51", "lo"), "af52": ("127.0.0.52", "lo")}


def build_targets(root: Path) -> dict:
    targets, idx = {}, 0
    for env_key, (addr, iface) in train_environments().items():
        plo = 58000 + idx * 40
        targets[(env_key, "custom")] = rc.RobTarget(addr=addr, control_port=TRAIN_PORT["custom"], passive_lo=plo,
                                                    passive_hi=plo + 39, iface=iface, server="custom", env_key=env_key)
        idx += 1
    return targets


def _think():
    time.sleep(_RNG.uniform(0.4, 2.2))


def _raw_conn(t, creds, pace=False, timeout=12.0):
    """One connection; send (user,pw) list. pace=True adds human think-time. Returns (ok, fails)."""
    ok = fail = 0
    try:
        s = socket.create_connection((t.addr, t.control_port), timeout=timeout); s.settimeout(timeout); s.recv(4096)
    except OSError:
        return 0, 0
    for k, (user, pw) in enumerate(creds):
        if pace and k:
            _think()
        try:
            s.sendall(f"USER {user}\r\n".encode()); s.recv(4096)
            s.sendall(f"PASS {pw}\r\n".encode()); resp = s.recv(4096)
            ok += int(resp[:3] == b"230"); fail += int(resp[:3] != b"230")
        except OSError:
            break
    try:
        s.sendall(b"QUIT\r\n"); s.close()
    except OSError:
        pass
    return ok, fail


# ---- benign (typos; human paced) -------------------------------------------


def bn_typo_then_success(t, n=2):
    creds = [(USER, pw) for pw in typos_of(PASSWORD, n)] + [(USER, PASSWORD)]
    o, f = _raw_conn(t, creds, pace=True)
    return {"sessions": 1, "successes": o, "failures": f, "label_reason": "benign_typo_then_success"}


def bn_typo_give_up(t, n=3):
    _o, f = _raw_conn(t, [(USER, pw) for pw in typos_of(PASSWORD, n)], pace=True)
    return {"sessions": 1, "failures": f, "label_reason": "benign_typo_give_up"}


def bn_alt_password_then_success(t, n=2):
    """Hard benign case: a couple of plausible ALTERNATE (non-typo) passwords, then success."""
    creds = [(USER, pw) for pw in ALT_PW[:n]] + [(USER, PASSWORD)]
    o, f = _raw_conn(t, creds, pace=True)
    return {"sessions": 1, "successes": o, "failures": f, "label_reason": "benign_alt_then_success"}


def bn_clean(t):
    o, _f = _raw_conn(t, [(USER, PASSWORD)]); return {"sessions": 1, "successes": o, "label_reason": "benign_clean"}


def bn_repeated(t, n=3):
    ok = 0
    for _ in range(n):
        o, _f = _raw_conn(t, [(USER, PASSWORD)]); ok += o; time.sleep(0.3)
    return {"sessions": n, "successes": ok, "label_reason": "benign_repeated"}


def bn_reconnect_success(t):
    _o1, f1 = _raw_conn(t, [(USER, typos_of(PASSWORD, 1)[0])], pace=True)
    o2, _f2 = _raw_conn(t, [(USER, PASSWORD)])
    return {"sessions": 2, "successes": o2, "failures": f1, "label_reason": "benign_reconnect_success"}


# ---- attacks (dictionary; single-session, fast) ----------------------------


def at_dict_fail(t, n=8):
    _o, f = _raw_conn(t, [(USER, DICT_PW[i % len(DICT_PW)]) for i in range(n)])
    return {"sessions": 1, "failures": f, "label_reason": "attack_dict_fail"}


def at_dict_then_success(t, n=8):
    """Single-session fail-then-success: dictionary guesses then the correct password."""
    creds = [(USER, DICT_PW[i % len(DICT_PW)]) for i in range(n)] + [(USER, PASSWORD)]
    o, f = _raw_conn(t, creds)
    return {"sessions": 1, "successes": o, "failures": f, "label_reason": "attack_dict_then_success"}


def at_dict_sweep_users(t, n=8):
    _o, f = _raw_conn(t, [(["admin", "root", "oracle", "ftp", "guest", "test"][i % 6], DICT_PW[i % len(DICT_PW)]) for i in range(n)])
    return {"sessions": 1, "failures": f, "label_reason": "attack_dict_sweep"}


def at_dict_slow(t, n=6):
    creds = [(USER, DICT_PW[i % len(DICT_PW)]) for i in range(n)]
    o = f = 0
    r_o, r_f = _raw_conn(t, creds, pace=False)   # fast dictionary; timing similar-fast
    return {"sessions": 1, "failures": r_f, "label_reason": "attack_dict"}


def at_dict_multi(t, sessions=5):
    total_f = 0
    for i in range(sessions):
        _o, f = _raw_conn(t, [(USER, DICT_PW[i % len(DICT_PW)])]); total_f += f
    return {"sessions": sessions, "failures": total_f, "label_reason": "attack_dict_multi"}


def specs() -> list[rc.Spec]:
    envs = list(train_environments())

    def e(i):
        return envs[i % len(envs)]

    plan = [
        # benign: typos, alternates, clean, repeated
        ("Benign", "typo_then_success_1", "mistype", "raw-socket", "custom", "passive", lambda t: bn_typo_then_success(t, 1)),
        ("Benign", "typo_then_success_2", "mistype", "raw-socket", "custom", "passive", lambda t: bn_typo_then_success(t, 2)),
        ("Benign", "typo_then_success_3", "mistype", "raw-socket", "custom", "passive", lambda t: bn_typo_then_success(t, 3)),
        ("Benign", "typo_then_success_2b", "mistype", "raw-socket", "custom", "passive", lambda t: bn_typo_then_success(t, 2)),
        ("Benign", "typo_give_up_2", "gave_up", "raw-socket", "custom", "passive", lambda t: bn_typo_give_up(t, 2)),
        ("Benign", "typo_give_up_3", "gave_up", "raw-socket", "custom", "passive", lambda t: bn_typo_give_up(t, 3)),
        ("Benign", "typo_give_up_4", "gave_up", "raw-socket", "custom", "passive", lambda t: bn_typo_give_up(t, 4)),
        ("Benign", "alt_then_success_1", "alt_pw", "raw-socket", "custom", "passive", lambda t: bn_alt_password_then_success(t, 1)),
        ("Benign", "alt_then_success_2", "alt_pw", "raw-socket", "custom", "passive", lambda t: bn_alt_password_then_success(t, 2)),
        ("Benign", "clean_1", "clean", "raw-socket", "custom", "passive", bn_clean),
        ("Benign", "clean_2", "clean", "raw-socket", "custom", "passive", bn_clean),
        ("Benign", "repeated_2", "repeated", "raw-socket", "custom", "passive", lambda t: bn_repeated(t, 2)),
        ("Benign", "repeated_3", "repeated", "raw-socket", "custom", "passive", lambda t: bn_repeated(t, 3)),
        ("Benign", "reconnect_success", "reconnect", "raw-socket", "custom", "passive", bn_reconnect_success),
        ("Benign", "typo_then_success_1b", "mistype", "raw-socket", "custom", "passive", lambda t: bn_typo_then_success(t, 1)),
        ("Benign", "typo_give_up_2b", "gave_up", "raw-socket", "custom", "passive", lambda t: bn_typo_give_up(t, 2)),
        # attacks: single-session dictionary (incl fail-then-success)
        ("FTP-BruteForce", "dict_fail_6", "single_dict", "raw-socket", "custom", "passive", lambda t: at_dict_fail(t, 6)),
        ("FTP-BruteForce", "dict_fail_10", "single_dict", "raw-socket", "custom", "passive", lambda t: at_dict_fail(t, 10)),
        ("FTP-BruteForce", "dict_fail_14", "single_dict", "raw-socket", "custom", "passive", lambda t: at_dict_fail(t, 14)),
        ("FTP-BruteForce", "dict_then_success_6", "dict_success", "raw-socket", "custom", "passive", lambda t: at_dict_then_success(t, 6)),
        ("FTP-BruteForce", "dict_then_success_10", "dict_success", "raw-socket", "custom", "passive", lambda t: at_dict_then_success(t, 10)),
        ("FTP-BruteForce", "dict_then_success_4", "dict_success", "raw-socket", "custom", "passive", lambda t: at_dict_then_success(t, 4)),
        ("FTP-BruteForce", "dict_sweep_8", "single_dict", "raw-socket", "custom", "passive", lambda t: at_dict_sweep_users(t, 8)),
        ("FTP-BruteForce", "dict_sweep_12", "single_dict", "raw-socket", "custom", "passive", lambda t: at_dict_sweep_users(t, 12)),
        ("FTP-BruteForce", "dict_multi_5", "multi_dict", "raw-socket", "custom", "passive", lambda t: at_dict_multi(t, 5)),
        ("FTP-BruteForce", "dict_multi_7", "multi_dict", "raw-socket", "custom", "passive", lambda t: at_dict_multi(t, 7)),
        ("FTP-BruteForce", "dict_fail_8", "single_dict", "raw-socket", "custom", "passive", lambda t: at_dict_fail(t, 8)),
        ("FTP-BruteForce", "dict_then_success_8", "dict_success", "raw-socket", "custom", "passive", lambda t: at_dict_then_success(t, 8)),
    ]
    return [rc.Spec(sc, lab, fam, cl, srv, e(i), mode, fn)
            for i, (lab, sc, fam, cl, srv, mode, fn) in enumerate(plan)]
