"""
EXPERIMENTAL cross-session / source-level FTP features (analysis only; nothing frozen touched).

Single-session features cannot separate "a legitimate user failed a few times and left"
from "a short brute force" -- the two look identical inside one session. The hypothesis
here is that *cross-session* behaviour can: an attacker persists across many
sessions/connections with sustained failure and credential variation, while a legitimate
user makes few sessions and either succeeds or gives up once.

A capture is treated as one SOURCE's window; each FTP control TCP stream in it is a
SESSION. These features aggregate ACROSS the sessions in the window (session count,
failure rate across sessions, unique/varying credentials, time between sessions,
per-session success history, persistence, session rate). Everything is computed directly
from the PCAP with NO label and NO model prediction. Genuinely undefined measurements
(e.g. time-between-sessions with < 2 sessions) are NaN, never zero-filled.

Meaningful across FTP implementations (standard cleartext control dialogue + wire
timestamps). Under FTPS/TLS the control channel is encrypted and these are unavailable
(handled separately). Nothing here imports into or changes ml.py / live_capture.py /
pcap_validation.py / ftp_behavioral.py; the existing 30 + 15 features are preserved.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_RESP_RE = re.compile(rb"^(\d{3})[ -]")
_CMD_RE = re.compile(rb"^([A-Za-z]{3,4})(?:\s+(.*))?$")
_KNOWN_VERBS = {"USER", "PASS", "QUIT", "RETR", "STOR", "LIST", "NLST", "MLSD", "PWD", "SYST",
                "TYPE", "NOOP", "CWD", "CDUP", "PASV", "EPSV", "PORT", "MKD", "RMD", "DELE",
                "SIZE", "MDTM", "STAT", "FEAT", "OPTS", "AUTH", "APPE", "STOU"}

# Cross-session / source-level feature schema (order matters for the model).
CROSS_FEATURES = [
    "ftpx_sessions_per_source",              # number of FTP control connections (sessions) in the window
    "ftpx_failed_attempts_across_sessions",  # total 530s across all sessions
    "ftpx_failure_rate_across_sessions",     # total 530 / total PASS (NaN if no attempts)
    "ftpx_unique_usernames_across_sessions", # distinct USER strings across sessions
    "ftpx_unique_passwords_across_sessions", # distinct PASS strings across sessions
    "ftpx_credential_variation",             # distinct (user,pass) pairs / total attempts (NaN if none)
    "ftpx_mean_time_between_sessions_s",     # mean gap between session starts (NaN if < 2 sessions)
    "ftpx_failed_session_count",             # sessions with >=1 fail and NO success
    "ftpx_successful_session_count",         # sessions with a 230
    "ftpx_frac_sessions_successful",         # successful_sessions / sessions (NaN if 0 sessions)
    "ftpx_session_rate_per_s",               # sessions / window span (NaN if span 0)
    "ftpx_source_persistence_s",             # last session start - first session start
    "ftpx_max_fails_in_session",             # max 530 within a single session
]


def _stream_key(pk, TCP, IP):
    a = (pk[IP].src, pk[TCP].sport)
    b = (pk[IP].dst, pk[TCP].dport)
    return tuple(sorted((a, b)))


def _lines(payload: bytes):
    for line in payload.split(b"\r\n"):
        line = line.strip()
        if line:
            yield line


def cross_session_features_for_pcap(pcap_path) -> dict:
    """Aggregate cross-session behaviour over the control connections in one capture."""
    from scapy.all import rdpcap, TCP, IP, Raw

    pkts = rdpcap(str(Path(pcap_path)))

    # per control stream (session): timeline of commands/responses
    sessions: dict = {}
    for pk in pkts:
        if TCP not in pk or IP not in pk or Raw not in pk:
            continue
        t = float(pk.time)
        sk = _stream_key(pk, TCP, IP)
        payload = bytes(pk[Raw].load)
        cur_user = None
        for line in _lines(payload):
            rm = _RESP_RE.match(line)
            if rm:
                s = sessions.setdefault(sk, _new_session())
                code = int(rm.group(1)); s["codes"].append(code)
                s["t0"] = min(s["t0"], t); s["t1"] = max(s["t1"], t)
                continue
            cm = _CMD_RE.match(line)
            if cm:
                verb = cm.group(1).decode("latin-1").upper()
                if verb not in _KNOWN_VERBS:
                    continue
                s = sessions.setdefault(sk, _new_session())
                s["t0"] = min(s["t0"], t); s["t1"] = max(s["t1"], t)
                arg = (cm.group(2) or b"").decode("latin-1")
                if verb == "USER":
                    cur_user = arg; s["users"].add(arg)
                elif verb == "PASS":
                    s["passwords"].add(arg); s["attempts"] += 1
                    s["cred_pairs"].add((cur_user, arg))
                elif verb == "QUIT":
                    s["quit"] = True

    if not sessions:
        # no FTP control traffic at all -> counts are honest zeros; ratios/timing NaN
        return {f: (0.0 if f in _COUNT_ZERO else np.nan) for f in CROSS_FEATURES}

    # per-session outcomes
    starts, fails_list, succ_sessions, failed_only_sessions = [], [], 0, 0
    total_attempts = total_fails = 0
    all_users, all_pass, all_pairs = set(), set(), set()
    for s in sessions.values():
        starts.append(s["t0"])
        n_fail = sum(1 for c in s["codes"] if c == 530)
        has_succ = any(c == 230 for c in s["codes"])
        fails_list.append(n_fail)
        total_fails += n_fail
        total_attempts += s["attempts"]
        all_users |= s["users"]; all_pass |= s["passwords"]; all_pairs |= s["cred_pairs"]
        if has_succ:
            succ_sessions += 1
        elif n_fail > 0:
            failed_only_sessions += 1

    n_sessions = len(sessions)
    starts.sort()
    span = float(starts[-1] - starts[0]) if len(starts) >= 2 else 0.0
    gaps = np.diff(starts) if len(starts) >= 2 else np.array([])
    return {
        "ftpx_sessions_per_source": float(n_sessions),
        "ftpx_failed_attempts_across_sessions": float(total_fails),
        "ftpx_failure_rate_across_sessions": float(total_fails / total_attempts) if total_attempts else np.nan,
        "ftpx_unique_usernames_across_sessions": float(len(all_users)),
        "ftpx_unique_passwords_across_sessions": float(len(all_pass)),
        "ftpx_credential_variation": float(len(all_pairs) / total_attempts) if total_attempts else np.nan,
        "ftpx_mean_time_between_sessions_s": float(np.mean(gaps)) if len(gaps) else np.nan,
        "ftpx_failed_session_count": float(failed_only_sessions),
        "ftpx_successful_session_count": float(succ_sessions),
        "ftpx_frac_sessions_successful": float(succ_sessions / n_sessions) if n_sessions else np.nan,
        "ftpx_session_rate_per_s": float(n_sessions / span) if span > 0 else np.nan,
        "ftpx_source_persistence_s": span,
        "ftpx_max_fails_in_session": float(max(fails_list)) if fails_list else 0.0,
    }


# counts that are genuinely zero when there is no FTP control traffic (not fabricated)
_COUNT_ZERO = {"ftpx_sessions_per_source", "ftpx_failed_attempts_across_sessions",
               "ftpx_unique_usernames_across_sessions", "ftpx_unique_passwords_across_sessions",
               "ftpx_failed_session_count", "ftpx_successful_session_count",
               "ftpx_source_persistence_s", "ftpx_max_fails_in_session"}


def _new_session():
    return {"codes": [], "users": set(), "passwords": set(), "cred_pairs": set(),
            "attempts": 0, "quit": False, "t0": float("inf"), "t1": float("-inf")}
