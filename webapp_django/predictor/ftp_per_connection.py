"""
EXPERIMENTAL per-connection FTP brute-force features (analysis only; nothing frozen touched).

The cross-session detector failed its 2nd independent test because it leaned on
*sessions-per-source*: a genuine SINGLE-SESSION brute force (many attempts packed into one
connection) has few sessions, so the model called it benign. The fix is to describe what a
single CONNECTION is doing -- attempts, failures, credential variation, rate and burstiness
*within* a connection -- so an attack is detectable no matter how many connections the
attacker chooses.

Each capture's FTP control TCP streams (connections) are parsed; per-connection statistics
are computed and aggregated to the capture (max / mean across connections). Everything is
label-free, prediction-free, and never zero-filled: a genuinely undefined measurement
(e.g. an intra-connection inter-attempt gap when no connection has >= 2 attempts) is NaN.

These features are the *counterpart* of the cross-session block: cross-session captures
"failures spread across many connections", per-connection captures "failures packed into
one connection". Together they cover the whole single<->multi-session spectrum, so session
count alone cannot decide the label. Meaningful across FTP server implementations (standard
cleartext control dialogue + wire timestamps); unavailable under FTPS (handled separately).
Nothing here changes ml.py / live_capture.py / pcap_validation.py / ftp_behavioral.py.
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

# Per-connection feature schema (order matters for the model).
PC_FEATURES = [
    "ftppc_max_attempts_per_conn",      # max PASS attempts in any single connection
    "ftppc_mean_attempts_per_conn",     # mean PASS attempts per connection
    "ftppc_max_fails_per_conn",         # max 530s in any single connection
    "ftppc_max_attempt_rate_per_conn",  # max attempts/second within a connection (NaN if no timed conn)
    "ftppc_min_interattempt_within_conn",  # min gap between attempts within a connection (NaN if none has >=2)
    "ftppc_max_distinct_pw_per_conn",   # max distinct PASS strings in a connection
    "ftppc_max_distinct_user_per_conn", # max distinct USER strings in a connection
    "ftppc_mean_conn_duration_s",       # mean control-connection duration
    "ftppc_max_conn_duration_s",        # max control-connection duration
    "ftppc_max_fail_ratio_per_conn",    # max fails/attempts over connections with attempts (NaN if none)
    "ftppc_frac_conns_with_failure",    # fraction of connections that had >=1 failure
]

_COUNT_ZERO = {"ftppc_max_attempts_per_conn", "ftppc_mean_attempts_per_conn", "ftppc_max_fails_per_conn",
               "ftppc_max_distinct_pw_per_conn", "ftppc_max_distinct_user_per_conn",
               "ftppc_mean_conn_duration_s", "ftppc_max_conn_duration_s", "ftppc_frac_conns_with_failure"}


def _stream_key(pk, TCP, IP):
    a = (pk[IP].src, pk[TCP].sport); b = (pk[IP].dst, pk[TCP].dport)
    return tuple(sorted((a, b)))


def _lines(payload: bytes):
    for line in payload.split(b"\r\n"):
        line = line.strip()
        if line:
            yield line


def _new_conn():
    return {"pass_times": [], "passwords": [], "users": set(), "n_fail": 0, "t0": float("inf"), "t1": float("-inf")}


def per_connection_features_for_pcap(pcap_path) -> dict:
    """Aggregate per-connection brute-force behaviour over the control connections in a capture."""
    from scapy.all import rdpcap, TCP, IP, Raw

    pkts = rdpcap(str(Path(pcap_path)))
    conns: dict = {}
    for pk in pkts:
        if TCP not in pk or IP not in pk or Raw not in pk:
            continue
        t = float(pk.time); sk = _stream_key(pk, TCP, IP); payload = bytes(pk[Raw].load)
        cur_user = None
        for line in _lines(payload):
            rm = _RESP_RE.match(line)
            if rm:
                c = conns.setdefault(sk, _new_conn()); c["t0"] = min(c["t0"], t); c["t1"] = max(c["t1"], t)
                if int(rm.group(1)) == 530:
                    c["n_fail"] += 1
                continue
            cm = _CMD_RE.match(line)
            if cm:
                verb = cm.group(1).decode("latin-1").upper()
                if verb not in _KNOWN_VERBS:
                    continue
                c = conns.setdefault(sk, _new_conn()); c["t0"] = min(c["t0"], t); c["t1"] = max(c["t1"], t)
                arg = (cm.group(2) or b"").decode("latin-1")
                if verb == "USER":
                    cur_user = arg; c["users"].add(arg)
                elif verb == "PASS":
                    c["pass_times"].append(t); c["passwords"].append(arg)

    if not conns:
        return {f: (0.0 if f in _COUNT_ZERO else np.nan) for f in PC_FEATURES}

    attempts, fails, distinct_pw, distinct_user, durations = [], [], [], [], []
    rates, min_gaps, fail_ratios, conns_with_fail = [], [], [], 0
    for c in conns.values():
        n_att = len(c["pass_times"]); attempts.append(n_att); fails.append(c["n_fail"])
        distinct_pw.append(len(set(c["passwords"]))); distinct_user.append(len(c["users"]))
        dur = (c["t1"] - c["t0"]) if (c["t1"] >= c["t0"]) else 0.0
        durations.append(float(dur))
        if n_att >= 1 and dur > 0:
            rates.append(n_att / dur)
        if n_att >= 2:
            gaps = np.diff(sorted(c["pass_times"]))
            gaps = gaps[gaps >= 0]
            if len(gaps):
                min_gaps.append(float(np.min(gaps)))
        if n_att > 0:
            fail_ratios.append(c["n_fail"] / n_att)
        if c["n_fail"] > 0:
            conns_with_fail += 1

    n_conn = len(conns)
    return {
        "ftppc_max_attempts_per_conn": float(max(attempts)) if attempts else 0.0,
        "ftppc_mean_attempts_per_conn": float(np.mean(attempts)) if attempts else 0.0,
        "ftppc_max_fails_per_conn": float(max(fails)) if fails else 0.0,
        "ftppc_max_attempt_rate_per_conn": float(max(rates)) if rates else np.nan,
        "ftppc_min_interattempt_within_conn": float(min(min_gaps)) if min_gaps else np.nan,
        "ftppc_max_distinct_pw_per_conn": float(max(distinct_pw)) if distinct_pw else 0.0,
        "ftppc_max_distinct_user_per_conn": float(max(distinct_user)) if distinct_user else 0.0,
        "ftppc_mean_conn_duration_s": float(np.mean(durations)) if durations else 0.0,
        "ftppc_max_conn_duration_s": float(max(durations)) if durations else 0.0,
        "ftppc_max_fail_ratio_per_conn": float(max(fail_ratios)) if fail_ratios else np.nan,
        "ftppc_frac_conns_with_failure": float(conns_with_fail / n_conn) if n_conn else np.nan,
    }
