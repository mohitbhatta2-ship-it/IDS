"""
EXPERIMENTAL *extended* FTP behavioural features (analysis only; nothing frozen touched).

The independent vsFTPD validation showed the 30+15 behavioural model catches brute
force perfectly but raises false positives on genuinely benign failed-login sessions
(mistype / give-up). The existing 15 behavioural features are mostly COUNTS (how many
failed logins, etc.), which cannot separate "benign user failed a few times" from "a
small brute force" -- they have the same counts.

This module adds NEW features that describe the *dynamics* of the session, computed
directly from the PCAP with NO label information, NO model prediction, and NO
zero-fill (a genuinely undefined measurement -- e.g. inter-attempt timing with <2
attempts -- is returned as NaN, which HistGradientBoosting handles natively):

  * timing between login attempts (humans are slow/irregular; tools are fast/regular);
  * credential variation (a benign user retries ONE username and a few passwords; a
    brute force sweeps many distinct passwords / usernames);
  * per-connection attempt load and reconnect pacing;
  * what happens AFTER a successful auth (a real user runs commands and transfers data;
    an attacker who guesses right often does little);
  * session duration and graceful QUIT vs drop.

These are meaningful across FTP server implementations (they read the standard cleartext
control dialogue + wire timestamps). Under FTPS/TLS the control channel is encrypted and
these -- like the 15 -- are unavailable; encrypted traffic is handled separately.

Nothing here imports into or changes ml.py / live_capture.py / pcap_validation.py /
ftp_behavioral.py; the existing 30 packet features and 15 behavioural features are
preserved and used unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_DATA_CMDS = {"RETR", "STOR", "STOU", "APPE", "LIST", "NLST", "MLSD"}
_KNOWN_VERBS = _DATA_CMDS | {"USER", "PASS", "QUIT", "PWD", "SYST", "TYPE", "NOOP", "CWD",
                             "CDUP", "PASV", "EPSV", "PORT", "EPRT", "MKD", "RMD", "DELE",
                             "RNFR", "RNTO", "SIZE", "MDTM", "STAT", "FEAT", "OPTS", "AUTH",
                             "ABOR", "REST", "XPWD"}
_RESP_RE = re.compile(rb"^(\d{3})[ -]")
_CMD_RE = re.compile(rb"^([A-Za-z]{3,4})(?:\s+(.*))?$")

# Extended behavioural feature schema (order matters for the model).
EXT_FEATURES = [
    "ftpx_interattempt_mean_s",    # mean seconds between consecutive PASS attempts (NaN if <2)
    "ftpx_interattempt_min_s",     # min gap between attempts (NaN if <2) -- tiny => automated
    "ftpx_interattempt_cv",        # coeff. of variation of gaps (NaN if <2) -- regularity
    "ftpx_distinct_passwords",     # distinct PASS argument strings
    "ftpx_distinct_usernames",     # distinct USER argument strings
    "ftpx_password_reuse_ratio",   # 1 - distinct_pw/total_pw (NaN if no PASS) -- benign retries
    "ftpx_attempts_per_conn_max",  # max PASS attempts on a single control connection
    "ftpx_session_duration_s",     # control-channel span (last-first packet time)
    "ftpx_ended_with_quit",        # 1 if a QUIT was issued, else 0
    "ftpx_post_auth_commands",     # control commands issued AFTER the first 230
    "ftpx_post_auth_data_transfers",  # data commands (RETR/STOR/LIST...) after the first 230
    "ftpx_fail_run_before_success",   # consecutive 530s immediately before the first 230 (0 if none)
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


def ext_features_for_pcap(pcap_path) -> dict:
    """Compute the extended behavioural features for one PCAP (label-free, no zero-fill)."""
    from scapy.all import rdpcap, TCP, IP, Raw

    pkts = rdpcap(str(Path(pcap_path)))

    # ---- pass 1: classify streams as control (carry FTP verbs/codes) or not ----
    control_streams = set()
    parsed = []           # (time, stream, kind, value) for control packets
    data_bytes = []       # (time, stream, nbytes) for candidate data packets
    for pk in pkts:
        if TCP not in pk or IP not in pk:
            continue
        t = float(pk.time)
        sk = _stream_key(pk, TCP, IP)
        if Raw not in pk:
            continue
        payload = bytes(pk[Raw].load)
        saw_ftp = False
        events = []
        for line in _lines(payload):
            rm = _RESP_RE.match(line)
            if rm:
                events.append(("resp", int(rm.group(1)), None)); saw_ftp = True; continue
            cm = _CMD_RE.match(line)
            if cm:
                verb = cm.group(1).decode("latin-1").upper()
                if verb in _KNOWN_VERBS:
                    arg = (cm.group(2) or b"").decode("latin-1")
                    events.append(("cmd", verb, arg)); saw_ftp = True
        if saw_ftp:
            control_streams.add(sk)
            for kind, v, arg in events:
                parsed.append((t, sk, kind, v, arg))
        else:
            data_bytes.append((t, sk, len(payload)))

    # ---- pass 2: walk the control dialogue in time order ----
    parsed.sort(key=lambda e: e[0])
    pass_times, pass_args, user_args = [], [], []
    resp_seq = []                 # ordered response codes with time
    quit_seen = False
    per_conn_pass = {}
    cmd_times = []                # (time, verb) for all commands
    ctrl_times = []
    for t, sk, kind, v, arg in parsed:
        if kind == "resp":
            resp_seq.append((t, v))
            ctrl_times.append(t)
        else:  # cmd
            ctrl_times.append(t)
            cmd_times.append((t, v))
            if v == "PASS":
                pass_times.append(t); pass_args.append(arg)
                per_conn_pass[sk] = per_conn_pass.get(sk, 0) + 1
            elif v == "USER":
                user_args.append(arg)
            elif v == "QUIT":
                quit_seen = True

    # ---- timing between attempts ----
    if len(pass_times) >= 2:
        gaps = np.diff(sorted(pass_times))
        gaps = gaps[gaps >= 0]
        inter_mean = float(np.mean(gaps)) if len(gaps) else np.nan
        inter_min = float(np.min(gaps)) if len(gaps) else np.nan
        inter_cv = float(np.std(gaps) / inter_mean) if inter_mean and inter_mean > 0 else np.nan
    else:
        inter_mean = inter_min = inter_cv = np.nan

    # ---- credential variation ----
    total_pw = len(pass_args)
    distinct_pw = len(set(pass_args))
    distinct_user = len(set(user_args))
    reuse_ratio = (1.0 - distinct_pw / total_pw) if total_pw > 0 else np.nan

    # ---- session duration + first success ----
    duration = float(max(ctrl_times) - min(ctrl_times)) if len(ctrl_times) >= 2 else 0.0
    first_success_t = next((t for (t, c) in resp_seq if c == 230), None)

    # ---- number of 530 failures that occur before the first success ----
    # (auth-relevant codes only: 220 banner and 331 "need password" are ignored)
    fail_run = 0
    if first_success_t is not None:
        for (t, c) in resp_seq:
            if c == 230:
                break
            if c == 530:
                fail_run += 1

    # ---- post-auth activity ----
    if first_success_t is not None:
        post_cmds = sum(1 for (t, v) in cmd_times if t > first_success_t and v not in ("USER", "PASS"))
        post_data = sum(1 for (t, v) in cmd_times if t > first_success_t and v in _DATA_CMDS)
    else:
        post_cmds = 0
        post_data = 0

    attempts_per_conn_max = float(max(per_conn_pass.values())) if per_conn_pass else 0.0

    return {
        "ftpx_interattempt_mean_s": inter_mean,
        "ftpx_interattempt_min_s": inter_min,
        "ftpx_interattempt_cv": inter_cv,
        "ftpx_distinct_passwords": float(distinct_pw),
        "ftpx_distinct_usernames": float(distinct_user),
        "ftpx_password_reuse_ratio": reuse_ratio,
        "ftpx_attempts_per_conn_max": attempts_per_conn_max,
        "ftpx_session_duration_s": duration,
        "ftpx_ended_with_quit": 1.0 if quit_seen else 0.0,
        "ftpx_post_auth_commands": float(post_cmds),
        "ftpx_post_auth_data_transfers": float(post_data),
        "ftpx_fail_run_before_success": float(fail_run),
    }
