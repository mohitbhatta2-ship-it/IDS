"""
Experimental FTP application-layer BEHAVIOURAL feature extractor (analysis only).

The production 30-feature pipeline is CIC-derived packet/timing statistics, and
(see final_robustness) real benign and real FTP-BruteForce OVERLAP on the
CIC-artifact features -- so re-weighting cannot separate them. This module computes
NEW features from the FTP CONTROL channel itself (cleartext USER/PASS commands and
numeric response codes) that describe the *semantic* difference between the two
classes: authentication success vs repeated failure, data-transfer activity,
command variety, and reconnect behaviour.

Everything is measured directly from the PCAP -- no fabrication, no zero-fill, no
label information. Features are per-capture (aggregated over the capture's control
connections); a flow inherits its capture's behavioural context. Nothing here
imports into or changes ml.py / live_capture.py / pcap_validation.py; the existing
30-feature pipeline is untouched and used separately.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

# FTP verbs we care about (control-channel commands)
_DATA_CMDS = {"RETR", "STOR", "STOU", "APPE", "LIST", "NLST", "MLSD"}
_AUTH_CMDS = {"USER", "PASS"}
_RESP_RE = re.compile(rb"^(\d{3})[ -]")
_VERB_RE = re.compile(rb"^([A-Za-z]{3,4})(?:\s|$)")

# the experimental behavioural feature schema (order matters for the model)
BEHAV_FEATURES = [
    "ftp_login_attempts",        # number of PASS commands
    "ftp_failed_logins",         # number of 530 responses (login incorrect)
    "ftp_successful_logins",     # number of 230 responses (login ok)
    "ftp_failed_login_ratio",    # 530 / (230 + 530)
    "ftp_has_successful_auth",   # 1 if any 230 else 0
    "ftp_user_commands",         # number of USER commands
    "ftp_distinct_commands",     # distinct command verbs
    "ftp_total_commands",        # total command lines
    "ftp_data_setup_responses",  # number of 150 responses (data connection opening)
    "ftp_data_commands",         # RETR/STOR/APPE/LIST/NLST/MLSD count
    "ftp_error_responses_5xx",   # count of 5xx responses
    "ftp_ok_responses_2xx",      # count of 2xx responses
    "ftp_control_connections",   # distinct TCP streams that carried FTP control
    "ftp_reconnects",            # control_connections - 1
    "ftp_cmds_per_connection",   # total_commands / control_connections
]


def _iter_lines(payload: bytes):
    for line in payload.split(b"\r\n"):
        line = line.strip()
        if line:
            yield line


def behavioural_features_for_pcap(pcap_path) -> dict:
    """
    Parse the FTP control channel(s) of a PCAP and return the behavioural feature
    vector for the whole capture. Reads only what is on the wire; never zero-fills
    a missing measurement (a capture with no FTP control traffic returns explicit
    zeros for counts, which is the true measurement, not a fabricated fill).
    """
    from scapy.all import rdpcap, TCP, IP, Raw

    pcap_path = Path(pcap_path)
    cmds = Counter()
    resps = Counter()
    control_streams = set()
    total_commands = 0

    for pk in rdpcap(str(pcap_path)):
        if TCP not in pk or Raw not in pk or IP not in pk:
            continue
        payload = bytes(pk[Raw].load)
        saw_ftp = False
        for line in _iter_lines(payload):
            rm = _RESP_RE.match(line)
            if rm:
                resps[int(rm.group(1))] += 1
                saw_ftp = True
                continue
            vm = _VERB_RE.match(line)
            if vm:
                verb = vm.group(1).decode("latin-1").upper()
                # only count real FTP verbs (avoid counting file content that looks like a word)
                if verb in _AUTH_CMDS or verb in _DATA_CMDS or verb in {
                        "QUIT", "PWD", "SYST", "TYPE", "NOOP", "CWD", "CDUP", "PASV",
                        "PORT", "MKD", "RMD", "DELE", "RNFR", "RNTO", "SIZE", "MDTM",
                        "STAT", "FEAT", "OPTS", "AUTH", "ABOR", "REST", "XPWD"}:
                    cmds[verb] += 1
                    total_commands += 1
                    saw_ftp = True
        if saw_ftp:
            # identify the control stream by its ordered endpoints
            a = (pk[IP].src, pk[TCP].sport)
            b = (pk[IP].dst, pk[TCP].dport)
            control_streams.add(tuple(sorted((a, b))))

    n_530 = resps.get(530, 0)
    n_230 = resps.get(230, 0)
    n_150 = resps.get(150, 0)
    n_conn = max(1, len(control_streams))
    resp_5xx = sum(v for k, v in resps.items() if 500 <= k < 600)
    resp_2xx = sum(v for k, v in resps.items() if 200 <= k < 300)
    data_cmds = sum(cmds.get(c, 0) for c in _DATA_CMDS)
    login_attempts = cmds.get("PASS", 0)

    return {
        "ftp_login_attempts": float(login_attempts),
        "ftp_failed_logins": float(n_530),
        "ftp_successful_logins": float(n_230),
        "ftp_failed_login_ratio": float(n_530 / (n_230 + n_530)) if (n_230 + n_530) else 0.0,
        "ftp_has_successful_auth": 1.0 if n_230 > 0 else 0.0,
        "ftp_user_commands": float(cmds.get("USER", 0)),
        "ftp_distinct_commands": float(len(cmds)),
        "ftp_total_commands": float(total_commands),
        "ftp_data_setup_responses": float(n_150),
        "ftp_data_commands": float(data_cmds),
        "ftp_error_responses_5xx": float(resp_5xx),
        "ftp_ok_responses_2xx": float(resp_2xx),
        "ftp_control_connections": float(len(control_streams)),
        "ftp_reconnects": float(max(0, len(control_streams) - 1)),
        "ftp_cmds_per_connection": float(total_commands / n_conn),
    }


def has_ftp_control(features: dict) -> bool:
    """True if the capture actually carried FTP control traffic (commands or codes)."""
    return (features.get("ftp_total_commands", 0) > 0
            or features.get("ftp_ok_responses_2xx", 0) > 0
            or features.get("ftp_error_responses_5xx", 0) > 0)
