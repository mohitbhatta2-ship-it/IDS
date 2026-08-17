"""
EXPERIMENTAL authentication-forensics FTP features (analysis only; nothing frozen touched).

The one case every prior experiment could not separate: a benign user who **mistypes** and
then logs in vs an attacker who **fails then succeeds** in a single session. Failure COUNT
and session count cannot tell them apart. The realistic difference is in *what the failed
passwords look like*:

  * a human who mistypes produces attempts that are EDIT-DISTANCE-CLOSE to the correct
    password (dropped/added/transposed characters, wrong case) -- typos;
  * an attacker produces DICTIONARY guesses that are unrelated strings, edit-distance-far
    from each other and from the eventually-correct password.

These features quantify that from the cleartext control channel with NO label, NO
prediction, NO threshold and NO zero-fill (a genuinely undefined measurement -- e.g. the
distance from a failed attempt to the successful one when there was no success -- is NaN):

  * distinct failed passwords, and distinct passwords before the first success;
  * min / mean Levenshtein distance from failed attempts to the successful password;
  * mean Levenshtein distance between consecutive attempts (typo cluster vs dictionary);
  * password length spread; password reuse ratio;
  * inter-attempt timing (human pacing vs automation) and its regularity.

Meaningful across FTP server implementations (standard cleartext dialogue + wire
timestamps); unavailable under FTPS (encrypted, handled separately). Nothing here changes
ml.py / live_capture.py / pcap_validation.py / ftp_behavioral.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_RESP_RE = re.compile(rb"^(\d{3})[ -]")
_CMD_RE = re.compile(rb"^([A-Za-z]{3,4})(?:\s+(.*))?$")

FORENSIC_FEATURES = [
    "ftpaf_distinct_failed_pw",          # distinct passwords that received a 530
    "ftpaf_distinct_pw_before_success",  # distinct passwords tried before the first 230
    "ftpaf_fails_before_success",        # 530s before the first 230 (0 if none/no success)
    "ftpaf_min_editdist_fail_to_success",   # min Levenshtein(failed_pw, successful_pw) (NaN if no success/no fails)
    "ftpaf_mean_editdist_fail_to_success",  # mean Levenshtein(failed_pw, successful_pw) (NaN if no success/no fails)
    "ftpaf_mean_editdist_consecutive",   # mean Levenshtein between consecutive distinct attempts (NaN if <2)
    "ftpaf_max_pw_len_spread",           # max - min password length over attempts
    "ftpaf_pw_reuse_ratio",              # 1 - distinct/total attempts (NaN if no attempts)
    "ftpaf_mean_interattempt_s",         # mean seconds between PASS attempts (NaN if <2)
    "ftpaf_interattempt_cv",             # coeff. of variation of gaps (NaN if <2 or mean 0)
]

_ZERO_OK = {"ftpaf_distinct_failed_pw", "ftpaf_distinct_pw_before_success",
            "ftpaf_fails_before_success", "ftpaf_max_pw_len_spread"}


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return max(m, n)
    d = list(range(n + 1))
    for i in range(1, m + 1):
        prev = d[0]; d[0] = i
        ca = a[i - 1]
        for j in range(1, n + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (0 if ca == b[j - 1] else 1))
            prev = cur
    return d[n]


def _lines(payload: bytes):
    for line in payload.split(b"\r\n"):
        line = line.strip()
        if line:
            yield line


def forensic_features_for_pcap(pcap_path) -> dict:
    """Authentication-forensics features from the cleartext FTP control channel(s)."""
    from scapy.all import rdpcap, TCP, IP, Raw

    pkts = rdpcap(str(Path(pcap_path)))
    # ordered auth events across the capture: (time, kind, value)
    events = []           # ("pass", time, pw) or ("resp", time, code)
    for pk in pkts:
        if TCP not in pk or IP not in pk or Raw not in pk:
            continue
        t = float(pk.time)
        for line in _lines(bytes(pk[Raw].load)):
            rm = _RESP_RE.match(line)
            if rm:
                events.append(("resp", t, int(rm.group(1)))); continue
            cm = _CMD_RE.match(line)
            if cm and cm.group(1).decode("latin-1").upper() == "PASS":
                events.append(("pass", t, (cm.group(2) or b"").decode("latin-1")))

    pass_events = [(t, pw) for (k, t, pw) in events if k == "pass"]
    resp_events = [(t, c) for (k, t, c) in events if k == "resp"]

    # associate each PASS with the next response code (its outcome)
    passwords, pass_times, outcomes = [], [], []
    ri = 0
    resp_sorted = sorted(resp_events)
    for (t, pw) in pass_events:
        passwords.append(pw); pass_times.append(t)
        # next 2xx/5xx response after this PASS time
        code = None
        for (rt, rc) in resp_sorted:
            if rt >= t and rc in (230, 530) or (rt >= t and 500 <= rc < 600):
                code = rc; break
        outcomes.append(code)

    total = len(passwords)
    failed_pw = [pw for pw, oc in zip(passwords, outcomes) if oc == 530]
    success_pw = next((pw for pw, oc in zip(passwords, outcomes) if oc == 230), None)
    distinct_failed = len(set(failed_pw))
    # distinct passwords before the first success
    before, seen_success = set(), False
    fails_before = 0
    for pw, oc in zip(passwords, outcomes):
        if oc == 230:
            break
        before.add(pw)
        if oc == 530:
            fails_before += 1

    # edit distances from failed attempts to the successful password
    if success_pw is not None and failed_pw:
        dists = [_lev(pw, success_pw) for pw in set(failed_pw)]
        min_ed = float(min(dists)); mean_ed = float(np.mean(dists))
    else:
        min_ed = mean_ed = np.nan

    # edit distance between consecutive DISTINCT attempts (typo cluster vs dictionary spread)
    consec = []
    for i in range(1, len(passwords)):
        if passwords[i] != passwords[i - 1]:
            consec.append(_lev(passwords[i], passwords[i - 1]))
    mean_consec = float(np.mean(consec)) if consec else np.nan

    lengths = [len(pw) for pw in passwords]
    len_spread = float(max(lengths) - min(lengths)) if lengths else 0.0
    reuse = float(1.0 - len(set(passwords)) / total) if total else np.nan

    if len(pass_times) >= 2:
        gaps = np.diff(sorted(pass_times)); gaps = gaps[gaps >= 0]
        mean_gap = float(np.mean(gaps)) if len(gaps) else np.nan
        cv = float(np.std(gaps) / mean_gap) if (mean_gap and mean_gap > 0) else np.nan
    else:
        mean_gap = cv = np.nan

    return {
        "ftpaf_distinct_failed_pw": float(distinct_failed),
        "ftpaf_distinct_pw_before_success": float(len(before)),
        "ftpaf_fails_before_success": float(fails_before),
        "ftpaf_min_editdist_fail_to_success": min_ed,
        "ftpaf_mean_editdist_fail_to_success": mean_ed,
        "ftpaf_mean_editdist_consecutive": mean_consec,
        "ftpaf_max_pw_len_spread": len_spread,
        "ftpaf_pw_reuse_ratio": reuse,
        "ftpaf_mean_interattempt_s": mean_gap,
        "ftpaf_interattempt_cv": cv,
    }
