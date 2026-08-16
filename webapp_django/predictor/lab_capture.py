"""
Realistic-PCAP DATA COLLECTION framework (experimental; production untouched).

This module captures REAL FTP packets in a self-contained local lab so we can grow
the realistic-PCAP set for a future, larger leakage-controlled retraining
experiment. It does NOT fabricate, synthesise, duplicate, replay or post-edit any
capture:

  * The target is a real ``pyftpdlib`` FTP server bound to **127.0.0.1** (loopback
    only — the "local lab"). Nothing external is ever contacted.
  * Traffic is captured by **real ``tcpdump``** on the loopback interface, BPF-
    filtered to the lab's own ports, so only the intended lab traffic is stored.
  * Benign and brute-force sessions are driven by a real ``ftplib`` client making
    real TCP connections and real FTP command exchanges against that server.
  * Ground truth is the **scenario that generated the traffic** (which function
    ran), never a model prediction. Each capture is an independent new network
    interaction — a fresh ``tcpdump`` + fresh FTP connections, never a copy.

Nothing here imports into or changes ml.py / live_capture.py / pcap_validation.py
(that pipeline is used read-only for extraction) or any frozen result. It writes
only under ``validation/realistic_pcaps/``.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from ftplib import FTP, error_perm, error_temp
from pathlib import Path

# ---------------------------------------------------------------------------
# The lab target — identified up front, loopback only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabTarget:
    host: str = "127.0.0.1"           # loopback ONLY — never an external host
    control_port: int = 21
    passive_lo: int = 60000
    passive_hi: int = 60040
    user: str = "labuser"
    password: str = "labpass"
    iface: str = "lo"

    def describe(self) -> str:
        return (f"local lab FTP target {self.host}:{self.control_port} "
                f"(passive {self.passive_lo}-{self.passive_hi}) on iface {self.iface}")

    def is_loopback(self) -> bool:
        return self.host in ("127.0.0.1", "::1", "localhost")

    def bpf(self) -> str:
        # only the lab's own control + passive-data ports are captured
        return (f"tcp port {self.control_port} or "
                f"tcp portrange {self.passive_lo}-{self.passive_hi}")


def default_target() -> LabTarget:
    return LabTarget()


# ---------------------------------------------------------------------------
# The FTP lab server (real pyftpdlib, run as an isolated subprocess)
# ---------------------------------------------------------------------------

# Runs in a clean child interpreter (no Django) so shutdown is a clean kill.
_SERVER_CODE = r"""
import sys, logging
logging.disable(logging.CRITICAL)
root, port, plo, phi, user, pw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), \
    int(sys.argv[4]), sys.argv[5], sys.argv[6]
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
auth = DummyAuthorizer()
auth.add_user(user, pw, root, perm="elradfmw")
h = FTPHandler
h.authorizer = auth
h.passive_ports = range(plo, phi + 1)
h.masquerade_address = "127.0.0.1"
# Permissive lab server: allow many login attempts per control connection so the
# "single connection, many attempts" brute-force pattern is a real, coherent
# scenario rather than an artificial early disconnect. (Fresh-connection patterns
# are unaffected -- each makes one attempt then quits.)
h.max_login_attempts = 512
FTPServer(("127.0.0.1", port), h).serve_forever()
"""

SAFE_TEST_FILE = "labfile.txt"      # a safe, local test file for downloads


@dataclass
class FtpLabServer:
    target: LabTarget
    root: Path
    proc: subprocess.Popen | None = None

    def __post_init__(self):
        self.root = Path(self.root)
        (self.root / "uploads").mkdir(parents=True, exist_ok=True)
        # a small, safe local test file to exercise real downloads
        (self.root / SAFE_TEST_FILE).write_text(
            "This is a safe local lab test file for FTP transfer tests.\n" * 40)

    # -- lifecycle (task 1: start + verify) --------------------------------

    def start(self, timeout: float = 8.0) -> "FtpLabServer":
        t = self.target
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _SERVER_CODE, str(self.root), str(t.control_port),
             str(t.passive_lo), str(t.passive_hi), t.user, t.password],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if not self._wait_accepting(timeout):
            err = b""
            if self.proc.poll() is not None and self.proc.stderr:
                err = self.proc.stderr.read()
            self.stop()
            raise RuntimeError(f"FTP lab server did not come up on "
                               f"{t.host}:{t.control_port}: {err.decode('utf-8','ignore')}")
        return self

    def verify(self) -> bool:
        """Task 1 — confirm the target really accepts an FTP control connection."""
        t = self.target
        try:
            ftp = FTP()
            ftp.connect(t.host, t.control_port, timeout=4)
            banner = ftp.getwelcome()
            ftp.login(t.user, t.password)
            ftp.voidcmd("NOOP")
            ftp.quit()
            return bool(banner)
        except Exception:  # noqa: BLE001
            return False

    def _wait_accepting(self, timeout: float) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            with socket.socket() as s:
                s.settimeout(0.5)
                try:
                    s.connect((self.target.host, self.target.control_port))
                    return True
                except OSError:
                    time.sleep(0.1)
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


# ---------------------------------------------------------------------------
# tcpdump capture (task 2 + 5) — real capture, clean stop, lab ports only
# ---------------------------------------------------------------------------


@dataclass
class Tcpdump:
    pcap_path: Path
    target: LabTarget
    proc: subprocess.Popen | None = None
    started_at: float = 0.0
    stopped_at: float = 0.0

    def start(self, settle: float = 0.8, timeout: float = 6.0) -> "Tcpdump":
        self.pcap_path = Path(self.pcap_path)
        self.pcap_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            ["tcpdump", "-i", self.target.iface, "-w", str(self.pcap_path),
             "-U", "-n", self.target.bpf()],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # tcpdump prints "listening on ..." once the capture socket is open
        end = time.time() + timeout
        while time.time() < end:
            line = self.proc.stderr.readline() if self.proc.stderr else b""
            if b"listening on" in line:
                break
            if self.proc.poll() is not None:
                err = self.proc.stderr.read() if self.proc.stderr else b""
                raise RuntimeError(f"tcpdump failed to start: {err.decode('utf-8','ignore')}")
        time.sleep(settle)          # ensure the socket is really capturing
        self.started_at = time.time()
        return self

    def stop(self, drain: float = 0.8):
        time.sleep(drain)           # let final packets be captured + flushed
        self.stopped_at = time.time()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    @property
    def wall_duration(self) -> float:
        return max(0.0, self.stopped_at - self.started_at)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


# ---------------------------------------------------------------------------
# Real FTP scenarios — benign (task 8) and brute force (task 9)
# Each returns a stats dict; every call makes NEW real connections.
# ---------------------------------------------------------------------------


def _connect(target: LabTarget, timeout: float = 5.0) -> FTP:
    ftp = FTP()
    ftp.connect(target.host, target.control_port, timeout=timeout)
    return ftp


# ---- benign ---------------------------------------------------------------


def benign_login(t: LabTarget) -> dict:
    ftp = _connect(t); ftp.login(t.user, t.password); ftp.quit()
    return {"attempts": 1, "successes": 1, "commands": ["USER", "PASS", "QUIT"]}


def benign_login_list(t: LabTarget) -> dict:
    ftp = _connect(t); ftp.login(t.user, t.password)
    ftp.retrlines("LIST", lambda _l: None)
    ftp.quit()
    return {"attempts": 1, "successes": 1, "commands": ["USER", "PASS", "LIST", "QUIT"]}


def benign_login_commands(t: LabTarget) -> dict:
    ftp = _connect(t); ftp.login(t.user, t.password)
    for cmd in ("PWD", "SYST", "TYPE I", "TYPE A", "NOOP"):
        try:
            ftp.voidcmd(cmd) if cmd.startswith(("TYPE", "NOOP")) else ftp.sendcmd(cmd)
        except Exception:  # noqa: BLE001
            pass
    ftp.quit()
    return {"attempts": 1, "successes": 1,
            "commands": ["USER", "PASS", "PWD", "SYST", "TYPE", "NOOP", "QUIT"]}


def benign_download(t: LabTarget) -> dict:
    ftp = _connect(t); ftp.login(t.user, t.password)
    buf = []
    ftp.retrbinary(f"RETR {SAFE_TEST_FILE}", buf.append)
    ftp.quit()
    return {"attempts": 1, "successes": 1, "bytes_down": sum(len(b) for b in buf),
            "commands": ["USER", "PASS", "RETR", "QUIT"]}


def benign_upload(t: LabTarget) -> dict:
    import io
    ftp = _connect(t); ftp.login(t.user, t.password)
    payload = b"safe lab upload payload\n" * 30
    ftp.storbinary(f"STOR uploads/upload_{int(time.time()*1000)}.txt", io.BytesIO(payload))
    ftp.quit()
    return {"attempts": 1, "successes": 1, "bytes_up": len(payload),
            "commands": ["USER", "PASS", "STOR", "QUIT"]}


def benign_multi_session(t: LabTarget, sessions: int = 4) -> dict:
    for _ in range(sessions):
        ftp = _connect(t); ftp.login(t.user, t.password)
        ftp.retrlines("LIST", lambda _l: None)
        ftp.quit()
        time.sleep(0.15)
    return {"attempts": sessions, "successes": sessions, "sessions": sessions,
            "commands": ["USER", "PASS", "LIST", "QUIT"] * sessions}


def benign_long_session(t: LabTarget, hold: float = 0.6, rounds: int = 6) -> dict:
    ftp = _connect(t); ftp.login(t.user, t.password)
    for _ in range(rounds):
        ftp.voidcmd("NOOP")
        ftp.retrlines("LIST", lambda _l: None)
        time.sleep(hold)            # real idle => real longer duration
    ftp.quit()
    return {"attempts": 1, "successes": 1, "rounds": rounds,
            "commands": ["USER", "PASS", "NOOP", "LIST", "QUIT"]}


def benign_short_session(t: LabTarget) -> dict:
    ftp = _connect(t); ftp.login(t.user, t.password); ftp.voidcmd("NOOP"); ftp.quit()
    return {"attempts": 1, "successes": 1, "commands": ["USER", "PASS", "NOOP", "QUIT"]}


# ---- brute force (real failed logins against the lab target only) ---------


def _try_login(t: LabTarget, user: str, pw: str) -> bool:
    """One real FTP login attempt on a fresh connection. True iff it succeeded."""
    ftp = _connect(t)
    try:
        ftp.login(user, pw)
        ok = True
    except (error_perm, error_temp):
        ok = False
    finally:
        try:
            ftp.quit()
        except Exception:  # noqa: BLE001
            try:
                ftp.close()
            except Exception:  # noqa: BLE001
                pass
    return ok


def _bruteforce(t: LabTarget, creds, delay: float, pattern: str) -> dict:
    """
    Run a real credential-guessing sequence against the LAB target only.
    ``creds`` is a list of (user, password) pairs, all wrong on purpose.
    ``pattern`` = 'new_conn' (fresh connection per attempt) or
    'single_conn' (one control connection, repeated USER/PASS).
    """
    assert t.is_loopback(), "brute force is only ever run against the loopback lab"
    attempts = successes = failures = 0
    users = set()
    if pattern == "single_conn":
        reconnects = 0
        ftp = _connect(t)
        for user, pw in creds:
            attempts += 1; users.add(user)
            try:
                ftp.sendcmd(f"USER {user}")
                ftp.sendcmd(f"PASS {pw}")
                successes += 1
            except (error_perm, error_temp):
                failures += 1
            except (EOFError, OSError):
                # server dropped the control connection -- a real brute-force
                # tool would reconnect and keep going. Count the attempt as a
                # failure and resume on a fresh connection.
                failures += 1
                reconnects += 1
                try:
                    ftp.close()
                except Exception:  # noqa: BLE001
                    pass
                ftp = _connect(t)
            if delay:
                time.sleep(delay)
        try:
            ftp.quit()
        except Exception:  # noqa: BLE001
            ftp.close()
        return {"attempts": attempts, "successes": successes, "failures": failures,
                "distinct_usernames": len(users), "delay_s": delay,
                "pattern": pattern, "reconnects": reconnects}
    else:
        for user, pw in creds:
            attempts += 1; users.add(user)
            if _try_login(t, user, pw):
                successes += 1
            else:
                failures += 1
            if delay:
                time.sleep(delay)
    return {"attempts": attempts, "successes": successes, "failures": failures,
            "distinct_usernames": len(users), "delay_s": delay, "pattern": pattern}


# password / username pools (all invalid against the lab creds)
_WRONG_PW = ["123456", "password", "admin", "root", "qwerty", "letmein",
             "ftp", "test123", "welcome", "changeme", "pass1234", "iloveyou"]
_WRONG_USERS = ["admin", "root", "ftpadmin", "anonymous", "user", "guest",
                "administrator", "oracle"]


def bf_slow_failed(t: LabTarget) -> dict:
    creds = [("admin", pw) for pw in _WRONG_PW[:6]]
    return _bruteforce(t, creds, delay=0.8, pattern="new_conn")


def bf_fast_failed(t: LabTarget) -> dict:
    creds = [("admin", pw) for pw in _WRONG_PW]
    return _bruteforce(t, creds, delay=0.02, pattern="new_conn")


def bf_diff_usernames(t: LabTarget) -> dict:
    creds = [(u, "password") for u in _WRONG_USERS]
    return _bruteforce(t, creds, delay=0.1, pattern="new_conn")


def bf_diff_passwords(t: LabTarget) -> dict:
    creds = [("labuser", pw) for pw in _WRONG_PW]        # right user, wrong pws
    return _bruteforce(t, creds, delay=0.1, pattern="new_conn")


def bf_many_attempts(t: LabTarget) -> dict:
    creds = [(u, pw) for u in _WRONG_USERS[:4] for pw in _WRONG_PW[:6]]  # 24
    return _bruteforce(t, creds, delay=0.02, pattern="new_conn")


def bf_few_attempts(t: LabTarget) -> dict:
    creds = [("root", pw) for pw in _WRONG_PW[:3]]
    return _bruteforce(t, creds, delay=0.3, pattern="new_conn")


def bf_single_conn_many(t: LabTarget) -> dict:
    creds = [("admin", pw) for pw in _WRONG_PW[:8]]
    return _bruteforce(t, creds, delay=0.1, pattern="single_conn")


def bf_multi_burst(t: LabTarget) -> dict:
    total = {"attempts": 0, "successes": 0, "failures": 0, "distinct_usernames": 0}
    users = set()
    for burst in range(3):
        creds = [(_WRONG_USERS[burst], pw) for pw in _WRONG_PW[:4]]
        r = _bruteforce(t, creds, delay=0.05, pattern="new_conn")
        for k in ("attempts", "successes", "failures"):
            total[k] += r[k]
        users.add(_WRONG_USERS[burst])
        time.sleep(0.3)             # real gap between bursts
    total["distinct_usernames"] = len(users)
    total["pattern"] = "multi_burst"
    total["delay_s"] = 0.05
    return total


# ---- the declarative scenario catalog ------------------------------------

BENIGN_SCENARIOS = [
    ("successful_login", benign_login),
    ("login_then_listing", benign_login_list),
    ("login_then_commands", benign_login_commands),
    ("file_download", benign_download),
    ("file_upload", benign_upload),
    ("multiple_sessions", benign_multi_session),
    ("long_session", benign_long_session),
    ("short_session", benign_short_session),
]

BRUTEFORCE_SCENARIOS = [
    ("slow_failed_logins", bf_slow_failed),
    ("fast_failed_logins", bf_fast_failed),
    ("different_usernames", bf_diff_usernames),
    ("different_passwords", bf_diff_passwords),
    ("many_attempts", bf_many_attempts),
    ("few_attempts", bf_few_attempts),
    ("single_connection_many_attempts", bf_single_conn_many),
    ("multi_burst", bf_multi_burst),
]


# ---------------------------------------------------------------------------
# PCAP verification (task 6) — independent of the model, no predictions
# ---------------------------------------------------------------------------


@dataclass
class PcapCheck:
    pcap: str
    exists: bool = False
    readable: bool = False
    packets: int = 0
    has_ftp_traffic: bool = False
    src_ok: bool = False
    dst_ok: bool = False
    protocol_ok: bool = False
    port_ok: bool = False
    duration_s: float = 0.0
    duration_valid: bool = False
    errors: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.exists and self.readable and self.packets > 0
                and self.has_ftp_traffic and self.src_ok and self.dst_ok
                and self.protocol_ok and self.port_ok and self.duration_valid)


def verify_pcap(pcap_path, target: LabTarget) -> PcapCheck:
    """
    Verify a capture WITHOUT any model: it exists, is readable, has packets,
    carries FTP traffic on the lab port, is loopback-only TCP, and has a valid
    (finite, non-negative) duration.
    """
    from scapy.all import rdpcap, TCP, IP

    p = Path(pcap_path)
    chk = PcapCheck(pcap=p.name, exists=p.is_file())
    if not chk.exists:
        chk.errors.append("file does not exist")
        return chk
    try:
        pkts = rdpcap(str(p))
        chk.readable = True
    except Exception as e:  # noqa: BLE001
        chk.errors.append(f"unreadable: {e}")
        return chk

    chk.packets = len(pkts)
    if chk.packets == 0:
        chk.errors.append("no packets")
        return chk

    times, tcp_pkts = [], 0
    src_all_lab = dst_all_lab = True
    saw_ftp_port = False
    for pk in pkts:
        times.append(float(pk.time))
        if IP in pk:
            if pk[IP].src != target.host:
                src_all_lab = False
            if pk[IP].dst != target.host:
                dst_all_lab = False
        if TCP in pk:
            tcp_pkts += 1
            if target.control_port in (pk[TCP].sport, pk[TCP].dport):
                saw_ftp_port = True

    chk.protocol_ok = tcp_pkts > 0
    chk.port_ok = saw_ftp_port
    chk.has_ftp_traffic = saw_ftp_port and tcp_pkts > 0
    chk.src_ok = src_all_lab
    chk.dst_ok = dst_all_lab
    if times:
        d = max(times) - min(times)
        chk.duration_s = float(d)
        chk.duration_valid = (d >= 0) and (d == d) and (d != float("inf"))
    if not chk.protocol_ok:
        chk.errors.append("no TCP packets")
    if not chk.port_ok:
        chk.errors.append(f"no traffic on FTP port {target.control_port}")
    if not (chk.src_ok and chk.dst_ok):
        chk.errors.append("traffic not confined to loopback lab host")
    return chk


# ---------------------------------------------------------------------------
# Capture metadata (task 7) + manifest (task 13)
# ---------------------------------------------------------------------------


@dataclass
class CaptureMeta:
    capture_id: str
    label: str
    scenario: str
    timestamp: str
    source: str
    destination: str
    client_tool: str
    attempts: int
    capture_duration_s: float
    pcap_filename: str
    validation_status: str
    # extra scenario detail kept for the report / future retraining
    detail: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("detail", None)
        return d


MANIFEST_COLUMNS = [
    "capture_id", "label", "scenario", "timestamp", "source", "destination",
    "client_tool", "attempts", "capture_duration_s", "pcap_filename",
    "validation_status",
]
