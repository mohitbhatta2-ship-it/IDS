"""
Realistic-PCAP DIVERSIFICATION framework v2 (experimental; production untouched).

Builds a substantially more diverse REAL FTP corpus than v1 by varying genuine
behaviour -- not by sleeping to fake variety. Diversity comes from real axes:

  * CLIENTS: python-ftplib, curl, wget, and a hand-written raw-socket FTP speaker
    -- genuinely different implementations, command sequencing and timing.
  * SERVER VARIANTS (all real pyftpdlib, different *behaviour*): ``ratelimited``
    (the default 3 s ``auth_failed_timeout`` -> a defended server that answers
    failed logins slowly), ``permissive`` (no rate limit, fast answers), and
    ``throttled`` (a real ``ThrottledDTPHandler`` bandwidth cap -> genuinely
    slower, differently-segmented transfers). Slow/fast pacing is therefore a
    property of the real server, never an injected ``sleep``.
  * MODE: passive vs active data connections (a real protocol difference).
  * ENVIRONMENT: three loopback addresses (127.0.0.1/.2/.3) -> genuinely
    different source/destination, all provably inside 127.0.0.0/8.

Every capture is a fresh ``tcpdump`` over fresh connections against a server this
process starts and controls, on loopback only -- nothing external is contacted,
and the ground-truth label is the scenario/folder, never a prediction.

Reuses v1's ``Tcpdump`` (interface/BPF-driven, environment-agnostic). Nothing here
imports into or changes ml.py / live_capture.py / pcap_validation.py (used read
only for extraction) or any frozen v1 result.
"""

from __future__ import annotations

import ipaddress
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from ftplib import FTP, error_perm, error_temp
from pathlib import Path

from .lab_capture import Tcpdump          # reuse the v1 capture primitive (unchanged)

# ---------------------------------------------------------------------------
# Environments, server variants, ports
# ---------------------------------------------------------------------------

ENV_ADDR = {"env_a": "127.0.0.1", "env_b": "127.0.0.2", "env_c": "127.0.0.3"}
VARIANT_PORT = {"ratelimited": 21, "permissive": 2121, "throttled": 2221}
SERVER_VARIANTS = tuple(VARIANT_PORT)

USER = "labuser"
PASSWORD = "labpass"
SAFE_TEST_FILE = "labfile.txt"


@dataclass(frozen=True)
class V2Target:
    """One (environment, server-variant) endpoint. Provides iface+bpf for Tcpdump."""
    addr: str
    control_port: int
    passive_lo: int
    passive_hi: int
    variant: str
    env_key: str
    user: str = USER
    password: str = PASSWORD
    iface: str = "lo"

    def is_loopback_net(self) -> bool:
        return ipaddress.ip_address(self.addr) in ipaddress.ip_network("127.0.0.0/8")

    def bpf(self) -> str:
        # confine capture to THIS endpoint: its host, control port, active-mode
        # data port 20, and its own passive range. Loopback only.
        return (f"host {self.addr} and (tcp port {self.control_port} "
                f"or tcp port 20 or tcp portrange {self.passive_lo}-{self.passive_hi})")


def build_targets(root: Path) -> dict[tuple[str, str], V2Target]:
    """A V2Target for every (env, variant), each with a unique passive range."""
    targets, idx = {}, 0
    for env_key in ENV_ADDR:
        for variant in SERVER_VARIANTS:
            plo = 60000 + idx * 40
            targets[(env_key, variant)] = V2Target(
                addr=ENV_ADDR[env_key], control_port=VARIANT_PORT[variant],
                passive_lo=plo, passive_hi=plo + 39, variant=variant, env_key=env_key)
            idx += 1
    return targets


# ---------------------------------------------------------------------------
# Server (real pyftpdlib; behaviour varies by variant), one isolated subprocess
# ---------------------------------------------------------------------------

_SERVER_CODE = r"""
import sys, logging
logging.disable(logging.CRITICAL)
addr, port, plo, phi, user, pw, variant, root = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler, ThrottledDTPHandler
from pyftpdlib.servers import FTPServer
auth = DummyAuthorizer()
auth.add_user(user, pw, root, perm="elradfmw")
h = FTPHandler
h.authorizer = auth
h.passive_ports = range(plo, phi + 1)
h.masquerade_address = addr
h.banner = "pyftpdlib lab (%s) ready" % variant
if variant == "ratelimited":
    h.auth_failed_timeout = 3          # real defended-server delay on bad login
    h.max_login_attempts = 3
elif variant == "permissive":
    h.auth_failed_timeout = 0          # fast answers, many attempts allowed
    h.max_login_attempts = 9999
elif variant == "throttled":
    h.auth_failed_timeout = 0
    h.max_login_attempts = 9999
    d = ThrottledDTPHandler
    d.read_limit = 8192                # real 8 KB/s data cap -> real slow transfers
    d.write_limit = 8192
    h.dtp_handler = d
FTPServer((addr, port), h).serve_forever()
"""


@dataclass
class LabServer:
    target: V2Target
    root: Path
    proc: subprocess.Popen | None = None

    def start(self, timeout: float = 8.0) -> "LabServer":
        t = self.target
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _SERVER_CODE, t.addr, str(t.control_port),
             str(t.passive_lo), str(t.passive_hi), t.user, t.password, t.variant,
             str(self.root)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if not self._wait(timeout):
            err = self.proc.stderr.read() if self.proc.stderr else b""
            self.stop()
            raise RuntimeError(f"server {t.addr}:{t.control_port} ({t.variant}) "
                               f"did not start: {err.decode('utf-8','ignore')}")
        return self

    def verify(self) -> bool:
        t = self.target
        try:
            ftp = FTP(); ftp.connect(t.addr, t.control_port, timeout=4)
            ok = bool(ftp.getwelcome()); ftp.login(t.user, t.password); ftp.quit()
            return ok
        except Exception:  # noqa: BLE001
            return False

    def _wait(self, timeout):
        end = time.time() + timeout
        while time.time() < end:
            with socket.socket() as s:
                s.settimeout(0.4)
                try:
                    s.connect((self.target.addr, self.target.control_port)); return True
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


class ServerPool:
    """Starts one real server per (env, variant) needed and reuses it."""

    def __init__(self, targets, root: Path):
        self.targets = targets
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "uploads").mkdir(exist_ok=True)
        (self.root / SAFE_TEST_FILE).write_text(
            "safe local lab test file for FTP transfer diversity tests.\n" * 60)
        (self.root / "subdir").mkdir(exist_ok=True)
        (self.root / "subdir" / "nested.txt").write_text("nested lab file\n" * 20)
        self.servers: dict[tuple[str, str], LabServer] = {}

    def get(self, env_key, variant) -> V2Target:
        key = (env_key, variant)
        if key not in self.servers:
            self.servers[key] = LabServer(self.targets[key], self.root).start()
            if not self.servers[key].verify():
                raise RuntimeError(f"server {key} failed verification")
        return self.targets[key]

    def stop_all(self):
        for s in self.servers.values():
            s.stop()
        self.servers.clear()


# ---------------------------------------------------------------------------
# Clients -- real, different implementations
# ---------------------------------------------------------------------------


def _ftp(t: V2Target, passive=True, timeout=6.0) -> FTP:
    f = FTP(); f.connect(t.addr, t.control_port, timeout=timeout)
    f.set_pasv(passive)
    return f


# ---- benign (ftplib) ------------------------------------------------------

def b_login(t, passive=True):
    f = _ftp(t, passive); f.login(t.user, t.password); f.quit()
    return {"attempts": 1, "successes": 1}


def b_login_fail_then_success(t, passive=True):
    f = _ftp(t, passive)
    fails = 0
    try:
        f.login(t.user, "not-the-password")
    except (error_perm, error_temp):
        fails = 1
        f.close()
        f = _ftp(t, passive)
    f.login(t.user, t.password); f.quit()
    return {"attempts": 1 + fails, "successes": 1, "failed_benign_logins": fails}


def b_list(t, passive=True):
    f = _ftp(t, passive); f.login(t.user, t.password)
    f.retrlines("LIST", lambda _l: None); f.quit()
    return {"attempts": 1, "successes": 1}


def b_download(t, passive=True):
    f = _ftp(t, passive); f.login(t.user, t.password)
    buf = []; f.retrbinary(f"RETR {SAFE_TEST_FILE}", buf.append); f.quit()
    return {"attempts": 1, "successes": 1, "bytes_down": sum(len(b) for b in buf)}


def b_upload(t, passive=True):
    import io
    f = _ftp(t, passive); f.login(t.user, t.password)
    payload = b"safe lab upload payload\n" * 40
    f.storbinary(f"STOR uploads/up_{int(time.time()*1000)}.txt", io.BytesIO(payload))
    f.quit()
    return {"attempts": 1, "successes": 1, "bytes_up": len(payload)}


def b_multi_session(t, passive=True, sessions=3):
    for _ in range(sessions):
        f = _ftp(t, passive); f.login(t.user, t.password)
        f.retrlines("LIST", lambda _l: None); f.quit()
    return {"attempts": sessions, "successes": sessions, "sessions": sessions}


def b_reconnect(t, passive=True):
    f = _ftp(t, passive); f.login(t.user, t.password); f.quit()
    f = _ftp(t, passive); f.login(t.user, t.password)      # genuine reconnect
    f.retrlines("LIST", lambda _l: None); f.quit()
    return {"attempts": 2, "successes": 2, "reconnects": 1}


def b_interactive(t, passive=True):
    """A longer, genuinely multi-command interactive session (no sleeps)."""
    f = _ftp(t, passive); f.login(t.user, t.password)
    for cmd in ("PWD", "SYST", "TYPE I", "TYPE A", "NOOP"):
        try:
            f.voidcmd(cmd) if cmd.startswith(("TYPE", "NOOP")) else f.sendcmd(cmd)
        except Exception:  # noqa: BLE001
            pass
    f.cwd("subdir"); f.retrlines("LIST", lambda _l: None)
    buf = []; f.retrbinary("RETR nested.txt", buf.append)
    f.cwd(".."); f.quit()
    return {"attempts": 1, "successes": 1, "bytes_down": sum(len(b) for b in buf)}


def b_short(t, passive=True):
    f = _ftp(t, passive); f.login(t.user, t.password); f.quit()
    return {"attempts": 1, "successes": 1}


# ---- benign (curl / wget: real external client binaries) ------------------

def _url(t, path=""):
    return f"ftp://{t.user}:{t.password}@{t.addr}:{t.control_port}/{path}"


def b_curl_download(t, passive=True):
    mode = "--ftp-pasv" if passive else "--ftp-port -"
    r = subprocess.run(["curl", "-s", "-S", *mode.split(), _url(t, SAFE_TEST_FILE),
                        "-o", "/dev/null"], capture_output=True, text=True)
    return {"attempts": 1, "successes": int(r.returncode == 0), "client_rc": r.returncode}


def b_curl_list(t, passive=True):
    mode = "--ftp-pasv" if passive else "--ftp-port -"
    r = subprocess.run(["curl", "-s", "-S", *mode.split(), _url(t, "")],
                       capture_output=True, text=True)
    return {"attempts": 1, "successes": int(r.returncode == 0), "client_rc": r.returncode}


def b_wget_download(t, passive=True):
    r = subprocess.run(["wget", "-q", "-O", "/dev/null", _url(t, SAFE_TEST_FILE)],
                       capture_output=True, text=True)
    return {"attempts": 1, "successes": int(r.returncode == 0), "client_rc": r.returncode}


def b_wget_list(t, passive=True):
    import tempfile, os
    tmp = tempfile.mktemp()
    r = subprocess.run(["wget", "-q", "-O", tmp, _url(t, "")], capture_output=True, text=True)
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return {"attempts": 1, "successes": int(r.returncode == 0), "client_rc": r.returncode}


# ---- benign (raw socket: our own FTP speaker) -----------------------------

def _raw_cmd(sock, line, read=True):
    sock.sendall((line + "\r\n").encode())
    if read:
        try:
            return sock.recv(4096)
        except OSError:
            return b""
    return b""


def b_raw_login(t, passive=True):
    s = socket.create_connection((t.addr, t.control_port), timeout=6)
    s.recv(4096)
    _raw_cmd(s, f"USER {t.user}"); _raw_cmd(s, f"PASS {t.password}")
    _raw_cmd(s, "PWD"); _raw_cmd(s, "NOOP"); _raw_cmd(s, "QUIT", read=False)
    s.close()
    return {"attempts": 1, "successes": 1}


# ---- brute force (ftplib) -------------------------------------------------

WRONG_PW = ["123456", "password", "admin", "root", "qwerty", "letmein", "ftp",
            "test123", "welcome", "changeme", "pass1234", "iloveyou", "1234",
            "monkey", "dragon", "master"]
WRONG_USERS = ["admin", "root", "ftpadmin", "anonymous", "user", "guest",
               "administrator", "oracle", "ftpuser", "webadmin"]


def _one_login(t, user, pw, passive=True) -> bool:
    f = _ftp(t, passive)
    try:
        f.login(user, pw); ok = True
    except (error_perm, error_temp):
        ok = False
    finally:
        try:
            f.quit()
        except Exception:  # noqa: BLE001
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass
    return ok


def bf_newconn(t, creds, passive=True):
    """Fresh connection per attempt (pacing = real server response timing)."""
    assert t.is_loopback_net()
    a = s = fail = 0; users = set()
    for u, pw in creds:
        a += 1; users.add(u)
        if _one_login(t, u, pw, passive):
            s += 1
        else:
            fail += 1
    return {"attempts": a, "successes": s, "failures": fail,
            "distinct_usernames": len(users), "pattern": "new_conn"}


def bf_singleconn(t, creds, passive=True):
    """One control connection, many USER/PASS (needs a permissive server)."""
    assert t.is_loopback_net()
    a = s = fail = rec = 0; users = set()
    f = _ftp(t, passive)
    for u, pw in creds:
        a += 1; users.add(u)
        try:
            f.sendcmd(f"USER {u}"); f.sendcmd(f"PASS {pw}"); s += 1
        except (error_perm, error_temp):
            fail += 1
        except (EOFError, OSError):
            fail += 1; rec += 1
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass
            f = _ftp(t, passive)
    try:
        f.quit()
    except Exception:  # noqa: BLE001
        f.close()
    return {"attempts": a, "successes": s, "failures": fail,
            "distinct_usernames": len(users), "pattern": "single_conn", "reconnects": rec}


def bf_multiburst(t, users, pws, passive=True):
    assert t.is_loopback_net()
    a = fail = 0; seen = set()
    for u in users:
        for pw in pws:
            a += 1; seen.add(u)
            if not _one_login(t, u, pw, passive):
                fail += 1
    return {"attempts": a, "successes": a - fail, "failures": fail,
            "distinct_usernames": len(seen), "pattern": "multi_burst",
            "bursts": len(users)}


# ---- brute force (curl / raw socket) --------------------------------------

def bf_curl(t, creds, passive=True):
    """Each guess is a fresh real curl process (a genuinely different client)."""
    assert t.is_loopback_net()
    mode = "--ftp-pasv" if passive else "--ftp-port -"
    a = s = fail = 0; users = set()
    for u, pw in creds:
        a += 1; users.add(u)
        url = f"ftp://{u}:{pw}@{t.addr}:{t.control_port}/"
        r = subprocess.run(["curl", "-s", "-S", *mode.split(), url],
                           capture_output=True, text=True)
        if r.returncode == 0:
            s += 1
        else:
            fail += 1
    return {"attempts": a, "successes": s, "failures": fail,
            "distinct_usernames": len(users), "pattern": "curl_new_conn"}


def bf_raw_pipelined(t, creds, passive=True):
    """One socket, pipelined USER/PASS -- genuinely different client timing."""
    assert t.is_loopback_net()
    a = s = fail = rec = 0; users = set()
    sock = socket.create_connection((t.addr, t.control_port), timeout=6); sock.recv(4096)
    for u, pw in creds:
        a += 1; users.add(u)
        try:
            _raw_cmd(sock, f"USER {u}")
            resp = _raw_cmd(sock, f"PASS {pw}")
            if resp[:3] == b"230":
                s += 1
            else:
                fail += 1
            if not resp:
                raise OSError("closed")
        except OSError:
            fail += 1; rec += 1
            try:
                sock.close()
            except OSError:
                pass
            sock = socket.create_connection((t.addr, t.control_port), timeout=6)
            sock.recv(4096)
    try:
        _raw_cmd(sock, "QUIT", read=False); sock.close()
    except OSError:
        pass
    return {"attempts": a, "successes": s, "failures": fail,
            "distinct_usernames": len(users), "pattern": "raw_pipelined", "reconnects": rec}


# ---------------------------------------------------------------------------
# Verification (task: independent, model-free, strict loopback)
# ---------------------------------------------------------------------------


@dataclass
class PcapCheck:
    pcap: str
    exists: bool = False
    readable: bool = False
    packets: int = 0
    has_ftp_traffic: bool = False
    loopback_only: bool = False
    expected_host_present: bool = False
    tcp_only: bool = False
    port_ok: bool = False
    duration_s: float = 0.0
    duration_valid: bool = False
    errors: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.exists and self.readable and self.packets > 0 and self.has_ftp_traffic
                and self.loopback_only and self.expected_host_present and self.tcp_only
                and self.port_ok and self.duration_valid)


def verify_pcap(pcap_path, target: V2Target) -> PcapCheck:
    from scapy.all import rdpcap, TCP, UDP, IP

    p = Path(pcap_path)
    c = PcapCheck(pcap=p.name, exists=p.is_file())
    if not c.exists:
        c.errors.append("missing"); return c
    try:
        pkts = rdpcap(str(p)); c.readable = True
    except Exception as e:  # noqa: BLE001
        c.errors.append(f"unreadable: {e}"); return c
    c.packets = len(pkts)
    if not pkts:
        c.errors.append("no packets"); return c

    loop_net = ipaddress.ip_network("127.0.0.0/8")
    times, tcp, non_tcp, saw_port, saw_host, loop_ok = [], 0, 0, False, False, True
    for pk in pkts:
        times.append(float(pk.time))
        if IP in pk:
            for a in (pk[IP].src, pk[IP].dst):
                if ipaddress.ip_address(a) not in loop_net:
                    loop_ok = False
            if target.addr in (pk[IP].src, pk[IP].dst):
                saw_host = True
        if TCP in pk:
            tcp += 1
            if target.control_port in (pk[TCP].sport, pk[TCP].dport):
                saw_port = True
        elif UDP in pk:
            non_tcp += 1
    c.loopback_only = loop_ok
    c.expected_host_present = saw_host
    c.tcp_only = non_tcp == 0
    c.port_ok = saw_port
    c.has_ftp_traffic = saw_port and tcp > 0
    if times:
        d = max(times) - min(times)
        c.duration_s = float(d)
        c.duration_valid = d >= 0 and d == d and d != float("inf")
    if not c.loopback_only:
        c.errors.append("non-loopback address present")
    if not c.port_ok:
        c.errors.append(f"no traffic on control port {target.control_port}")
    if not c.tcp_only:
        c.errors.append("non-TCP payload present")
    if not c.duration_valid:
        c.errors.append("invalid duration")
    return c


# ---------------------------------------------------------------------------
# Metadata + manifest
# ---------------------------------------------------------------------------


@dataclass
class CaptureMeta:
    capture_id: str
    label: str
    scenario: str
    client: str
    server: str
    environment: str
    interface: str
    mode: str
    attempts: int
    start_time: str
    end_time: str
    packet_count: int
    duration_s: float
    source: str
    destination: str
    capture_command: str
    verification_status: str
    detail: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = asdict(self); d.pop("detail", None); return d


MANIFEST_COLUMNS = [
    "capture_id", "label", "scenario", "client", "server", "environment",
    "interface", "mode", "attempts", "start_time", "end_time", "packet_count",
    "duration_s", "source", "destination", "capture_command", "verification_status",
]


# ---------------------------------------------------------------------------
# The capture matrix -- genuine axis combinations, sized ~40 benign / ~40 bf
# ---------------------------------------------------------------------------


@dataclass
class Spec:
    label: str
    scenario: str
    client: str
    server: str          # variant key
    env: str             # env key
    mode: str            # passive | active
    fn: object           # callable(target) -> stats
    params: dict = field(default_factory=dict)


def _cyc(seq, i):
    return seq[i % len(seq)]


def benign_specs() -> list[Spec]:
    envs = list(ENV_ADDR)
    specs, i = [], 0

    # ftplib behaviours across rotating (server, env, mode)
    ftplib_behaviours = [
        ("successful_login", b_login),
        ("failed_then_successful_login", b_login_fail_then_success),
        ("directory_listing", b_list),
        ("download", b_download),
        ("upload", b_upload),
        ("multiple_sessions", b_multi_session),
        ("reconnect", b_reconnect),
        ("interactive_session", b_interactive),
        ("short_session", b_short),
    ]
    for name, fn in ftplib_behaviours:
        for server in ("ratelimited", "permissive", "throttled"):
            mode = _cyc(("passive", "active"), i)
            specs.append(Spec("Benign", name, "python-ftplib", server,
                              _cyc(envs, i), mode, fn)); i += 1

    # curl behaviours (passive + active) across envs/servers
    for name, fn in (("download", b_curl_download), ("directory_listing", b_curl_list)):
        for mode in ("passive", "active"):
            for server in ("permissive", "throttled"):
                specs.append(Spec("Benign", name, "curl", server,
                                  _cyc(envs, i), mode, fn)); i += 1

    # wget behaviours (passive only -- wget FTP is passive)
    for name, fn in (("download", b_wget_download), ("directory_listing", b_wget_list)):
        for server in ("permissive", "throttled"):
            specs.append(Spec("Benign", name, "wget", server,
                              _cyc(envs, i), "passive", fn)); i += 1

    # raw-socket login across envs
    for env in envs:
        specs.append(Spec("Benign", "raw_socket_login", "raw-socket", "permissive",
                          env, "passive", b_raw_login)); i += 1
    return specs


def _creds_user(user, pws):
    return [(user, pw) for pw in pws]


def _creds_users(users, pw):
    return [(u, pw) for u in users]


def bruteforce_specs() -> list[Spec]:
    envs = list(ENV_ADDR)
    specs, i = [], 0

    def add(scenario, client, server, mode, fn, **params):
        nonlocal i
        specs.append(Spec("FTP-BruteForce", scenario, client, server,
                          _cyc(envs, i), mode, fn, params)); i += 1

    # ftplib, fresh-connection, varied pacing via SERVER behaviour + attempt count
    add("slow_pacing_defended_server", "python-ftplib", "ratelimited", "passive",
        lambda t, **p: bf_newconn(t, _creds_user("admin", WRONG_PW[:3])))
    add("slow_pacing_defended_server_active", "python-ftplib", "ratelimited", "active",
        lambda t, **p: bf_newconn(t, _creds_user("root", WRONG_PW[3:6]), passive=False))
    add("medium_pacing_defended_server", "python-ftplib", "ratelimited", "passive",
        lambda t, **p: bf_newconn(t, _creds_user("admin", WRONG_PW[:5])))
    add("fast_pacing_permissive_server", "python-ftplib", "permissive", "passive",
        lambda t, **p: bf_newconn(t, _creds_user("admin", WRONG_PW)))
    add("fast_pacing_permissive_active", "python-ftplib", "permissive", "active",
        lambda t, **p: bf_newconn(t, _creds_user("admin", WRONG_PW), passive=False))
    add("different_usernames", "python-ftplib", "permissive", "passive",
        lambda t, **p: bf_newconn(t, _creds_users(WRONG_USERS, "password")))
    add("different_passwords_valid_user", "python-ftplib", "permissive", "passive",
        lambda t, **p: bf_newconn(t, _creds_user("labuser", WRONG_PW)))
    add("few_attempts", "python-ftplib", "ratelimited", "passive",
        lambda t, **p: bf_newconn(t, _creds_user("root", WRONG_PW[:3])))
    add("many_attempts", "python-ftplib", "permissive", "passive",
        lambda t, **p: bf_newconn(t, [(u, pw) for u in WRONG_USERS[:4] for pw in WRONG_PW[:6]]))
    add("single_connection_many", "python-ftplib", "permissive", "passive",
        lambda t, **p: bf_singleconn(t, _creds_user("admin", WRONG_PW)))
    add("single_connection_active", "python-ftplib", "permissive", "active",
        lambda t, **p: bf_singleconn(t, _creds_user("admin", WRONG_PW[:8]), passive=False))
    add("multi_burst", "python-ftplib", "permissive", "passive",
        lambda t, **p: bf_multiburst(t, WRONG_USERS[:4], WRONG_PW[:4]))
    add("multi_burst_large", "python-ftplib", "permissive", "passive",
        lambda t, **p: bf_multiburst(t, WRONG_USERS[:5], WRONG_PW[:5]))
    add("repeated_connections", "python-ftplib", "permissive", "passive",
        lambda t, **p: bf_newconn(t, _creds_user("guest", WRONG_PW[:8])))

    # curl-driven brute force (different client), varied server/mode/creds
    add("curl_fast", "curl", "permissive", "passive",
        lambda t, **p: bf_curl(t, _creds_user("admin", WRONG_PW[:8])))
    add("curl_active", "curl", "permissive", "active",
        lambda t, **p: bf_curl(t, _creds_user("root", WRONG_PW[:6]), passive=False))
    add("curl_different_usernames", "curl", "permissive", "passive",
        lambda t, **p: bf_curl(t, _creds_users(WRONG_USERS[:8], "letmein")))
    add("curl_defended_server", "curl", "ratelimited", "passive",
        lambda t, **p: bf_curl(t, _creds_user("admin", WRONG_PW[:3])))

    # raw-socket pipelined brute force (different client timing)
    add("raw_pipelined_fast", "raw-socket", "permissive", "passive",
        lambda t, **p: bf_raw_pipelined(t, _creds_user("admin", WRONG_PW)))
    add("raw_pipelined_usernames", "raw-socket", "permissive", "passive",
        lambda t, **p: bf_raw_pipelined(t, _creds_users(WRONG_USERS, "password")))
    add("raw_pipelined_valid_user", "raw-socket", "permissive", "passive",
        lambda t, **p: bf_raw_pipelined(t, _creds_user("labuser", WRONG_PW)))

    # a second pass to broaden env/server coverage and reach ~40 -- each row
    # still differs by (scenario, server, env, client, creds), never a re-run
    add("fast_pacing_throttled_server", "python-ftplib", "throttled", "passive",
        lambda t, **p: bf_newconn(t, _creds_user("admin", WRONG_PW[:10])))
    add("different_usernames_valid_pw_shape", "python-ftplib", "permissive", "active",
        lambda t, **p: bf_newconn(t, _creds_users(WRONG_USERS, "admin"), passive=False))
    add("many_attempts_mixed", "python-ftplib", "permissive", "passive",
        lambda t, **p: bf_newconn(t, [(u, pw) for u in WRONG_USERS[:5] for pw in WRONG_PW[:5]]))
    add("single_connection_usernames", "python-ftplib", "permissive", "passive",
        lambda t, **p: bf_singleconn(t, _creds_users(WRONG_USERS, "root")))
    add("curl_many", "curl", "permissive", "passive",
        lambda t, **p: bf_curl(t, [(u, pw) for u in WRONG_USERS[:3] for pw in WRONG_PW[:4]]))
    add("raw_pipelined_many", "raw-socket", "permissive", "passive",
        lambda t, **p: bf_raw_pipelined(t, [(u, pw) for u in WRONG_USERS[:4] for pw in WRONG_PW[:5]]))
    add("multi_burst_defended", "python-ftplib", "ratelimited", "passive",
        lambda t, **p: bf_multiburst(t, WRONG_USERS[:2], WRONG_PW[:2]))
    add("few_attempts_curl", "curl", "ratelimited", "active",
        lambda t, **p: bf_curl(t, _creds_user("oracle", WRONG_PW[:2]), passive=False))

    # third pass: broaden throttled-server + env + client coverage to ~40,
    # each row distinct in (scenario, server, env, client, mode, credentials)
    add("throttled_server_usernames", "python-ftplib", "throttled", "passive",
        lambda t, **p: bf_newconn(t, _creds_users(WRONG_USERS[:8], "qwerty")))
    add("throttled_server_passwords", "python-ftplib", "throttled", "active",
        lambda t, **p: bf_newconn(t, _creds_user("administrator", WRONG_PW[:8]), passive=False))
    add("throttled_single_connection", "python-ftplib", "throttled", "passive",
        lambda t, **p: bf_singleconn(t, _creds_user("root", WRONG_PW[:10])))
    add("curl_throttled_server", "curl", "throttled", "passive",
        lambda t, **p: bf_curl(t, _creds_user("ftpadmin", WRONG_PW[:6])))
    add("raw_pipelined_throttled", "raw-socket", "throttled", "passive",
        lambda t, **p: bf_raw_pipelined(t, _creds_user("webadmin", WRONG_PW[:10])))
    add("valid_user_wrong_pw_curl", "curl", "permissive", "passive",
        lambda t, **p: bf_curl(t, _creds_user("labuser", WRONG_PW[:8])))
    add("valid_user_wrong_pw_active", "python-ftplib", "permissive", "active",
        lambda t, **p: bf_newconn(t, _creds_user("labuser", WRONG_PW[:10]), passive=False))
    add("mixed_users_passwords_large", "python-ftplib", "permissive", "passive",
        lambda t, **p: bf_newconn(t, [(u, pw) for u in WRONG_USERS[:6] for pw in WRONG_PW[:6]]))
    add("multi_burst_curl", "curl", "permissive", "active",
        lambda t, **p: bf_curl(t, [(u, pw) for u in WRONG_USERS[:3] for pw in WRONG_PW[:3]], passive=False))
    add("single_connection_defended", "python-ftplib", "ratelimited", "passive",
        lambda t, **p: bf_singleconn(t, _creds_user("admin", WRONG_PW[:3])))
    add("raw_valid_user_active_env", "raw-socket", "permissive", "passive",
        lambda t, **p: bf_raw_pipelined(t, _creds_users(WRONG_USERS[:6], "changeme")))
    add("fast_pacing_many_users", "python-ftplib", "permissive", "passive",
        lambda t, **p: bf_newconn(t, _creds_users(WRONG_USERS, "dragon")))
    return specs
