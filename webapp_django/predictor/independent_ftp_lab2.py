"""
Second independent FTP validation lab (experimental; all models frozen; TEST-ONLY).

A genuinely DIFFERENT setup from every prior corpus AND from the first vsFTPD independent
test -- built to validate the cross-session detector on fresh ground:

  * **Different server**: ``pure-ftpd`` (never used before; the first independent test used
    vsftpd, training used pyftpdlib + a bespoke raw-socket server). Pure-FTPd's escalating
    anti-bruteforce failure delay gives a completely different timing/session structure.
  * **Different client**: ``ncftp`` (new), alongside curl / Python ftplib / raw sockets.
  * **Different network namespace + subnet + ports**: a new ``ip netns`` (``ivlab2``) on
    ``10.88.0.0/24`` (vs the first test's 10.77.0.x), ports 2222 / 2323.
  * **Fresh scenario code** with deliberate SESSION-STRUCTURE diversity: single-session
    attacks (many attempts in one connection) AND multi-session attacks (many reconnects),
    plus benign multi-session use -- specifically to stress whether the cross-session model
    over-relies on session count.
  * **FTPS/TLS** as a separate category where available.

Fresh captures only; ground truth from the scenario folder, never a prediction. SAFETY:
only ever talks to ``10.88.0.2`` (private, created and owned here). Nothing here changes
ml.py / live_capture.py / pcap_validation.py / ftp_behavioral.py / any frozen model.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field, asdict
from ftplib import FTP, error_perm, error_temp
from pathlib import Path

LAB_NET = ipaddress.ip_network("10.88.0.0/24")
HOST_ADDR = "10.88.0.1"
SERVER_ADDR = "10.88.0.2"
NETNS = "ivlab2"
VETH_H = "veth-iv2h"
VETH_N = "veth-iv2n"
PORT_PLAIN = 2222
PORT_TLS = 2323
PASV_PLAIN = (30000, 30099)
PASV_TLS = (30200, 30299)

PUSERS = {"pureuser": "PurePass123", "pureuser2": "PurePass456", "pureuser3": "PurePass789"}
PRIMARY_USER, PRIMARY_PW = "pureuser", "PurePass123"
WRONG_PW = ["123456", "password", "admin", "qwerty", "letmein", "root", "test123", "secret"]
WRONG_USERS = ["admin", "root", "oracle", "ftp", "guest", "test", "www", "mysql"]
FAIL_TIMEOUT = 30.0            # pure-ftpd delays failed logins; allow generous time


def is_private_lab(addr: str) -> bool:
    try:
        return ipaddress.ip_address(addr) in LAB_NET
    except ValueError:
        return False


def _run(cmd, check=True, ns=False):
    full = (["ip", "netns", "exec", NETNS] + cmd) if ns else cmd
    r = subprocess.run(full, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {' '.join(full)}\n{r.stderr}")
    return r


@dataclass
class IvLab2:
    workdir: Path
    cert: Path = field(init=False)
    tls_ok: bool = False
    procs: list = field(default_factory=list)

    def __post_init__(self):
        self.workdir = Path(self.workdir); self.workdir.mkdir(parents=True, exist_ok=True)
        self.cert = Path("/etc/ssl/private/pure-ftpd.pem")

    def ensure_users(self):
        for name, pw in PUSERS.items():
            if subprocess.run(["id", name], capture_output=True).returncode != 0:
                _run(["useradd", "-m", "-d", f"/home/{name}", "-s", "/bin/bash", name])
            _run(["bash", "-c", f"echo '{name}:{pw}' | chpasswd"])
            home = Path(f"/home/{name}/pub"); home.mkdir(parents=True, exist_ok=True)
            (home / "readme.txt").write_text(f"pure-ftpd independent validation lab file for {name}.\n" * 6)
            (home / "data.bin").write_bytes(os.urandom(4096))
            _run(["chown", "-R", name, f"/home/{name}"])

    def ensure_cert(self):
        try:
            self.cert.parent.mkdir(parents=True, exist_ok=True)
            if not self.cert.is_file():
                _run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", str(self.cert),
                      "-out", str(self.cert), "-days", "2", "-nodes", "-subj", "/CN=ivlab2"])
                os.chmod(self.cert, 0o600)
            self.tls_ok = True
        except Exception:  # noqa: BLE001
            self.tls_ok = False

    def setup_network(self):
        self.teardown_network()
        _run(["ip", "netns", "add", NETNS])
        _run(["ip", "link", "add", VETH_H, "type", "veth", "peer", "name", VETH_N])
        _run(["ip", "link", "set", VETH_N, "netns", NETNS])
        _run(["ip", "addr", "add", f"{HOST_ADDR}/24", "dev", VETH_H]); _run(["ip", "link", "set", VETH_H, "up"])
        _run(["ip", "addr", "add", f"{SERVER_ADDR}/24", "dev", VETH_N], ns=True)
        _run(["ip", "link", "set", VETH_N, "up"], ns=True); _run(["ip", "link", "set", "lo", "up"], ns=True)

    def teardown_network(self):
        subprocess.run(["ip", "netns", "del", NETNS], capture_output=True)
        subprocess.run(["ip", "link", "del", VETH_H], capture_output=True)

    def _start_pureftpd(self, port, pasv, tls):
        cmd = ["ip", "netns", "exec", NETNS, "/usr/sbin/pure-ftpd", "-S", f"{SERVER_ADDR},{port}",
               "-E", "-j", "-p", f"{pasv[0]}:{pasv[1]}", "-P", SERVER_ADDR]
        if tls:
            cmd += ["-Y", "1"]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.procs.append(p)
        return self._wait_port(port)

    def _wait_port(self, port, timeout=8.0):
        end = time.time() + timeout
        while time.time() < end:
            with socket.socket() as s:
                s.settimeout(0.4)
                try:
                    s.connect((SERVER_ADDR, port)); return True
                except OSError:
                    time.sleep(0.1)
        return False

    def verify_plain(self):
        f = FTP(); f.connect(SERVER_ADDR, PORT_PLAIN, timeout=8)
        ok = bool(f.getwelcome()); f.login(PRIMARY_USER, PRIMARY_PW); f.quit()
        return ok

    def setup(self):
        if not is_private_lab(SERVER_ADDR):
            raise SystemExit(f"REFUSING: {SERVER_ADDR} not in private lab network {LAB_NET}.")
        self.ensure_users(); self.ensure_cert(); self.setup_network()
        if not self._start_pureftpd(PORT_PLAIN, PASV_PLAIN, tls=False):
            err = self.procs[-1].stderr.read() if self.procs[-1].stderr else b""
            raise RuntimeError(f"pure-ftpd plaintext did not start: {err.decode('utf-8', 'ignore')}")
        if self.tls_ok:
            self.tls_ok = self._start_pureftpd(PORT_TLS, PASV_TLS, tls=True)
        if not self.verify_plain():
            raise RuntimeError("pure-ftpd plaintext failed verification")
        return self

    def stop(self):
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        self.procs.clear()
        subprocess.run(["pkill", "-9", "pure-ftpd"], capture_output=True)
        self.teardown_network()


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def bpf() -> str:
    return (f"host {SERVER_ADDR} and (tcp port {PORT_PLAIN} or tcp port {PORT_TLS} "
            f"or tcp port 20 or tcp portrange {PASV_PLAIN[0]}-{PASV_TLS[1]})")


@dataclass
class Capture:
    pcap_path: Path
    proc: subprocess.Popen | None = None
    started_at: float = 0.0
    stopped_at: float = 0.0

    def start(self, settle=0.7, timeout=6.0):
        self.pcap_path = Path(self.pcap_path); self.pcap_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(["tcpdump", "-i", VETH_H, "-w", str(self.pcap_path), "-U", "-n", bpf()],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        end = time.time() + timeout
        while time.time() < end:
            line = self.proc.stderr.readline() if self.proc.stderr else b""
            if b"listening on" in line:
                break
            if self.proc.poll() is not None:
                raise RuntimeError("tcpdump failed")
        time.sleep(settle); self.started_at = time.time(); return self

    def stop(self, drain=0.8):
        time.sleep(drain); self.stopped_at = time.time()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


# ---------------------------------------------------------------------------
# Fresh clients / helpers (ncftp / ftplib / curl / raw) -- pure-ftpd delays failures
# ---------------------------------------------------------------------------


def _ncftp(script_lines, port=PORT_PLAIN, user=PRIMARY_USER, pw=PRIMARY_PW, timeout=40):
    script = f"open -u {user} -p {pw} -P {port} {SERVER_ADDR}\n" + "\n".join(script_lines) + "\nbye\n"
    return subprocess.run(["ncftp"], input=script, capture_output=True, text=True, timeout=timeout)


def _ftplib(port=PORT_PLAIN, passive=True, timeout=FAIL_TIMEOUT):
    f = FTP(); f.connect(SERVER_ADDR, port, timeout=timeout); f.set_pasv(passive); return f


def _raw_multi_attempt_single_conn(creds, port=PORT_PLAIN):
    """Many attempts on ONE connection (single-session). Returns (ok, fails)."""
    ok = fail = 0
    s = socket.create_connection((SERVER_ADDR, port), timeout=FAIL_TIMEOUT); s.settimeout(FAIL_TIMEOUT); s.recv(4096)
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


def _raw_session(creds, port=PORT_PLAIN):
    """One fresh connection with the given creds (a single session)."""
    return _raw_multi_attempt_single_conn(creds, port)


# ---------------------------------------------------------------------------
# BENIGN scenarios
# ---------------------------------------------------------------------------


def bn_clean_login(t):
    r = _ncftp(["ls"]); return {"sessions": 1, "successes": int(r.returncode == 0), "client": "ncftp"}


def bn_transfer(t):
    r = _ncftp(["get pub/readme.txt /dev/null"]); return {"sessions": 1, "successes": int(r.returncode == 0), "client": "ncftp"}


def bn_mistype_success(t, n=1):
    ok, fails = _raw_multi_attempt_single_conn([(PRIMARY_USER, WRONG_PW[i]) for i in range(n)] + [(PRIMARY_USER, PRIMARY_PW)])
    return {"sessions": 1, "successes": ok, "failures": fails, "client": "raw-socket"}


def bn_giveup(t, n=2):
    _ok, fails = _raw_multi_attempt_single_conn([(PRIMARY_USER, WRONG_PW[i]) for i in range(n)])
    return {"sessions": 1, "successes": 0, "failures": fails, "client": "raw-socket"}


def bn_reconnect_after_mistake(t):
    _o1, f1 = _raw_session([(PRIMARY_USER, WRONG_PW[0])])
    o2, _f2 = _raw_session([(PRIMARY_USER, PRIMARY_PW)])
    return {"sessions": 2, "successes": o2, "failures": f1, "client": "raw-socket"}


def bn_repeated_normal(t, n=3):
    ok = 0
    for _ in range(n):
        r = _ncftp(["ls"]); ok += int(r.returncode == 0)
    return {"sessions": n, "successes": ok, "client": "ncftp"}


def bn_multi_user(t):
    ok = 0
    for u, pw in (("pureuser2", "PurePass456"), ("pureuser3", "PurePass789")):
        o, _f = _raw_session([(u, pw)]); ok += o
    return {"sessions": 2, "successes": ok, "client": "raw-socket"}


def bn_curl_get(t):
    r = subprocess.run(["curl", "-s", "-S", "--ftp-pasv",
                        f"ftp://{PRIMARY_USER}:{PRIMARY_PW}@{SERVER_ADDR}:{PORT_PLAIN}/pub/readme.txt", "-o", "/dev/null"],
                       capture_output=True, text=True, timeout=30)
    return {"sessions": 1, "successes": int(r.returncode == 0), "client": "curl"}


# ---------------------------------------------------------------------------
# ATTACKER scenarios (session-structure diversity is the point)
# ---------------------------------------------------------------------------


def at_single_session_brute(t, n=3):
    """SINGLE-SESSION attack: many attempts in ONE connection (the key stress case)."""
    _ok, fails = _raw_multi_attempt_single_conn([(PRIMARY_USER, WRONG_PW[i % len(WRONG_PW)]) for i in range(n)])
    return {"sessions": 1, "successes": 0, "failures": fails, "client": "raw-socket", "structure": "single_session"}


def at_multi_session_brute(t, n=6):
    """MULTI-SESSION attack: many reconnects, one failed guess each."""
    total_f = 0
    for i in range(n):
        _o, f = _raw_session([(PRIMARY_USER, WRONG_PW[i % len(WRONG_PW)])]); total_f += f
    return {"sessions": n, "successes": 0, "failures": total_f, "client": "raw-socket", "structure": "multi_session"}


def at_credential_sweep(t, n=6):
    total_f = 0
    for i in range(n):
        _o, f = _raw_session([(WRONG_USERS[i % len(WRONG_USERS)], WRONG_PW[i % len(WRONG_PW)])]); total_f += f
    return {"sessions": n, "successes": 0, "failures": total_f, "client": "raw-socket", "structure": "multi_session"}


def at_slow_brute(t, n=5, delay=1.0):
    total_f = 0
    for i in range(n):
        _o, f = _raw_session([(PRIMARY_USER, WRONG_PW[i % len(WRONG_PW)])]); total_f += f
        time.sleep(delay)
    return {"sessions": n, "successes": 0, "failures": total_f, "client": "raw-socket", "structure": "multi_session"}


def at_eventual_success(t, n_wrong=5):
    total_f = 0
    for i in range(n_wrong):
        _o, f = _raw_session([(PRIMARY_USER, WRONG_PW[i % len(WRONG_PW)])]); total_f += f
    o, _f = _raw_session([(PRIMARY_USER, PRIMARY_PW)])
    return {"sessions": n_wrong + 1, "successes": o, "failures": total_f, "client": "raw-socket", "structure": "multi_session"}


def at_reconnect_bursts(t, bursts=2, per_burst=3):
    total_f = 0
    for b in range(bursts):
        for i in range(per_burst):
            _o, f = _raw_session([(PRIMARY_USER, WRONG_PW[(b * per_burst + i) % len(WRONG_PW)])]); total_f += f
        time.sleep(0.5)
    return {"sessions": bursts * per_burst, "successes": 0, "failures": total_f, "client": "raw-socket", "structure": "multi_session"}


def at_single_session_curl(t):
    """Single-session attack via curl (a few passwords, connection-per-try is curl's model)."""
    fails = 0
    for pw in WRONG_PW[:3]:
        r = subprocess.run(["curl", "-s", "-S", "--ftp-pasv", f"ftp://{PRIMARY_USER}:{pw}@{SERVER_ADDR}:{PORT_PLAIN}/"],
                           capture_output=True, text=True, timeout=40)
        fails += int(r.returncode != 0)
    return {"sessions": 3, "successes": 0, "failures": fails, "client": "curl", "structure": "multi_session"}


# ---------------------------------------------------------------------------
# FTPS / TLS (separate category; behavioural features unavailable)
# ---------------------------------------------------------------------------


def tls_login_get(t):
    r = subprocess.run(["curl", "-s", "-S", "--ftp-ssl", "--insecure", "--ftp-pasv",
                        "-u", f"{PRIMARY_USER}:{PRIMARY_PW}", f"ftp://{SERVER_ADDR}:{PORT_TLS}/pub/readme.txt", "-o", "/dev/null"],
                       capture_output=True, text=True, timeout=30)
    return {"sessions": 1, "successes": int(r.returncode == 0), "client": "curl-ftps", "encrypted": True}


def tls_brute(t, n=3):
    fails = 0
    for i in range(n):
        r = subprocess.run(["curl", "-s", "-S", "--ftp-ssl", "--insecure", "--ftp-pasv",
                            "-u", f"{PRIMARY_USER}:{WRONG_PW[i % len(WRONG_PW)]}", f"ftp://{SERVER_ADDR}:{PORT_TLS}/"],
                           capture_output=True, text=True, timeout=40)
        fails += int(r.returncode != 0)
    return {"sessions": n, "successes": 0, "failures": fails, "client": "curl-ftps", "encrypted": True}


# ---------------------------------------------------------------------------
# Verifier + metadata + specs
# ---------------------------------------------------------------------------


@dataclass
class PcapCheck:
    pcap: str
    packets: int = 0
    duration_s: float = 0.0
    tcp_only: bool = False
    private_only: bool = False
    port_ok: bool = False
    ok: bool = False
    errors: list = field(default_factory=list)


def verify_pcap(pcap_path) -> PcapCheck:
    from scapy.all import rdpcap, TCP, UDP, IP
    p = Path(pcap_path); c = PcapCheck(pcap=p.name)
    if not p.is_file():
        c.errors.append("missing"); return c
    try:
        pkts = rdpcap(str(p))
    except Exception as e:  # noqa: BLE001
        c.errors.append(f"unreadable: {e}"); return c
    c.packets = len(pkts)
    if not pkts:
        c.errors.append("no packets"); return c
    times, tcp, non_tcp, priv, saw_port = [], 0, 0, True, False
    ports = {PORT_PLAIN, PORT_TLS, 20} | set(range(PASV_PLAIN[0], PASV_TLS[1] + 1))
    for pk in pkts:
        times.append(float(pk.time))
        if IP in pk:
            for a in (pk[IP].src, pk[IP].dst):
                if not is_private_lab(a):
                    priv = False
        if TCP in pk:
            tcp += 1
            if pk[TCP].sport in ports or pk[TCP].dport in ports:
                saw_port = True
        elif UDP in pk:
            non_tcp += 1
    c.private_only, c.port_ok, c.tcp_only = priv, saw_port, non_tcp == 0
    d = (max(times) - min(times)) if times else 0.0; c.duration_s = float(d)
    if not priv:
        c.errors.append("non-private address present")
    if not saw_port:
        c.errors.append("no expected-port traffic")
    if not c.tcp_only:
        c.errors.append("non-TCP payload present")
    c.ok = not c.errors
    return c


@dataclass
class CaptureMeta:
    capture_id: str
    scenario: str
    label: str
    client: str
    server: str
    environment: str
    interface: str
    mode: str
    scenario_family: str
    session_structure: str
    encrypted: bool
    sessions: int
    duration_s: float
    packet_count: int
    capture_timestamp: str
    source: str
    destination: str
    capture_command: str
    verification_status: str
    detail: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = asdict(self); d.pop("detail", None); return d


MANIFEST_COLUMNS = ["capture_id", "scenario", "label", "client", "server", "environment", "interface",
                    "mode", "scenario_family", "session_structure", "encrypted", "sessions", "duration_s",
                    "packet_count", "capture_timestamp", "source", "destination", "capture_command",
                    "verification_status"]


@dataclass
class Spec:
    scenario: str
    label: str
    family: str
    client: str
    structure: str
    encrypted: bool
    fn: object


def specs(tls_ok=True) -> list[Spec]:
    plan = [
        # ---- benign (cleartext) ----
        Spec("clean_login", "Benign", "clean", "ncftp", "single_session", False, bn_clean_login),
        Spec("clean_login_2", "Benign", "clean", "ncftp", "single_session", False, bn_clean_login),
        Spec("transfer", "Benign", "activity", "ncftp", "single_session", False, bn_transfer),
        Spec("curl_get", "Benign", "activity", "curl", "single_session", False, bn_curl_get),
        Spec("mistype_success_1", "Benign", "mistype", "raw-socket", "single_session", False, lambda t: bn_mistype_success(t, 1)),
        Spec("mistype_success_2", "Benign", "mistype", "raw-socket", "single_session", False, lambda t: bn_mistype_success(t, 2)),
        Spec("giveup_2", "Benign", "gave_up", "raw-socket", "single_session", False, lambda t: bn_giveup(t, 2)),
        Spec("giveup_3", "Benign", "gave_up", "raw-socket", "single_session", False, lambda t: bn_giveup(t, 3)),
        Spec("reconnect_after_mistake", "Benign", "reconnect", "raw-socket", "multi_session", False, bn_reconnect_after_mistake),
        Spec("repeated_normal_2", "Benign", "repeated", "ncftp", "multi_session", False, lambda t: bn_repeated_normal(t, 2)),
        Spec("repeated_normal_3", "Benign", "repeated", "ncftp", "multi_session", False, lambda t: bn_repeated_normal(t, 3)),
        Spec("multi_user", "Benign", "multi_user", "raw-socket", "multi_session", False, bn_multi_user),
        # ---- attacker (cleartext) ----
        Spec("single_session_brute_3", "FTP-BruteForce", "single_session_brute", "raw-socket", "single_session", False, lambda t: at_single_session_brute(t, 3)),
        Spec("single_session_brute_4", "FTP-BruteForce", "single_session_brute", "raw-socket", "single_session", False, lambda t: at_single_session_brute(t, 4)),
        Spec("single_session_curl", "FTP-BruteForce", "single_session_brute", "curl", "multi_session", False, at_single_session_curl),
        Spec("multi_session_brute_6", "FTP-BruteForce", "multi_session_brute", "raw-socket", "multi_session", False, lambda t: at_multi_session_brute(t, 6)),
        Spec("multi_session_brute_8", "FTP-BruteForce", "multi_session_brute", "raw-socket", "multi_session", False, lambda t: at_multi_session_brute(t, 8)),
        Spec("credential_sweep_6", "FTP-BruteForce", "cred_sweep", "raw-socket", "multi_session", False, lambda t: at_credential_sweep(t, 6)),
        Spec("slow_brute_5", "FTP-BruteForce", "slow_brute", "raw-socket", "multi_session", False, lambda t: at_slow_brute(t, 5)),
        Spec("eventual_success_5", "FTP-BruteForce", "eventual_success", "raw-socket", "multi_session", False, lambda t: at_eventual_success(t, 5)),
        Spec("eventual_success_4", "FTP-BruteForce", "eventual_success", "raw-socket", "multi_session", False, lambda t: at_eventual_success(t, 4)),
        Spec("reconnect_bursts", "FTP-BruteForce", "bursts", "raw-socket", "multi_session", False, at_reconnect_bursts),
    ]
    if tls_ok:
        plan += [
            Spec("ftps_login", "Benign", "tls_benign", "curl-ftps", "single_session", True, tls_login_get),
            Spec("ftps_login_2", "Benign", "tls_benign", "curl-ftps", "single_session", True, tls_login_get),
            Spec("ftps_brute_3", "FTP-BruteForce", "tls_brute", "curl-ftps", "multi_session", True, lambda t: tls_brute(t, 3)),
        ]
    return plan
