"""
Fourth independent FTP validation lab (experimental; all models frozen; TEST-ONLY).

The FRESH corpus for the auth-forensics experiment: benign **typo-then-success /
typo-give-up** vs attacker **single-session dictionary brute force (incl.
dictionary-then-success)** on a genuinely different stack -- pure-ftpd server (as in the
2nd independent test but a NEW netns/subnet/port and entirely new typo-vs-dictionary
scenarios, so hash/session-disjoint), ncftp/raw clients, network ``10.111.0.0/24``, port
2525. Ground truth from the scenario folder. SAFETY: only ever talks to ``10.111.0.2``.
Nothing here changes ml.py / live_capture.py / pcap_validation.py / ftp_behavioral.py.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field, asdict
from ftplib import FTP
from pathlib import Path

from .auth_forensics_capture import typos_of

LAB_NET = ipaddress.ip_network("10.111.0.0/24")
HOST_ADDR = "10.111.0.1"
SERVER_ADDR = "10.111.0.2"
NETNS = "ivlab4"
VETH_H = "veth-iv4h"
VETH_N = "veth-iv4n"
PORT = 2525
PASV = (41000, 41100)

PRIMARY_USER, PRIMARY_PW = "forensuser", "Str0ngPass77"
DICT_PW = ["123456", "password", "admin", "qwerty", "letmein", "root", "test123", "secret",
           "111111", "dragon", "monkey", "shadow", "welcome", "hunter2"]
WRONG_USERS = ["admin", "root", "oracle", "ftp", "guest", "test"]
FAIL_TIMEOUT = 30.0


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
class IvLab4:
    workdir: Path
    procs: list = field(default_factory=list)

    def __post_init__(self):
        self.workdir = Path(self.workdir); self.workdir.mkdir(parents=True, exist_ok=True)

    def ensure_users(self):
        if subprocess.run(["id", PRIMARY_USER], capture_output=True).returncode != 0:
            _run(["useradd", "-m", "-d", f"/home/{PRIMARY_USER}", "-s", "/bin/bash", PRIMARY_USER])
        _run(["bash", "-c", f"echo '{PRIMARY_USER}:{PRIMARY_PW}' | chpasswd"])
        home = Path(f"/home/{PRIMARY_USER}/pub"); home.mkdir(parents=True, exist_ok=True)
        (home / "readme.txt").write_text("auth-forensics independent lab file.\n" * 6)
        _run(["chown", "-R", PRIMARY_USER, f"/home/{PRIMARY_USER}"])

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

    def start(self):
        cmd = ["ip", "netns", "exec", NETNS, "/usr/sbin/pure-ftpd", "-S", f"{SERVER_ADDR},{PORT}",
               "-E", "-j", "-p", f"{PASV[0]}:{PASV[1]}", "-P", SERVER_ADDR]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE); self.procs.append(p)
        if not self._wait_port(PORT):
            err = p.stderr.read() if p.stderr else b""
            raise RuntimeError(f"pure-ftpd did not start: {err.decode('utf-8', 'ignore')}")

    def _wait_port(self, port, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            with socket.socket() as s:
                s.settimeout(0.4)
                try:
                    s.connect((SERVER_ADDR, port)); return True
                except OSError:
                    time.sleep(0.1)
        return False

    def verify(self):
        f = FTP(); f.connect(SERVER_ADDR, PORT, timeout=8); ok = bool(f.getwelcome())
        f.login(PRIMARY_USER, PRIMARY_PW); f.quit(); return ok

    def setup(self):
        if not is_private_lab(SERVER_ADDR):
            raise SystemExit(f"REFUSING: {SERVER_ADDR} not in private lab network {LAB_NET}.")
        self.ensure_users(); self.setup_network(); self.start()
        if not self.verify():
            raise RuntimeError("pure-ftpd failed verification")
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


def bpf() -> str:
    return f"host {SERVER_ADDR} and (tcp port {PORT} or tcp port 20 or tcp portrange {PASV[0]}-{PASV[1]})"


@dataclass
class Capture:
    pcap_path: Path
    proc: subprocess.Popen | None = None

    def start(self, settle=0.8, timeout=6.0):
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
        time.sleep(settle); return self

    def stop(self, drain=0.8):
        time.sleep(drain)
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def _raw_conn(creds, pace=False, timeout=FAIL_TIMEOUT):
    import random as _r
    ok = fail = 0
    try:
        s = socket.create_connection((SERVER_ADDR, PORT), timeout=timeout); s.settimeout(timeout); s.recv(4096)
    except OSError:
        return 0, 0
    for k, (user, pw) in enumerate(creds):
        if pace and k:
            time.sleep(_r.uniform(0.4, 1.8))
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


def _ncftp(lines):
    script = f"open -u {PRIMARY_USER} -p {PRIMARY_PW} -P {PORT} {SERVER_ADDR}\n" + "\n".join(lines) + "\nbye\n"
    return subprocess.run(["ncftp"], input=script, capture_output=True, text=True, timeout=30)


# ---- benign (typos) --------------------------------------------------------


def bn_clean(t):
    r = _ncftp(["ls"]); return {"sessions": 1, "successes": int(r.returncode == 0), "client": "ncftp"}


def bn_typo_then_success(t, n=2):
    o, f = _raw_conn([(PRIMARY_USER, pw) for pw in typos_of(PRIMARY_PW, n)] + [(PRIMARY_USER, PRIMARY_PW)], pace=True)
    return {"sessions": 1, "successes": o, "failures": f, "client": "raw-socket"}


def bn_typo_give_up(t, n=3):
    _o, f = _raw_conn([(PRIMARY_USER, pw) for pw in typos_of(PRIMARY_PW, n)], pace=True)
    return {"sessions": 1, "failures": f, "client": "raw-socket"}


def bn_repeated(t, n=2):
    ok = 0
    for _ in range(n):
        r = _ncftp(["ls"]); ok += int(r.returncode == 0)
    return {"sessions": n, "successes": ok, "client": "ncftp"}


# ---- attacker (dictionary, single-session incl fail-then-success) ----------


def at_dict_fail(t, n=6):
    _o, f = _raw_conn([(PRIMARY_USER, DICT_PW[i % len(DICT_PW)]) for i in range(n)])
    return {"sessions": 1, "failures": f, "client": "raw-socket", "structure": "single_session"}


def at_dict_then_success(t, n=6):
    o, f = _raw_conn([(PRIMARY_USER, DICT_PW[i % len(DICT_PW)]) for i in range(n)] + [(PRIMARY_USER, PRIMARY_PW)])
    return {"sessions": 1, "successes": o, "failures": f, "client": "raw-socket", "structure": "single_session"}


def at_dict_sweep(t, n=6):
    _o, f = _raw_conn([(WRONG_USERS[i % len(WRONG_USERS)], DICT_PW[i % len(DICT_PW)]) for i in range(n)])
    return {"sessions": 1, "failures": f, "client": "raw-socket", "structure": "single_session"}


def at_dict_multi(t, n=5):
    total_f = 0
    for i in range(n):
        _o, f = _raw_conn([(PRIMARY_USER, DICT_PW[i % len(DICT_PW)])]); total_f += f
    return {"sessions": n, "failures": total_f, "client": "raw-socket", "structure": "multi_session"}


# ---- verifier + metadata + specs -------------------------------------------


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
    ports = {PORT, 20} | set(range(PASV[0], PASV[1] + 1))
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
    fn: object


def specs() -> list[Spec]:
    return [
        Spec("clean_login", "Benign", "clean", "ncftp", "single_session", bn_clean),
        Spec("clean_login_2", "Benign", "clean", "ncftp", "single_session", bn_clean),
        Spec("repeated_2", "Benign", "repeated", "ncftp", "multi_session", lambda t: bn_repeated(t, 2)),
        Spec("typo_then_success_1", "Benign", "mistype", "raw-socket", "single_session", lambda t: bn_typo_then_success(t, 1)),
        Spec("typo_then_success_2", "Benign", "mistype", "raw-socket", "single_session", lambda t: bn_typo_then_success(t, 2)),
        Spec("typo_then_success_3", "Benign", "mistype", "raw-socket", "single_session", lambda t: bn_typo_then_success(t, 3)),
        Spec("typo_then_success_2b", "Benign", "mistype", "raw-socket", "single_session", lambda t: bn_typo_then_success(t, 2)),
        Spec("typo_give_up_2", "Benign", "gave_up", "raw-socket", "single_session", lambda t: bn_typo_give_up(t, 2)),
        Spec("typo_give_up_3", "Benign", "gave_up", "raw-socket", "single_session", lambda t: bn_typo_give_up(t, 3)),
        Spec("typo_give_up_4", "Benign", "gave_up", "raw-socket", "single_session", lambda t: bn_typo_give_up(t, 4)),
        # attacker single-session dictionary (incl fail-then-success) -- the decisive cases
        Spec("dict_fail_6", "FTP-BruteForce", "single_dict", "raw-socket", "single_session", lambda t: at_dict_fail(t, 6)),
        Spec("dict_fail_10", "FTP-BruteForce", "single_dict", "raw-socket", "single_session", lambda t: at_dict_fail(t, 10)),
        Spec("dict_then_success_5", "FTP-BruteForce", "dict_success", "raw-socket", "single_session", lambda t: at_dict_then_success(t, 5)),
        Spec("dict_then_success_8", "FTP-BruteForce", "dict_success", "raw-socket", "single_session", lambda t: at_dict_then_success(t, 8)),
        Spec("dict_then_success_4", "FTP-BruteForce", "dict_success", "raw-socket", "single_session", lambda t: at_dict_then_success(t, 4)),
        Spec("dict_sweep_6", "FTP-BruteForce", "single_dict", "raw-socket", "single_session", lambda t: at_dict_sweep(t, 6)),
        Spec("dict_multi_5", "FTP-BruteForce", "multi_dict", "raw-socket", "multi_session", lambda t: at_dict_multi(t, 5)),
    ]
