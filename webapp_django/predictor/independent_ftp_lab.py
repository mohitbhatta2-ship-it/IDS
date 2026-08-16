"""
Independent FTP validation lab (experimental; all models frozen; TEST-ONLY).

A DELIBERATELY DIFFERENT capture framework from ``robustness_capture`` /
``robust_train_capture``: it does NOT reuse any of their scenario functions, server
implementations, addresses, ports, or client tools. The point is to test the robust
behavioural candidate against traffic produced by a genuinely different stack:

  * **Real production server**: ``vsftpd`` (never used in any prior corpus, which used
    only ``pyftpdlib`` and a bespoke raw-socket server).
  * **Different client tool**: ``lftp`` (never used before), alongside curl / Python
    ftplib / raw sockets for variety.
  * **Non-loopback private network**: a real ``veth`` pair between the host network
    namespace and an isolated ``ip netns`` (``10.77.0.0/24``, RFC1918). Traffic
    traverses a genuine separate interface (``veth-ivh``), not ``lo``. This addresses
    the single-host / loopback-only limitation of every earlier corpus.
  * **FTPS/TLS**: a separate vsftpd instance with explicit AUTH TLS, captured on its
    own port and clearly marked -- the cleartext behavioural features are then
    genuinely unavailable (the parser sees no USER/PASS/230).

Fresh scenario logic (below) is written from scratch. Ground truth is the scenario
folder, never a model prediction. Nothing here modifies ml.py / live_capture.py /
pcap_validation.py or any frozen model.

SAFETY: the lab only ever talks to ``10.77.0.2`` (private, created and owned here,
reachable only through our own veth). Every target is refused unless it is inside the
private lab network.
"""

from __future__ import annotations

import ipaddress
import io
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from ftplib import FTP, error_perm, error_temp
from pathlib import Path

# ---------------------------------------------------------------------------
# Lab constants -- all NEW (disjoint from every prior corpus)
# ---------------------------------------------------------------------------

LAB_NET = ipaddress.ip_network("10.77.0.0/24")
HOST_ADDR = "10.77.0.1"          # root-ns end of the veth (client side)
SERVER_ADDR = "10.77.0.2"        # netns end of the veth (vsftpd side)
NETNS = "ivlab"
VETH_H = "veth-ivh"              # host side (where we capture)
VETH_N = "veth-ivn"              # namespace side
PORT_PLAIN = 2121
PORT_TLS = 2131
PASV_PLAIN = (63000, 63099)
PASV_TLS = (63100, 63199)
SAFE_FILE = "readme.txt"

USERS = {"ftplabuser": "Labpass123", "ftplabalice": "AlicePw456", "ftplabbob": "BobPw789"}
PRIMARY_USER, PRIMARY_PW = "ftplabuser", "Labpass123"
WRONG_PW = ["123456", "password", "admin", "qwerty", "letmein", "root", "test123",
            "secret", "111111", "dragon", "monkey", "shadow"]
WRONG_USERS = ["admin", "root", "oracle", "ftp", "guest", "test", "www", "mysql"]


def is_private_lab(addr: str) -> bool:
    """True iff addr is inside the private lab network we created and own."""
    try:
        return ipaddress.ip_address(addr) in LAB_NET
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Lab lifecycle: users, TLS cert, netns + veth, vsftpd (plain + TLS)
# ---------------------------------------------------------------------------


def _run(cmd, check=True, ns=False):
    full = (["ip", "netns", "exec", NETNS] + cmd) if ns else cmd
    r = subprocess.run(full, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {' '.join(full)}\n{r.stderr}")
    return r


@dataclass
class IvLab:
    workdir: Path
    cert: Path = field(init=False)
    key: Path = field(init=False)
    plain_conf: Path = field(init=False)
    tls_conf: Path = field(init=False)
    procs: list = field(default_factory=list)

    def __post_init__(self):
        self.workdir = Path(self.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cert = self.workdir / "vsftpd.crt"
        self.key = self.workdir / "vsftpd.key"
        self.plain_conf = self.workdir / "vsftpd_plain.conf"
        self.tls_conf = self.workdir / "vsftpd_tls.conf"

    # -- users + content -------------------------------------------------
    def ensure_users(self):
        for name, pw in USERS.items():
            if subprocess.run(["id", name], capture_output=True).returncode != 0:
                _run(["useradd", "-m", "-d", f"/home/{name}", "-s", "/bin/bash", name])
            _run(["bash", "-c", f"echo '{name}:{pw}' | chpasswd"])
            root = Path(f"/srv/ftplab_{name}")
            root.mkdir(parents=True, exist_ok=True)
            (root / SAFE_FILE).write_text(f"independent FTP validation lab file for {name}.\n" * 8)
            (root / "data.bin").write_bytes(os.urandom(4096))
            _run(["chown", "-R", name, str(root)])
        Path("/var/run/vsftpd/empty").mkdir(parents=True, exist_ok=True)

    def ensure_cert(self):
        if not (self.cert.is_file() and self.key.is_file()):
            _run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", str(self.key),
                  "-out", str(self.cert), "-days", "2", "-nodes", "-subj", "/CN=ivftplab"])

    # -- network ----------------------------------------------------------
    def setup_network(self):
        self.teardown_network()          # idempotent
        _run(["ip", "netns", "add", NETNS])
        _run(["ip", "link", "add", VETH_H, "type", "veth", "peer", "name", VETH_N])
        _run(["ip", "link", "set", VETH_N, "netns", NETNS])
        _run(["ip", "addr", "add", f"{HOST_ADDR}/24", "dev", VETH_H])
        _run(["ip", "link", "set", VETH_H, "up"])
        _run(["ip", "addr", "add", f"{SERVER_ADDR}/24", "dev", VETH_N], ns=True)
        _run(["ip", "link", "set", VETH_N, "up"], ns=True)
        _run(["ip", "link", "set", "lo", "up"], ns=True)

    def teardown_network(self):
        subprocess.run(["ip", "netns", "del", NETNS], capture_output=True)
        subprocess.run(["ip", "link", "del", VETH_H], capture_output=True)

    # -- servers ----------------------------------------------------------
    def _write_confs(self):
        common = (f"listen=YES\nlisten_address={SERVER_ADDR}\nlocal_enable=YES\n"
                  "write_enable=YES\nanonymous_enable=NO\npasv_enable=YES\n"
                  f"pasv_address={SERVER_ADDR}\nseccomp_sandbox=NO\n"
                  "secure_chroot_dir=/var/run/vsftpd/empty\npam_service_name=vsftpd\n"
                  "user_sub_token=$USER\nlocal_root=/srv/ftplab_$USER\n"
                  "allow_writeable_chroot=YES\n")
        self.plain_conf.write_text(common + f"listen_port={PORT_PLAIN}\n"
                                   f"pasv_min_port={PASV_PLAIN[0]}\npasv_max_port={PASV_PLAIN[1]}\n")
        self.tls_conf.write_text(common + f"listen_port={PORT_TLS}\n"
                                 f"pasv_min_port={PASV_TLS[0]}\npasv_max_port={PASV_TLS[1]}\n"
                                 "ssl_enable=YES\nforce_local_logins_ssl=YES\nforce_local_data_ssl=NO\n"
                                 f"rsa_cert_file={self.cert}\nrsa_private_key_file={self.key}\n"
                                 "ssl_ciphers=HIGH\nrequire_ssl_reuse=NO\n")

    def start_servers(self):
        self._write_confs()
        for conf, port in ((self.plain_conf, PORT_PLAIN), (self.tls_conf, PORT_TLS)):
            p = subprocess.Popen(["ip", "netns", "exec", NETNS, "/usr/sbin/vsftpd", str(conf)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.procs.append(p)
            if not self._wait_port(port):
                err = p.stderr.read() if p.stderr else b""
                raise RuntimeError(f"vsftpd on {port} did not start: {err.decode('utf-8', 'ignore')}")

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
        f = FTP(); f.connect(SERVER_ADDR, PORT_PLAIN, timeout=5)
        ok = bool(f.getwelcome()); f.login(PRIMARY_USER, PRIMARY_PW); f.quit()
        return ok

    def stop(self):
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        self.procs.clear()
        subprocess.run(["pkill", "-9", "vsftpd"], capture_output=True)
        self.teardown_network()

    # -- full setup -------------------------------------------------------
    def setup(self):
        if not is_private_lab(SERVER_ADDR):
            raise SystemExit(f"REFUSING: {SERVER_ADDR} is not inside the private lab network {LAB_NET}.")
        self.ensure_users(); self.ensure_cert(); self.setup_network(); self.start_servers()
        if not self.verify_plain():
            raise RuntimeError("plaintext vsftpd failed verification")
        return self


# ---------------------------------------------------------------------------
# tcpdump capture (host side of the veth -- genuinely non-loopback)
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
        self.pcap_path = Path(self.pcap_path)
        self.pcap_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(["tcpdump", "-i", VETH_H, "-w", str(self.pcap_path),
                                      "-U", "-n", bpf()], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        end = time.time() + timeout
        while time.time() < end:
            line = self.proc.stderr.readline() if self.proc.stderr else b""
            if b"listening on" in line:
                break
            if self.proc.poll() is not None:
                err = self.proc.stderr.read() if self.proc.stderr else b""
                raise RuntimeError(f"tcpdump failed: {err.decode('utf-8', 'ignore')}")
        time.sleep(settle); self.started_at = time.time(); return self

    def stop(self, drain=0.7):
        time.sleep(drain); self.stopped_at = time.time()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


# ---------------------------------------------------------------------------
# Fresh client helpers (lftp / ftplib / curl / raw socket) -- NEW code
# ---------------------------------------------------------------------------


def _ftplib(port=PORT_PLAIN, passive=True, timeout=6.0):
    f = FTP(); f.connect(SERVER_ADDR, port, timeout=timeout); f.set_pasv(passive); return f


def _lftp(script, port=PORT_PLAIN, user=PRIMARY_USER, pw=PRIMARY_PW, timeout=8):
    url = f"ftp://{user}:{pw}@{SERVER_ADDR}:{port}"
    cmd = ["lftp", "-e", f"set net:max-retries 1; set net:timeout {timeout}; "
           f"set ftp:ssl-allow false; {script}; bye", url]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)


def _raw_attempts(creds, port=PORT_PLAIN, delay=0.0):
    """Send USER/PASS pairs over raw sockets, reconnecting as needed.

    vsftpd drops a connection after 3 failed logins (its default), so a real
    brute-forcer reconnects. We proactively reconnect every 3 attempts and also
    recover from a mid-attempt drop. Returns (successes, failures).
    """
    ok = fail = 0

    def _open():
        c = socket.create_connection((SERVER_ADDR, port), timeout=6); c.recv(4096); return c

    s = _open(); used = 0
    for user, pw in creds:
        if used >= 3:
            try:
                s.close()
            except OSError:
                pass
            s = _open(); used = 0
        try:
            s.sendall(f"USER {user}\r\n".encode()); s.recv(4096)
            s.sendall(f"PASS {pw}\r\n".encode()); resp = s.recv(4096)
        except OSError:
            s = _open(); used = 0
            s.sendall(f"USER {user}\r\n".encode()); s.recv(4096)
            s.sendall(f"PASS {pw}\r\n".encode()); resp = s.recv(4096)
        used += 1
        if resp[:3] == b"230":
            ok += 1
        else:
            fail += 1
        if delay:
            time.sleep(delay)
    try:
        s.sendall(b"QUIT\r\n"); s.recv(4096)
    except OSError:
        pass
    s.close()
    return ok, fail


def _ftplib_guess_loop(pairs):
    """Try (user, pw) pairs via ftplib, reconnecting every 3 attempts. Returns (ok, fail)."""
    ok = fail = 0
    f = _ftplib(); used = 0
    for user, pw in pairs:
        if used >= 3:
            try:
                f.quit()
            except Exception:  # noqa: BLE001
                pass
            f = _ftplib(); used = 0
        try:
            f.sendcmd(f"USER {user}"); f.sendcmd(f"PASS {pw}"); ok += 1
        except (error_perm, error_temp):
            fail += 1
        except (EOFError, OSError):
            f = _ftplib(); used = 0
            try:
                f.sendcmd(f"USER {user}"); f.sendcmd(f"PASS {pw}"); ok += 1
            except (error_perm, error_temp):
                fail += 1
            except (EOFError, OSError):
                fail += 1
        used += 1
    try:
        f.quit()
    except Exception:  # noqa: BLE001
        pass
    return ok, fail


# ---------------------------------------------------------------------------
# BENIGN scenarios (label Benign) -- fresh implementations
# ---------------------------------------------------------------------------


def bn_login_ls(t):
    """lftp: clean login + directory listing."""
    r = _lftp("ls")
    return {"attempts": 1, "successes": int(r.returncode == 0), "client": "lftp", "commands": ["USER", "PASS", "LIST"]}


def bn_wrong_then_success(t, n=1):
    """ftplib: mistype n passwords, then log in and LIST (reconnecting past the 3-try drop)."""
    _ok, fails = _ftplib_guess_loop([(PRIMARY_USER, WRONG_PW[i]) for i in range(n)])
    f = _ftplib()                       # fresh connection for the genuine success
    f.sendcmd(f"USER {PRIMARY_USER}"); f.sendcmd(f"PASS {PRIMARY_PW}")
    f.retrlines("LIST", lambda _l: None); f.quit()
    return {"attempts": n + 1, "successes": 1, "failed_before_success": fails, "client": "python-ftplib"}


def bn_several_wrong_giveup(t, n=4):
    """raw socket: several wrong passwords, then give up (QUIT) -- never authenticates."""
    _ok, fails = _raw_attempts([(PRIMARY_USER, WRONG_PW[i]) for i in range(n)])
    return {"attempts": n, "successes": 0, "failures": fails, "client": "raw-socket"}


def bn_transfer(t):
    """lftp: download then upload a file (real data-channel transfers)."""
    r = _lftp(f"get {SAFE_FILE} -o /dev/null; put /etc/hostname -o up_$$.txt; rm -f up_$$.txt")
    return {"attempts": 1, "successes": int(r.returncode == 0), "client": "lftp", "commands": ["RETR", "STOR"]}


def bn_command_heavy(t):
    """ftplib: a session with many control commands."""
    f = _ftplib(); f.login(PRIMARY_USER, PRIMARY_PW)
    for c in ("PWD", "SYST", "TYPE I", "NOOP", f"SIZE {SAFE_FILE}", f"MDTM {SAFE_FILE}", "STAT", "CWD /", "CDUP"):
        try:
            f.voidcmd(c) if c in ("TYPE I", "NOOP") else f.sendcmd(c)
        except Exception:  # noqa: BLE001
            pass
    f.quit()
    return {"attempts": 1, "successes": 1, "client": "python-ftplib"}


def bn_reconnect(t):
    """lftp: two separate login+transfer sessions (reconnect)."""
    r1 = _lftp(f"get {SAFE_FILE} -o /dev/null")
    r2 = _lftp("ls")
    return {"attempts": 2, "successes": int(r1.returncode == 0) + int(r2.returncode == 0),
            "reconnects": 1, "client": "lftp"}


def bn_curl_get(t):
    """curl: fetch a file (passive)."""
    r = subprocess.run(["curl", "-s", "-S", "--ftp-pasv",
                        f"ftp://{PRIMARY_USER}:{PRIMARY_PW}@{SERVER_ADDR}:{PORT_PLAIN}/{SAFE_FILE}", "-o", "/dev/null"],
                       capture_output=True, text=True)
    return {"attempts": 1, "successes": int(r.returncode == 0), "client": "curl"}


def bn_multi_user(t):
    """ftplib: separate benign sessions as two other valid users."""
    ok = 0
    for u in ("ftplabalice", "ftplabbob"):
        f = _ftplib()
        try:
            f.login(u, USERS[u]); f.retrlines("LIST", lambda _l: None); ok += 1; f.quit()
        except Exception:  # noqa: BLE001
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass
    return {"attempts": 2, "successes": ok, "distinct_users": 2, "client": "python-ftplib"}


def bn_active_transfer(t):
    """ftplib active mode (PORT) transfer -- server connects back over the veth."""
    f = _ftplib(passive=False); f.login(PRIMARY_USER, PRIMARY_PW)
    buf = []
    try:
        f.retrbinary(f"RETR {SAFE_FILE}", buf.append)
    except Exception:  # noqa: BLE001
        pass
    f.quit()
    return {"attempts": 1, "successes": 1, "mode": "active", "client": "python-ftplib"}


# ---------------------------------------------------------------------------
# ATTACK scenarios (label FTP-BruteForce) -- fresh implementations
# ---------------------------------------------------------------------------


def at_slow_brute(t, n=6, delay=0.4):
    """raw socket: slow brute force, one wrong guess every `delay` seconds."""
    _ok, fails = _raw_attempts([(PRIMARY_USER, WRONG_PW[i % len(WRONG_PW)]) for i in range(n)], delay=delay)
    return {"attempts": n, "successes": 0, "failures": fails, "client": "raw-socket", "rate": "slow"}


def at_fast_brute(t, n=12):
    """ftplib: fast brute force, many wrong guesses back-to-back (reconnecting)."""
    _ok, fails = _ftplib_guess_loop([(PRIMARY_USER, WRONG_PW[i % len(WRONG_PW)]) for i in range(n)])
    return {"attempts": n, "successes": 0, "failures": fails, "client": "python-ftplib", "rate": "fast"}


def at_diff_usernames(t):
    """raw socket: brute force across many usernames (all wrong)."""
    _ok, fails = _raw_attempts([(u, "password") for u in WRONG_USERS])
    return {"attempts": len(WRONG_USERS), "successes": 0, "failures": fails, "client": "raw-socket"}


def at_eventual_success(t, n=8):
    """ftplib: keep guessing, EVENTUALLY hit the correct password (reconnecting)."""
    pairs = [(PRIMARY_USER, WRONG_PW[i % len(WRONG_PW)]) for i in range(n)] + [(PRIMARY_USER, PRIMARY_PW)]
    ok, fails = _ftplib_guess_loop(pairs)
    return {"attempts": n + 1, "successes": ok, "failures": fails, "client": "python-ftplib"}


def at_multi_conn(t, n=8):
    """lftp: many separate connections, each a single failed guess (reconnecting attack)."""
    fails = 0
    for i in range(n):
        r = _lftp("ls", user=PRIMARY_USER, pw=WRONG_PW[i % len(WRONG_PW)])
        fails += int(r.returncode != 0)
    return {"attempts": n, "successes": 0, "failures": fails, "connections": n, "client": "lftp"}


def at_curl_brute(t):
    """curl: brute force with several wrong passwords."""
    fails = 0
    for pw in WRONG_PW[:6]:
        r = subprocess.run(["curl", "-s", "-S", "--ftp-pasv",
                            f"ftp://{PRIMARY_USER}:{pw}@{SERVER_ADDR}:{PORT_PLAIN}/"], capture_output=True, text=True)
        fails += int(r.returncode != 0)
    return {"attempts": 6, "successes": 0, "failures": fails, "client": "curl"}


# ---------------------------------------------------------------------------
# FTPS / TLS scenarios (encrypted control -> behavioural features UNAVAILABLE)
# ---------------------------------------------------------------------------


def tls_login_get(t):
    """curl FTPS (AUTH TLS): successful encrypted login + download."""
    r = subprocess.run(["curl", "-s", "-S", "--ftp-ssl", "--insecure", "--ftp-pasv",
                        "-u", f"{PRIMARY_USER}:{PRIMARY_PW}",
                        f"ftp://{SERVER_ADDR}:{PORT_TLS}/{SAFE_FILE}", "-o", "/dev/null"],
                       capture_output=True, text=True)
    return {"attempts": 1, "successes": int(r.returncode == 0), "client": "curl-ftps", "encrypted": True}


def tls_brute(t, n=5):
    """curl FTPS: failed encrypted logins (brute force under TLS)."""
    fails = 0
    for i in range(n):
        r = subprocess.run(["curl", "-s", "-S", "--ftp-ssl", "--insecure", "--ftp-pasv",
                            "-u", f"{PRIMARY_USER}:{WRONG_PW[i % len(WRONG_PW)]}",
                            f"ftp://{SERVER_ADDR}:{PORT_TLS}/"], capture_output=True, text=True)
        fails += int(r.returncode != 0)
    return {"attempts": n, "successes": 0, "failures": fails, "client": "curl-ftps", "encrypted": True}


# ---------------------------------------------------------------------------
# Private-range verifier (accepts ONLY the lab network we created)
# ---------------------------------------------------------------------------


@dataclass
class PcapCheck:
    pcap: str
    packets: int = 0
    duration_s: float = 0.0
    tcp_only: bool = False
    private_only: bool = False
    host_present: bool = False
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
    times, tcp, non_tcp, priv, saw_host, saw_port = [], 0, 0, True, False, False
    ports = {PORT_PLAIN, PORT_TLS, 20} | set(range(PASV_PLAIN[0], PASV_TLS[1] + 1))
    for pk in pkts:
        times.append(float(pk.time))
        if IP in pk:
            for a in (pk[IP].src, pk[IP].dst):
                if not is_private_lab(a):
                    priv = False
            if SERVER_ADDR in (pk[IP].src, pk[IP].dst):
                saw_host = True
        if TCP in pk:
            tcp += 1
            if pk[TCP].sport in ports or pk[TCP].dport in ports:
                saw_port = True
        elif UDP in pk:
            non_tcp += 1
    c.private_only, c.host_present, c.port_ok = priv, saw_host, saw_port
    c.tcp_only = non_tcp == 0
    d = (max(times) - min(times)) if times else 0.0
    c.duration_s = float(d)
    if not priv:
        c.errors.append("non-private (external) address present")
    if not saw_port:
        c.errors.append("no traffic on an expected lab port")
    if not c.tcp_only:
        c.errors.append("non-TCP payload present")
    if not (d >= 0 and d == d and d != float("inf")):
        c.errors.append("invalid duration")
    c.ok = not c.errors
    return c


# ---------------------------------------------------------------------------
# Metadata + scenario matrix
# ---------------------------------------------------------------------------


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
    encrypted: bool
    attempts: int
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


MANIFEST_COLUMNS = ["capture_id", "scenario", "label", "client", "server", "environment",
                    "interface", "mode", "scenario_family", "encrypted", "attempts", "duration_s",
                    "packet_count", "capture_timestamp", "source", "destination",
                    "capture_command", "verification_status"]


@dataclass
class Spec:
    scenario: str
    label: str
    family: str
    client: str
    mode: str
    encrypted: bool
    fn: object


def specs() -> list[Spec]:
    plain = "vsftpd(plaintext)"
    return [
        # -------- BENIGN (cleartext) --------
        Spec("clean_login_ls", "Benign", "clean", "lftp", "passive", False, bn_login_ls),
        Spec("one_wrong_then_success", "Benign", "mistype", "python-ftplib", "passive", False, lambda t: bn_wrong_then_success(t, 1)),
        Spec("two_wrong_then_success", "Benign", "mistype", "python-ftplib", "passive", False, lambda t: bn_wrong_then_success(t, 2)),
        Spec("three_wrong_then_success", "Benign", "mistype", "python-ftplib", "passive", False, lambda t: bn_wrong_then_success(t, 3)),
        Spec("several_wrong_giveup_3", "Benign", "gave_up", "raw-socket", "passive", False, lambda t: bn_several_wrong_giveup(t, 3)),
        Spec("several_wrong_giveup_5", "Benign", "gave_up", "raw-socket", "passive", False, lambda t: bn_several_wrong_giveup(t, 5)),
        Spec("file_transfer", "Benign", "activity", "lftp", "passive", False, bn_transfer),
        Spec("command_heavy", "Benign", "activity", "python-ftplib", "passive", False, bn_command_heavy),
        Spec("reconnect_session", "Benign", "reconnect", "lftp", "passive", False, bn_reconnect),
        Spec("curl_get", "Benign", "activity", "curl", "passive", False, bn_curl_get),
        Spec("multi_user", "Benign", "multi_user", "python-ftplib", "passive", False, bn_multi_user),
        Spec("active_transfer", "Benign", "activity", "python-ftplib", "active", False, bn_active_transfer),
        Spec("clean_login_ls_2", "Benign", "clean", "lftp", "passive", False, bn_login_ls),
        Spec("file_transfer_2", "Benign", "activity", "lftp", "passive", False, bn_transfer),
        # -------- ATTACK (cleartext) --------
        Spec("slow_brute_6", "FTP-BruteForce", "slow_brute", "raw-socket", "passive", False, lambda t: at_slow_brute(t, 6, 0.4)),
        Spec("slow_brute_8", "FTP-BruteForce", "slow_brute", "raw-socket", "passive", False, lambda t: at_slow_brute(t, 8, 0.3)),
        Spec("fast_brute_12", "FTP-BruteForce", "fast_brute", "python-ftplib", "passive", False, lambda t: at_fast_brute(t, 12)),
        Spec("fast_brute_16", "FTP-BruteForce", "fast_brute", "python-ftplib", "passive", False, lambda t: at_fast_brute(t, 16)),
        Spec("different_usernames", "FTP-BruteForce", "user_enum", "raw-socket", "passive", False, at_diff_usernames),
        Spec("eventual_success_6", "FTP-BruteForce", "eventual_success", "python-ftplib", "passive", False, lambda t: at_eventual_success(t, 6)),
        Spec("eventual_success_10", "FTP-BruteForce", "eventual_success", "python-ftplib", "passive", False, lambda t: at_eventual_success(t, 10)),
        Spec("multi_conn_brute_8", "FTP-BruteForce", "multi_conn", "lftp", "passive", False, lambda t: at_multi_conn(t, 8)),
        Spec("multi_conn_brute_6", "FTP-BruteForce", "multi_conn", "lftp", "passive", False, lambda t: at_multi_conn(t, 6)),
        Spec("curl_brute", "FTP-BruteForce", "fast_brute", "curl", "passive", False, at_curl_brute),
        # -------- FTPS / TLS (encrypted; behavioural unavailable) --------
        Spec("ftps_login_get", "Benign", "tls_benign", "curl-ftps", "passive", True, tls_login_get),
        Spec("ftps_login_get_2", "Benign", "tls_benign", "curl-ftps", "passive", True, tls_login_get),
        Spec("ftps_brute_5", "FTP-BruteForce", "tls_brute", "curl-ftps", "passive", True, lambda t: tls_brute(t, 5)),
        Spec("ftps_brute_7", "FTP-BruteForce", "tls_brute", "curl-ftps", "passive", True, lambda t: tls_brute(t, 7)),
    ]
