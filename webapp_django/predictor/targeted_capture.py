"""
TARGETED benign real-PCAP collection (experimental; models frozen, no retraining).

Collects a new benign-heavy real FTP corpus aimed squarely at the STRUCTURED
benign false positives diagnosed in ``final_robustness`` -- active-mode sessions,
command-heavy sessions (MKD/RMD/RNFR/RNTO/DELE/APPE/SIZE/MDTM/NLST/STAT, multi
command), upload/download/append/delete transfers, and reconnect / multi-session
patterns -- across different real FTP server implementations/configurations
(a hand-written raw-socket server + pyftpdlib in permissive and bandwidth
throttled configs), varied clients, ports, and session patterns.

It reuses the existing ``independent_capture`` client scenario functions and
``Tcpdump`` primitive unchanged, and the existing ``pcap_validation`` extraction.
Diversity is genuine: no synthetic packets, no duplicated PCAPs, no injected
sleeps, no fabricated features. Loopback / host-local lab only; ground truth is
the scenario folder. NEW addresses/ports keep it disjoint from v1/v2/independent.
"""

from __future__ import annotations

import ipaddress
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from ftplib import FTP
from pathlib import Path

from . import independent_capture as ic
from .independent_capture import Tcpdump, host_eth0_ip, is_controlled_local
from . import custom_ftp_server

USER = "labuser"
PASSWORD = "labpass"

# NEW loopback addresses (not used by v1/v2/independent) + the host-local address.
def environments() -> dict[str, tuple[str, str]]:
    envs = {"te8": ("127.0.0.8", "lo"), "te9": ("127.0.0.9", "lo"), "te10": ("127.0.0.10", "lo")}
    eth0 = host_eth0_ip()
    if eth0:
        envs["hostip2"] = (eth0, "lo")     # host-local address, still lo (documented)
    return envs

# NEW control ports (independent used 2130/2140/2150)
SERVER_PORT = {"custom": 2230, "permissive": 2240, "throttled": 2250}


@dataclass(frozen=True)
class TargetedTarget:
    addr: str
    control_port: int
    passive_lo: int
    passive_hi: int
    iface: str
    server: str
    env_key: str
    user: str = USER
    password: str = PASSWORD

    def bpf(self) -> str:
        return (f"host {self.addr} and (tcp port {self.control_port} "
                f"or tcp port 20 or tcp portrange {self.passive_lo}-{self.passive_hi})")

    def is_controlled_local(self) -> bool:
        return is_controlled_local(self.addr)


def build_targets(root: Path) -> dict[tuple[str, str], TargetedTarget]:
    targets, idx = {}, 0
    for env_key, (addr, iface) in environments().items():
        for server in SERVER_PORT:
            plo = 63500 + idx * 40
            targets[(env_key, server)] = TargetedTarget(
                addr=addr, control_port=SERVER_PORT[server], passive_lo=plo,
                passive_hi=plo + 39, iface=iface, server=server, env_key=env_key)
            idx += 1
    return targets


# ---------------------------------------------------------------------------
# Servers: custom raw-socket (different implementation) + pyftpdlib configs
# ---------------------------------------------------------------------------

_PYFTPD_CODE = r"""
import sys, logging
logging.disable(logging.CRITICAL)
addr, port, plo, phi, user, pw, variant, root = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler, ThrottledDTPHandler
from pyftpdlib.servers import FTPServer
auth = DummyAuthorizer(); auth.add_user(user, pw, root, perm="elradfmw")
h = FTPHandler; h.authorizer = auth; h.passive_ports = range(plo, phi + 1)
h.masquerade_address = addr; h.auth_failed_timeout = 0; h.max_login_attempts = 9999
h.banner = "pyftpdlib targeted-lab (%s)" % variant
if variant == "throttled":
    d = ThrottledDTPHandler; d.read_limit = 8192; d.write_limit = 8192
    h.dtp_handler = d
FTPServer((addr, port), h).serve_forever()
"""


@dataclass
class ServerProc:
    target: TargetedTarget
    root: Path
    proc: subprocess.Popen | None = None

    def start(self, timeout: float = 8.0) -> "ServerProc":
        t = self.target
        if t.server == "custom":
            cmd = [sys.executable, custom_ftp_server.__file__, t.addr, str(t.control_port),
                   str(self.root), t.user, t.password, str(t.passive_lo), str(t.passive_hi)]
        else:
            cmd = [sys.executable, "-c", _PYFTPD_CODE, t.addr, str(t.control_port),
                   str(t.passive_lo), str(t.passive_hi), t.user, t.password, t.server, str(self.root)]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if not self._wait(timeout):
            err = self.proc.stderr.read() if self.proc.stderr else b""
            self.stop()
            raise RuntimeError(f"server {t.server}@{t.addr}:{t.control_port} did not start: "
                               f"{err.decode('utf-8','ignore')}")
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
    def __init__(self, targets, root: Path):
        self.targets = targets
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ic.SAFE_FILE).write_text("targeted benign lab file for FTP transfer.\n" * 50)
        (self.root / "data.bin").write_bytes(b"\x00\x01\x02\x03" * 500)
        self.servers: dict = {}

    def get(self, env_key, server) -> TargetedTarget:
        key = (env_key, server)
        if key not in self.servers:
            self.servers[key] = ServerProc(self.targets[key], self.root).start()
            if not self.servers[key].verify():
                raise RuntimeError(f"server {key} failed verification")
        return self.targets[key]

    def stop_all(self):
        for s in self.servers.values():
            s.stop()
        self.servers.clear()


# ---------------------------------------------------------------------------
# A couple of extra benign behaviours (command-heavy) for breadth
# ---------------------------------------------------------------------------


def b_command_workout(t, passive=True):
    """A long benign session mixing directory, transfer and metadata commands."""
    import io
    f = ic._ftp(t, passive); f.login(t.user, t.password)
    for c in ("PWD", "SYST", "STAT", "TYPE I", "NOOP"):
        try:
            f.voidcmd(c) if c.startswith(("TYPE", "NOOP")) else f.sendcmd(c)
        except Exception:  # noqa: BLE001
            pass
    try:
        f.sendcmd("MKD wdir"); f.cwd("wdir")
    except Exception:  # noqa: BLE001
        pass
    f.storbinary("STOR w.txt", io.BytesIO(b"workout payload\n" * 20))
    try:
        f.sendcmd("SIZE w.txt")
    except Exception:  # noqa: BLE001
        pass
    buf = []; f.retrbinary("RETR w.txt", buf.append)
    for c in ("DELE w.txt", "CDUP", "RMD wdir"):
        try:
            f.sendcmd(c)
        except Exception:  # noqa: BLE001
            pass
    f.retrlines("LIST", lambda _l: None); f.quit()
    return {"attempts": 1, "successes": 1, "bytes_down": sum(len(b) for b in buf),
            "commands": ["MKD", "STOR", "SIZE", "RETR", "DELE", "RMD", "LIST"]}


def b_mixed_transfer(t, passive=True):
    """Download + upload + append in one benign session."""
    import io
    f = ic._ftp(t, passive); f.login(t.user, t.password)
    d1 = []; f.retrbinary(f"RETR {ic.SAFE_FILE}", d1.append)
    f.storbinary("STOR m.txt", io.BytesIO(b"mixed\n" * 15))
    f.storbinary("APPE m.txt", io.BytesIO(b"more\n" * 15))
    d2 = []; f.retrbinary("RETR m.txt", d2.append)
    f.quit()
    return {"attempts": 1, "successes": 1,
            "bytes_down": sum(len(b) for b in d1 + d2), "commands": ["RETR", "STOR", "APPE"]}


def b_download_delete(t, passive=True):
    """Benign download + delete of a file the client itself created (works on any
    server: DELE targets a real file, so no 550)."""
    import io
    f = ic._ftp(t, passive); f.login(t.user, t.password)
    name = f"dd_{int(time.time()*1000)}.txt"
    f.storbinary(f"STOR {name}", io.BytesIO(b"download-delete payload\n" * 15))
    buf = []; f.retrbinary(f"RETR {name}", buf.append)
    try:
        f.delete(name)
    except Exception:  # noqa: BLE001
        pass
    f.quit()
    return {"attempts": 1, "successes": 1, "bytes_down": sum(len(b) for b in buf),
            "commands": ["STOR", "RETR", "DELE"]}


def b_multi_session_varied(t, passive=True, n=4):
    """Several benign sessions doing different things (list, download, commands)."""
    import io
    acts = 0
    for i in range(n):
        f = ic._ftp(t, passive); f.login(t.user, t.password)
        if i % 3 == 0:
            f.retrlines("LIST", lambda _l: None)
        elif i % 3 == 1:
            b = []; f.retrbinary(f"RETR {ic.SAFE_FILE}", b.append)
        else:
            f.storbinary(f"STOR s{i}.txt", io.BytesIO(b"session\n" * 10))
        f.quit(); acts += 1
    return {"attempts": n, "successes": n, "sessions": n}


# ---------------------------------------------------------------------------
# The targeted benign scenario matrix (>=30, active/command/transfer/reconnect)
# ---------------------------------------------------------------------------


@dataclass
class Spec:
    scenario: str
    client: str
    server: str
    env: str
    mode: str
    pattern: str
    fn: object


def benign_specs() -> list[Spec]:
    envs = list(environments())

    def e(i):
        return envs[i % len(envs)]

    # (scenario, client, server, mode, pattern, fn)
    plan = [
        # --- active-mode benign (the highest-FP mode) ---
        ("active_download", "python-ftplib", "custom", "active", "single", ic.b_active_download),
        ("active_download_pyftpdlib", "python-ftplib", "permissive", "active", "single", ic.b_active_download),
        ("active_download_throttled", "python-ftplib", "throttled", "active", "single", ic.b_active_download),
        ("active_listing", "python-ftplib", "custom", "active", "single", ic.b_nlst_listing),
        ("active_multi_command", "python-ftplib", "permissive", "active", "single", ic.b_multi_command),
        ("active_upload_then_download", "python-ftplib", "custom", "active", "single", ic.b_upload_then_download),
        ("active_mixed_transfer", "python-ftplib", "throttled", "active", "single", b_mixed_transfer),
        ("active_command_workout", "python-ftplib", "permissive", "active", "single", b_command_workout),
        # --- command-heavy benign ---
        ("mkd_rmd_session", "python-ftplib", "custom", "passive", "single", ic.b_login_mkd_rmd),
        ("upload_rename", "python-ftplib", "permissive", "passive", "single", ic.b_upload_rename),
        ("size_mdtm_stat", "python-ftplib", "throttled", "passive", "single", ic.b_size_mdtm),
        ("multi_command_session", "python-ftplib", "custom", "passive", "single", ic.b_multi_command),
        ("nlst_listing", "python-ftplib", "permissive", "passive", "single", ic.b_nlst_listing),
        ("command_workout", "python-ftplib", "custom", "passive", "single", b_command_workout),
        ("command_workout_pyftpdlib", "python-ftplib", "permissive", "passive", "single", b_command_workout),
        ("command_workout_throttled", "python-ftplib", "throttled", "passive", "single", b_command_workout),
        # --- transfers: download/upload/append/delete/mixed ---
        ("append_upload", "python-ftplib", "custom", "passive", "single", ic.b_append_upload),
        ("append_upload_throttled", "python-ftplib", "throttled", "passive", "single", ic.b_append_upload),
        ("download_delete", "python-ftplib", "permissive", "passive", "single", b_download_delete),
        ("upload_then_download", "python-ftplib", "custom", "passive", "single", ic.b_upload_then_download),
        ("upload_then_download_throttled", "python-ftplib", "throttled", "passive", "single", ic.b_upload_then_download),
        ("binary_download", "python-ftplib", "permissive", "passive", "single", ic.b_passive_binary_download),
        ("mixed_transfer", "python-ftplib", "custom", "passive", "single", b_mixed_transfer),
        ("mixed_transfer_throttled", "python-ftplib", "throttled", "passive", "single", b_mixed_transfer),
        ("curl_download_custom", "curl", "custom", "passive", "single", ic.b_curl_download),
        ("curl_download_pyftpdlib", "curl", "permissive", "passive", "single", ic.b_curl_download),
        ("curl_download_throttled", "curl", "throttled", "passive", "single", ic.b_curl_download),
        ("wget_download_custom", "wget", "custom", "passive", "single", ic.b_wget_download),
        ("wget_download_pyftpdlib", "wget", "permissive", "passive", "single", ic.b_wget_download),
        ("wget_download_throttled", "wget", "throttled", "passive", "single", ic.b_wget_download),
        # --- reconnect / multi-session ---
        ("reconnect_session", "python-ftplib", "custom", "passive", "reconnect", ic.b_reconnect),
        ("reconnect_pyftpdlib", "python-ftplib", "permissive", "passive", "reconnect", ic.b_reconnect),
        ("repeated_sessions", "python-ftplib", "custom", "passive", "repeated", ic.b_repeated_sessions),
        ("repeated_sessions_pyftpdlib", "python-ftplib", "permissive", "passive", "repeated", ic.b_repeated_sessions),
        ("multi_session_varied", "python-ftplib", "custom", "passive", "repeated", b_multi_session_varied),
        ("multi_session_varied_throttled", "python-ftplib", "throttled", "passive", "repeated", b_multi_session_varied),
        # --- a few more command/transfer combos for breadth ---
        ("upload_rename_custom", "python-ftplib", "custom", "passive", "single", ic.b_upload_rename),
        ("size_mdtm_custom", "python-ftplib", "custom", "passive", "single", ic.b_size_mdtm),
        ("download_delete_custom", "python-ftplib", "custom", "passive", "single", b_download_delete),
        ("active_append", "python-ftplib", "custom", "active", "single", ic.b_append_upload),
        ("active_size_mdtm", "python-ftplib", "permissive", "active", "single", ic.b_size_mdtm),
    ]
    specs = []
    for i, (scenario, client, server, mode, pattern, fn) in enumerate(plan):
        specs.append(Spec(scenario, client, server, e(i), mode, pattern, fn))
    return specs


# ---------------------------------------------------------------------------
# Verification + metadata (reuse the independent verifier semantics)
# ---------------------------------------------------------------------------

verify_pcap = ic.verify_pcap


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
    connection_pattern: str
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


MANIFEST_COLUMNS = [
    "capture_id", "scenario", "label", "client", "server", "environment", "interface",
    "mode", "connection_pattern", "attempts", "duration_s", "packet_count",
    "capture_timestamp", "source", "destination", "capture_command", "verification_status",
]
