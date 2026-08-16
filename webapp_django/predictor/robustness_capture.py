"""
Robustness stress-test capture framework (experimental; models frozen).

Collects a NEW, deliberately MESSY real FTP corpus to stress-test the 45-feature
behavioural model -- specifically the cases the clean earlier corpora never
contained, where ``ftp_failed_logins`` / ``ftp_has_successful_auth`` are NOT a clean
proxy for the label:

  * benign users who mistype 1-3 passwords then log in successfully;
  * benign failed login followed by disconnect (never authenticates);
  * benign sessions as several different valid users;
  * brute force that EVENTUALLY guesses a correct password (attack that succeeds);
  * interrupted sessions with incomplete control-channel evidence.

Fresh real captures only, controlled local lab, NEW addresses/ports (disjoint from
every previous corpus). Ground truth is the scenario folder, never a prediction.
Reuses ``independent_capture`` primitives (Tcpdump, verify, custom server) and the
multi-user ``custom_ftp_server``; nothing here changes ml.py / live_capture.py /
pcap_validation.py.
"""

from __future__ import annotations

import io
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from ftplib import FTP, error_perm, error_temp
from pathlib import Path

from . import independent_capture as ic, custom_ftp_server
from .independent_capture import Tcpdump, host_eth0_ip, is_controlled_local

USER, PASSWORD = "labuser", "labpass"
EXTRA_USERS = {"alice": "alicepw", "bob": "bobpw"}         # additional valid users
ALL_USERS = {USER: PASSWORD, **EXTRA_USERS}
WRONG_PW = ["123456", "password", "admin", "qwerty", "letmein", "root", "test123", "secret"]

SERVER_PORT = {"custom": 2330, "permissive": 2340}


def environments() -> dict[str, tuple[str, str]]:
    envs = {"rb11": ("127.0.0.11", "lo"), "rb12": ("127.0.0.12", "lo"), "rb13": ("127.0.0.13", "lo")}
    eth0 = host_eth0_ip()
    if eth0:
        envs["rbhost"] = (eth0, "lo")
    return envs


@dataclass(frozen=True)
class RobTarget:
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


def build_targets(root: Path) -> dict:
    targets, idx = {}, 0
    for env_key, (addr, iface) in environments().items():
        for server in SERVER_PORT:
            plo = 64000 + idx * 40
            targets[(env_key, server)] = RobTarget(addr=addr, control_port=SERVER_PORT[server],
                                                   passive_lo=plo, passive_hi=plo + 39, iface=iface,
                                                   server=server, env_key=env_key)
            idx += 1
    return targets


# ---------------------------------------------------------------------------
# Servers (multi-user): custom raw-socket + pyftpdlib
# ---------------------------------------------------------------------------

_PYFTPD_CODE = r"""
import sys, logging
logging.disable(logging.CRITICAL)
addr, port, plo, phi, users_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
auth = DummyAuthorizer()
for pair in users_s.split(","):
    u, p = pair.split(":", 1); auth.add_user(u, p, sys.argv[6], perm="elradfmw")
h = FTPHandler; h.authorizer = auth; h.passive_ports = range(plo, phi + 1)
h.masquerade_address = addr; h.auth_failed_timeout = 0; h.max_login_attempts = 9999
FTPServer((addr, port), h).serve_forever()
"""


@dataclass
class ServerProc:
    target: RobTarget
    root: Path
    proc: subprocess.Popen | None = None

    def start(self, timeout=8.0):
        t = self.target
        users_s = ",".join(f"{u}:{p}" for u, p in ALL_USERS.items())
        if t.server == "custom":
            extra = ",".join(f"{u}:{p}" for u, p in EXTRA_USERS.items())
            cmd = [sys.executable, custom_ftp_server.__file__, t.addr, str(t.control_port),
                   str(self.root), USER, PASSWORD, str(t.passive_lo), str(t.passive_hi), extra]
        else:
            cmd = [sys.executable, "-c", _PYFTPD_CODE, t.addr, str(t.control_port),
                   str(t.passive_lo), str(t.passive_hi), users_s, str(self.root)]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if not self._wait(timeout):
            err = self.proc.stderr.read() if self.proc.stderr else b""
            self.stop(); raise RuntimeError(f"server {t.server}@{t.addr}:{t.control_port} down: {err.decode('l1','ignore') if False else err.decode('utf-8','ignore')}")
        return self

    def verify(self):
        t = self.target
        try:
            f = FTP(); f.connect(t.addr, t.control_port, timeout=4); f.login(USER, PASSWORD); f.quit(); return True
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
        self.targets, self.root = targets, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ic.SAFE_FILE).write_text("robustness lab test file.\n" * 50)
        (self.root / "data.bin").write_bytes(b"\x00\x01\x02\x03" * 400)
        self.servers = {}

    def get(self, env_key, server) -> RobTarget:
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
# Low-level auth helpers (manual USER/PASS so we control success/failure)
# ---------------------------------------------------------------------------


def _connect(t, passive=True, timeout=6.0) -> FTP:
    f = FTP(); f.connect(t.addr, t.control_port, timeout=timeout); f.set_pasv(passive); return f


def _try_pass(f, user, pw) -> bool:
    """Send USER/PASS manually; return True on 230 success, False on 530."""
    try:
        f.sendcmd(f"USER {user}")
        f.sendcmd(f"PASS {pw}")
        return True
    except (error_perm, error_temp):
        return False


# ---------------------------------------------------------------------------
# MESSY benign scenarios (label Benign)
# ---------------------------------------------------------------------------


def b_n_wrong_then_success(t, passive=True, n=1):
    """Benign user mistypes n passwords, then logs in successfully and does activity."""
    f = _connect(t, passive)
    fails = 0
    f.sendcmd(f"USER {USER}")
    for i in range(n):
        try:
            f.sendcmd(f"PASS {WRONG_PW[i]}")
        except (error_perm, error_temp):
            fails += 1
    f.sendcmd(f"PASS {PASSWORD}")             # correct -> 230
    f.retrlines("LIST", lambda _l: None)
    f.quit()
    return {"attempts": n + 1, "successes": 1, "failed_before_success": fails, "label_reason": "benign_mistype"}


def b_failed_then_disconnect(t, passive=True, n=2):
    """Benign user fails login n times then gives up (QUIT) -- never authenticates."""
    f = _connect(t, passive)
    f.sendcmd(f"USER {USER}")
    fails = 0
    for i in range(n):
        try:
            f.sendcmd(f"PASS {WRONG_PW[i]}")
        except (error_perm, error_temp):
            fails += 1
    try:
        f.sendcmd("QUIT")
    except Exception:  # noqa: BLE001
        pass
    try:
        f.close()
    except Exception:  # noqa: BLE001
        pass
    return {"attempts": n, "successes": 0, "failures": fails, "label_reason": "benign_gave_up"}


def b_multi_user(t, passive=True):
    """Several benign sessions as different valid users."""
    ok = 0
    for u, pw in ALL_USERS.items():
        f = _connect(t, passive)
        if _try_pass(f, u, pw):
            ok += 1
            f.retrlines("LIST", lambda _l: None)
        try:
            f.quit()
        except Exception:  # noqa: BLE001
            f.close()
    return {"attempts": len(ALL_USERS), "successes": ok, "distinct_users": len(ALL_USERS)}


def b_wrong_user_then_correct(t, passive=True):
    """Benign: typo'd username fails, then correct user/pass succeeds."""
    f = _connect(t, passive)
    try:
        f.sendcmd("USER nosuchuser"); f.sendcmd(f"PASS {PASSWORD}")
    except (error_perm, error_temp):
        pass
    _try_pass(f, USER, PASSWORD)
    f.retrlines("LIST", lambda _l: None); f.quit()
    return {"attempts": 2, "successes": 1, "label_reason": "benign_typo_user"}


def b_mixed_activity(t, passive=True):
    f = _connect(t, passive); _try_pass(f, USER, PASSWORD)
    for c in ("PWD", "SYST", "TYPE I", "NOOP"):
        try:
            f.voidcmd(c) if c.startswith(("TYPE", "NOOP")) else f.sendcmd(c)
        except Exception:  # noqa: BLE001
            pass
    buf = []; f.retrbinary(f"RETR {ic.SAFE_FILE}", buf.append)
    f.storbinary("STOR act.txt", io.BytesIO(b"activity\n" * 20))
    f.retrlines("LIST", lambda _l: None); f.quit()
    return {"attempts": 1, "successes": 1, "bytes_down": sum(len(b) for b in buf)}


def b_transfer(t, passive=True):
    f = _connect(t, passive); _try_pass(f, USER, PASSWORD)
    b = []; f.retrbinary("RETR data.bin", b.append)
    f.storbinary("STOR up.txt", io.BytesIO(b"x" * 400)); f.quit()
    return {"attempts": 1, "successes": 1, "bytes_down": sum(len(x) for x in b)}


def b_command_heavy(t, passive=True):
    f = _connect(t, passive); _try_pass(f, USER, PASSWORD)
    for c in ("MKD d", "SIZE " + ic.SAFE_FILE, "STAT", "RMD d", "CWD /", "CDUP"):
        try:
            f.sendcmd(c)
        except Exception:  # noqa: BLE001
            pass
    f.quit()
    return {"attempts": 1, "successes": 1}


def b_reconnect(t, passive=True):
    for _ in range(2):
        f = _connect(t, passive); _try_pass(f, USER, PASSWORD)
        f.retrlines("LIST", lambda _l: None); f.quit()
    return {"attempts": 2, "successes": 2, "reconnects": 1}


def b_active_success(t, passive=True):
    f = _connect(t, passive=False); _try_pass(f, USER, PASSWORD)
    b = []; f.retrbinary(f"RETR {ic.SAFE_FILE}", b.append); f.quit()
    return {"attempts": 1, "successes": 1, "mode": "active"}


def b_interrupted(t, passive=True):
    """Incomplete control evidence: connect + USER, then close before auth completes."""
    f = _connect(t, passive)
    try:
        f.sendcmd(f"USER {USER}")
    except Exception:  # noqa: BLE001
        pass
    try:
        f.sock.close()
    except Exception:  # noqa: BLE001
        pass
    return {"attempts": 0, "successes": 0, "label_reason": "benign_interrupted_no_auth"}


def b_curl_benign(t, passive=True):
    url = f"ftp://{USER}:{PASSWORD}@{t.addr}:{t.control_port}/{ic.SAFE_FILE}"
    r = subprocess.run(["curl", "-s", "-S", "--ftp-pasv", url, "-o", "/dev/null"], capture_output=True, text=True)
    return {"attempts": 1, "successes": int(r.returncode == 0), "client_rc": r.returncode}


def b_wget_benign(t, passive=True):
    url = f"ftp://{USER}:{PASSWORD}@{t.addr}:{t.control_port}/{ic.SAFE_FILE}"
    r = subprocess.run(["wget", "-q", "-O", "/dev/null", url], capture_output=True, text=True)
    return {"attempts": 1, "successes": int(r.returncode == 0), "client_rc": r.returncode}


# ---------------------------------------------------------------------------
# MESSY brute-force scenarios (label FTP-BruteForce)
# ---------------------------------------------------------------------------


def bf_all_fail(t, passive=True, n=8):
    f = _connect(t, passive); fails = 0
    for i in range(n):
        try:
            f.sendcmd(f"USER {USER}"); f.sendcmd(f"PASS {WRONG_PW[i % len(WRONG_PW)]}")
        except (error_perm, error_temp):
            fails += 1
    try:
        f.quit()
    except Exception:  # noqa: BLE001
        f.close()
    return {"attempts": n, "successes": 0, "failures": fails, "label_reason": "bf_all_fail"}


def bf_eventual_success(t, passive=True, n_wrong=6):
    """Attack that keeps guessing and EVENTUALLY hits the correct password."""
    f = _connect(t, passive); fails = 0
    for i in range(n_wrong):
        try:
            f.sendcmd(f"USER {USER}"); f.sendcmd(f"PASS {WRONG_PW[i % len(WRONG_PW)]}")
        except (error_perm, error_temp):
            fails += 1
    ok = _try_pass(f, USER, PASSWORD)          # finally guesses right
    try:
        f.quit()
    except Exception:  # noqa: BLE001
        f.close()
    return {"attempts": n_wrong + 1, "successes": int(ok), "failures": fails, "label_reason": "bf_eventual_success"}


def bf_eventual_success_newconn(t, passive=True, n_wrong=6):
    """Reconnecting attack; the final fresh connection guesses the right password."""
    fails = 0
    for i in range(n_wrong):
        f = _connect(t, passive)
        if not _try_pass(f, USER, WRONG_PW[i % len(WRONG_PW)]):
            fails += 1
        try:
            f.quit()
        except Exception:  # noqa: BLE001
            f.close()
    f = _connect(t, passive); ok = _try_pass(f, USER, PASSWORD)
    try:
        f.quit()
    except Exception:  # noqa: BLE001
        f.close()
    return {"attempts": n_wrong + 1, "successes": int(ok), "failures": fails, "label_reason": "bf_eventual_success_reconnect"}


def bf_multiuser_eventual(t, passive=True):
    """Guess across usernames; one of them (a real user) eventually succeeds."""
    creds = [("admin", "x"), ("root", "y"), ("alice", "wrong"), ("bob", "wrong"), ("alice", "alicepw")]
    fails = ok = 0
    for u, pw in creds:
        f = _connect(t, passive)
        if _try_pass(f, u, pw):
            ok += 1
        else:
            fails += 1
        try:
            f.quit()
        except Exception:  # noqa: BLE001
            f.close()
    return {"attempts": len(creds), "successes": ok, "failures": fails, "label_reason": "bf_multiuser_eventual_success"}


def bf_different_users_fail(t, passive=True):
    fails = 0
    for u in ("admin", "root", "ftp", "guest", "oracle", "test"):
        f = _connect(t, passive)
        if not _try_pass(f, u, "password"):
            fails += 1
        try:
            f.quit()
        except Exception:  # noqa: BLE001
            f.close()
    return {"attempts": 6, "successes": 0, "failures": fails, "label_reason": "bf_users_fail"}


def bf_single_conn_many(t, passive=True, n=10):
    f = _connect(t, passive); fails = 0
    for i in range(n):
        try:
            f.sendcmd(f"USER admin"); f.sendcmd(f"PASS {WRONG_PW[i % len(WRONG_PW)]}")
        except (error_perm, error_temp):
            fails += 1
        except (EOFError, OSError):
            fails += 1; f = _connect(t, passive)
    try:
        f.quit()
    except Exception:  # noqa: BLE001
        f.close()
    return {"attempts": n, "successes": 0, "failures": fails}


def bf_curl(t, passive=True):
    fails = 0
    for pw in WRONG_PW[:6]:
        r = subprocess.run(["curl", "-s", "-S", "--ftp-pasv", f"ftp://admin:{pw}@{t.addr}:{t.control_port}/"],
                           capture_output=True, text=True)
        fails += int(r.returncode != 0)
    return {"attempts": 6, "successes": 0, "failures": fails}


def bf_raw(t, passive=True):
    fails = 0
    s = socket.create_connection((t.addr, t.control_port), timeout=6); s.recv(4096)
    for pw in WRONG_PW:
        s.sendall(f"USER admin\r\n".encode()); s.recv(4096)
        s.sendall(f"PASS {pw}\r\n".encode()); resp = s.recv(4096)
        fails += int(resp[:3] != b"230")
    s.close()
    return {"attempts": len(WRONG_PW), "successes": 0, "failures": fails}


# ---------------------------------------------------------------------------
# Verification (reuse independent verifier: controlled-local, TCP, port, finite)
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
    scenario_family: str
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


MANIFEST_COLUMNS = ["capture_id", "scenario", "label", "client", "server", "environment", "interface",
                    "mode", "scenario_family", "attempts", "duration_s", "packet_count",
                    "capture_timestamp", "source", "destination", "capture_command", "verification_status"]


# ---------------------------------------------------------------------------
# Scenario matrix (messy benign + messy brute force)
# ---------------------------------------------------------------------------


@dataclass
class Spec:
    scenario: str
    label: str
    family: str
    client: str
    server: str
    env: str
    mode: str
    fn: object


def specs() -> list[Spec]:
    envs = list(environments())

    def e(i):
        return envs[i % len(envs)]

    plan = [
        # --- messy benign ---
        ("Benign", "one_wrong_then_success", "mistype", "python-ftplib", "custom", "passive", lambda t: b_n_wrong_then_success(t, n=1)),
        ("Benign", "two_wrong_then_success", "mistype", "python-ftplib", "permissive", "passive", lambda t: b_n_wrong_then_success(t, n=2)),
        ("Benign", "three_wrong_then_success", "mistype", "python-ftplib", "custom", "passive", lambda t: b_n_wrong_then_success(t, n=3)),
        ("Benign", "two_wrong_then_success_active", "mistype", "python-ftplib", "custom", "active", lambda t: b_n_wrong_then_success(t, passive=False, n=2)),
        ("Benign", "failed_then_disconnect", "gave_up", "python-ftplib", "custom", "passive", lambda t: b_failed_then_disconnect(t, n=2)),
        ("Benign", "failed_then_disconnect_3", "gave_up", "python-ftplib", "permissive", "passive", lambda t: b_failed_then_disconnect(t, n=3)),
        ("Benign", "multi_user_session", "multi_user", "python-ftplib", "custom", "passive", b_multi_user),
        ("Benign", "multi_user_session_pyftpd", "multi_user", "python-ftplib", "permissive", "passive", b_multi_user),
        ("Benign", "wrong_user_then_correct", "typo_user", "python-ftplib", "custom", "passive", b_wrong_user_then_correct),
        ("Benign", "mixed_activity", "activity", "python-ftplib", "custom", "passive", b_mixed_activity),
        ("Benign", "mixed_activity_pyftpd", "activity", "python-ftplib", "permissive", "passive", b_mixed_activity),
        ("Benign", "transfer_session", "activity", "python-ftplib", "custom", "passive", b_transfer),
        ("Benign", "command_heavy", "activity", "python-ftplib", "custom", "passive", b_command_heavy),
        ("Benign", "reconnect_session", "reconnect", "python-ftplib", "permissive", "passive", b_reconnect),
        ("Benign", "active_success", "activity", "python-ftplib", "custom", "active", b_active_success),
        ("Benign", "interrupted_no_auth", "incomplete", "python-ftplib", "custom", "passive", b_interrupted),
        ("Benign", "interrupted_no_auth_pyftpd", "incomplete", "python-ftplib", "permissive", "passive", b_interrupted),
        ("Benign", "curl_benign", "activity", "curl", "custom", "passive", b_curl_benign),
        ("Benign", "wget_benign", "activity", "wget", "permissive", "passive", b_wget_benign),
        ("Benign", "one_wrong_then_success_pyftpd", "mistype", "python-ftplib", "permissive", "passive", lambda t: b_n_wrong_then_success(t, n=1)),
        # --- messy brute force ---
        ("FTP-BruteForce", "all_fail", "all_fail", "python-ftplib", "custom", "passive", lambda t: bf_all_fail(t, n=8)),
        ("FTP-BruteForce", "all_fail_pyftpd", "all_fail", "python-ftplib", "permissive", "passive", lambda t: bf_all_fail(t, n=8)),
        ("FTP-BruteForce", "eventual_success", "eventual_success", "python-ftplib", "custom", "passive", lambda t: bf_eventual_success(t, n_wrong=6)),
        ("FTP-BruteForce", "eventual_success_pyftpd", "eventual_success", "python-ftplib", "permissive", "passive", lambda t: bf_eventual_success(t, n_wrong=6)),
        ("FTP-BruteForce", "eventual_success_active", "eventual_success", "python-ftplib", "custom", "active", lambda t: bf_eventual_success(t, passive=False, n_wrong=4)),
        ("FTP-BruteForce", "eventual_success_reconnect", "eventual_success", "python-ftplib", "custom", "passive", lambda t: bf_eventual_success_newconn(t, n_wrong=6)),
        ("FTP-BruteForce", "multiuser_eventual_success", "eventual_success", "python-ftplib", "permissive", "passive", bf_multiuser_eventual),
        ("FTP-BruteForce", "different_users_fail", "all_fail", "python-ftplib", "custom", "passive", bf_different_users_fail),
        ("FTP-BruteForce", "single_conn_many", "all_fail", "python-ftplib", "custom", "passive", lambda t: bf_single_conn_many(t, n=10)),
        ("FTP-BruteForce", "single_conn_many_pyftpd", "all_fail", "python-ftplib", "permissive", "passive", lambda t: bf_single_conn_many(t, n=10)),
        ("FTP-BruteForce", "reconnecting_fail", "all_fail", "python-ftplib", "permissive", "passive", bf_different_users_fail),
        ("FTP-BruteForce", "curl_bruteforce", "all_fail", "curl", "custom", "passive", bf_curl),
        ("FTP-BruteForce", "raw_bruteforce", "all_fail", "raw-socket", "custom", "passive", bf_raw),
        ("FTP-BruteForce", "eventual_success_raw", "eventual_success", "raw-socket", "permissive", "passive", lambda t: bf_eventual_success(t, n_wrong=5)),
    ]
    return [Spec(sc, lab, fam, cl, srv, e(i), mode, fn) for i, (lab, sc, fam, cl, srv, mode, fn) in enumerate(plan)]
