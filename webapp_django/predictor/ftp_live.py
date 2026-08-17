"""
Experimental LIVE FTP-BruteForce detector (candidate model only; production untouched).

This wires the validated **per-connection + cross-session** FTP candidate
(``validation/models/ftp-per-connection/candidate_pc_unweighted.pkl``, 67 features) into the
live-capture pipeline, so real FTP traffic off an interface can be classified as
FTP-BruteForce vs Benign with the behavioural signal the production 30-feature model does not
have. It reuses the *verified* live core for the 30 packet-flow features
(``feature_calculator.calculate_features``, exactly as ``live_capture`` and ``pcap_validation``
do) and the *frozen, validated* FTP extractors for the application-layer features
(``ftp_behavioral`` / ``ftp_per_connection`` / ``ftp_cross_session``). Nothing here retrains,
and nothing writes to ``webapp_data/Results/Models/`` -- the production models and
``ml.py`` / ``live_capture.py`` are not modified.

Pipeline (per the brief):

    packet capture -> flow building -> FTP behavioural extraction -> per-connection features
    -> preprocessing -> candidate model -> prediction

The 30 packet features and their order are preserved (they are the first 30 columns of the
candidate's 67). Application-layer features are computed only from cleartext FTP control
traffic actually seen on the wire; when they are unavailable -- a non-FTP flow, or an
encrypted **FTPS** control channel -- they are left **NaN (never zero-filled)** and the record
is flagged ``behavioral_available: false`` with a reason, and the event is logged. The
candidate (a HistGradientBoosting model) consumes NaN natively, so the flow is still scored
from its packet features alone rather than on fabricated values. No heuristics, thresholds or
hard-coded attack rules are applied -- the label comes only from the model.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import ml, live_capture, classes, retraining as rt, retraining_per_connection as rpc, \
    ftp_behavioral as fb, ftp_per_connection as fpc, ftp_cross_session as fcs, pcap_validation as pv

logger = logging.getLogger("predictor.ftp_live")

# Explicit, non-production model selection. This key is deliberately NOT registered in
# ml.MODEL_REGISTRY, so the production prediction paths and the production model files are
# entirely unaffected; only the live console routes it here.
CANDIDATE_KEY = "ftp_pc_candidate"
CANDIDATE_NAME = "FTP-BruteForce behavioural detector (experimental)"
CANDIDATE_REL = "validation/models/ftp-per-connection/candidate_pc_unweighted.pkl"

FEATURES_PC: list[str] = list(rpc.FEATURES_PC)     # 67 = 30 packet + 13 behav + 11 pc + 13 cross
APP_FEATURES: list[str] = FEATURES_PC[30:]         # 37 application-layer features (after the 30 packet)

FTP_LABEL = "FTP-BruteForce"

# How many buffered control packets to keep per FTP session (client<->server window), and how
# many sessions to track, so a busy link cannot exhaust memory. Old sessions are evicted.
MAX_FTP_PKTS = 6000
MAX_FTP_SESSIONS = 512

_RESP_RE = re.compile(rb"^(\d{3})[ -]")
_CMD_RE = re.compile(rb"^([A-Za-z]{3,4})(?:\s+(.*))?$")
_KNOWN_VERBS = {"USER", "PASS", "QUIT", "RETR", "STOR", "LIST", "NLST", "MLSD", "PWD", "SYST",
                "TYPE", "NOOP", "CWD", "CDUP", "PASV", "EPSV", "PORT", "MKD", "RMD", "DELE",
                "SIZE", "MDTM", "STAT", "FEAT", "OPTS", "AUTH", "APPE", "STOU", "XPWD", "ABOR", "REST"}
# Explicit FTPS: an AUTH TLS/SSL on the control channel, or implicit-FTPS ports.
_AUTH_TLS_RE = re.compile(rb"^AUTH\s+(TLS|SSL|TLS-C|TLS-P)\b", re.IGNORECASE)
FTPS_PORTS = {990}

_cache: dict = {}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Candidate model (loaded separately from ml._cache; production cache untouched)
# ---------------------------------------------------------------------------

def candidate_path() -> Path:
    return rpc.repo_root() / CANDIDATE_REL


def load_candidate():
    """Load and cache the experimental candidate and its encoded->name decoder."""
    with _cache_lock:
        if "model" in _cache:
            return _cache["model"], _cache["decode"]
        import joblib
        path = candidate_path()
        if not path.is_file():
            raise live_capture.CaptureError(
                f"Experimental FTP candidate model is missing: {path}. It is a validation "
                "artifact stored in the repo under validation/models/ftp-per-connection/."
            )
        model = joblib.load(path)
        _cache["model"] = model
        _cache["decode"] = rt._encoded_to_name()
        return model, _cache["decode"]


def available_model() -> dict:
    """UI descriptor for the live console dropdown (never added to ml.MODEL_REGISTRY)."""
    return {
        "key": CANDIDATE_KEY,
        "name": CANDIDATE_NAME,
        "experimental": True,
        "present": candidate_path().is_file(),
        "n_features": len(FEATURES_PC),
        "macro_f1": None,
        "blurb": "Validated per-connection + cross-session FTP candidate. Adds cleartext FTP "
                 "behavioural features; encrypted FTPS is flagged as unavailable.",
    }


# ---------------------------------------------------------------------------
# FTP control-channel parsing helpers
# ---------------------------------------------------------------------------

def _payload_role(payload: bytes) -> str | None:
    """'resp' if the payload begins an FTP response, 'cmd' for a known verb, else None."""
    for raw in payload.split(b"\r\n"):
        line = raw.strip()
        if not line:
            continue
        if _RESP_RE.match(line):
            return "resp"
        m = _CMD_RE.match(line)
        if m and m.group(1).decode("latin-1").upper() in _KNOWN_VERBS:
            return "cmd"
    return None


def _is_auth_tls(payload: bytes) -> bool:
    for raw in payload.split(b"\r\n"):
        if _AUTH_TLS_RE.match(raw.strip()):
            return True
    return False


def _stream_key(src_ip, src_port, dst_ip, dst_port):
    return tuple(sorted(((src_ip, src_port), (dst_ip, dst_port))))


def app_features_from_pcap(pcap_path):
    """
    Return ``(app_features, behav)`` where ``app_features`` is the 37 non-packet columns of
    FEATURES_PC (behavioural + per-connection + cross-session) computed by the frozen,
    validated FTP extractors, and ``behav`` is the full 15-feature behavioural dict (used for
    the "is this actually FTP?" check and the session summary). NaN measurements are preserved
    (never zero-filled).
    """
    behav = fb.behavioural_features_for_pcap(pcap_path)
    pc = fpc.per_connection_features_for_pcap(pcap_path)
    cross = fcs.cross_session_features_for_pcap(pcap_path)
    merged = {**behav, **pc, **cross}
    return {f: float(merged[f]) for f in APP_FEATURES}, behav


# ---------------------------------------------------------------------------
# Live capture session (subclass of the verified core; only FTP bits added)
# ---------------------------------------------------------------------------

class FtpLiveCaptureSession(live_capture.CaptureSession):
    """
    A live-capture session that classifies each finished flow with the FTP candidate,
    augmenting the 30 packet features with cleartext-FTP behavioural / per-connection /
    cross-session features when they are available on the wire.
    """

    def __init__(self, interface: str | None, model_key: str = CANDIDATE_KEY):
        super().__init__(interface, CANDIDATE_KEY)
        # stream_key -> {"session": (client_ip, server_ip), "ftps": bool}
        self._ftp_streams: dict = {}
        # session_key (client_ip, server_ip) -> {"pkts": deque, "last": float}
        self._ftp_sessions: dict = {}
        self._ftp_lock = threading.Lock()
        self.ftp_detections = 0
        self.behavioral_unavailable = 0

    # -- capture thread: warm the CANDIDATE (not a production registry key) ----

    def _run(self) -> None:
        from scapy.all import sniff

        try:
            load_candidate()
        except Exception as exc:  # noqa: BLE001
            self.error = f"Candidate model could not be loaded: {exc}"
            return

        try:
            sniff(
                iface=self.interface,
                prn=self._handle,
                store=False,
                stop_filter=lambda _pkt: self._stop.is_set(),
            )
        except PermissionError:
            self.error = (
                "Permission denied opening the interface. Live capture needs root / "
                "CAP_NET_RAW (e.g. run the server with sudo, or grant the capability)."
            )
        except OSError as exc:
            self.error = f"Capture failed: {exc}"
        except Exception as exc:  # noqa: BLE001
            self.error = f"Capture stopped unexpectedly: {exc}"
        finally:
            self._flush_all()
            self.stopped_at = datetime.now(timezone.utc)

    # -- packet handling: buffer FTP control traffic, then run the flow core ---

    def _handle(self, pkt) -> None:
        # Buffer FTP control packets BEFORE the flow core may finalize this flow, so the
        # session buffer already contains this packet when _classify runs.
        try:
            self._buffer_ftp(pkt)
        except Exception:  # noqa: BLE001 - buffering must never break capture
            pass
        super()._handle(pkt)

    def _buffer_ftp(self, pkt) -> None:
        from scapy.layers.inet import IP, TCP
        from scapy.packet import Raw

        if IP not in pkt or TCP not in pkt:
            return
        ip = pkt[IP]; tcp = pkt[TCP]
        sk = _stream_key(ip.src, tcp.sport, ip.dst, tcp.dport)
        payload = bytes(pkt[Raw].load) if Raw in pkt else b""

        known = self._ftp_streams.get(sk)
        if known is None:
            # Encrypted FTPS control channel: implicit-FTPS port, or explicit AUTH TLS/SSL.
            if tcp.dport in FTPS_PORTS or tcp.sport in FTPS_PORTS or _is_auth_tls(payload):
                server_ip = ip.dst if (tcp.dport in FTPS_PORTS or _is_auth_tls(payload)) else ip.src
                client_ip = ip.src if server_ip == ip.dst else ip.dst
                self._ftp_streams[sk] = {"session": (client_ip, server_ip), "ftps": True}
                return
            role = _payload_role(payload)
            if role is None:
                return  # not (yet) a recognised FTP control stream
            if role == "resp":
                server_ip, client_ip = ip.src, ip.dst
            else:  # command -> sender is the client
                client_ip, server_ip = ip.src, ip.dst
            known = {"session": (client_ip, server_ip), "ftps": False}
            self._ftp_streams[sk] = known

        if known.get("ftps"):
            return  # encrypted: nothing cleartext to buffer

        session_key = known["session"]
        with self._ftp_lock:
            sess = self._ftp_sessions.get(session_key)
            if sess is None:
                if len(self._ftp_sessions) >= MAX_FTP_SESSIONS:
                    oldest = min(self._ftp_sessions, key=lambda k: self._ftp_sessions[k]["last"])
                    self._ftp_sessions.pop(oldest, None)
                sess = {"pkts": deque(maxlen=MAX_FTP_PKTS), "last": 0.0}
                self._ftp_sessions[session_key] = sess
            sess["pkts"].append(pkt)
            sess["last"] = float(pkt.time)

    def _session_for_flow(self, flow):
        """Return (session_key, is_ftps) for a flow, or (None, False) if it is not FTP."""
        sk = _stream_key(flow.src_ip, flow.src_port, flow.dst_ip, flow.dst_port)
        known = self._ftp_streams.get(sk)
        if known is None:
            return None, False
        return known["session"], bool(known.get("ftps"))

    # -- classification: candidate model over 67 features, no zero-fill --------

    def _classify(self, flow) -> None:
        from feature_calculator import calculate_features

        try:
            feat30 = calculate_features(flow)
        except Exception as exc:  # noqa: BLE001 - one bad flow must not stop capture
            self.error = f"Feature calculation error: {exc}"
            return

        session_key, is_ftps = self._session_for_flow(flow)
        row, behavioral_available, reason, session_info = self._build_row(feat30, session_key, is_ftps)

        try:
            pred = self._predict(row)
        except Exception as exc:  # noqa: BLE001
            self.error = f"Classification error: {exc}"
            return

        proto = live_capture._PROTO_NAMES.get(flow.protocol, str(flow.protocol))
        record = {
            "src_ip": flow.src_ip, "src_port": flow.src_port,
            "dst_ip": flow.dst_ip, "dst_port": flow.dst_port,
            "protocol": proto, "packets": flow.total_packets,
            "duration_us": feat30.get("Flow Duration", 0.0),
            "label": pred["label"], "confidence": pred["confidence"],
            "is_attack": pred["is_attack"], "family": pred["family"],
            "family_name": pred["family_name"], "top": pred["top"],
            "is_ftp_bruteforce": pred["label"] == FTP_LABEL,
            "behavioral_available": behavioral_available,
            "behavioral_reason": reason,
            "session": session_info,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        with self._lock:
            self._seq += 1
            record["seq"] = self._seq
            self.classified += 1
            if pred["is_attack"]:
                self.attacks += 1
            if record["is_ftp_bruteforce"]:
                self.ftp_detections += 1
            if not behavioral_available:
                self.behavioral_unavailable += 1
            self._recent.appendleft(record)

    def _build_row(self, feat30: dict, session_key, is_ftps: bool):
        """
        Assemble the 67-feature row. The 30 packet features come from the flow (real values,
        order preserved); the 37 application-layer features default to NaN and are only filled
        from cleartext FTP control traffic actually captured. Never zero-filled.
        """
        row = {f: float(feat30.get(f, np.nan)) for f in ml.FEATURES}  # 30 packet, in order
        for f in APP_FEATURES:
            row[f] = np.nan  # unavailable until proven otherwise

        behavioral_available = False
        reason = "no-ftp-control"
        session_info: dict = {}

        if is_ftps:
            reason = "encrypted-ftps"
            logger.info(
                "FTP behavioural features unavailable (encrypted FTPS control channel) for "
                "session %s; scoring from packet features only.", session_key,
            )
            return row, behavioral_available, reason, session_info

        if session_key is None:
            return row, behavioral_available, reason, session_info

        with self._ftp_lock:
            sess = self._ftp_sessions.get(session_key)
            pkts = list(sess["pkts"]) if sess else []

        if not pkts:
            return row, behavioral_available, reason, session_info

        appfeat, behav = self._extract_app_features(pkts)
        if appfeat is None or not fb.has_ftp_control(behav):
            reason = "no-ftp-control"
            return row, behavioral_available, reason, session_info

        for f in APP_FEATURES:
            row[f] = appfeat[f]  # may be NaN (a genuinely undefined measurement)
        behavioral_available = True
        reason = "ok"
        session_info = {
            "client_ip": session_key[0], "server_ip": session_key[1],
            "control_connections": int(behav.get("ftp_control_connections", 0)),
            "login_attempts": int(behav.get("ftp_login_attempts", 0)),
            "failed_logins": int(behav.get("ftp_failed_logins", 0)),
            "successful_logins": int(behav.get("ftp_successful_logins", 0)),
        }
        return row, behavioral_available, reason, session_info

    def _extract_app_features(self, pkts):
        """Write buffered packets to a temp pcap and run the frozen FTP extractors on it."""
        import os
        import tempfile
        from scapy.utils import wrpcap

        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(prefix="ftp_live_", suffix=".pcap")
            os.close(fd)
            wrpcap(tmp, pkts)
            return app_features_from_pcap(tmp)
        except Exception as exc:  # noqa: BLE001 - degrade to "unavailable" rather than crash
            logger.warning("FTP behavioural extraction failed: %s", exc)
            return None, {}
        finally:
            if tmp is not None:
                try:
                    Path(tmp).unlink()
                except OSError:
                    pass

    def _predict(self, row: dict) -> dict:
        model, decode = load_candidate()
        # Preserve NaN: build the frame directly from the row (no .get(f, 0.0) fill).
        frame = pd.DataFrame([row])[FEATURES_PC]
        proba = model.predict_proba(frame)[0]
        order = np.argsort(proba)[::-1]

        def name_at(i):
            return decode.get(int(model.classes_[i]), f"Class {model.classes_[i]}")

        ranked = [{"label": name_at(i), "confidence": float(proba[i])} for i in order if proba[i] > 0]
        top = ranked[:3] if ranked else [{"label": name_at(order[0]), "confidence": 0.0}]
        for r in top:
            r["family"] = classes.family_of(r["label"])
            r["family_name"] = classes.family_name(r["label"])
        verdict = top[0]
        return {
            "label": verdict["label"], "confidence": verdict["confidence"],
            "is_attack": verdict["label"] != ml.BENIGN_LABEL,
            "family": classes.family_of(verdict["label"]),
            "family_name": classes.family_name(verdict["label"]),
            "top": top,
        }

    # -- status snapshot (adds the FTP-specific fields) ------------------------

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            recent = [r for r in self._recent if r["seq"] > since]
            return {
                "running": self.running,
                "interface": self.interface or "auto",
                "model_key": CANDIDATE_KEY,
                "model_name": CANDIDATE_NAME,
                "detector": "ftp",
                "packets": self.packets,
                "flows": self.classified,
                "attacks": self.attacks,
                "ftp_detections": self.ftp_detections,
                "behavioral_unavailable": self.behavioral_unavailable,
                "open_flows": len(self._flows),
                "ftp_sessions": len(self._ftp_sessions),
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "stopped_at": self.stopped_at.isoformat(timespec="seconds") if self.stopped_at else None,
                "error": self.error,
                "recent": recent,
            }


class FtpCaptureManager(live_capture.CaptureManager):
    """Process-wide holder for the single active FTP live-capture session."""

    def start(self, interface: str | None, model_key: str = CANDIDATE_KEY) -> FtpLiveCaptureSession:
        with self._lock:
            if self._session is not None and self._session.running:
                raise live_capture.CaptureError("A capture is already running. Stop it first.")
            session = FtpLiveCaptureSession(interface, CANDIDATE_KEY)
            session.start()
            self._session = session
            return session


manager = FtpCaptureManager()
