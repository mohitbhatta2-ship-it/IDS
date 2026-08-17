"""
Transparent internal FTP specialisation for the production Live Capture system.

The live console selects a normal **production** model, exactly as before. This module makes
FTP detection automatic and invisible to the user: the single live-capture session routes each
finished flow to the appropriate feature/model pipeline internally --

    * non-FTP traffic          -> the user-selected production model (30 packet features),
                                  byte-for-byte the existing behaviour (``ml.predict_one``);
    * cleartext FTP traffic    -> the validated FTP per-connection detector
                                  (``validation/models/ftp-per-connection/candidate_pc_unweighted.pkl``,
                                  67 features = the same 30 packet features, in the same order,
                                  plus the frozen behavioural / per-connection / cross-session
                                  features);
    * FTPS / encrypted FTP     -> the production packet-feature path, with behavioural features
                                  left UNAVAILABLE and never fabricated / zero-filled.

The routing layer only *chooses the pipeline*; it never decides "attack". Every label comes
from a model's ``predict_proba`` -- the production model for non-FTP/FTPS, the FTP candidate
for cleartext FTP. No heuristics, thresholds or hard-coded attack rules are added.

Nothing here retrains or writes to ``webapp_data/Results/Models/``. The FTP candidate is loaded
from the repo's ``validation/models/…`` artifact through a private cache; the production
``ml.MODEL_REGISTRY`` and ``ml._cache`` are untouched (the FTP model is NOT registered there,
so it can never be selected as a normal model). The 30 packet features and their order are
preserved (they are the first 30 columns of the candidate's 67). The verified ``Live/`` core
computes those 30 features exactly as ``live_capture`` and ``pcap_validation`` do.

Integration mechanism: this module installs :class:`UnifiedLiveCaptureSession` as
``live_capture.manager.session_factory`` (a backward-compatible hook), so the existing
``live_capture.manager`` -- and therefore the existing views, endpoints and dashboard --
transparently gain FTP specialisation without any UI or API change.
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
    ftp_behavioral as fb, ftp_per_connection as fpc, ftp_cross_session as fcs  # noqa: F401

logger = logging.getLogger("predictor.ftp_live")

# Internal-only identifier for the FTP candidate. It is deliberately NOT a key in
# ml.MODEL_REGISTRY and is never shown in the model dropdown -- FTP routing is automatic.
CANDIDATE_ID = "ftp_pc_candidate"
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
# FTP candidate model (loaded separately from ml._cache; production cache untouched)
# ---------------------------------------------------------------------------

def candidate_path() -> Path:
    return rpc.repo_root() / CANDIDATE_REL


def load_candidate():
    """Load and cache the validated FTP candidate and its encoded->name decoder."""
    with _cache_lock:
        if "model" in _cache:
            return _cache["model"], _cache["decode"]
        import joblib
        path = candidate_path()
        if not path.is_file():
            raise live_capture.CaptureError(
                f"Validated FTP candidate model is missing: {path}. It is a validation "
                "artifact stored in the repo under validation/models/ftp-per-connection/."
            )
        model = joblib.load(path)
        _cache["model"] = model
        _cache["decode"] = rt._encoded_to_name()
        return model, _cache["decode"]


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
# Unified live capture session (subclass of the verified core; routing added)
# ---------------------------------------------------------------------------

class UnifiedLiveCaptureSession(live_capture.CaptureSession):
    """
    The production live-capture session with transparent internal FTP specialisation.

    A single session runs the user-selected production ``model_key``. Each finished flow is
    routed by *protocol/session identification*: non-FTP and FTPS flows are scored by the
    production model exactly as before; cleartext FTP flows are scored by the validated FTP
    candidate over the 67-feature vector (30 packet + behavioural + per-connection +
    cross-session). The label always comes from a model prediction, never from the router.
    """

    def __init__(self, interface: str | None, model_key: str = ml.DEFAULT_MODEL):
        super().__init__(interface, model_key)     # model_key is a REAL production registry key
        # stream_key -> {"session": (client_ip, server_ip), "ftps": bool}
        self._ftp_streams: dict = {}
        # session_key (client_ip, server_ip) -> {"pkts": deque, "last": float}
        self._ftp_sessions: dict = {}
        self._ftp_lock = threading.Lock()
        self._ftp_enabled = True          # downgraded in _run if the candidate cannot load
        self.ftp_flows = 0                # flows routed to (or recognised as) FTP
        self.ftp_detections = 0           # flows the FTP model called FTP-BruteForce
        self.behavioral_unavailable = 0   # FTP-relevant flows scored without behavioural features

    # -- capture thread: warm BOTH the production model and the FTP candidate --

    def _run(self) -> None:
        # Warm the FTP candidate up-front so cleartext FTP flows do not pay the unpickling
        # cost mid-sweep. If it is unavailable, degrade gracefully to production-only routing
        # (still a fully working NIDS) rather than failing the capture.
        try:
            load_candidate()
            self._ftp_enabled = True
        except Exception as exc:  # noqa: BLE001
            self._ftp_enabled = False
            logger.warning(
                "FTP detector unavailable; production model handles all flows: %s", exc)
        # The parent warms the production model (self.model_key) and runs the sniff loop,
        # driving our overridden _handle / _classify.
        super()._run()

    # -- packet handling: buffer FTP control traffic, then run the flow core ---

    def _handle(self, pkt) -> None:
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

    # -- classification: route to production or FTP model (never a rule) -------

    def _classify(self, flow) -> None:
        from feature_calculator import calculate_features

        try:
            feat30 = calculate_features(flow)
        except Exception as exc:  # noqa: BLE001 - one bad flow must not stop capture
            self.error = f"Feature calculation error: {exc}"
            return

        session_key, is_ftps = self._session_for_flow(flow)

        routed = "production"
        ftp_relevant = False
        behavioral_available = False
        reason = None
        session_info: dict = {}

        if is_ftps:
            # FTPS: behavioural features are unavailable (encrypted) and must NOT be
            # fabricated. Score with the production packet-feature path.
            ftp_relevant = True
            reason = "encrypted-ftps"
            logger.info(
                "FTP behavioural features unavailable (encrypted FTPS control channel) for "
                "session %s; scoring with the production model on packet features.", session_key)
            pred = self._predict_production(feat30)
        elif self._ftp_enabled and session_key is not None:
            # A recognised cleartext FTP session: try the behavioural pipeline.
            row, avail, r, info = self._build_row(feat30, session_key)
            if avail:
                routed = "ftp"
                ftp_relevant = True
                behavioral_available = True
                reason = "ok"
                session_info = info
                pred = self._predict_candidate(row)
            else:
                # Recognised as FTP but no usable cleartext control yet (e.g. a data
                # connection): fall back to the production model, no fabrication.
                ftp_relevant = True
                reason = r
                pred = self._predict_production(feat30)
        else:
            # Ordinary non-FTP traffic: unchanged production behaviour.
            pred = self._predict_production(feat30)

        proto = live_capture._PROTO_NAMES.get(flow.protocol, str(flow.protocol))
        record = {
            "src_ip": flow.src_ip, "src_port": flow.src_port,
            "dst_ip": flow.dst_ip, "dst_port": flow.dst_port,
            "protocol": proto, "packets": flow.total_packets,
            "duration_us": feat30.get("Flow Duration", 0.0),
            "label": pred["label"], "confidence": pred["confidence"],
            "is_attack": pred["is_attack"], "family": pred["family"],
            "family_name": pred["family_name"], "top": pred.get("top", []),
            "routed": routed,
            "ftp_relevant": ftp_relevant,
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
            if ftp_relevant:
                self.ftp_flows += 1
                if not behavioral_available:
                    self.behavioral_unavailable += 1
            if record["is_ftp_bruteforce"]:
                self.ftp_detections += 1
            self._recent.appendleft(record)

    def _predict_production(self, feat30: dict) -> dict:
        """The existing production path -- byte-for-byte the pre-integration behaviour."""
        return ml.predict_one(feat30, self.model_key)

    def _build_row(self, feat30: dict, session_key):
        """
        Assemble the 67-feature row for a cleartext FTP session. The 30 packet features come
        from the flow (real values, order preserved); the 37 application-layer features default
        to NaN and are only filled from cleartext FTP control traffic actually captured. Never
        zero-filled. Returns ``(row, behavioral_available, reason, session_info)``.
        """
        row = {f: float(feat30.get(f, np.nan)) for f in ml.FEATURES}  # 30 packet, in order
        for f in APP_FEATURES:
            row[f] = np.nan  # unavailable until proven otherwise

        with self._ftp_lock:
            sess = self._ftp_sessions.get(session_key)
            pkts = list(sess["pkts"]) if sess else []

        if not pkts:
            return row, False, "no-ftp-control", {}

        appfeat, behav = self._extract_app_features(pkts)
        if appfeat is None or not fb.has_ftp_control(behav):
            return row, False, "no-ftp-control", {}

        for f in APP_FEATURES:
            row[f] = appfeat[f]  # may be NaN (a genuinely undefined measurement)
        session_info = {
            "client_ip": session_key[0], "server_ip": session_key[1],
            "control_connections": int(behav.get("ftp_control_connections", 0)),
            "login_attempts": int(behav.get("ftp_login_attempts", 0)),
            "failed_logins": int(behav.get("ftp_failed_logins", 0)),
            "successful_logins": int(behav.get("ftp_successful_logins", 0)),
        }
        return row, True, "ok", session_info

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

    def _predict_candidate(self, row: dict) -> dict:
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

    # -- status snapshot: production model identity + FTP routing counters -----

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            recent = [r for r in self._recent if r["seq"] > since]
            return {
                "running": self.running,
                "interface": self.interface or "auto",
                "model_key": self.model_key,
                "model_name": ml.MODEL_REGISTRY.get(self.model_key, {}).get("name", self.model_key),
                "packets": self.packets,
                "flows": self.classified,
                "attacks": self.attacks,
                "ftp_flows": self.ftp_flows,
                "ftp_detections": self.ftp_detections,
                "behavioral_unavailable": self.behavioral_unavailable,
                "open_flows": len(self._flows),
                "ftp_sessions": len(self._ftp_sessions),
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "stopped_at": self.stopped_at.isoformat(timespec="seconds") if self.stopped_at else None,
                "error": self.error,
                "recent": recent,
            }


# Backward-compatible alias.
FtpLiveCaptureSession = UnifiedLiveCaptureSession


def install(manager: "live_capture.CaptureManager | None" = None) -> None:
    """
    Install the unified session as a capture manager's factory so the existing live console
    transparently gains FTP specialisation. Idempotent; defaults to the process-wide
    ``live_capture.manager``. Backward-compatible: the production 30-feature path is unchanged
    for non-FTP/FTPS flows.
    """
    mgr = manager if manager is not None else live_capture.manager
    mgr.session_factory = UnifiedLiveCaptureSession


# Install on import so any code path that reaches the live console (views, dashboard) uses the
# unified session. Non-FTP behaviour is identical to before; only FTP flows are specialised.
install()
