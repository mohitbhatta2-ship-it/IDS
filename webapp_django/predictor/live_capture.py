"""
Live packet capture for the web app.

This is the web-facing wrapper around the *verified* live feature-extraction
core in the repository's ``Live/`` package. It deliberately reuses only the three
dependency-light, parity-checked modules from there --

    Live/flow.py               (the Flow accumulator)
    Live/feature_extractor.py  (update_flow: packet -> accumulators)
    Live/feature_calculator.py (calculate_features: accumulators -> 30 features)

-- and classifies each finished flow through :func:`predictor.ml.predict_one`,
the exact same preprocessing + model pipeline the manual and dataset pages use.
Nothing here re-implements feature maths or model logic.

What is *not* reused: ``Live/flow_builder.py`` and ``Live/flow_manager.py``. Both
eagerly ``import config`` (which resolves the data bundle at import time and can
raise) and keep flow state in a module global with a CSV writer and their own
predictor -- all wrong for a long-lived, multi-threaded server. The flow table
and expiry are therefore owned per-session here, mirroring flow_builder's keying
exactly but scoped to an instance and guarded by a lock.

Everything scapy-related is imported lazily inside :meth:`CaptureSession._run`
so that importing this module -- and therefore starting Django -- never fails on
a host without scapy or without capture privileges. The page degrades to a clear
"unavailable" message instead of taking the rest of the app down with it.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from . import ml

# How many recent classified flows the status endpoint keeps for the UI.
MAX_RECENT = 200

# Flow lifetime, matching CICFlowMeter / the Live CLI default (seconds, capture
# clock). A flow idle longer than this is expired and classified.
FLOW_TIMEOUT = 120.0

# Hard cap on concurrently tracked flows so a busy link cannot exhaust memory.
MAX_FLOWS = 20000

# Protocol numbers we assemble into flows (TCP, UDP). Everything else is ignored,
# exactly as the CLI does.
_PROTO_NAMES = {6: "TCP", 17: "UDP"}


def _locate_live_package() -> Path | None:
    """
    Find the repository's ``Live/`` directory so its modules can be imported.

    Walks up from this file looking for a ``Live`` folder that actually contains
    the feature calculator -- the same defensive style as ml._locate_data_root.
    Returns None if it cannot be found, so the caller can report a clean error.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "Live"
        if (candidate / "feature_calculator.py").is_file():
            return candidate
    return None


def _ensure_live_on_path() -> Path:
    """
    Put the repository's ``Live/`` directory on sys.path so its modules import,
    and return it. Raises CaptureError if it cannot be found.
    """
    live_dir = _locate_live_package()
    if live_dir is None:
        raise CaptureError(
            "Could not locate the Live/ package next to the web app; live "
            "capture is unavailable."
        )
    if str(live_dir) not in sys.path:
        sys.path.insert(0, str(live_dir))
    return live_dir


class CaptureError(RuntimeError):
    """Raised when a capture cannot be started (no scapy, bad interface, ...)."""


class CaptureSession:
    """
    One live-capture run: a background sniff thread feeding a flow table, with
    finished flows classified through ``ml.predict_one`` and pushed to a bounded
    buffer the status endpoint reads.
    """

    def __init__(self, interface: str | None, model_key: str):
        self.interface = interface or None          # None => scapy's default
        self.model_key = model_key
        self.started_at = datetime.now(timezone.utc)
        self.stopped_at: datetime | None = None

        # Counters and the recent-flows buffer are read by request threads and
        # written by the capture thread, so every access goes through _lock.
        self._lock = threading.Lock()
        self._flows: dict = {}
        self._recent: deque = deque(maxlen=MAX_RECENT)
        self.packets = 0
        self.classified = 0
        self.attacks = 0

        self._last_cleanup: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self._seq = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Spawn the capture thread. Raises CaptureError if scapy is missing."""
        _ensure_live_on_path()

        # Lazy import: only now do we require scapy. Keeping it out of module
        # import means the web app starts fine without it.
        try:
            from scapy.all import sniff  # noqa: F401
            from scapy.layers.inet import IP  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            raise CaptureError(
                "Scapy is not available, so live capture cannot run here. "
                f"({exc})"
            ) from exc

        self._thread = threading.Thread(target=self._run, name="ids-capture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 6.0) -> None:
        """Signal the thread to stop, flush open flows, and wait for it briefly."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        if self.stopped_at is None:
            self.stopped_at = datetime.now(timezone.utc)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- capture thread ----------------------------------------------------

    def _run(self) -> None:
        from scapy.all import sniff

        # Warm the model once, on this thread, so the first classified flow does
        # not pay the unpickling cost mid-sweep and so a missing model surfaces
        # as a capture error rather than a silent stall.
        try:
            ml._load(self.model_key)
        except Exception as exc:  # noqa: BLE001
            self.error = f"Model could not be loaded: {exc}"
            return

        try:
            # stop_filter is evaluated per packet, so Stop takes effect when the
            # next packet arrives. That is fine here: stop() does not wait on the
            # sniff loop beyond a short join, the thread is a daemon so it never
            # blocks process exit, and open flows are flushed in the finally below.
            sniff(
                iface=self.interface,
                prn=self._handle,
                store=False,
                stop_filter=lambda _pkt: self._stop.is_set(),
            )
        except PermissionError:
            self.error = (
                "Permission denied opening the interface. Live capture needs "
                "root / CAP_NET_RAW (e.g. run the server with sudo, or grant the "
                "capability)."
            )
        except OSError as exc:
            self.error = f"Capture failed: {exc}"
        except Exception as exc:  # noqa: BLE001 - never let the thread die silently
            self.error = f"Capture stopped unexpectedly: {exc}"
        finally:
            # Score whatever is still open so a short capture is not lost.
            self._flush_all()
            self.stopped_at = datetime.now(timezone.utc)

    def _handle(self, pkt) -> None:
        from scapy.layers.inet import IP

        if IP not in pkt:
            return

        flow, key = self._get_or_create_flow(pkt)
        if flow is None:
            return

        # update_flow is the verified extractor from Live/.
        from feature_extractor import update_flow

        update_flow(flow, pkt)

        with self._lock:
            self.packets += 1

        now = float(pkt.time)
        if self._last_cleanup is None:
            self._last_cleanup = now
        elif now - self._last_cleanup >= 5.0:
            self._sweep(now)
            self._last_cleanup = now

    # -- flow table (mirrors Live/flow_builder, but instance-scoped) --------

    @staticmethod
    def _make_key(src_ip, dst_ip, src_port, dst_port, proto):
        return (src_ip, dst_ip, src_port, dst_port, proto)

    def _get_or_create_flow(self, pkt):
        from flow import Flow
        from scapy.layers.inet import IP

        ip = pkt[IP]
        proto = ip.proto
        if proto == 6:
            transport = pkt["TCP"]
        elif proto == 17:
            transport = pkt["UDP"]
        else:
            return None, None

        forward_key = self._make_key(ip.src, ip.dst, transport.sport, transport.dport, proto)
        reverse_key = self._make_key(ip.dst, ip.src, transport.dport, transport.sport, proto)

        with self._lock:
            if forward_key in self._flows:
                return self._flows[forward_key], forward_key
            if reverse_key in self._flows:
                return self._flows[reverse_key], reverse_key

            if len(self._flows) >= MAX_FLOWS:
                oldest = min(self._flows, key=lambda k: self._flows[k].last_seen or 0)
                del self._flows[oldest]

            flow = Flow(
                src_ip=ip.src,
                dst_ip=ip.dst,
                src_port=transport.sport,
                dst_port=transport.dport,
                protocol=proto,
            )
            self._flows[forward_key] = flow
            return flow, forward_key

    def _sweep(self, now: float) -> None:
        """Expire and classify every flow idle longer than the timeout."""
        with self._lock:
            expired = [
                key
                for key, flow in self._flows.items()
                if flow.last_seen is not None and (now - float(flow.last_seen)) > FLOW_TIMEOUT
            ]
            flows = [self._flows.pop(key) for key in expired]

        for flow in flows:
            self._classify(flow)

    def _flush_all(self) -> None:
        with self._lock:
            flows = list(self._flows.values())
            self._flows.clear()
        for flow in flows:
            self._classify(flow)

    # -- classification (reuses the offline pipeline) ----------------------

    def _classify(self, flow) -> None:
        from feature_calculator import calculate_features

        try:
            features = calculate_features(flow)
            result = ml.predict_one(features, self.model_key)
        except Exception as exc:  # noqa: BLE001 - one bad flow must not stop capture
            self.error = f"Classification error: {exc}"
            return

        proto = _PROTO_NAMES.get(flow.protocol, str(flow.protocol))
        record = {
            "src_ip": flow.src_ip,
            "src_port": flow.src_port,
            "dst_ip": flow.dst_ip,
            "dst_port": flow.dst_port,
            "protocol": proto,
            "packets": flow.total_packets,
            "duration_us": features.get("Flow Duration", 0.0),
            "label": result["label"],
            "confidence": result["confidence"],
            "is_attack": result["is_attack"],
            "family": result["family"],
            "family_name": result["family_name"],
            # Wall-clock time the flow was classified, for the UI timeline.
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        with self._lock:
            self._seq += 1
            record["seq"] = self._seq
            self.classified += 1
            if result["is_attack"]:
                self.attacks += 1
            self._recent.appendleft(record)

    # -- status snapshot (read by the request thread) ----------------------

    def snapshot(self, since: int = 0) -> dict:
        """
        A JSON-serialisable view of the session. ``since`` lets the client fetch
        only records newer than the highest seq it has already shown.
        """
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
                "open_flows": len(self._flows),
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "stopped_at": self.stopped_at.isoformat(timespec="seconds") if self.stopped_at else None,
                "error": self.error,
                "recent": recent,
            }


class CaptureManager:
    """Process-wide holder for the single active capture session."""

    def __init__(self):
        self._lock = threading.Lock()
        self._session: CaptureSession | None = None

    def start(self, interface: str | None, model_key: str) -> CaptureSession:
        with self._lock:
            if self._session is not None and self._session.running:
                raise CaptureError("A capture is already running. Stop it first.")
            session = CaptureSession(interface, model_key)
            session.start()   # may raise CaptureError (no scapy) before we store it
            self._session = session
            return session

    def stop(self) -> CaptureSession | None:
        with self._lock:
            session = self._session
        if session is not None:
            session.stop()
        return session

    @property
    def session(self) -> CaptureSession | None:
        return self._session


# One manager per process. gunicorn with multiple workers would give each worker
# its own manager; capture is a single-host developer/operator tool, so that is
# acceptable and documented rather than coordinated across workers.
manager = CaptureManager()


def list_interfaces() -> list[str]:
    """
    Available capture interface names, or an empty list if scapy is unavailable.
    Never raises -- the page falls back to the "Auto" option alone.
    """
    try:
        _ensure_live_on_path()
        from scapy.all import get_if_list

        return sorted(get_if_list())
    except Exception:  # noqa: BLE001
        return []


def capture_supported() -> bool:
    """True when scapy can be imported, i.e. capture is at least possible here."""
    try:
        _ensure_live_on_path()
        import scapy.all  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False
