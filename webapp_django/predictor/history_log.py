"""
Prediction history, kept in an append-only log file.

Every classification run appends one JSON object on its own line (JSONL) to
``logs/predictions.jsonl``. A log file rather than a database table because the
history is write-once, read-often evidence of what the app was asked to
classify: it survives independently of the database, can be copied off the
machine, opened in a text editor, or read straight back into pandas for a
report, and a run is never lost to a schema migration.

The format is deliberately forgiving. A line that cannot be parsed -- a write
cut short by a restart, say -- is skipped rather than taking the whole page
down with it, so the worst a crash mid-write costs is the run being written.

Where the file lives is settings.HISTORY_LOG_PATH (env: IDS_HISTORY_LOG). On a
host with an ephemeral filesystem the log is only as durable as the container,
exactly like the SQLite file next to it -- point IDS_HISTORY_LOG at a mounted
disk to keep history across deploys.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from django.conf import settings

try:  # POSIX only; the app also has to run on Windows for development.
    import fcntl
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None


MANUAL = "manual"
BATCH = "batch"

KIND_LABELS = {MANUAL: "Manual entry", BATCH: "Dataset upload"}

# The page reads the tail of the log, not all of it, so a long-running instance
# does not pay for its entire history on every request. Runs older than this
# stay in the file -- they are just not rendered.
SCAN_LINES = 500

# Rotate at ~4 MB (tens of thousands of runs) and keep one previous file, so an
# instance left running does not fill its disk with history.
MAX_BYTES = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# One recorded run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """
    One line of the log, parsed. Mirrors what the history table needs; the
    properties are the same ones the template used when this was a model, so
    the presentation logic did not have to move into the template.
    """

    at: datetime | None = None
    kind: str = BATCH
    model_key: str = ""
    model_name: str = ""

    # Manual entry
    predicted_label: str = ""
    confidence: float | None = None
    features: dict | None = None

    # Dataset upload
    source_filename: str = ""
    row_count: int | None = None
    attack_count: int | None = None
    accuracy: float | None = None
    macro_f1: float | None = None

    @property
    def kind_display(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)

    @property
    def is_manual(self) -> bool:
        return self.kind == MANUAL

    @property
    def is_attack(self) -> bool:
        return bool(self.predicted_label) and self.predicted_label != "Benign"

    @property
    def was_evaluated(self) -> bool:
        """
        True when the upload carried a usable Label column, so accuracy and
        macro F1 were computed. Checked explicitly against None because a
        genuinely poor run can score 0.0, which is a real result rather than
        a missing one.
        """
        return self.accuracy is not None

    @property
    def blank_reason(self) -> str:
        """Why this run has no evaluation scores, phrased for a tooltip."""
        if self.is_manual:
            return (
                "Manual entries classify a single flow with no ground truth, "
                "so there is nothing to score against."
            )
        return (
            "This upload had no Label column, so there was no ground truth to "
            "score the predictions against. Re-upload a file that includes one "
            "to get accuracy and macro F1."
        )

    @property
    def attack_share(self) -> float:
        if not self.row_count:
            return 0.0
        return (self.attack_count or 0) / self.row_count

    @property
    def attack_pct(self) -> float:
        return 100.0 * self.attack_share

    def as_json(self) -> str:
        payload = {
            "at": self.at.isoformat() if self.at else None,
            "kind": self.kind,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "features": self.features,
            "source_filename": self.source_filename,
            "row_count": self.row_count,
            "attack_count": self.attack_count,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
        }
        # Drop the keys a run of this kind never fills in, so a manual entry
        # does not carry ten nulls and the file stays readable by eye.
        return json.dumps({k: v for k, v in payload.items() if v not in (None, "")})


def _parse(line: str) -> Run | None:
    """A malformed or truncated line is skipped, never raised."""
    try:
        raw = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None

    at = None
    stamp = raw.get("at")
    if isinstance(stamp, str):
        try:
            at = datetime.fromisoformat(stamp)
        except ValueError:
            at = None

    known = {f for f in Run.__dataclass_fields__ if f != "at"}
    return Run(at=at, **{k: v for k, v in raw.items() if k in known})


# ---------------------------------------------------------------------------
# The file itself
# ---------------------------------------------------------------------------


def log_path() -> Path:
    """Read per call, so a test can point the log somewhere temporary."""
    return Path(settings.HISTORY_LOG_PATH)


def _rotated_path() -> Path:
    path = log_path()
    return path.with_name(path.name + ".1")


def _files() -> list[Path]:
    """Oldest first: the rotated file, then the live one."""
    return [p for p in (_rotated_path(), log_path()) if p.exists()]


def _rotate_if_large(path: Path) -> None:
    try:
        if path.stat().st_size < MAX_BYTES:
            return
        os.replace(path, _rotated_path())
    except OSError:  # pragma: no cover - the append below reports the real problem
        pass


def record(**fields) -> Run:
    """
    Append one run to the log and return it.

    Recording history must never be the reason a prediction fails, so an
    unwritable log is swallowed: the caller still gets its answer, and the
    history page is simply missing that run.
    """
    run = Run(at=datetime.now(timezone.utc), **fields)
    path = log_path()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_if_large(path)
        with path.open("a", encoding="utf-8") as handle:
            # Two gunicorn workers can append at the same moment. The lock
            # keeps their lines from interleaving; it is released on close.
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(run.as_json() + "\n")
    except OSError:
        pass

    return run


def recent(limit: int = SCAN_LINES) -> list[Run]:
    """The most recent runs, newest first. Unreadable lines are skipped."""
    lines: deque[str] = deque(maxlen=limit)
    for path in _files():
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        lines.append(line)
        except OSError:
            continue

    runs = [run for run in (_parse(line) for line in lines) if run is not None]
    runs.reverse()
    return runs


def count() -> int:
    """Total runs on file, including any older than the read window."""
    total = 0
    for path in _files():
        try:
            with path.open("rb") as handle:
                total += sum(1 for line in handle if line.strip())
        except OSError:
            continue
    return total


def clear() -> None:
    """Delete the log and its rotated predecessor."""
    for path in (log_path(), _rotated_path()):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def raw_text() -> str:
    """The whole log, oldest first -- what the Download button serves."""
    chunks = []
    for path in _files():
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "".join(chunks)


# ---------------------------------------------------------------------------
# What the history page shows above the table
# ---------------------------------------------------------------------------


@dataclass
class Summary:
    runs: int = 0
    flows: int = 0
    attacks: int = 0
    scored_runs: int = 0
    mean_accuracy: float | None = None
    models: list = field(default_factory=list)
    last_at: datetime | None = None

    @property
    def attack_share(self) -> float:
        return (self.attacks / self.flows) if self.flows else 0.0

    @property
    def attack_pct(self) -> float:
        return 100.0 * self.attack_share


def summarise(runs: list[Run]) -> Summary:
    """Roll a list of runs up into the headline tiles."""
    accuracies = [r.accuracy for r in runs if r.was_evaluated]
    stamps = [r.at for r in runs if r.at]
    return Summary(
        runs=len(runs),
        flows=sum(r.row_count or 0 for r in runs),
        attacks=sum(r.attack_count or 0 for r in runs),
        scored_runs=len(accuracies),
        mean_accuracy=(sum(accuracies) / len(accuracies)) if accuracies else None,
        models=sorted({r.model_name for r in runs if r.model_name}),
        last_at=max(stamps) if stamps else None,
    )
