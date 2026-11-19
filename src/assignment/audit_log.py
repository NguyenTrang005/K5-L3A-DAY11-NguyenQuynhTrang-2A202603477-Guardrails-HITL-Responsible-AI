"""
Assignment 11 — Audit log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str | None], deque[tuple[dict, float]]] = (
            defaultdict(deque)
        )

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Track the start of a request with its redacted input."""
        entry = {
            "user_id": user_id,
            "request_id": request_id,
            "input": content_filter(text)["redacted"],
            "timestamp": utc_now_iso(),
        }
        self._open[(user_id, request_id)].append((entry, perf_counter()))

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Match a response to its request and store the completed record."""
        key = (user_id, request_id)
        pending = self._open.get(key)
        if pending:
            entry, started = pending.popleft()
            latency_ms = round((perf_counter() - started) * 1000, 2)
            if not pending:
                del self._open[key]
        else:
            entry = {
                "user_id": user_id,
                "request_id": request_id,
                "input": None,
                "timestamp": utc_now_iso(),
            }
            latency_ms = None

        entry.update(
            output=content_filter(text)["redacted"],
            blocked=blocked,
            layer=layer,
            latency_ms=latency_ms,
        )
        self.logs.append(entry)

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
