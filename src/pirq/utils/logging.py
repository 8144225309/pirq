"""PIRQ Logging Utilities.

JSONL logging for event tracking and debugging.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class JsonlLogger:
    """Append-only JSONL logger for PIRQ events."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure log directory exists."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log an event with optional data.

        Args:
            event: Event type (e.g., "session_start", "pirq_check", "run_start")
            data: Optional event data
        """
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": event,
            "pid": os.getpid(),
        }

        if data:
            entry.update(data)

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except IOError:
            pass  # Best effort logging

    def log_session_start(self, task: Optional[str] = None) -> None:
        """Log session start event."""
        self.log("session_start", {"task": task})

    def log_session_end(self, reason: str = "normal") -> None:
        """Log session end event."""
        self.log("session_end", {"reason": reason})

    def log_pirq_check(self, gate: str, status: str, message: str) -> None:
        """Log a PIRQ gate check."""
        self.log("pirq_check", {
            "gate": gate,
            "status": status,
            "message": message,
        })

    def log_pirq_blocked(self, gates: list, reason: str) -> None:
        """Log when PIRQs block execution."""
        self.log("pirq_blocked", {
            "gates": gates,
            "reason": reason,
        })

    def log_run_start(self, prompt_hash: str, task: Optional[str] = None) -> None:
        """Log Claude invocation start."""
        self.log("run_start", {
            "prompt_hash": prompt_hash,
            "task": task,
        })

    def log_run_end(
        self,
        tokens_used: int,
        duration_ms: int,
        exit_code: int,
        orch_tag: Optional[str] = None,
    ) -> None:
        """Log Claude invocation end."""
        self.log("run_end", {
            "tokens_used": tokens_used,
            "duration_ms": duration_ms,
            "exit_code": exit_code,
            "orch_tag": orch_tag,
        })

    def log_error(self, error: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an error."""
        data = {"error": error}
        if context:
            data["context"] = context
        self.log("error", data)

    def read_recent(self, n: int = 50) -> list:
        """Read the last N log entries."""
        if not self.log_path.exists():
            return []

        entries = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-n:]:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass

        return entries


class TokenLogger:
    """Specialized logger for token usage tracking."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure log directory exists."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        task: Optional[str] = None,
        prompt_num: Optional[int] = None,
    ) -> None:
        """Log token usage for a run."""
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "task": task,
            "prompt_num": prompt_num,
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except IOError:
            pass

    def get_total_usage(self, since: Optional[datetime] = None) -> int:
        """Get total token usage, optionally since a date."""
        if not self.log_path.exists():
            return 0

        total = 0
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)

                        if since:
                            entry_time = datetime.fromisoformat(
                                entry["ts"].replace("Z", "+00:00")
                            )
                            if entry_time < since:
                                continue

                        total += entry.get("total_tokens", 0)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except IOError:
            pass

        return total
