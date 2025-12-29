"""PIRQ Loop Detection Gate - Prevents infinite loops.

Detects when Claude is stuck in a loop by tracking:
- Output hashes (same output repeated)
- Error patterns (same error repeated)
- Lack of progress (no meaningful changes)
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..config import Config
from .base import Gate, GateResult, GateStatus


@dataclass
class LoopRecord:
    """Record of a single run output."""
    timestamp: str
    output_hash: Optional[str]  # None for empty outputs
    error_hash: Optional[str]
    files_changed: int


class LoopDetector:
    """Tracks outputs to detect loops."""

    def __init__(
        self,
        state_path: Path,
        same_output_threshold: int = 3,
        same_error_threshold: int = 3,
        history_size: int = 10,
    ):
        self.state_path = state_path
        self.same_output_threshold = same_output_threshold
        self.same_error_threshold = same_error_threshold
        self.history_size = history_size
        self._records: List[LoopRecord] = []
        self._load_state()

    def _load_state(self) -> None:
        """Load state from file."""
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                self._records = [
                    LoopRecord(**r) for r in data.get("records", [])
                ]
            except (json.JSONDecodeError, TypeError):
                self._records = []

    def _save_state(self) -> None:
        """Save state to file."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "records": [
                {
                    "timestamp": r.timestamp,
                    "output_hash": r.output_hash,
                    "error_hash": r.error_hash,
                    "files_changed": r.files_changed,
                }
                for r in self._records
            ]
        }
        self.state_path.write_text(json.dumps(data, indent=2))

    def _hash_text(self, text: str) -> str:
        """Create a short hash of text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def record(
        self,
        output: str,
        error: Optional[str] = None,
        files_changed: int = 0,
    ) -> None:
        """Record a run output."""
        # Don't hash empty output - it would cause false positives
        # (all empty outputs hash to same value)
        output_hash = self._hash_text(output) if output.strip() else None

        record = LoopRecord(
            timestamp=datetime.utcnow().isoformat() + "Z",
            output_hash=output_hash,
            error_hash=self._hash_text(error) if error else None,
            files_changed=files_changed,
        )

        self._records.append(record)

        # Trim to history size
        if len(self._records) > self.history_size:
            self._records = self._records[-self.history_size:]

        self._save_state()

    def check_loop(self) -> Dict[str, Any]:
        """Check if we're in a loop.

        Returns:
            Dict with 'detected', 'type', and 'details'
        """
        if len(self._records) < 2:
            return {"detected": False}

        # Check same output (skip None hashes - empty outputs don't count)
        output_hashes = [r.output_hash for r in self._records if r.output_hash]
        if not output_hashes:
            return {"detected": False}

        latest_hash = output_hashes[-1]
        same_output_count = output_hashes.count(latest_hash)

        if same_output_count >= self.same_output_threshold:
            return {
                "detected": True,
                "type": "same_output",
                "details": f"Same output seen {same_output_count} times",
                "hash": latest_hash,
            }

        # Check same error
        error_hashes = [r.error_hash for r in self._records if r.error_hash]
        if error_hashes:
            latest_error = error_hashes[-1]
            same_error_count = error_hashes.count(latest_error)

            if same_error_count >= self.same_error_threshold:
                return {
                    "detected": True,
                    "type": "same_error",
                    "details": f"Same error seen {same_error_count} times",
                    "hash": latest_error,
                }

        # NOTE: Removed "no_progress" check - it was too aggressive.
        # Research/query prompts legitimately don't change files.
        # Loop detection should focus on same_output and same_error patterns.

        return {"detected": False}

    def clear(self) -> None:
        """Clear loop detection state."""
        self._records = []
        if self.state_path.exists():
            self.state_path.unlink()


class LoopDetectGate(Gate):
    """Gate that detects and prevents loops."""

    name = "loop_detect"

    def __init__(self, config: Config, pirq_dir: Optional[Path] = None):
        super().__init__(config)
        self.pirq_dir = pirq_dir or Path.cwd() / ".pirq"

        # Get loop detection settings from config
        loop_config = getattr(config, 'loop_detection', None)
        if loop_config:
            same_output = getattr(loop_config, 'same_output_threshold', 3)
            same_error = getattr(loop_config, 'same_error_threshold', 3)
        else:
            same_output = 3
            same_error = 3

        self.detector = LoopDetector(
            state_path=self.pirq_dir / "runtime" / "loop_state.json",
            same_output_threshold=same_output,
            same_error_threshold=same_error,
        )

    def check(self) -> GateResult:
        """Check if we're in a detected loop."""
        result = self.detector.check_loop()

        if result["detected"]:
            return GateResult(
                status=GateStatus.BLOCK,
                message=f"Loop detected: {result['type']}",
                data={
                    "type": result["type"],
                    "details": result["details"],
                    "hash": result.get("hash"),
                },
            )

        return GateResult(
            status=GateStatus.CLEAR,
            message="No loop detected",
            data={"history_size": len(self.detector._records)},
        )

    def record_run(
        self,
        output: str,
        error: Optional[str] = None,
        files_changed: int = 0,
    ) -> None:
        """Record a run for loop detection."""
        self.detector.record(output, error, files_changed)

    def clear(self) -> None:
        """Clear loop detection state (use after successful recovery)."""
        self.detector.clear()
