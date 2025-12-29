"""PIRQ State Management.

Manages the semaphore.json file that tracks overall system state
and individual gate results.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from .gates.base import GateResult, GateStatus


@dataclass
class GateState:
    """State of a single gate."""

    status: str  # "clear", "warn", "block"
    message: str
    last_check: str
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: GateResult) -> "GateState":
        return cls(
            status=result.status.value,
            message=result.message,
            last_check=datetime.utcnow().isoformat() + "Z",
            data=result.data or {},
        )


@dataclass
class Semaphore:
    """Overall system state (semaphore.json)."""

    state: str  # "GO", "WAIT", "STOP"
    reason: Optional[str]
    timestamp: str
    blocked_by: List[str] = field(default_factory=list)
    pirqs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Semaphore":
        """Create default GO state."""
        return cls(
            state="GO",
            reason=None,
            timestamp=datetime.utcnow().isoformat() + "Z",
            blocked_by=[],
            pirqs={},
        )

    @classmethod
    def load(cls, path: Path) -> "Semaphore":
        """Load semaphore from file."""
        if not path.exists():
            return cls.default()

        try:
            data = json.loads(path.read_text())
            return cls(
                state=data.get("state", "GO"),
                reason=data.get("reason"),
                timestamp=data.get("timestamp", datetime.utcnow().isoformat() + "Z"),
                blocked_by=data.get("blocked_by", []),
                pirqs=data.get("pirqs", {}),
            )
        except (json.JSONDecodeError, IOError):
            return cls.default()

    def save(self, path: Path) -> None:
        """Save semaphore to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "state": self.state,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "blocked_by": self.blocked_by,
            "pirqs": self.pirqs,
        }
        path.write_text(json.dumps(data, indent=2))

    def update_from_results(self, results: Dict[str, GateResult]) -> None:
        """Update semaphore from gate check results."""
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.blocked_by = []
        self.pirqs = {}

        any_blocked = False

        for gate_name, result in results.items():
            self.pirqs[gate_name] = {
                "status": result.status.value,
                "message": result.message,
                "last_check": self.timestamp,
                "data": result.data or {},
            }

            if result.status == GateStatus.BLOCK:
                any_blocked = True
                self.blocked_by.append(gate_name)

        if any_blocked:
            self.state = "WAIT"
            self.reason = f"Blocked by: {', '.join(self.blocked_by)}"
        else:
            self.state = "GO"
            self.reason = None

    @property
    def is_go(self) -> bool:
        return self.state == "GO"

    @property
    def is_wait(self) -> bool:
        return self.state == "WAIT"

    @property
    def is_stop(self) -> bool:
        return self.state == "STOP"

    def set_stop(self, reason: str) -> None:
        """Set STOP state."""
        self.state = "STOP"
        self.reason = reason
        self.timestamp = datetime.utcnow().isoformat() + "Z"


class StateManager:
    """Manages PIRQ state files."""

    def __init__(self, pirq_dir: Path):
        self.pirq_dir = pirq_dir
        self.runtime_dir = pirq_dir / "runtime"
        self.semaphore_path = self.runtime_dir / "semaphore.json"

    def ensure_dirs(self) -> None:
        """Ensure runtime directories exist."""
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        (self.pirq_dir / "logs").mkdir(parents=True, exist_ok=True)

    def load_semaphore(self) -> Semaphore:
        """Load current semaphore state."""
        return Semaphore.load(self.semaphore_path)

    def save_semaphore(self, semaphore: Semaphore) -> None:
        """Save semaphore state."""
        self.ensure_dirs()
        semaphore.save(self.semaphore_path)

    def update_from_gates(self, results: Dict[str, GateResult]) -> Semaphore:
        """Update semaphore from gate results."""
        semaphore = self.load_semaphore()
        semaphore.update_from_results(results)
        self.save_semaphore(semaphore)
        return semaphore

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of current state for display."""
        semaphore = self.load_semaphore()

        return {
            "state": semaphore.state,
            "reason": semaphore.reason,
            "timestamp": semaphore.timestamp,
            "blocked_by": semaphore.blocked_by,
            "gates": {
                name: {
                    "status": info.get("status", "unknown"),
                    "message": info.get("message", ""),
                }
                for name, info in semaphore.pirqs.items()
            },
        }
