"""PIRQ Gate Base Classes.

Gates are pre-execution checks that run before every Claude invocation.
Each gate returns CLEAR, WARN, or BLOCK status.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any

from ..config import Config


class GateStatus(Enum):
    """Gate check result status."""

    CLEAR = "clear"   # All good, proceed
    WARN = "warn"     # Warning but allow
    BLOCK = "block"   # Block execution


@dataclass
class GateResult:
    """Result of a gate check."""

    status: GateStatus
    message: str
    data: Optional[Dict[str, Any]] = field(default_factory=dict)

    @property
    def is_blocked(self) -> bool:
        return self.status == GateStatus.BLOCK

    @property
    def is_warning(self) -> bool:
        return self.status == GateStatus.WARN

    @property
    def is_clear(self) -> bool:
        return self.status == GateStatus.CLEAR

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "message": self.message,
            "data": self.data or {},
        }

    @classmethod
    def clear(cls, message: str = "OK", data: Optional[Dict[str, Any]] = None) -> "GateResult":
        return cls(GateStatus.CLEAR, message, data)

    @classmethod
    def warn(cls, message: str, data: Optional[Dict[str, Any]] = None) -> "GateResult":
        return cls(GateStatus.WARN, message, data)

    @classmethod
    def block(cls, message: str, data: Optional[Dict[str, Any]] = None) -> "GateResult":
        return cls(GateStatus.BLOCK, message, data)


class Gate:
    """Base class for PIRQ gates.

    Subclasses must implement:
    - name: str - unique identifier for the gate
    - check() -> GateResult - perform the gate check
    """

    name: str = "base"

    def __init__(self, config: Config):
        self.config = config

    def check(self) -> GateResult:
        """Perform the gate check.

        Returns:
            GateResult with status, message, and optional data
        """
        raise NotImplementedError("Subclasses must implement check()")

    def is_enabled(self) -> bool:
        """Check if this gate is enabled in config."""
        gate_config = getattr(self.config.pirqs, self.name.replace("-", "_"), None)
        if gate_config is None:
            return True  # Default to enabled
        return gate_config.enabled

    def get_strictness(self) -> str:
        """Get strictness level for this gate."""
        gate_config = getattr(self.config.pirqs, self.name.replace("-", "_"), None)
        if gate_config is None:
            return "warn"
        return getattr(gate_config, "strictness", "warn")
