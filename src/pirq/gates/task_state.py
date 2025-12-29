"""PIRQ Task State Gate - Tracks task completion and stall states.

Prevents re-running completed or stalled tasks:
- Warns if task was already completed recently
- Blocks if task is marked as needing human review
- Blocks after consecutive failures (stall detection)
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..config import Config
from ..logs import AuditLogger
from .base import Gate, GateResult, GateStatus


class TaskStateGate(Gate):
    """Gate that tracks task completion/stall state."""

    name = "task_state"

    def __init__(
        self,
        config: Config,
        audit_logger: AuditLogger,
        stall_threshold: int = 3,
    ):
        super().__init__(config)
        self.audit_logger = audit_logger

        # Get config values
        task_config = getattr(config, 'task_state', None)
        if task_config:
            self.stall_threshold = getattr(task_config, 'stall_threshold', stall_threshold)
            self.block_on_done = getattr(task_config, 'block_on_done', False)
            self.block_on_stall = getattr(task_config, 'block_on_stall', True)
        else:
            self.stall_threshold = stall_threshold
            self.block_on_done = False
            self.block_on_stall = True

        # Current prompt being checked (set by orchestrator)
        self._current_prompt: Optional[str] = None

    def set_prompt(self, prompt: str) -> None:
        """Set the current prompt for checking."""
        self._current_prompt = prompt

    def _get_prompt_hash(self, prompt: str) -> str:
        """Compute prompt hash for matching."""
        return f"sha256:{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"

    def check(self) -> GateResult:
        """Check if task state allows execution."""
        if not self._current_prompt:
            return GateResult(
                status=GateStatus.CLEAR,
                message="No prompt to check",
                data={},
            )

        prompt_hash = self._get_prompt_hash(self._current_prompt)

        # Get history for this prompt
        history = self.audit_logger.get_prompt_history(prompt_hash)

        if not history:
            return GateResult(
                status=GateStatus.CLEAR,
                message="New task",
                data={"prompt_hash": prompt_hash},
            )

        # Filter to start entries only (not completion records)
        starts = [h for h in history if h.get('timestamp_start')]
        if not starts:
            return GateResult(
                status=GateStatus.CLEAR,
                message="New task",
                data={"prompt_hash": prompt_hash},
            )

        # Get the most recent run's final state
        # We need to pair starts with their completions
        last_start = starts[-1]
        last_entry_id = last_start.get('entry_id')

        # Find completion for this entry
        completions = [h for h in history if h.get('command') == '_complete']
        last_completion = None
        for c in completions:
            if c.get('entry_id') == last_entry_id:
                last_completion = c
                break

        if last_completion:
            orch_action = last_completion.get('orch_action')
            exit_code = last_completion.get('exit_code')
        else:
            orch_action = None
            exit_code = None

        # Check if already completed
        if orch_action == 'DONE':
            hours_ago = self._hours_since(last_start.get('timestamp_start'))
            status = GateStatus.BLOCK if self.block_on_done else GateStatus.WARN
            return GateResult(
                status=status,
                message=f"Task completed {hours_ago:.1f}h ago. Use --force to retry.",
                data={
                    "prompt_hash": prompt_hash,
                    "hours_ago": hours_ago,
                    "last_orch_action": orch_action,
                },
            )

        # Check if blocked/aborted
        if orch_action in ['BLOCKED', 'ABORT', 'WAIT']:
            return GateResult(
                status=GateStatus.BLOCK,
                message=f"Task marked {orch_action}. Needs human review.",
                data={
                    "prompt_hash": prompt_hash,
                    "last_orch_action": orch_action,
                },
            )

        # Check for stall (consecutive failures)
        recent_completions = [
            c for c in completions
            if c.get('entry_id') in [s.get('entry_id') for s in starts[-self.stall_threshold:]]
        ]
        failures = [c for c in recent_completions if c.get('exit_code', 0) != 0]

        if len(failures) >= self.stall_threshold:
            status = GateStatus.BLOCK if self.block_on_stall else GateStatus.WARN
            return GateResult(
                status=status,
                message=f"Task stalled: {len(failures)} consecutive failures. Use --force to retry.",
                data={
                    "prompt_hash": prompt_hash,
                    "failure_count": len(failures),
                    "stall_threshold": self.stall_threshold,
                },
            )

        # Check warning for previous attempts
        attempt_count = len(starts)
        if attempt_count > 1:
            return GateResult(
                status=GateStatus.WARN,
                message=f"Task attempted {attempt_count} times before",
                data={
                    "prompt_hash": prompt_hash,
                    "attempt_count": attempt_count,
                },
            )

        return GateResult(
            status=GateStatus.CLEAR,
            message="Task state OK",
            data={"prompt_hash": prompt_hash},
        )

    def _hours_since(self, timestamp: Optional[str]) -> float:
        """Calculate hours since a timestamp."""
        if not timestamp:
            return 0.0

        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).replace(tzinfo=None)
            delta = datetime.utcnow() - ts
            return delta.total_seconds() / 3600
        except (ValueError, TypeError):
            return 0.0

    def is_enabled(self) -> bool:
        """Check if task state tracking is enabled."""
        task_config = getattr(self.config, 'task_state', None)
        if task_config:
            return getattr(task_config, 'enabled', True)
        # Check in pirqs config
        gate_config = getattr(self.config.pirqs, self.name, None)
        if gate_config:
            return gate_config.enabled
        return True  # Enabled by default
