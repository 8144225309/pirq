"""PIRQ Rate Limit Gate - Prevents runaway invocations.

Limits the rate of PIRQ invocations to prevent:
- Accidental infinite loops
- Token budget exhaustion
- System overload
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import Config
from ..logs import AuditLogger
from .base import Gate, GateResult, GateStatus


class RateLimitGate(Gate):
    """Gate that prevents runaway invocations."""

    name = "rate_limit"

    def __init__(
        self,
        config: Config,
        audit_logger: AuditLogger,
        max_per_hour: int = 30,
        max_per_minute: int = 5,
        cooldown_seconds: int = 0,
    ):
        super().__init__(config)
        self.audit_logger = audit_logger

        # Get config values or use defaults
        rate_config = getattr(config, 'rate_limit', None)
        if rate_config:
            self.max_per_hour = getattr(rate_config, 'max_per_hour', max_per_hour)
            self.max_per_minute = getattr(rate_config, 'max_per_minute', max_per_minute)
            self.cooldown_seconds = getattr(rate_config, 'cooldown_seconds', cooldown_seconds)
        else:
            self.max_per_hour = max_per_hour
            self.max_per_minute = max_per_minute
            self.cooldown_seconds = cooldown_seconds

    def check(self) -> GateResult:
        """Check if rate limits allow execution."""
        # Check hourly rate
        hour_invocations = self.audit_logger.get_recent_invocations(hours=1)
        hour_count = len(hour_invocations)

        if hour_count >= self.max_per_hour:
            return GateResult(
                status=GateStatus.BLOCK,
                message=f"Rate limit: {hour_count} invocations in last hour (max {self.max_per_hour})",
                data={
                    "hour_count": hour_count,
                    "max_per_hour": self.max_per_hour,
                    "type": "hourly",
                },
            )

        # Check minute rate
        minute_invocations = self.audit_logger.get_recent_invocations(minutes=1)
        minute_count = len(minute_invocations)

        if minute_count >= self.max_per_minute:
            return GateResult(
                status=GateStatus.BLOCK,
                message=f"Rate limit: {minute_count} invocations in last minute (max {self.max_per_minute})",
                data={
                    "minute_count": minute_count,
                    "max_per_minute": self.max_per_minute,
                    "type": "per_minute",
                },
            )

        # Check cooldown
        if self.cooldown_seconds > 0:
            last = self.audit_logger.get_last_invocation()
            if last and last.get('timestamp_start'):
                try:
                    last_time = datetime.fromisoformat(
                        last['timestamp_start'].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    elapsed = (datetime.utcnow() - last_time).total_seconds()

                    if elapsed < self.cooldown_seconds:
                        remaining = self.cooldown_seconds - elapsed
                        return GateResult(
                            status=GateStatus.WARN,
                            message=f"Cooldown: {remaining:.0f}s remaining",
                            data={
                                "elapsed": elapsed,
                                "cooldown": self.cooldown_seconds,
                                "remaining": remaining,
                            },
                        )
                except (ValueError, KeyError):
                    pass

        return GateResult(
            status=GateStatus.CLEAR,
            message=f"Rate limit OK ({hour_count}/h, {minute_count}/min)",
            data={
                "hour_count": hour_count,
                "minute_count": minute_count,
                "max_per_hour": self.max_per_hour,
                "max_per_minute": self.max_per_minute,
            },
        )

    def is_enabled(self) -> bool:
        """Check if rate limiting is enabled."""
        rate_config = getattr(self.config, 'rate_limit', None)
        if rate_config:
            return getattr(rate_config, 'enabled', True)
        # Check in pirqs config
        gate_config = getattr(self.config.pirqs, self.name, None)
        if gate_config:
            return gate_config.enabled
        return True  # Enabled by default
