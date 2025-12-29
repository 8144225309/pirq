"""PIRQ Token Gate - Budget enforcement.

Blocks execution when token budget is exceeded.

Enhanced features:
- Pre-execution token estimation from prompt length
- Block if estimated usage would exceed budget
- Token velocity tracking (tokens/hour) for anomaly detection
- Real usage syncing from Claude Code logs
- Time-aware pacing (adjusts thresholds based on time remaining)
- Emergency reserve protection
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

from ..config import Config, TurboConfig
from ..tokens import TokenTracker
from ..tokens.pacing import (
    PacingCalculator,
    PacingConfig,
    PacingStatus,
    ReserveConfig,
    dollars_to_tokens,
)
from .base import Gate, GateResult, GateStatus

if TYPE_CHECKING:
    from ..logs import AuditLogger
    from ..claude.logs import ClaudeLogReader, UsageSummary


class TokenGate(Gate):
    """Gate that enforces token budget limits with estimation and velocity tracking."""

    name = "tokens"

    # Default estimation multipliers
    CHARS_PER_TOKEN = 4  # Rough estimate: 4 chars = 1 token
    OUTPUT_MULTIPLIER = 3  # Expect output to be ~3x input tokens

    def __init__(
        self,
        config: Config,
        pirq_dir: Optional[Path] = None,
        audit_logger: Optional["AuditLogger"] = None,
    ):
        super().__init__(config)
        self.pirq_dir = pirq_dir or Path.cwd() / ".pirq"
        self.audit_logger = audit_logger

        # Get token settings from config
        token_config = getattr(config, 'tokens', None)
        if token_config:
            self.budget = token_config.budget
            self.reset_day = getattr(token_config, 'reset_day', 1)

            # New flexible thresholds
            self.warn_at_percent_used = getattr(token_config, 'warn_at_percent_used', 80.0)
            self.warn_at_percent_remaining = getattr(token_config, 'warn_at_percent_remaining', 0.0)
            self.warn_at_tokens = getattr(token_config, 'warn_at_tokens', 0)
            self.warn_at_dollars = getattr(token_config, 'warn_at_dollars', 0.0)

            self.block_at_percent_used = getattr(token_config, 'block_at_percent_used', 95.0)
            self.block_at_percent_remaining = getattr(token_config, 'block_at_percent_remaining', 0.0)
            self.block_at_tokens = getattr(token_config, 'block_at_tokens', 0)
            self.block_at_dollars = getattr(token_config, 'block_at_dollars', 0.0)

            # Legacy fallback
            if getattr(token_config, 'warn_percent', 0) > 0:
                self.warn_at_percent_used = float(token_config.warn_percent)
            if getattr(token_config, 'block_percent', 0) > 0:
                self.block_at_percent_used = float(token_config.block_percent)

            # Enhanced settings
            self.max_velocity = getattr(token_config, 'max_velocity', 10000)
            self.estimate_before_run = getattr(token_config, 'estimate_before_run', True)
        else:
            # Defaults
            self.budget = 50000
            self.reset_day = 1
            self.warn_at_percent_used = 80.0
            self.warn_at_percent_remaining = 0.0
            self.warn_at_tokens = 0
            self.warn_at_dollars = 0.0
            self.block_at_percent_used = 95.0
            self.block_at_percent_remaining = 0.0
            self.block_at_tokens = 0
            self.block_at_dollars = 0.0
            self.max_velocity = 10000
            self.estimate_before_run = True

        # Turbo config
        turbo_config = getattr(config, 'turbo', None)
        self.turbo_config = turbo_config or TurboConfig()

        self.tracker = TokenTracker(
            log_path=self.pirq_dir / "logs" / "tokens.jsonl",
            budget=self.budget,
            warn_percent=self.warn_at_percent_used,
            block_percent=self.block_at_percent_used,
            reset_day=self.reset_day,
        )

        # Pacing configuration
        if token_config:
            self.pacing_enabled = getattr(token_config, 'pacing_enabled', True)
            self.reserve_mode = getattr(token_config, 'reserve_mode', 'soft')

            pacing_config = PacingConfig(
                ahead_threshold=getattr(token_config, 'pace_warn_threshold', 150.0),
            )
            reserve_config = ReserveConfig(
                reserve_tokens=getattr(token_config, 'reserve_tokens', 0),
                reserve_percent=getattr(token_config, 'reserve_percent', 5.0),
                reserve_dollars=getattr(token_config, 'reserve_dollars', 0.0),
                reserve_mode=getattr(token_config, 'reserve_mode', 'soft'),
            )
        else:
            self.pacing_enabled = True
            self.reserve_mode = 'soft'
            pacing_config = PacingConfig()
            reserve_config = ReserveConfig()

        self.pacing_calculator = PacingCalculator(
            pacing_config=pacing_config,
            reserve_config=reserve_config,
        )

        # Claude Code log reader (lazy-loaded)
        self._claude_reader: Optional["ClaudeLogReader"] = None

        # Current prompt for estimation (set by orchestrator)
        self._current_prompt: Optional[str] = None

    @property
    def claude_reader(self) -> "ClaudeLogReader":
        """Get Claude log reader (lazy-loaded)."""
        if self._claude_reader is None:
            from ..claude.logs import ClaudeLogReader
            self._claude_reader = ClaudeLogReader()
        return self._claude_reader

    def set_prompt(self, prompt: str) -> None:
        """Set the current prompt for token estimation."""
        self._current_prompt = prompt

    def _get_warn_threshold_tokens(self) -> int:
        """Get warn threshold in tokens, from any configured format.

        Priority (first non-zero wins):
        1. warn_at_tokens (absolute)
        2. warn_at_dollars (converted to tokens)
        3. warn_at_percent_remaining (% of budget)
        4. warn_at_percent_used (default - converted to remaining)
        """
        if self.warn_at_tokens > 0:
            return self.warn_at_tokens
        if self.warn_at_dollars > 0:
            return dollars_to_tokens(self.warn_at_dollars)
        if self.warn_at_percent_remaining > 0:
            return int(self.budget * self.warn_at_percent_remaining / 100)
        # Default: percent used -> convert to tokens remaining
        return int(self.budget * (100 - self.warn_at_percent_used) / 100)

    def _get_block_threshold_tokens(self) -> int:
        """Get block threshold in tokens, from any configured format.

        Priority (first non-zero wins):
        1. block_at_tokens (absolute)
        2. block_at_dollars (converted to tokens)
        3. block_at_percent_remaining (% of budget)
        4. block_at_percent_used (default - converted to remaining)
        """
        if self.block_at_tokens > 0:
            return self.block_at_tokens
        if self.block_at_dollars > 0:
            return dollars_to_tokens(self.block_at_dollars)
        if self.block_at_percent_remaining > 0:
            return int(self.budget * self.block_at_percent_remaining / 100)
        # Default: percent used -> convert to tokens remaining
        return int(self.budget * (100 - self.block_at_percent_used) / 100)

    def is_turbo_active(self) -> bool:
        """Check if turbo mode should be active.

        Turbo mode activates when:
        1. Enabled in config
        2. Within X days/hours of reset
        3. Still have enough tokens remaining (min_remaining_percent)
        """
        if not self.turbo_config.enabled:
            return False

        pacing = self.get_pacing_status()

        # Check time condition
        days_trigger = self.turbo_config.activate_days_before_reset
        hours_trigger = self.turbo_config.activate_hours_before_reset

        if hours_trigger > 0:
            hours_remaining = pacing.days_remaining * 24
            if hours_remaining > hours_trigger:
                return False
        elif pacing.days_remaining > days_trigger:
            return False

        # Check minimum remaining
        remaining_percent = 100 - pacing.percent_used
        if remaining_percent < self.turbo_config.min_remaining_percent:
            return False

        return True

    def get_turbo_status(self) -> Dict[str, Any]:
        """Get detailed turbo mode status."""
        pacing = self.get_pacing_status()
        is_active = self.is_turbo_active()

        # Calculate when turbo would activate
        days_trigger = self.turbo_config.activate_days_before_reset
        hours_trigger = self.turbo_config.activate_hours_before_reset

        if hours_trigger > 0:
            trigger_desc = f"{hours_trigger} hours before reset"
        else:
            trigger_desc = f"{days_trigger} days before reset"

        # Calculate available for turbo (remaining minus reserve if not allowed to dip)
        if self.turbo_config.allow_reserve_dip:
            available = pacing.remaining
        else:
            available = max(0, pacing.remaining - pacing.reserve)

        return {
            "enabled": self.turbo_config.enabled,
            "active": is_active,
            "days_remaining": pacing.days_remaining,
            "hours_remaining": pacing.days_remaining * 24,
            "percent_remaining": 100 - pacing.percent_used,
            "min_remaining_percent": self.turbo_config.min_remaining_percent,
            "trigger": trigger_desc,
            "available_tokens": available,
            "reserve_protected": not self.turbo_config.allow_reserve_dip,
        }

    def check(self) -> GateResult:
        """Check if token budget allows execution.

        Checks (in order):
        1. Turbo mode - if active, lift restrictions
        2. Block threshold - hard stop
        3. Reserve - emergency fund check
        4. Warn threshold - soft warning
        5. All clear
        """
        # Get usage - prefer real Claude usage if available
        real_usage = self.get_real_usage()
        if real_usage:
            used = real_usage.total_tokens
        else:
            used = self.tracker.get_period_usage().total_tokens

        # Get pacing status (includes reserve calculation)
        pacing = self.get_pacing_status(used_override=used)
        remaining = pacing.remaining

        # Get thresholds (in tokens)
        warn_threshold = self._get_warn_threshold_tokens()
        block_threshold = self._get_block_threshold_tokens()

        # Add turbo status to data
        is_turbo = self.is_turbo_active()
        data = self._pacing_to_data(pacing)
        data["turbo_active"] = is_turbo
        data["warn_threshold_tokens"] = warn_threshold
        data["block_threshold_tokens"] = block_threshold

        # Check 1: Turbo mode active - lift restrictions (but not reserve)
        if is_turbo:
            # In turbo mode, only block if completely exhausted or in hard reserve
            if used >= self.budget:
                return GateResult(
                    status=GateStatus.BLOCK,
                    message=f"Budget exhausted (turbo mode active)",
                    data=data,
                )
            if pacing.in_reserve and self.reserve_mode == "hard":
                return GateResult(
                    status=GateStatus.BLOCK,
                    message=f"Reserve reached (turbo mode can't override hard reserve)",
                    data=data,
                )
            # Turbo mode - clear to proceed
            return GateResult(
                status=GateStatus.CLEAR,
                message=f"[TURBO] {remaining:,} tokens available for research/maintenance/cosmetic",
                data=data,
            )

        # Check 2: Block threshold
        if remaining <= block_threshold:
            return GateResult(
                status=GateStatus.BLOCK,
                message=f"Below block threshold ({remaining:,} < {block_threshold:,} tokens)",
                data=data,
            )

        # Check 3: Reserve check
        if pacing.in_reserve:
            if self.reserve_mode == "hard":
                return GateResult(
                    status=GateStatus.BLOCK,
                    message=f"Emergency reserve reached ({pacing.reserve_remaining:,} tokens left)",
                    data=data,
                )
            else:
                return GateResult(
                    status=GateStatus.WARN,
                    message=f"Using emergency reserve: {used - pacing.effective_budget:,} of {pacing.reserve:,} tokens",
                    data=data,
                )

        # Check 4: Warn threshold
        if remaining <= warn_threshold:
            return GateResult(
                status=GateStatus.WARN,
                message=f"Below warn threshold ({remaining:,} < {warn_threshold:,} tokens)",
                data=data,
            )

        # Check 5: All clear
        percent_remaining = 100 - pacing.percent_used
        return GateResult(
            status=GateStatus.CLEAR,
            message=f"{percent_remaining:.0f}% remaining ({remaining:,} tokens)",
            data=data,
        )

    def _pacing_to_data(self, pacing: PacingStatus) -> Dict[str, Any]:
        """Convert PacingStatus to gate result data dict."""
        return {
            "budget": pacing.budget,
            "used": pacing.used,
            "remaining": pacing.remaining,
            "percent_used": round(pacing.percent_used, 1),
            "reserve": pacing.reserve,
            "effective_budget": pacing.effective_budget,
            "in_reserve": pacing.in_reserve,
            "days_elapsed": round(pacing.days_elapsed, 1),
            "days_remaining": round(pacing.days_remaining, 1),
            "percent_period_elapsed": round(pacing.percent_period_elapsed, 1),
            "expected_usage": pacing.expected_usage,
            "pace_percent": round(pacing.pace_percent, 1),
            "pace_status": pacing.pace_status,
            "projected_end_usage": pacing.projected_end_usage,
            "daily_burn_rate": round(pacing.daily_burn_rate, 0),
            "safe_daily_rate": round(pacing.safe_daily_rate, 0),
            "used_dollars": round(pacing.used_dollars, 2),
            "remaining_dollars": round(pacing.remaining_dollars, 2),
        }

    def get_pacing_status(self, used_override: Optional[int] = None) -> PacingStatus:
        """Get current pacing status.

        Args:
            used_override: Override the used token count (e.g., from Claude logs)

        Returns:
            PacingStatus with all pacing metrics
        """
        if used_override is not None:
            used = used_override
        else:
            real_usage = self.get_real_usage()
            if real_usage:
                used = real_usage.total_tokens
            else:
                used = self.tracker.get_period_usage().total_tokens

        period_start = self.tracker._get_period_start()
        period_end = self.tracker._get_next_reset()

        return self.pacing_calculator.calculate(
            budget=self.budget,
            used=used,
            period_start=period_start,
            period_end=period_end,
        )

    def estimate_tokens(self, prompt: str) -> int:
        """Estimate total tokens for a prompt (input + expected output).

        Uses a simple heuristic:
        - Input tokens ≈ len(prompt) / 4
        - Output tokens ≈ input * 3 (typical Claude response ratio)
        """
        input_estimate = len(prompt) // self.CHARS_PER_TOKEN
        output_estimate = input_estimate * self.OUTPUT_MULTIPLIER
        return input_estimate + output_estimate

    def _calculate_velocity(self) -> int:
        """Calculate tokens used per hour from audit log."""
        if not self.audit_logger:
            return 0

        try:
            recent = self.audit_logger.get_recent_invocations(hours=1)
            return sum(e.get('tokens_used', 0) for e in recent)
        except Exception:
            return 0

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "unknown",
        task: Optional[str] = None,
    ) -> None:
        """Record token usage from a run."""
        self.tracker.log_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            task=task,
        )

    def sync_from_claude(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Sync token usage from Claude Code logs.

        Args:
            since: Only include usage after this timestamp.
                   Defaults to start of current budget period.

        Returns:
            Dict with sync details including token counts
        """
        # Default to current budget period
        if since is None:
            since = self.tracker._get_period_start()

        try:
            usage = self.claude_reader.get_total_usage(since=since)
            sync_result = self.tracker.sync_with_claude(usage)
            return {
                "success": True,
                "synced": True,
                **sync_result,
            }
        except Exception as e:
            return {
                "success": False,
                "synced": False,
                "error": str(e),
            }

    def get_real_usage(self, since: Optional[datetime] = None) -> Optional["UsageSummary"]:
        """Get real usage from Claude Code logs without syncing.

        Args:
            since: Only include usage after this timestamp.
                   Defaults to start of current budget period.

        Returns:
            UsageSummary or None if Claude logs unavailable
        """
        if since is None:
            since = self.tracker._get_period_start()

        try:
            if not self.claude_reader.is_available():
                return None
            return self.claude_reader.get_total_usage(since=since)
        except Exception:
            return None

    def is_claude_available(self) -> bool:
        """Check if Claude Code logs are available."""
        try:
            return self.claude_reader.is_available()
        except Exception:
            return False
