"""PIRQ Token Tracker - Usage tracking and budget enforcement.

Tracks token usage across sessions with monthly reset support.
Supports syncing with Claude Code's actual usage logs.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..claude.logs import UsageSummary


@dataclass
class TokenUsage:
    """Token usage summary."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    period_start: datetime
    period_end: datetime
    run_count: int


@dataclass
class BudgetStatus:
    """Current budget status."""
    budget: int
    used: int
    remaining: int
    percent_used: float
    is_warn: bool
    is_blocked: bool
    reset_date: datetime
    turbo_eligible: bool


class TokenTracker:
    """Tracks token usage and enforces budgets."""

    def __init__(
        self,
        log_path: Path,
        budget: int = 50000,
        warn_percent: float = 80.0,
        block_percent: float = 95.0,
        reset_day: int = 1,
        turbo_hours_before_reset: int = 48,
    ):
        self.log_path = log_path
        self.budget = budget
        self.warn_percent = warn_percent
        self.block_percent = block_percent
        self.reset_day = reset_day
        self.turbo_hours = turbo_hours_before_reset
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure log directory exists."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_period_start(self, now: Optional[datetime] = None) -> datetime:
        """Get the start of the current billing period."""
        now = now or datetime.utcnow()

        # Start of this month's period
        if now.day >= self.reset_day:
            return datetime(now.year, now.month, self.reset_day)
        else:
            # Start is last month
            if now.month == 1:
                return datetime(now.year - 1, 12, self.reset_day)
            else:
                return datetime(now.year, now.month - 1, self.reset_day)

    def _get_next_reset(self, now: Optional[datetime] = None) -> datetime:
        """Get the next reset date."""
        now = now or datetime.utcnow()
        period_start = self._get_period_start(now)

        # Next month
        if period_start.month == 12:
            return datetime(period_start.year + 1, 1, self.reset_day)
        else:
            return datetime(period_start.year, period_start.month + 1, self.reset_day)

    def log_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "unknown",
        task: Optional[str] = None,
        prompt_num: Optional[int] = None,
    ) -> None:
        """Log token usage for a run."""
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": model,
            "task": task,
            "prompt_num": prompt_num,
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except IOError:
            pass  # Best effort logging

    def get_period_usage(self, now: Optional[datetime] = None) -> TokenUsage:
        """Get token usage for the current billing period."""
        now = now or datetime.utcnow()
        period_start = self._get_period_start(now)
        period_end = self._get_next_reset(now)

        input_total = 0
        output_total = 0
        run_count = 0

        if self.log_path.exists():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            ts = datetime.fromisoformat(
                                entry["ts"].replace("Z", "+00:00")
                            ).replace(tzinfo=None)

                            if ts >= period_start:
                                input_total += entry.get("input_tokens", 0)
                                output_total += entry.get("output_tokens", 0)
                                run_count += 1
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            except IOError:
                pass

        return TokenUsage(
            input_tokens=input_total,
            output_tokens=output_total,
            total_tokens=input_total + output_total,
            period_start=period_start,
            period_end=period_end,
            run_count=run_count,
        )

    def get_budget_status(self, now: Optional[datetime] = None) -> BudgetStatus:
        """Get current budget status."""
        now = now or datetime.utcnow()
        usage = self.get_period_usage(now)
        next_reset = self._get_next_reset(now)

        remaining = max(0, self.budget - usage.total_tokens)
        percent_used = (usage.total_tokens / self.budget) * 100 if self.budget > 0 else 0

        # Check turbo eligibility (within N hours of reset)
        hours_until_reset = (next_reset - now).total_seconds() / 3600
        turbo_eligible = hours_until_reset <= self.turbo_hours

        return BudgetStatus(
            budget=self.budget,
            used=usage.total_tokens,
            remaining=remaining,
            percent_used=percent_used,
            is_warn=percent_used >= self.warn_percent,
            is_blocked=percent_used >= self.block_percent and not turbo_eligible,
            reset_date=next_reset,
            turbo_eligible=turbo_eligible,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text (~4 chars per token)."""
        return len(text) // 4

    def parse_claude_output(self, output: str) -> Optional[Dict[str, int]]:
        """Parse token usage from Claude JSON output.

        Returns:
            Dict with input_tokens and output_tokens, or None if not parseable
        """
        try:
            data = json.loads(output)

            # Claude output format varies, check common fields
            if "usage" in data:
                usage = data["usage"]
                return {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                }

            # Direct fields
            if "input_tokens" in data:
                return {
                    "input_tokens": data.get("input_tokens", 0),
                    "output_tokens": data.get("output_tokens", 0),
                }

        except (json.JSONDecodeError, TypeError):
            pass

        return None

    def get_usage_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent usage entries."""
        if not self.log_path.exists():
            return []

        entries = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass

        return entries

    def sync_with_claude(self, usage: "UsageSummary") -> Dict[str, Any]:
        """Sync token tracking with actual Claude Code usage.

        This replaces estimated usage with real usage from Claude Code logs.

        Args:
            usage: UsageSummary from ClaudeLogReader

        Returns:
            Dict with sync details
        """
        # Record the sync event
        sync_entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "type": "claude_sync",
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_tokens": usage.cache_creation_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "total_tokens": usage.total_tokens,
            "session_count": usage.session_count,
            "message_count": usage.message_count,
            "estimated_cost_usd": round(usage.estimated_cost_usd, 4),
            "first_timestamp": usage.first_timestamp.isoformat() if usage.first_timestamp else None,
            "last_timestamp": usage.last_timestamp.isoformat() if usage.last_timestamp else None,
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sync_entry) + "\n")
        except IOError:
            pass

        return sync_entry

    def get_claude_synced_usage(self, now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Get the most recent Claude sync data for current period.

        Returns:
            Most recent sync entry, or None if no sync in current period
        """
        now = now or datetime.utcnow()
        period_start = self._get_period_start(now)

        if not self.log_path.exists():
            return None

        latest_sync = None
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("type") != "claude_sync":
                            continue

                        ts = datetime.fromisoformat(
                            entry["ts"].replace("Z", "+00:00")
                        ).replace(tzinfo=None)

                        if ts >= period_start:
                            latest_sync = entry
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except IOError:
            pass

        return latest_sync
