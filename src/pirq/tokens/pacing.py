"""PIRQ Budget Pacing System.

Time-aware budget enforcement that adapts to period progress:
- 50% usage at mid-month = normal (on pace)
- 80% usage at mid-month = warning (ahead of pace)
- 90% usage on last day = fine (using allocation)
- Emergency reserve to prevent complete exhaustion
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# Token pricing (Claude Sonnet 4)
PRICING = {
    "input": 3.00 / 1_000_000,
    "output": 15.00 / 1_000_000,
    "cache_read": 0.30 / 1_000_000,
    "cache_create": 3.75 / 1_000_000,
    "blended": 4.00 / 1_000_000,  # Average for quick estimates
}


@dataclass
class PacingConfig:
    """Pacing threshold configuration."""

    # Pace percentage thresholds
    under_threshold: float = 80.0     # Below = under pace (good)
    on_pace_threshold: float = 110.0  # Below = on pace (normal)
    ahead_threshold: float = 150.0    # Below = ahead (warning)
    # Above ahead_threshold = critical

    # Time-adjusted multipliers
    final_day_multiplier: float = 2.0   # 2x more permissive on last day
    final_week_multiplier: float = 1.3  # 1.3x in final week (3 days)


@dataclass
class ReserveConfig:
    """Emergency reserve configuration."""

    # Reserve can be expressed multiple ways (first non-zero wins)
    reserve_tokens: int = 0           # Absolute tokens
    reserve_percent: float = 5.0      # % of budget (default 5%)
    reserve_dollars: float = 0.0      # $ amount

    # Behavior
    reserve_mode: str = "soft"        # "soft" (warn) or "hard" (block)


@dataclass
class PacingStatus:
    """Complete pacing status snapshot."""

    # Time metrics
    period_start: datetime
    period_end: datetime
    days_total: float
    days_elapsed: float
    days_remaining: float
    percent_period_elapsed: float

    # Budget metrics
    budget: int
    used: int
    remaining: int
    percent_used: float

    # Reserve metrics
    reserve: int
    effective_budget: int  # budget - reserve
    reserve_remaining: int  # How much reserve is left
    in_reserve: bool       # Currently using reserve tokens

    # Pacing metrics
    expected_usage: int      # What you "should" have used by now
    pace_delta: int          # used - expected (negative = under budget)
    pace_percent: float      # (used / expected) * 100

    # Status
    pace_status: str         # "under", "on_pace", "ahead", "critical"

    # Projections
    projected_end_usage: int   # At current pace, where will you end?
    projected_end_percent: float
    daily_burn_rate: float     # tokens per day average
    safe_daily_rate: float     # tokens/day to stay on pace

    # Dollar equivalents
    used_dollars: float
    remaining_dollars: float
    budget_dollars: float
    daily_burn_dollars: float


def tokens_to_dollars(tokens: int, pricing_type: str = "blended") -> float:
    """Convert tokens to dollars."""
    return tokens * PRICING.get(pricing_type, PRICING["blended"])


def dollars_to_tokens(dollars: float, pricing_type: str = "blended") -> int:
    """Convert dollars to tokens."""
    rate = PRICING.get(pricing_type, PRICING["blended"])
    return int(dollars / rate) if rate > 0 else 0


def calculate_reserve(config: ReserveConfig, budget: int) -> int:
    """Calculate reserve in tokens, regardless of how user specified it."""
    if config.reserve_tokens > 0:
        return config.reserve_tokens

    if config.reserve_percent > 0:
        return int(budget * config.reserve_percent / 100)

    if config.reserve_dollars > 0:
        return dollars_to_tokens(config.reserve_dollars)

    return 0


class PacingCalculator:
    """Calculate budget pacing status with time awareness."""

    def __init__(
        self,
        pacing_config: Optional[PacingConfig] = None,
        reserve_config: Optional[ReserveConfig] = None,
    ):
        self.pacing = pacing_config or PacingConfig()
        self.reserve = reserve_config or ReserveConfig()

    def calculate(
        self,
        budget: int,
        used: int,
        period_start: datetime,
        period_end: datetime,
        now: Optional[datetime] = None,
    ) -> PacingStatus:
        """Calculate complete pacing status.

        Args:
            budget: Total token budget for period
            used: Tokens used so far
            period_start: Start of billing period
            period_end: End of billing period (next reset)
            now: Current time (defaults to utcnow)

        Returns:
            PacingStatus with all metrics
        """
        now = now or datetime.utcnow()

        # Time calculations
        total_seconds = (period_end - period_start).total_seconds()
        elapsed_seconds = max(0, (now - period_start).total_seconds())

        # Clamp elapsed to not exceed total (in case now > period_end)
        elapsed_seconds = min(elapsed_seconds, total_seconds)

        percent_elapsed = (elapsed_seconds / total_seconds * 100) if total_seconds > 0 else 100

        days_total = total_seconds / 86400
        days_elapsed = elapsed_seconds / 86400
        days_remaining = max(0, days_total - days_elapsed)

        # Reserve calculation
        reserve_tokens = calculate_reserve(self.reserve, budget)
        effective_budget = budget - reserve_tokens

        # Basic budget metrics
        remaining = max(0, budget - used)
        percent_used = (used / budget * 100) if budget > 0 else 0

        # Reserve status
        in_reserve = used > effective_budget
        reserve_remaining = max(0, budget - used) if in_reserve else reserve_tokens

        # Expected usage at this point in the period
        expected = int(effective_budget * percent_elapsed / 100)

        # Pacing metrics
        pace_delta = used - expected
        pace_percent = (used / expected * 100) if expected > 0 else 0

        # Determine pace status with time-adjusted thresholds
        thresholds = self._get_adjusted_thresholds(days_remaining)
        pace_status = self._determine_status(pace_percent, thresholds)

        # Projections
        if days_elapsed > 0:
            daily_rate = used / days_elapsed
            projected_end = int(daily_rate * days_total)
        else:
            daily_rate = 0
            projected_end = used  # Just started

        projected_percent = (projected_end / budget * 100) if budget > 0 else 0
        safe_daily = (effective_budget - used) / days_remaining if days_remaining > 0 else 0
        safe_daily = max(0, safe_daily)

        # Dollar conversions
        used_dollars = tokens_to_dollars(used)
        remaining_dollars = tokens_to_dollars(remaining)
        budget_dollars = tokens_to_dollars(budget)
        daily_burn_dollars = tokens_to_dollars(int(daily_rate))

        return PacingStatus(
            # Time
            period_start=period_start,
            period_end=period_end,
            days_total=days_total,
            days_elapsed=days_elapsed,
            days_remaining=days_remaining,
            percent_period_elapsed=percent_elapsed,

            # Budget
            budget=budget,
            used=used,
            remaining=remaining,
            percent_used=percent_used,

            # Reserve
            reserve=reserve_tokens,
            effective_budget=effective_budget,
            reserve_remaining=reserve_remaining,
            in_reserve=in_reserve,

            # Pacing
            expected_usage=expected,
            pace_delta=pace_delta,
            pace_percent=pace_percent,
            pace_status=pace_status,

            # Projections
            projected_end_usage=projected_end,
            projected_end_percent=projected_percent,
            daily_burn_rate=daily_rate,
            safe_daily_rate=safe_daily,

            # Dollars
            used_dollars=used_dollars,
            remaining_dollars=remaining_dollars,
            budget_dollars=budget_dollars,
            daily_burn_dollars=daily_burn_dollars,
        )

    def _get_adjusted_thresholds(self, days_remaining: float) -> dict:
        """Get time-adjusted thresholds based on days remaining."""
        multiplier = 1.0

        if days_remaining <= 1:
            multiplier = self.pacing.final_day_multiplier
        elif days_remaining <= 3:
            multiplier = self.pacing.final_week_multiplier

        return {
            "under": self.pacing.under_threshold * multiplier,
            "on_pace": self.pacing.on_pace_threshold * multiplier,
            "ahead": self.pacing.ahead_threshold * multiplier,
        }

    def _determine_status(self, pace_percent: float, thresholds: dict) -> str:
        """Determine pace status from percentage and thresholds."""
        if pace_percent < thresholds["under"]:
            return "under"
        elif pace_percent < thresholds["on_pace"]:
            return "on_pace"
        elif pace_percent < thresholds["ahead"]:
            return "ahead"
        else:
            return "critical"

    def get_recommendation(self, status: PacingStatus) -> str:
        """Get human-readable recommendation based on status."""
        if status.in_reserve:
            used_from_reserve = status.used - status.effective_budget
            return (
                f"Using emergency reserve: {used_from_reserve:,} of "
                f"{status.reserve:,} tokens ({status.reserve_remaining:,} left)"
            )

        if status.pace_status == "under":
            return (
                f"Under pace ({status.pace_percent:.0f}% of expected) - "
                f"room to spare"
            )

        if status.pace_status == "on_pace":
            return (
                f"On pace ({status.pace_percent:.0f}% of expected) - "
                f"tracking normally"
            )

        if status.pace_status == "ahead":
            # Calculate when budget would run out at current rate
            if status.daily_burn_rate > 0:
                days_until_exhausted = status.remaining / status.daily_burn_rate
                return (
                    f"Ahead of pace ({status.pace_percent:.0f}% of expected) - "
                    f"~{days_until_exhausted:.1f} days until budget exhausted"
                )
            return f"Ahead of pace ({status.pace_percent:.0f}% of expected)"

        # Critical
        return (
            f"Critical pace ({status.pace_percent:.0f}% of expected) - "
            f"burning tokens faster than sustainable"
        )
