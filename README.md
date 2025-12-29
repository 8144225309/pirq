# PIRQ

**Pre-execution Interrupt Request Queue** - A gating layer for Claude Code that enforces budget limits, tracks token usage, and prevents runaway AI sessions.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is PIRQ?

PIRQ sits between you and Claude, checking gates before every execution:

- **Token Gate** - Budget tracking, pacing, and reserve protection
- **Backup Gate** - Ensures git repository is in good state
- **Rate Limit Gate** - Prevents runaway loops
- **Session Gate** - Manages concurrent access

Think of it as a circuit breaker for your AI budget.

## Installation

```bash
# Clone the repository
git clone https://github.com/8144225309/pirq.git
cd pirq

# Install in development mode
pip install -e .

# Verify installation
pirq status
```

## Quick Start

```bash
# Check current status
pirq status

# Check all gates before running
pirq check

# Run a prompt through PIRQ
pirq run "Explain what this codebase does"

# Run with a specific model
pirq run "Quick question" --model haiku

# Output modes
pirq run "Explain this" --output brief    # 500 char summary
pirq run "Explain this" --output json     # Raw JSON for scripting

# Safety limits
pirq run "Refactor this" --max-turns 5    # Prevent infinite loops
pirq run "Review code" --tools Read,Grep  # Restrict available tools

# Check token budget and pacing
pirq tokens status

# See detailed pacing analysis
pirq tokens pace

# Check turbo mode (end-of-period token burning)
pirq turbo status
```

## Features

### Token Budget Management

```bash
# Set warn threshold (alert when 80% used)
pirq tokens warn --used 80

# Set block threshold (stop at 95% used)
pirq tokens block --used 95

# Configure emergency reserve
pirq tokens reserve --percent 5 --mode hard

# View all thresholds
pirq tokens thresholds
```

### Turbo Mode

Burn remaining tokens productively before your budget resets:

```bash
# Check if turbo mode is active
pirq turbo status

# Configure turbo activation
pirq turbo set --days 3 --min-remaining 20
```

When active, PIRQ suggests using remaining tokens on:
- Research tasks (explore docs, evaluate libraries)
- Maintenance tasks (cleanup, refactoring)
- Cosmetic tasks (formatting, comments)

### Pacing Analysis

PIRQ tracks your token velocity and projects end-of-period usage:

```bash
pirq tokens pace
```

Output:
```
PERIOD: Day 15 of 30
        15.0 days remaining
        50% of period elapsed

BUDGET PACING:
  Budget:      2,500,000 tokens | $10.00
  Expected:    1,250,000 tokens (at this point)
  Actual:        800,000 tokens | $3.20
  Remaining:   1,700,000 tokens | $6.80

PACE STATUS: [OK] UNDER
  Pace:        64% of expected usage
```

### Git Integration

PIRQ ensures your repository is backed up before running:

```bash
# Check git status
pirq git status

# View prompt history
pirq git log

# Rollback to previous prompt
pirq rollback
```

### Audit Logging

All executions are logged to `.pirq/logs/`:

```bash
# View command audit trail
pirq logs audit

# Verify log integrity (tamper-evident)
pirq logs verify

# View session details
pirq logs show
```

## Configuration

PIRQ stores configuration in `.pirq/config.json`:

```json
{
  "tokens": {
    "budget": 2500000,
    "warn_at_percent_used": 80.0,
    "block_at_percent_used": 95.0,
    "reserve_percent": 5.0,
    "reserve_mode": "soft"
  },
  "turbo": {
    "enabled": true,
    "activate_days_before_reset": 3,
    "min_remaining_percent": 20.0
  }
}
```

### Token Plans

```bash
# Configure for different Anthropic plans
pirq tokens configure --plan pro    # Pro plan limits
pirq tokens configure --plan max    # Max plan limits
pirq tokens configure --plan custom --budget 5000000
```

## Architecture

```
User Prompt
     │
     ▼
┌─────────┐
│  PIRQ   │ ◄── Gates check: tokens, backup, rate limit, session
└────┬────┘
     │ (if all gates pass)
     ▼
┌─────────┐
│ Claude  │
└────┬────┘
     │
     ▼
Response + Logging
```

## Documentation

- [Turbo Mode](docs/TURBO_MODE.md) - End-of-period token management
- [Architecture Decisions](docs/ARCHITECTURE_DECISIONS.md) - Design rationale
- [Design Specification](docs/DESIGN_PINNED.md) - Technical details

## Why PIRQ?

1. **Budget Control** - Stop before you overspend
2. **Pacing** - Track velocity, project end-of-period usage
3. **Safety** - Git backup gate prevents data loss
4. **Efficiency** - External gating is cheaper than AI self-monitoring
5. **Visibility** - Know exactly where your tokens go

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Check all gates
pirq check --json
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.
