# PIRQ

**Pre-execution Interrupt Request Queue** — A gating layer for Claude Code that enforces budget limits, manages token pacing, and provides execution control for AI-assisted development workflows.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

PIRQ intercepts Claude Code invocations and validates them against configurable gates before execution:

| Gate | Function |
|------|----------|
| **Token Gate** | Budget enforcement, usage pacing, reserve protection |
| **Backup Gate** | Repository state validation |
| **Rate Limit Gate** | Loop detection and execution throttling |
| **Session Gate** | Concurrent access management |

## Installation

```bash
pip install git+https://github.com/8144225309/pirq.git

# Verify installation
pirq status
```

Or for development:

```bash
git clone https://github.com/8144225309/pirq.git
cd pirq
pip install -e .
```

## Usage

### Basic Commands

```bash
# System status
pirq status

# Validate all gates
pirq check

# Execute prompt through PIRQ
pirq run "Explain this codebase"

# Model selection
pirq run "Quick question" --model haiku

# Output formats
pirq run "Analyze" --output brief    # Truncated
pirq run "Analyze" --output json     # Raw JSON
pirq run "Analyze" --output full     # Complete

# Execution limits
pirq run "Refactor" --max-turns 5
pirq run "Review" --tools Read,Grep
```

### Session Management

```bash
# Resume last session
pirq run --last "Continue"

# Auto-approve permissions
pirq run --yolo "Deploy changes"

# Combined
pirq run --last --yolo "Finish the task"
```

### Token Budget

```bash
# Current status with pacing
pirq tokens status

# Detailed pacing analysis
pirq tokens pace

# Configure thresholds
pirq tokens warn --used 80
pirq tokens block --used 95

# Emergency reserve
pirq tokens reserve --percent 5 --mode hard
```

### Turbo Mode

End-of-period token utilization:

```bash
pirq turbo status
pirq turbo set --days 3 --min-remaining 20
pirq turbo on
```

### Audit Trail

```bash
pirq logs audit      # Command history
pirq logs verify     # Integrity check
pirq logs show       # Session details
```

## Configuration

PIRQ configuration lives in `.pirq/config.json`:

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

### Plan Presets

```bash
pirq tokens configure --plan pro
pirq tokens configure --plan max
pirq tokens configure --plan custom --budget 5000000
```

## Architecture

```
User Prompt
     │
     ▼
┌─────────┐
│  PIRQ   │◄── Gate validation
└────┬────┘
     │ Pass
     ▼
┌─────────┐
│ Claude  │
└────┬────┘
     │
     ▼
Response + Audit Log
```

## License

MIT — See [LICENSE](LICENSE)
