"""PIRQ CLI Entry Point.

Usage:
    pirq                    Show status
    pirq check              Run all PIRQ gate checks
    pirq run <prompt>       Run Claude with gating
    pirq lock status        Show lock status
    pirq lock clean         Clean stale locks
    pirq init               Initialize .pirq directory
    pirq rollback [num]     Rollback to previous prompt
    pirq git log|status     Git-related commands
    pirq tokens status      Token budget with pacing info
    pirq tokens pace        Detailed pacing analysis
    pirq tokens reserve     Configure emergency reserve
    pirq tokens warn        Configure warn threshold
    pirq tokens block       Configure block threshold
    pirq tokens thresholds  Show current thresholds
    pirq tokens configure   Set plan and budget
    pirq tokens history     Usage history
    pirq tokens sync        Sync from Claude Code
    pirq turbo status       Show turbo mode status
    pirq turbo on           Enable turbo mode
    pirq turbo off          Disable turbo mode
    pirq turbo set          Configure turbo mode settings
    pirq logs show [id]     Show session logs
    pirq logs verify        Verify audit log integrity
    pirq logs audit         Show command audit trail
    pirq gates [name]       Show gate statuses
"""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_config, ensure_pirq_dir
from .logs import verify_audit_log, get_audit_summary
from .orchestrator import Orchestrator


def cmd_status(args, orch: Orchestrator) -> int:
    """Show current status."""
    status = orch.get_status()

    if args.json:
        print(json.dumps(status, indent=2))
        return 0

    # Human-readable output
    print(f"PIRQ Orchestrator v{__version__}")
    print("-" * 40)
    print(f"State:     {status['state']}")

    if status['reason']:
        print(f"Reason:    {status['reason']}")

    if status['session_active']:
        info = status['session_info']
        sessions = info.get('sessions', [])
        if sessions:
            print(f"Session:   {len(sessions)} active")
            for s in sessions:
                force_marker = " [FORCE]" if s.get('force') else ""
                print(f"           - PID {s.get('pid')}: {s.get('task', 'unknown')}{force_marker}")
        else:
            print(f"Session:   Active (PID {info.get('pid')}, task: {info.get('task')})")
    else:
        print("Session:   Idle")

    print(f"Timestamp: {status['timestamp']}")

    if status['blocked_by']:
        print(f"Blocked:   {', '.join(status['blocked_by'])}")

    # Gate summary
    if status['pirqs']:
        print("\nGates:")
        for name, info in status['pirqs'].items():
            status_icon = {
                "clear": "[OK]",
                "warn": "[!]",
                "block": "[X]",
            }.get(info.get('status', 'unknown'), "[?]")
            print(f"  {status_icon} {name}: {info.get('message', 'unknown')}")

    return 0


def cmd_check(args, orch: Orchestrator) -> int:
    """Run all PIRQ gate checks."""
    all_clear, results = orch.check_gates()

    if args.json:
        output = {
            "all_clear": all_clear,
            "gates": {name: r.to_dict() for name, r in results.items()},
        }
        print(json.dumps(output, indent=2))
        return 0 if all_clear else 2

    # Human-readable output
    print("PIRQ Gate Check")
    print("-" * 40)

    for name, result in results.items():
        status_icon = {
            "clear": "[OK]",
            "warn": "[!]",
            "block": "[X]",
        }.get(result.status.value, "[?]")

        print(f"{status_icon} {name}: {result.message}")

        if result.data:
            for key, value in result.data.items():
                print(f"    {key}: {value}")

    print("-" * 40)
    if all_clear:
        print("Result: ALL CLEAR - Ready to proceed")
        return 0
    else:
        blocked = [name for name, r in results.items() if r.is_blocked]
        print(f"Result: BLOCKED by {', '.join(blocked)}")
        return 2


def _format_run_output(raw_output: str, mode: str) -> tuple:
    """Format Claude output based on mode.

    Returns:
        Tuple of (formatted_output, metadata_dict)
    """
    if mode == "raw":
        return raw_output, {}

    if mode == "json":
        return raw_output, {}

    # Parse JSON for brief/normal/full modes
    metadata = {}
    try:
        data = json.loads(raw_output)
        if isinstance(data, dict) and "result" in data:
            result_text = data["result"]
            metadata = {
                "duration_ms": data.get("duration_ms"),
                "num_turns": data.get("num_turns"),
                "duration_api_ms": data.get("duration_api_ms"),
            }
        else:
            result_text = raw_output
    except (json.JSONDecodeError, TypeError):
        result_text = raw_output

    if mode == "brief":
        if len(result_text) > 500:
            result_text = result_text[:500] + f"... ({len(result_text)} chars total)"
        return result_text, metadata

    # normal and full modes return full text
    return result_text, metadata


def cmd_run(args, orch: Orchestrator) -> int:
    """Run Claude with PIRQ gating."""
    if not args.prompt:
        print("Error: Prompt required. Use: pirq run \"your prompt\"")
        return 1

    quiet = getattr(args, 'quiet', False)
    output_mode = getattr(args, 'output', 'normal')

    if not quiet:
        print("Running PIRQ orchestration...")
        print("-" * 40)

    # Collect Claude CLI pass-through parameters
    force = getattr(args, 'force', False)
    tools = getattr(args, 'tools', None)
    no_tools = getattr(args, 'no_tools', None)

    result = orch.run_once(
        args.prompt,
        task=args.task,
        model=getattr(args, 'model', None),
        force=force,
        max_turns=getattr(args, 'max_turns', None),
        system_prompt=getattr(args, 'system_prompt', None),
        verbose=getattr(args, 'verbose', False),
        continue_session=getattr(args, 'continue_session', False),
        resume_session=getattr(args, 'resume', None),
        fallback_model=getattr(args, 'fallback', None),
        allowed_tools=tools.split(',') if tools else None,
        blocked_tools=no_tools.split(',') if no_tools else None,
        timeout=getattr(args, 'timeout', None),
        last_session=getattr(args, 'last', False),
        yolo=getattr(args, 'yolo', False),
    )

    if result.blocked:
        if not quiet:
            print("BLOCKED by PIRQ gates:")
            for name, gate_result in result.gate_results.items():
                if gate_result.is_blocked:
                    print(f"  [X] {name}: {gate_result.message}")
            if not force:
                print()
                print("Tip: Use --force to bypass (with confirmation)")
        return 2

    if result.run_result:
        run = result.run_result
        if run.success:
            if quiet:
                return 0

            # Format output based on mode
            formatted_output, metadata = _format_run_output(run.output, output_mode)

            # Header info (skip for json/raw modes)
            if output_mode not in ("json", "raw"):
                print(f"[OK] Claude completed successfully")
                print(f"  Duration: {run.duration_ms}ms")
                if metadata.get("duration_api_ms"):
                    print(f"  API Duration: {metadata['duration_api_ms']}ms")
                if metadata.get("num_turns"):
                    print(f"  Turns: {metadata['num_turns']}")
                if run.orch_tag:
                    print(f"  ORCH Tag: {run.orch_tag}")
                print("-" * 40)
                print("Output:")

            print(formatted_output)
            return 0
        else:
            if not quiet:
                print(f"[X] Claude failed (exit code {run.exit_code})")
                if run.error:
                    print(f"Error: {run.error}")
            return 1

    return 0


def cmd_lock(args, orch: Orchestrator) -> int:
    """Manage session locks."""
    if args.lock_action == "status":
        lock_info = orch.session_gate.get_lock_info()
        if lock_info:
            if args.json:
                print(json.dumps(lock_info, indent=2))
            else:
                print("Session Lock: ACTIVE")
                for key, value in lock_info.items():
                    print(f"  {key}: {value}")
        else:
            if args.json:
                print(json.dumps({"active": False}))
            else:
                print("Session Lock: None")
        return 0

    elif args.lock_action == "clean":
        if args.force or not orch.session_gate.lock_file.exists():
            orch.session_gate.release()
            print("Lock cleaned.")
            return 0
        else:
            # Check if process is alive first
            lock_info = orch.session_gate.get_lock_info()
            if lock_info and "pid" in lock_info:
                from .gates.session import _is_process_alive
                if _is_process_alive(lock_info["pid"]):
                    print(f"Warning: Process {lock_info['pid']} appears to be running.")
                    print("Use --force to clean anyway.")
                    return 1
            orch.session_gate.release()
            print("Lock cleaned.")
            return 0

    return 0


def cmd_loop(args, orch: Orchestrator) -> int:
    """Loop detection management."""
    if args.loop_action == "status":
        # Show loop detection state
        detector = orch.loop_gate.detector
        records = detector._records

        if args.json:
            data = {
                "record_count": len(records),
                "records": [
                    {
                        "timestamp": r.timestamp,
                        "output_hash": r.output_hash,
                        "error_hash": r.error_hash,
                        "files_changed": r.files_changed,
                    }
                    for r in records
                ],
                "loop_check": detector.check_loop(),
            }
            print(json.dumps(data, indent=2))
        else:
            print("Loop Detection State")
            print("-" * 40)
            print(f"History: {len(records)} records")

            if records:
                print("\nRecent runs:")
                for r in records[-5:]:
                    ts = r.timestamp[:19]
                    print(f"  {ts}  hash={r.output_hash[:8]}  files={r.files_changed}")

            check = detector.check_loop()
            if check["detected"]:
                print(f"\n[X] Loop detected: {check['type']}")
                print(f"    {check['details']}")
            else:
                print("\n[OK] No loop detected")

        return 0

    elif args.loop_action == "clear":
        orch.loop_gate.clear()
        print("[OK] Loop detection state cleared")
        return 0

    return 0


def cmd_init(args, orch: Orchestrator) -> int:
    """Initialize .pirq directory."""
    config = load_config()
    ensure_pirq_dir(config)

    pirq_dir = config.pirq_dir or Path.cwd() / ".pirq"
    print(f"Initialized PIRQ in {pirq_dir}")
    print("\nCreated:")
    print(f"  {pirq_dir}/config.json")
    print(f"  {pirq_dir}/runtime/")
    print(f"  {pirq_dir}/logs/")

    return 0


def cmd_rollback(args, orch: Orchestrator) -> int:
    """Rollback to a previous prompt state."""
    git_mgr = orch.git_manager

    if not git_mgr.is_git_repo():
        print("Error: Not a git repository")
        return 1

    # Show history if requested
    if args.list:
        history = git_mgr.get_prompt_history(limit=20)
        if not history:
            print("No prompt history found")
            return 0

        print("Prompt History:")
        print("-" * 60)
        for tag in history:
            print(f"  #{tag.number}: {tag.summary or '(no summary)'}")
            print(f"         {tag.timestamp} [{tag.hash}]")
        return 0

    # Do the rollback
    target = args.prompt_num
    if target is not None:
        print(f"Rolling back to prompt #{target}...")
    else:
        print("Rolling back to previous prompt...")

    if git_mgr.rollback(target):
        print("[OK] Rollback successful")
        return 0
    else:
        print("[X] Rollback failed - tag not found or no history")
        return 1


def cmd_tokens(args, orch: Orchestrator) -> int:
    """Token budget commands."""
    tracker = orch.token_gate.tracker
    from .tokens.pacing import tokens_to_dollars

    if args.token_action == "pace":
        # Detailed pacing analysis
        pacing = orch.token_gate.get_pacing_status()

        if args.json:
            data = {
                "days_elapsed": round(pacing.days_elapsed, 1),
                "days_remaining": round(pacing.days_remaining, 1),
                "days_total": round(pacing.days_total, 1),
                "percent_period_elapsed": round(pacing.percent_period_elapsed, 1),
                "budget": pacing.budget,
                "used": pacing.used,
                "remaining": pacing.remaining,
                "expected_usage": pacing.expected_usage,
                "pace_delta": pacing.pace_delta,
                "pace_percent": round(pacing.pace_percent, 1),
                "pace_status": pacing.pace_status,
                "reserve": pacing.reserve,
                "effective_budget": pacing.effective_budget,
                "in_reserve": pacing.in_reserve,
                "projected_end_usage": pacing.projected_end_usage,
                "projected_end_percent": round(pacing.projected_end_percent, 1),
                "daily_burn_rate": round(pacing.daily_burn_rate, 0),
                "safe_daily_rate": round(pacing.safe_daily_rate, 0),
                "used_dollars": round(pacing.used_dollars, 2),
                "remaining_dollars": round(pacing.remaining_dollars, 2),
            }
            print(json.dumps(data, indent=2))
        else:
            print("=" * 50)
            print("          PIRQ BUDGET PACING ANALYSIS")
            print("=" * 50)

            # Time info
            print()
            print(f"PERIOD: Day {pacing.days_elapsed:.0f} of {pacing.days_total:.0f}")
            print(f"        {pacing.days_remaining:.1f} days remaining")
            print(f"        {pacing.percent_period_elapsed:.0f}% of period elapsed")

            # Budget vs Expected
            print()
            print("BUDGET PACING:")
            print(f"  Budget:      {pacing.budget:>12,} tokens | ${pacing.budget_dollars:,.2f}")
            print(f"  Expected:    {pacing.expected_usage:>12,} tokens (at this point)")
            print(f"  Actual:      {pacing.used:>12,} tokens | ${pacing.used_dollars:,.2f}")
            print(f"  Remaining:   {pacing.remaining:>12,} tokens | ${pacing.remaining_dollars:,.2f}")

            # Pace status
            print()
            if pacing.pace_delta < 0:
                delta_str = f"{abs(pacing.pace_delta):,} under expected"
            else:
                delta_str = f"{pacing.pace_delta:,} over expected"

            status_icons = {
                "under": "[OK]",
                "on_pace": "[OK]",
                "ahead": "[!]",
                "critical": "[X]",
            }
            icon = status_icons.get(pacing.pace_status, "[?]")

            print(f"PACE STATUS: {icon} {pacing.pace_status.upper()}")
            print(f"  Pace:        {pacing.pace_percent:.0f}% of expected usage")
            print(f"  Delta:       {delta_str}")

            # Reserve info
            print()
            print("RESERVE:")
            print(f"  Emergency reserve: {pacing.reserve:,} tokens ({pacing.reserve / pacing.budget * 100:.1f}%)")
            print(f"  Effective budget:  {pacing.effective_budget:,} tokens")
            if pacing.in_reserve:
                used_from_reserve = pacing.used - pacing.effective_budget
                print(f"  [!] USING RESERVE: {used_from_reserve:,} tokens into reserve")

            # Projections
            print()
            print("PROJECTIONS:")
            print(f"  At current rate:   {pacing.projected_end_usage:,} tokens by period end")
            print(f"                     ({pacing.projected_end_percent:.0f}% of budget)")
            print(f"  Your daily burn:   ~{pacing.daily_burn_rate:,.0f} tokens/day")
            print(f"  Safe daily rate:   ~{pacing.safe_daily_rate:,.0f} tokens/day")

            if pacing.daily_burn_rate > 0:
                days_until_exhausted = pacing.remaining / pacing.daily_burn_rate
                print(f"  Days until empty:  ~{days_until_exhausted:.1f} days")

            # Recommendation
            print()
            print("-" * 50)
            recommendation = orch.token_gate.pacing_calculator.get_recommendation(pacing)
            print(recommendation)

        return 0

    elif args.token_action == "reserve":
        # Configure or show reserve settings
        has_reserve_args = (
            args.threshold_percent is not None or
            args.threshold_tokens is not None or
            args.threshold_dollars is not None or
            args.reserve_mode
        )
        if has_reserve_args:
            # Set reserve config
            config_path = orch.pirq_dir / "config.json"
            try:
                if config_path.exists():
                    config_data = json.loads(config_path.read_text())
                else:
                    config_data = {}

                if "tokens" not in config_data:
                    config_data["tokens"] = {}

                if args.threshold_percent is not None:
                    config_data["tokens"]["reserve_percent"] = args.threshold_percent
                    config_data["tokens"]["reserve_tokens"] = 0
                    config_data["tokens"]["reserve_dollars"] = 0.0
                    print(f"[OK] Reserve set to {args.threshold_percent}% of budget")

                elif args.threshold_tokens is not None:
                    config_data["tokens"]["reserve_tokens"] = args.threshold_tokens
                    config_data["tokens"]["reserve_percent"] = 0.0
                    config_data["tokens"]["reserve_dollars"] = 0.0
                    print(f"[OK] Reserve set to {args.threshold_tokens:,} tokens")

                elif args.threshold_dollars is not None:
                    config_data["tokens"]["reserve_dollars"] = args.threshold_dollars
                    config_data["tokens"]["reserve_tokens"] = 0
                    config_data["tokens"]["reserve_percent"] = 0.0
                    print(f"[OK] Reserve set to ${args.threshold_dollars:.2f}")

                if args.reserve_mode:
                    config_data["tokens"]["reserve_mode"] = args.reserve_mode
                    print(f"[OK] Reserve mode set to: {args.reserve_mode}")

                config_path.write_text(json.dumps(config_data, indent=2))
                return 0

            except Exception as e:
                print(f"[X] Failed to save config: {e}")
                return 1

        else:
            # Show current reserve settings
            pacing = orch.token_gate.get_pacing_status()

            print("Emergency Reserve Configuration")
            print("-" * 40)
            print(f"Reserve:        {pacing.reserve:,} tokens")
            print(f"                {pacing.reserve / pacing.budget * 100:.1f}% of budget")
            print(f"                ${tokens_to_dollars(pacing.reserve):.2f}")
            print(f"Mode:           {orch.token_gate.reserve_mode}")
            print(f"Effective:      {pacing.effective_budget:,} tokens usable")

            if pacing.in_reserve:
                used_from_reserve = pacing.used - pacing.effective_budget
                print()
                print(f"[!] Currently in reserve: {used_from_reserve:,} tokens used")
                print(f"    Reserve remaining: {pacing.reserve_remaining:,} tokens")

            print()
            print("Usage:")
            print("  pirq tokens reserve --percent 5      # 5% of budget")
            print("  pirq tokens reserve --tokens 100000  # Fixed amount")
            print("  pirq tokens reserve --dollars 2.00   # Dollar amount")
            print("  pirq tokens reserve --mode hard      # Block at reserve")
            print("  pirq tokens reserve --mode soft      # Warn at reserve")

        return 0

    elif args.token_action == "warn":
        # Configure warn threshold
        has_args = (
            args.threshold_used is not None or
            args.threshold_remaining is not None or
            args.threshold_tokens is not None or
            args.threshold_dollars is not None
        )
        if has_args:
            config_path = orch.pirq_dir / "config.json"
            try:
                if config_path.exists():
                    config_data = json.loads(config_path.read_text())
                else:
                    config_data = {}

                if "tokens" not in config_data:
                    config_data["tokens"] = {}

                # Clear other warn settings, set the one specified
                config_data["tokens"]["warn_at_percent_used"] = 0.0
                config_data["tokens"]["warn_at_percent_remaining"] = 0.0
                config_data["tokens"]["warn_at_tokens"] = 0
                config_data["tokens"]["warn_at_dollars"] = 0.0

                if args.threshold_used is not None:
                    config_data["tokens"]["warn_at_percent_used"] = args.threshold_used
                    print(f"[OK] Warn threshold set to {args.threshold_used}% used")
                elif args.threshold_remaining is not None:
                    config_data["tokens"]["warn_at_percent_remaining"] = args.threshold_remaining
                    print(f"[OK] Warn threshold set to {args.threshold_remaining}% remaining")
                elif args.threshold_tokens is not None:
                    config_data["tokens"]["warn_at_tokens"] = args.threshold_tokens
                    print(f"[OK] Warn threshold set to {args.threshold_tokens:,} tokens remaining")
                elif args.threshold_dollars is not None:
                    config_data["tokens"]["warn_at_dollars"] = args.threshold_dollars
                    print(f"[OK] Warn threshold set to ${args.threshold_dollars:.2f} remaining")

                config_path.write_text(json.dumps(config_data, indent=2))
                return 0

            except Exception as e:
                print(f"[X] Failed to save config: {e}")
                return 1
        else:
            # Show usage
            print("Set warn threshold (when to start warning)")
            print("-" * 40)
            warn_tokens = orch.token_gate._get_warn_threshold_tokens()
            print(f"Current: warn at {warn_tokens:,} tokens remaining")
            print()
            print("Usage:")
            print("  pirq tokens warn --used 80       # Warn at 80% used (20% left)")
            print("  pirq tokens warn --remaining 20  # Same thing, different view")
            print("  pirq tokens warn --tokens 500000 # Warn at 500k tokens left")
            print("  pirq tokens warn --dollars 2.00  # Warn at $2 left")
            return 0

    elif args.token_action == "block":
        # Configure block threshold
        has_args = (
            args.threshold_used is not None or
            args.threshold_remaining is not None or
            args.threshold_tokens is not None or
            args.threshold_dollars is not None
        )
        if has_args:
            config_path = orch.pirq_dir / "config.json"
            try:
                if config_path.exists():
                    config_data = json.loads(config_path.read_text())
                else:
                    config_data = {}

                if "tokens" not in config_data:
                    config_data["tokens"] = {}

                # Clear other block settings, set the one specified
                config_data["tokens"]["block_at_percent_used"] = 0.0
                config_data["tokens"]["block_at_percent_remaining"] = 0.0
                config_data["tokens"]["block_at_tokens"] = 0
                config_data["tokens"]["block_at_dollars"] = 0.0

                if args.threshold_used is not None:
                    config_data["tokens"]["block_at_percent_used"] = args.threshold_used
                    print(f"[OK] Block threshold set to {args.threshold_used}% used")
                elif args.threshold_remaining is not None:
                    config_data["tokens"]["block_at_percent_remaining"] = args.threshold_remaining
                    print(f"[OK] Block threshold set to {args.threshold_remaining}% remaining")
                elif args.threshold_tokens is not None:
                    config_data["tokens"]["block_at_tokens"] = args.threshold_tokens
                    print(f"[OK] Block threshold set to {args.threshold_tokens:,} tokens remaining")
                elif args.threshold_dollars is not None:
                    config_data["tokens"]["block_at_dollars"] = args.threshold_dollars
                    print(f"[OK] Block threshold set to ${args.threshold_dollars:.2f} remaining")

                config_path.write_text(json.dumps(config_data, indent=2))
                return 0

            except Exception as e:
                print(f"[X] Failed to save config: {e}")
                return 1
        else:
            # Show usage
            print("Set block threshold (when to stop execution)")
            print("-" * 40)
            block_tokens = orch.token_gate._get_block_threshold_tokens()
            print(f"Current: block at {block_tokens:,} tokens remaining")
            print()
            print("Usage:")
            print("  pirq tokens block --used 95       # Block at 95% used (5% left)")
            print("  pirq tokens block --remaining 5   # Same thing, different view")
            print("  pirq tokens block --tokens 100000 # Block at 100k tokens left")
            print("  pirq tokens block --dollars 0.50  # Block at $0.50 left")
            return 0

    elif args.token_action == "thresholds":
        # Show all threshold settings
        pacing = orch.token_gate.get_pacing_status()
        warn_tokens = orch.token_gate._get_warn_threshold_tokens()
        block_tokens = orch.token_gate._get_block_threshold_tokens()

        if args.json:
            data = {
                "budget": pacing.budget,
                "warn": {
                    "tokens": warn_tokens,
                    "percent_remaining": warn_tokens / pacing.budget * 100 if pacing.budget > 0 else 0,
                    "percent_used": 100 - (warn_tokens / pacing.budget * 100) if pacing.budget > 0 else 0,
                    "dollars": tokens_to_dollars(warn_tokens),
                },
                "block": {
                    "tokens": block_tokens,
                    "percent_remaining": block_tokens / pacing.budget * 100 if pacing.budget > 0 else 0,
                    "percent_used": 100 - (block_tokens / pacing.budget * 100) if pacing.budget > 0 else 0,
                    "dollars": tokens_to_dollars(block_tokens),
                },
                "reserve": {
                    "tokens": pacing.reserve,
                    "percent": pacing.reserve / pacing.budget * 100 if pacing.budget > 0 else 0,
                    "mode": orch.token_gate.reserve_mode,
                },
            }
            print(json.dumps(data, indent=2))
        else:
            print("Token Budget Thresholds")
            print("=" * 50)
            print(f"Budget: {pacing.budget:,} tokens")
            print()

            # Warn threshold
            warn_pct_used = 100 - (warn_tokens / pacing.budget * 100) if pacing.budget > 0 else 0
            warn_pct_remaining = warn_tokens / pacing.budget * 100 if pacing.budget > 0 else 0
            print("WARN THRESHOLD:")
            print(f"  At {warn_tokens:,} tokens remaining")
            print(f"  = {warn_pct_used:.0f}% used / {warn_pct_remaining:.0f}% remaining")
            print(f"  = ${tokens_to_dollars(warn_tokens):.2f}")

            # Block threshold
            print()
            block_pct_used = 100 - (block_tokens / pacing.budget * 100) if pacing.budget > 0 else 0
            block_pct_remaining = block_tokens / pacing.budget * 100 if pacing.budget > 0 else 0
            print("BLOCK THRESHOLD:")
            print(f"  At {block_tokens:,} tokens remaining")
            print(f"  = {block_pct_used:.0f}% used / {block_pct_remaining:.0f}% remaining")
            print(f"  = ${tokens_to_dollars(block_tokens):.2f}")

            # Reserve
            print()
            reserve_pct = pacing.reserve / pacing.budget * 100 if pacing.budget > 0 else 0
            print("EMERGENCY RESERVE:")
            print(f"  {pacing.reserve:,} tokens ({reserve_pct:.1f}%)")
            print(f"  Mode: {orch.token_gate.reserve_mode}")

            # Current status
            print()
            print("-" * 50)
            if pacing.remaining <= block_tokens:
                print(f"[X] BLOCKED - Below block threshold")
            elif pacing.in_reserve:
                print(f"[!] IN RESERVE - Using emergency fund")
            elif pacing.remaining <= warn_tokens:
                print(f"[!] WARNING - Below warn threshold")
            else:
                print(f"[OK] CLEAR - {pacing.remaining:,} tokens remaining")

        return 0

    elif args.token_action == "configure":
        # Configure token budget
        from .config import PLAN_PRESETS

        if args.plan:
            plan = args.plan.lower()
            if plan not in PLAN_PRESETS and plan != "custom":
                print(f"Unknown plan: {plan}")
                print(f"Available plans: {', '.join(PLAN_PRESETS.keys())}, custom")
                return 1

            if plan == "custom" and not args.budget:
                print("Custom plan requires --budget")
                return 1

            # Determine budget
            if plan == "custom":
                budget = args.budget
            else:
                budget = PLAN_PRESETS[plan]

            # Update config file
            config_path = orch.pirq_dir / "config.json"
            try:
                if config_path.exists():
                    config_data = json.loads(config_path.read_text())
                else:
                    config_data = {}

                if "tokens" not in config_data:
                    config_data["tokens"] = {}

                config_data["tokens"]["plan"] = plan
                if budget >= 0:
                    config_data["tokens"]["budget"] = budget

                config_path.write_text(json.dumps(config_data, indent=2))

                if budget < 0:
                    print(f"[OK] Configured plan: {plan} (unlimited)")
                else:
                    print(f"[OK] Configured plan: {plan} ({budget:,} tokens/month)")
                print(f"     Config saved to: {config_path}")
                return 0

            except Exception as e:
                print(f"[X] Failed to save config: {e}")
                return 1
        else:
            # Show current config and options
            print("Token Budget Configuration")
            print("-" * 40)
            print(f"Current plan:   {orch.config.tokens.plan}")
            print(f"Current budget: {orch.config.tokens.budget:,} tokens")
            print()
            print("Available plans:")
            for plan, budget in PLAN_PRESETS.items():
                if budget < 0:
                    print(f"  {plan:12} unlimited")
                elif budget == 0:
                    print(f"  {plan:12} (blocked)")
                else:
                    print(f"  {plan:12} {budget:>12,} tokens/month")
            print()
            print("Usage:")
            print("  pirq tokens configure --plan max")
            print("  pirq tokens configure --plan custom --budget 1000000")
            return 0

    elif args.token_action == "status":
        # Get pacing status (includes real usage if available)
        pacing = orch.token_gate.get_pacing_status()
        real_usage = orch.token_gate.get_real_usage()
        plan = getattr(orch.config.tokens, 'plan', 'custom')
        budget = orch.config.tokens.budget

        if args.json:
            data = {
                "plan": plan,
                "budget": budget,
                "used": pacing.used,
                "remaining": pacing.remaining,
                "percent_used": round(pacing.percent_used, 1),
                "percent_remaining": round(100 - pacing.percent_used, 1),
                "estimated_cost_usd": round(pacing.used_dollars, 4),
                "source": "claude_code" if real_usage else "pirq_tracked",
                "pacing": {
                    "days_elapsed": round(pacing.days_elapsed, 1),
                    "days_remaining": round(pacing.days_remaining, 1),
                    "percent_period_elapsed": round(pacing.percent_period_elapsed, 1),
                    "expected_usage": pacing.expected_usage,
                    "pace_percent": round(pacing.pace_percent, 1),
                    "pace_status": pacing.pace_status,
                },
                "reserve": {
                    "reserve_tokens": pacing.reserve,
                    "effective_budget": pacing.effective_budget,
                    "in_reserve": pacing.in_reserve,
                },
            }
            if real_usage:
                data["breakdown"] = {
                    "input_tokens": real_usage.input_tokens,
                    "output_tokens": real_usage.output_tokens,
                    "cache_creation_tokens": real_usage.cache_creation_tokens,
                    "cache_read_tokens": real_usage.cache_read_tokens,
                    "session_count": real_usage.session_count,
                }
            print(json.dumps(data, indent=2))
        else:
            print("=" * 50)
            print("           PIRQ TOKEN BUDGET STATUS")
            print("=" * 50)

            # Plan info
            if budget > 0:
                print(f"Plan: {plan.upper()} ({budget:,} tokens | ~${pacing.budget_dollars:.2f}/month)")
            else:
                print(f"Plan: {plan.upper()} (unlimited)")

            print()

            # Main stats with dollars
            print(f"USED:        {pacing.used:,} tokens | ${pacing.used_dollars:.2f}")
            if budget > 0:
                print(f"REMAINING:   {pacing.remaining:,} tokens | ${pacing.remaining_dollars:.2f}")
                print(f"RESERVE:     {pacing.reserve:,} tokens | ${tokens_to_dollars(pacing.reserve):.2f}")
                print()

                # Visual progress bar
                bar_width = 30
                filled = min(bar_width, int(bar_width * pacing.percent_used / 100))
                bar = "#" * filled + "-" * (bar_width - filled)
                print(f"[{bar}] {pacing.percent_used:.1f}% of budget")
                print(f"                                {100 - pacing.percent_used:.1f}% remaining")
            else:
                print("REMAINING:   unlimited")

            # Pacing summary
            if budget > 0:
                print()
                print(f"PACING (Day {pacing.days_elapsed:.0f} of {pacing.days_total:.0f}):")
                print(f"  Expected:    {pacing.expected_usage:,} tokens ({pacing.percent_period_elapsed:.0f}%)")
                print(f"  Actual:      {pacing.used:,} tokens")

                status_icons = {
                    "under": "[OK] UNDER",
                    "on_pace": "[OK] ON PACE",
                    "ahead": "[!] AHEAD",
                    "critical": "[X] CRITICAL",
                }
                status_str = status_icons.get(pacing.pace_status, "[?]")
                print(f"  Status:      {status_str} ({pacing.pace_percent:.0f}% of expected)")

                print()
                print(f"  At this rate: {pacing.projected_end_usage:,} by end of period")
                print(f"  Safe daily:   ~{pacing.safe_daily_rate:,.0f} tokens/day")
                print(f"  Your daily:   ~{pacing.daily_burn_rate:,.0f} tokens/day")

            print()
            print("-" * 50)

            # Final status
            if pacing.in_reserve:
                used_from_reserve = pacing.used - pacing.effective_budget
                print(f"[!] USING RESERVE: {used_from_reserve:,} tokens into emergency reserve")
            elif pacing.pace_status == "critical":
                print("[X] CRITICAL - Burning tokens faster than sustainable")
            elif pacing.pace_status == "ahead":
                print("[!] AHEAD OF PACE - Consider slowing down")
            elif pacing.pace_status == "on_pace":
                print("[OK] ON PACE - Tracking normally")
            else:
                print("[OK] UNDER PACE - Room to spare")

            if not real_usage:
                print()
                print("Tip: Run 'pirq tokens sync' to fetch real usage from Claude Code")

        return 0

    elif args.token_action == "sync":
        # Sync from Claude Code logs
        print("Syncing token usage from Claude Code...")

        if not orch.token_gate.is_claude_available():
            print("[X] Claude Code logs not found")
            print(f"    Expected at: ~/.claude/projects/")
            return 1

        result = orch.token_gate.sync_from_claude()

        if result.get("success"):
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print("[OK] Synced from Claude Code")
                print("-" * 40)
                print(f"Input tokens:        {result.get('input_tokens', 0):,}")
                print(f"Output tokens:       {result.get('output_tokens', 0):,}")
                print(f"Cache creation:      {result.get('cache_creation_tokens', 0):,}")
                print(f"Cache read:          {result.get('cache_read_tokens', 0):,}")
                print(f"Total tokens:        {result.get('total_tokens', 0):,}")
                print(f"Estimated cost:      ${result.get('estimated_cost_usd', 0):.4f}")
                print(f"Sessions:            {result.get('session_count', 0)}")
                print(f"Messages:            {result.get('message_count', 0)}")
            return 0
        else:
            print(f"[X] Sync failed: {result.get('error', 'Unknown error')}")
            return 1

    elif args.token_action == "history":
        entries = tracker.get_usage_history(limit=args.limit or 20)
        if not entries:
            print("No token usage history")
            return 0

        print("Token Usage History")
        print("-" * 60)
        for entry in reversed(entries):
            ts = entry.get("ts", "?")[:19]
            entry_type = entry.get("type", "run")

            if entry_type == "claude_sync":
                total = entry.get("total_tokens", 0)
                cost = entry.get("estimated_cost_usd", 0)
                print(f"  {ts}  [SYNC] {total:>8} tokens (${cost:.4f})")
            else:
                total = entry.get("total_tokens", 0)
                model = entry.get("model", "?")
                print(f"  {ts}  {total:>8} tokens  ({model})")
        return 0

    return 0


def cmd_git(args, orch: Orchestrator) -> int:
    """Git-related commands."""
    git_mgr = orch.git_manager

    if not git_mgr.is_git_repo():
        print("Error: Not a git repository")
        return 1

    if args.git_action == "log":
        history = git_mgr.get_prompt_history(limit=args.limit or 10)
        if not history:
            print("No prompt history found")
            return 0

        print("Prompt History:")
        print("-" * 60)
        for tag in history:
            print(f"  #{tag.number}: {tag.summary or '(no summary)'}")
            print(f"         {tag.timestamp} [{tag.hash}]")
        return 0

    elif args.git_action == "status":
        status = git_mgr.get_status()
        if args.json:
            import json
            print(json.dumps({
                "is_repo": status.is_repo,
                "branch": status.branch,
                "has_changes": status.has_changes,
                "staged": status.staged_count,
                "unstaged": status.unstaged_count,
                "untracked": status.untracked_count,
                "last_commit": status.last_commit,
            }, indent=2))
        else:
            print(f"Branch: {status.branch}")
            print(f"Changes: {'Yes' if status.has_changes else 'No'}")
            if status.has_changes:
                print(f"  Staged: {status.staged_count}")
                print(f"  Unstaged: {status.unstaged_count}")
                print(f"  Untracked: {status.untracked_count}")
            if status.last_commit:
                print(f"Last commit: {status.last_commit}")
        return 0

    return 0


def cmd_logs(args, orch: Orchestrator) -> int:
    """Log inspection commands."""
    pirq_dir = orch.pirq_dir

    if args.logs_action == "show":
        # Show session logs
        sessions_dir = pirq_dir / "logs" / "sessions"
        if not sessions_dir.exists():
            print("No session logs found")
            return 0

        if args.session_id:
            # Show specific session
            session_files = list(sessions_dir.glob(f"*{args.session_id}*.jsonl"))
            if not session_files:
                print(f"Session not found: {args.session_id}")
                return 1

            session_file = session_files[0]
            print(f"Session: {session_file.stem}")
            print("-" * 60)

            with open(session_file, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if args.json:
                            print(json.dumps(entry, indent=2))
                        else:
                            _print_session_entry(entry)
            return 0
        else:
            # List recent sessions
            session_files = sorted(sessions_dir.glob("*.jsonl"), reverse=True)
            limit = args.limit or 10

            if not session_files:
                print("No session logs found")
                return 0

            print("Recent Sessions")
            print("-" * 60)

            for sf in session_files[:limit]:
                # Read first line to get session info
                with open(sf, "r") as f:
                    first_line = f.readline()
                    if first_line:
                        entry = json.loads(first_line)
                        session_id = entry.get("session_id", sf.stem)
                        started = entry.get("started_at", "?")[:19]
                        prompt = entry.get("invocation", {}).get("prompt", "")[:40]
                        print(f"  {session_id}")
                        print(f"    Started: {started}")
                        print(f"    Prompt:  {prompt}...")
            return 0

    elif args.logs_action == "verify":
        # Verify audit log integrity
        audit_path = pirq_dir / "logs" / "audit" / "commands.jsonl"
        if not audit_path.exists():
            print("No audit log found")
            return 0

        is_valid, broken_entries = verify_audit_log(audit_path)

        if args.json:
            print(json.dumps({
                "valid": is_valid,
                "broken_entries": broken_entries,
            }, indent=2))
        else:
            if is_valid:
                print("[OK] Audit log integrity verified")
                summary = get_audit_summary(audit_path)
                print(f"     Total entries: {summary['total_entries']}")
                print(f"     Date range: {summary['first_entry']} to {summary['last_entry']}")
            else:
                print("[X] Audit log TAMPERED")
                print(f"    Broken entries: {broken_entries}")
                return 2
        return 0

    elif args.logs_action == "audit":
        # Show command audit trail
        audit_path = pirq_dir / "logs" / "audit" / "commands.jsonl"
        if not audit_path.exists():
            print("No audit log found")
            return 0

        limit = args.limit or 20
        entries = []

        with open(audit_path, "r") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        # Show recent entries
        recent = entries[-limit:]

        if args.json:
            print(json.dumps(recent, indent=2))
        else:
            print("Command Audit Trail")
            print("-" * 60)

            for entry in recent:
                entry_id = entry.get("entry_id", "?")
                cmd = entry.get("command", "?")
                ts_start = (entry.get("timestamp_start") or "?")[:19]
                exit_code = entry.get("exit_code", "?")
                tokens = entry.get("tokens_used", 0)
                gates = "OK" if entry.get("gates_passed") else "BLOCKED"

                print(f"  [{entry_id}] {cmd}")
                print(f"      Time:   {ts_start}")
                print(f"      Gates:  {gates}")
                if exit_code != "?":
                    print(f"      Exit:   {exit_code}")
                if tokens:
                    print(f"      Tokens: {tokens}")
        return 0

    return 0


def cmd_gates(args, orch: Orchestrator) -> int:
    """Gate inspection commands."""
    if args.gate_name:
        # Show specific gate
        gate = None
        for g in orch.gates:
            if g.name == args.gate_name:
                gate = g
                break

        if not gate:
            print(f"Unknown gate: {args.gate_name}")
            return 1

        result = gate.check()

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status_icon = {
                "clear": "[OK]",
                "warn": "[!]",
                "block": "[X]",
            }.get(result.status.value, "[?]")

            print(f"Gate: {gate.name}")
            print("-" * 40)
            print(f"Status:  {status_icon} {result.status.value.upper()}")
            print(f"Message: {result.message}")

            if result.data:
                print("\nDetails:")
                for key, value in result.data.items():
                    print(f"  {key}: {value}")
        return 0

    else:
        # Show all gates
        print("PIRQ Gates")
        print("-" * 40)

        for gate in orch.gates:
            enabled = gate.is_enabled()
            result = gate.check() if enabled else None

            if not enabled:
                print(f"  [-] {gate.name}: disabled")
            else:
                status_icon = {
                    "clear": "[OK]",
                    "warn": "[!]",
                    "block": "[X]",
                }.get(result.status.value, "[?]")
                print(f"  {status_icon} {gate.name}: {result.message}")

        return 0


def cmd_turbo(args, orch: Orchestrator) -> int:
    """Turbo mode - burn remaining tokens before reset."""
    from .tokens.pacing import tokens_to_dollars

    if args.turbo_action == "status":
        turbo_status = orch.token_gate.get_turbo_status()
        pacing = orch.token_gate.get_pacing_status()

        if args.json:
            print(json.dumps(turbo_status, indent=2))
        else:
            print("=" * 50)
            print("               TURBO MODE")
            print("=" * 50)

            # Status
            if turbo_status["active"]:
                print("Status:     [ACTIVE]")
            elif not turbo_status["enabled"]:
                print("Status:     [DISABLED]")
            else:
                print("Status:     [INACTIVE]")

            print()
            print(f"Enabled:    {'Yes' if turbo_status['enabled'] else 'No'}")
            print(f"Trigger:    {turbo_status['trigger']}")
            print(f"Min %:      {turbo_status['min_remaining_percent']}% remaining required")

            # Time info
            print()
            print(f"Days left:  {turbo_status['days_remaining']:.1f}")
            print(f"Hours left: {turbo_status['hours_remaining']:.1f}")
            print(f"Remaining:  {turbo_status['percent_remaining']:.1f}%")

            # Available tokens
            print()
            print(f"Available:  {turbo_status['available_tokens']:,} tokens for turbo")
            print(f"            ${tokens_to_dollars(turbo_status['available_tokens']):.2f}")
            if turbo_status["reserve_protected"]:
                print(f"            (reserve protected)")
            else:
                print(f"            (reserve can be used)")

            # Usage suggestions when active
            if turbo_status["active"]:
                print()
                print("-" * 50)
                print("SUGGESTED TURBO TASKS:")
                print("  - Research: explore new patterns, read docs")
                print("  - Maintenance: cleanup, refactoring, deps")
                print("  - Cosmetic: formatting, comments, organization")
                print()
                print(f"Turbo deactivates: In {turbo_status['days_remaining']:.1f} days")

        return 0

    elif args.turbo_action == "on":
        config_path = orch.pirq_dir / "config.json"
        try:
            if config_path.exists():
                config_data = json.loads(config_path.read_text())
            else:
                config_data = {}

            if "turbo" not in config_data:
                config_data["turbo"] = {}

            config_data["turbo"]["enabled"] = True
            config_path.write_text(json.dumps(config_data, indent=2))
            print("[OK] Turbo mode enabled")
            print("     Run 'pirq turbo status' to check activation status")
            return 0

        except Exception as e:
            print(f"[X] Failed to save config: {e}")
            return 1

    elif args.turbo_action == "off":
        config_path = orch.pirq_dir / "config.json"
        try:
            if config_path.exists():
                config_data = json.loads(config_path.read_text())
            else:
                config_data = {}

            if "turbo" not in config_data:
                config_data["turbo"] = {}

            config_data["turbo"]["enabled"] = False
            config_path.write_text(json.dumps(config_data, indent=2))
            print("[OK] Turbo mode disabled")
            return 0

        except Exception as e:
            print(f"[X] Failed to save config: {e}")
            return 1

    elif args.turbo_action == "set":
        has_args = (
            args.days is not None or
            args.hours is not None or
            args.min_remaining is not None or
            args.allow_reserve
        )

        if has_args:
            config_path = orch.pirq_dir / "config.json"
            try:
                if config_path.exists():
                    config_data = json.loads(config_path.read_text())
                else:
                    config_data = {}

                if "turbo" not in config_data:
                    config_data["turbo"] = {}

                if args.days is not None:
                    config_data["turbo"]["activate_days_before_reset"] = args.days
                    config_data["turbo"]["activate_hours_before_reset"] = 0
                    print(f"[OK] Turbo activates {args.days} days before reset")

                if args.hours is not None:
                    config_data["turbo"]["activate_hours_before_reset"] = args.hours
                    config_data["turbo"]["activate_days_before_reset"] = 0
                    print(f"[OK] Turbo activates {args.hours} hours before reset")

                if args.min_remaining is not None:
                    config_data["turbo"]["min_remaining_percent"] = args.min_remaining
                    print(f"[OK] Turbo requires {args.min_remaining}% remaining")

                if args.allow_reserve:
                    config_data["turbo"]["allow_reserve_dip"] = True
                    print("[OK] Turbo can dip into reserve")

                config_path.write_text(json.dumps(config_data, indent=2))
                return 0

            except Exception as e:
                print(f"[X] Failed to save config: {e}")
                return 1
        else:
            # Show usage
            print("Configure turbo mode activation")
            print("-" * 40)
            turbo = orch.token_gate.turbo_config
            print(f"Current settings:")
            print(f"  Enabled:       {turbo.enabled}")
            if turbo.activate_hours_before_reset > 0:
                print(f"  Trigger:       {turbo.activate_hours_before_reset} hours before reset")
            else:
                print(f"  Trigger:       {turbo.activate_days_before_reset} days before reset")
            print(f"  Min remaining: {turbo.min_remaining_percent}%")
            print(f"  Reserve dip:   {turbo.allow_reserve_dip}")
            print()
            print("Usage:")
            print("  pirq turbo set --days 3          # Activate 3 days before reset")
            print("  pirq turbo set --hours 48        # Activate 48 hours before reset")
            print("  pirq turbo set --min-remaining 30 # Require 30% remaining")
            print("  pirq turbo set --allow-reserve   # Allow turbo to use reserve")
            return 0

    return 0


def _print_session_entry(entry: dict) -> None:
    """Pretty-print a session log entry."""
    event = entry.get("event", "unknown")

    if event == "session_start":
        print(f"SESSION START: {entry.get('session_id')}")
        print(f"  Prompt: {entry.get('invocation', {}).get('prompt', '')[:60]}...")

    elif event == "gates_checked":
        all_clear = entry.get("all_clear", False)
        icon = "[OK]" if all_clear else "[X]"
        print(f"  GATES: {icon} {'All clear' if all_clear else 'Blocked'}")

    elif event == "execution_start":
        print(f"  EXEC START: prompt #{entry.get('prompt_num', '?')}")

    elif event == "execution_end":
        exit_code = entry.get("exit_code", "?")
        orch_tag = entry.get("orch_tag", "")
        print(f"  EXEC END: exit={exit_code} orch={orch_tag}")

    elif event == "session_end":
        print(f"SESSION END: exit={entry.get('exit_code', '?')}")

    else:
        print(f"  {event}: {json.dumps(entry)[:60]}...")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="pirq",
        description="Self-orchestrating Claude automation with pre-execution gating",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"pirq {__version__}",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )

    subparsers = parser.add_subparsers(dest="command")

    # pirq status (default when no command)
    status_parser = subparsers.add_parser("status", help="Show current status")
    status_parser.add_argument("--json", action="store_true")

    # pirq check
    check_parser = subparsers.add_parser("check", help="Run all PIRQ gate checks")
    check_parser.add_argument("--json", action="store_true")

    # pirq run
    run_parser = subparsers.add_parser("run", help="Run Claude with gating")
    run_parser.add_argument("prompt", nargs="?", help="Prompt to send to Claude")
    run_parser.add_argument("--task", help="Task name for logging")
    run_parser.add_argument("--model", "-m", help="Model to use (haiku, sonnet, opus)")
    run_parser.add_argument("--force", "-f", action="store_true", help="Bypass gate blocks (with confirmation)")

    # Output modes
    run_parser.add_argument(
        "--output", "-o",
        choices=["brief", "normal", "full", "json", "raw"],
        default="normal",
        help="Output mode: brief (500 chars), normal (parsed), full (complete), json (raw JSON), raw (unprocessed)",
    )
    run_parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode - exit code only")

    # Tier 1: High-value Claude CLI pass-through
    run_parser.add_argument("--max-turns", type=int, help="Limit agentic turns (prevents infinite loops)")
    run_parser.add_argument("--system-prompt", help="Append to Claude's system prompt")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose Claude output")
    run_parser.add_argument("--continue", dest="continue_session", action="store_true", help="Continue last conversation")
    run_parser.add_argument("--resume", "-r", metavar="ID", help="Resume session by ID")

    # Tier 2: Medium-value Claude CLI pass-through
    run_parser.add_argument("--fallback", metavar="MODEL", help="Fallback model if primary overloaded")
    run_parser.add_argument("--tools", metavar="TOOL,...", help="Comma-separated allowed tools")
    run_parser.add_argument("--no-tools", metavar="TOOL,...", help="Comma-separated blocked tools")
    run_parser.add_argument("--timeout", type=int, metavar="SECS", help="Timeout in seconds")

    # Session shortcuts
    run_parser.add_argument("--last", "-l", action="store_true", help="Resume the most recent Claude session")
    run_parser.add_argument("--yolo", action="store_true", help="Auto-approve all permissions (--dangerously-skip-permissions)")

    # pirq loop
    loop_parser = subparsers.add_parser("loop", help="Loop detection management")
    loop_parser.add_argument(
        "loop_action",
        choices=["status", "clear"],
        help="Loop action: status (show state), clear (reset detection)",
    )
    loop_parser.add_argument("--json", action="store_true")

    # pirq lock
    lock_parser = subparsers.add_parser("lock", help="Manage session locks")
    lock_parser.add_argument(
        "lock_action",
        choices=["status", "clean"],
        help="Lock action",
    )
    lock_parser.add_argument("--force", action="store_true", help="Force clean")
    lock_parser.add_argument("--json", action="store_true")

    # pirq init
    subparsers.add_parser("init", help="Initialize .pirq directory")

    # pirq rollback
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to previous prompt")
    rollback_parser.add_argument("prompt_num", nargs="?", type=int, help="Prompt number to rollback to")
    rollback_parser.add_argument("--list", "-l", action="store_true", help="List prompt history")
    rollback_parser.add_argument("--json", action="store_true")

    # pirq git
    git_parser = subparsers.add_parser("git", help="Git-related commands")
    git_parser.add_argument(
        "git_action",
        choices=["log", "status"],
        help="Git action",
    )
    git_parser.add_argument("--limit", "-n", type=int, help="Limit for log entries")
    git_parser.add_argument("--json", action="store_true")

    # pirq tokens
    tokens_parser = subparsers.add_parser("tokens", help="Token budget commands")
    tokens_parser.add_argument(
        "token_action",
        choices=["status", "pace", "reserve", "warn", "block", "thresholds", "history", "sync", "configure"],
        help="Token action: status, pace, reserve, warn, block, thresholds, history, sync, configure",
    )
    tokens_parser.add_argument("--limit", "-n", type=int, help="Limit for history entries")
    tokens_parser.add_argument("--plan", help="Plan preset: free, pro, max, api, unlimited, custom")
    tokens_parser.add_argument("--budget", type=int, help="Custom budget in tokens (for --plan custom)")
    # Threshold arguments (for warn/block/reserve)
    tokens_parser.add_argument("--percent", dest="threshold_percent", type=float, help="Threshold as percent (reserve or warn/block)")
    tokens_parser.add_argument("--used", dest="threshold_used", type=float, help="Warn/block when X percent used")
    tokens_parser.add_argument("--remaining", dest="threshold_remaining", type=float, help="Warn/block when X percent remaining")
    tokens_parser.add_argument("--tokens", dest="threshold_tokens", type=int, help="Threshold in tokens")
    tokens_parser.add_argument("--dollars", dest="threshold_dollars", type=float, help="Threshold in dollars")
    tokens_parser.add_argument("--mode", dest="reserve_mode", choices=["soft", "hard"], help="Reserve mode: soft (warn) or hard (block)")
    tokens_parser.add_argument("--json", action="store_true")

    # pirq turbo
    turbo_parser = subparsers.add_parser("turbo", help="Turbo mode - burn remaining tokens before reset")
    turbo_parser.add_argument(
        "turbo_action",
        choices=["status", "on", "off", "set"],
        help="Turbo action: status, on (enable), off (disable), set (configure)",
    )
    turbo_parser.add_argument("--days", type=int, help="Activate X days before reset")
    turbo_parser.add_argument("--hours", type=int, help="Activate X hours before reset")
    turbo_parser.add_argument("--min-remaining", dest="min_remaining", type=float, help="Minimum percent remaining to activate")
    turbo_parser.add_argument("--allow-reserve", dest="allow_reserve", action="store_true", help="Allow turbo to dip into reserve")
    turbo_parser.add_argument("--json", action="store_true")

    # pirq logs
    logs_parser = subparsers.add_parser("logs", help="Log inspection commands")
    logs_parser.add_argument(
        "logs_action",
        choices=["show", "verify", "audit"],
        help="Log action: show (sessions), verify (audit integrity), audit (command trail)",
    )
    logs_parser.add_argument("session_id", nargs="?", help="Session ID to show (for 'show' action)")
    logs_parser.add_argument("--limit", "-n", type=int, help="Limit entries")
    logs_parser.add_argument("--json", action="store_true")

    # pirq gates
    gates_parser = subparsers.add_parser("gates", help="Gate inspection commands")
    gates_parser.add_argument("gate_name", nargs="?", help="Specific gate to inspect")
    gates_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    # Load config and create orchestrator
    try:
        config = load_config()
        orch = Orchestrator(config)
    except Exception as e:
        print(f"Error initializing PIRQ: {e}", file=sys.stderr)
        return 1

    # Route to command handler
    if args.command is None or args.command == "status":
        return cmd_status(args, orch)
    elif args.command == "check":
        return cmd_check(args, orch)
    elif args.command == "run":
        return cmd_run(args, orch)
    elif args.command == "lock":
        return cmd_lock(args, orch)
    elif args.command == "init":
        return cmd_init(args, orch)
    elif args.command == "rollback":
        return cmd_rollback(args, orch)
    elif args.command == "git":
        return cmd_git(args, orch)
    elif args.command == "tokens":
        return cmd_tokens(args, orch)
    elif args.command == "turbo":
        return cmd_turbo(args, orch)
    elif args.command == "logs":
        return cmd_logs(args, orch)
    elif args.command == "gates":
        return cmd_gates(args, orch)
    elif args.command == "loop":
        return cmd_loop(args, orch)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
