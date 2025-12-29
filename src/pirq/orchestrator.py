"""PIRQ Orchestrator - Main execution controller.

Manages the orchestration loop:
1. Check all PIRQ gates
2. If clear, invoke Claude
3. Parse output for ORCH tags
4. Handle results and update state
"""

import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from .config import Config, load_config, ensure_pirq_dir
from .gates import Gate, GateResult, GateStatus
from .gates.backup import BackupGate
from .gates.session import SessionGate
from .gates.tokens import TokenGate
from .gates.loop_detect import LoopDetectGate
from .gates.rate_limit import RateLimitGate
from .gates.task_state import TaskStateGate
from .git import GitManager
from .logs import AuditLogger, SessionLogger
from .orch import parse_orch_tags, get_final_action, OrchAction
from .state import StateManager, Semaphore
from .utils.logging import JsonlLogger


@dataclass
class RunResult:
    """Result of a Claude invocation."""

    success: bool
    output: str
    error: Optional[str]
    exit_code: int
    duration_ms: int
    tokens_used: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    orch_tag: Optional[str] = None


@dataclass
class OrchResult:
    """Result of an orchestration cycle."""

    blocked: bool
    gate_results: Dict[str, GateResult]
    run_result: Optional[RunResult] = None
    semaphore: Optional[Semaphore] = None


class Orchestrator:
    """Main PIRQ orchestrator."""

    def __init__(self, config: Optional[Config] = None, cwd: Optional[Path] = None):
        self.config = config or load_config()
        self.cwd = cwd or Path.cwd()

        # Ensure directories exist
        ensure_pirq_dir(self.config)

        # Initialize components
        pirq_dir = self.config.pirq_dir or self.cwd / ".pirq"
        self.pirq_dir = pirq_dir
        self.state_manager = StateManager(pirq_dir)
        self.logger = JsonlLogger(pirq_dir / "logs" / "orchestrator.jsonl")

        # Dual logging: audit trail + session details
        self.audit_logger = AuditLogger(pirq_dir / "logs")
        self.session_logger = SessionLogger(pirq_dir / "logs")

        # Initialize gates
        self.token_gate = TokenGate(self.config, pirq_dir, self.audit_logger)
        self.loop_gate = LoopDetectGate(self.config, pirq_dir)
        self.rate_gate = RateLimitGate(self.config, self.audit_logger)
        self.task_gate = TaskStateGate(self.config, self.audit_logger)
        self.session_gate = SessionGate(self.config)

        self.gates: List[Gate] = [
            BackupGate(self.config, self.cwd),
            self.token_gate,
            self.rate_gate,
            self.task_gate,
            self.loop_gate,
            self.session_gate,
        ]

        # Git manager for per-prompt commits
        self.git_manager = GitManager(self.cwd)

    def check_gates(self) -> Tuple[bool, Dict[str, GateResult]]:
        """Run all PIRQ gate checks.

        Returns:
            Tuple of (all_clear, results_dict)
        """
        results: Dict[str, GateResult] = {}
        all_clear = True

        for gate in self.gates:
            if not gate.is_enabled():
                results[gate.name] = GateResult.clear(f"{gate.name} disabled")
                continue

            result = gate.check()
            results[gate.name] = result

            # Log the check
            self.logger.log_pirq_check(
                gate.name,
                result.status.value,
                result.message,
            )

            if result.is_blocked:
                all_clear = False

        # Update semaphore
        semaphore = self.state_manager.update_from_gates(results)

        if not all_clear:
            blocked_gates = [name for name, r in results.items() if r.is_blocked]
            self.logger.log_pirq_blocked(
                blocked_gates,
                semaphore.reason or "Unknown",
            )

        return all_clear, results

    def run_once(
        self,
        prompt: str,
        task: Optional[str] = None,
        model: Optional[str] = None,
        force: bool = False,
        # Claude CLI pass-through parameters
        max_turns: Optional[int] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
        continue_session: bool = False,
        resume_session: Optional[str] = None,
        fallback_model: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        blocked_tools: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        last_session: bool = False,
        yolo: bool = False,
        auto_mode: bool = False,
    ) -> OrchResult:
        """Execute a single orchestration cycle.

        Args:
            prompt: The prompt to send to Claude
            task: Optional task name for logging
            model: Optional model override (haiku, sonnet, opus)
            force: If True, bypass gate blocks (with user confirmation)
            max_turns: Limit agentic turns (prevents infinite loops)
            system_prompt: Append to Claude's system prompt
            verbose: Enable verbose Claude output
            continue_session: Continue last conversation
            resume_session: Resume specific session by ID
            fallback_model: Fallback model if primary overloaded
            allowed_tools: List of allowed tools
            blocked_tools: List of blocked tools
            timeout: Timeout in seconds
            last_session: Resume the most recent session
            yolo: Auto-approve all permissions
            auto_mode: Full auto mode (yolo + no questions + default max-turns)

        Returns:
            OrchResult with gate results and run result
        """
        # Store pass-through params for _invoke_claude
        self._claude_params = {
            "max_turns": max_turns,
            "system_prompt": system_prompt,
            "verbose": verbose,
            "continue_session": continue_session,
            "resume_session": resume_session,
            "fallback_model": fallback_model,
            "allowed_tools": allowed_tools,
            "blocked_tools": blocked_tools,
            "timeout": timeout,
            "last_session": last_session,
            "yolo": yolo,
            "auto_mode": auto_mode,
        }

        # FIRST: Acquire session lock to prevent concurrent runs
        # This must happen BEFORE gate checks to avoid race conditions
        if not self.session_gate.acquire(task or "single-run", force=force):
            # Another session is running - return immediately
            return OrchResult(
                blocked=True,
                gate_results={"run_complete": GateResult.block(
                    "Another PIRQ session is running in this directory",
                    {"session_active": True}
                )},
                run_result=None,
                semaphore=self.state_manager.load_semaphore(),
            )

        # Everything below is wrapped in try/finally to ensure lock release
        run_result = None
        prompt_num = None
        files_changed = 0
        session_id = None
        audit_entry_id = None
        gate_results = {}

        try:
            # Start session logging
            session_id = self.session_logger.start_session(
                prompt=prompt,
                cwd=self.cwd,
                model=model or "claude",
            )

            # Set prompt for gates that need it
            self.task_gate.set_prompt(prompt)
            self.token_gate.set_prompt(prompt)

            # Check gates
            all_clear, gate_results = self.check_gates()
            blocked_by = [name for name, r in gate_results.items() if r.is_blocked]

            # Log gate results to session
            self.session_logger.log_gate_results(session_id, all_clear, gate_results)

            # Start audit trail entry
            audit_entry_id = self.audit_logger.log_command_start(
                command="run",
                prompt=prompt,
                session_id=session_id,
                gates_passed=all_clear,
                gates_blocked_by=blocked_by,
            )

            if not all_clear and not force:
                # Blocked by gates - will release lock in finally
                return OrchResult(
                    blocked=True,
                    gate_results=gate_results,
                    semaphore=self.state_manager.load_semaphore(),
                )
            elif not all_clear and force:
                # Force bypass - log to legacy logger
                self.logger.log("force_bypass", {
                    "gates_bypassed": blocked_by,
                    "prompt": prompt[:100],
                })
            # Log session start (legacy logger)
            self.logger.log_session_start(task)

            # Git: Tag pre-prompt state for rollback
            prompt_num = self.git_manager.on_prompt_start()
            git_tag_pre = f"pirq-prompt-{prompt_num}-pre" if prompt_num else None

            # Log execution start
            self.session_logger.log_execution_start(
                session_id,
                prompt_num=prompt_num or 0,
                git_tag_pre=git_tag_pre,
            )

            if prompt_num:
                self.logger.log("git_prompt_start", {"prompt_num": prompt_num})

            # Invoke Claude with pass-through parameters
            run_result = self._invoke_claude(prompt, model=model, **self._claude_params)

            # Git: Commit post-prompt changes
            git_tag_post = None
            if prompt_num:
                summary = self._extract_summary(run_result)
                committed = self.git_manager.on_prompt_complete(
                    prompt_num,
                    summary=summary,
                    success=run_result.success,
                )
                if committed:
                    git_tag_post = f"pirq-prompt-{prompt_num}"
                    self.logger.log("git_prompt_complete", {
                        "prompt_num": prompt_num,
                        "summary": summary,
                    })

            # Get files changed for logging
            git_status = self.git_manager.get_status()
            files_changed = git_status.staged_count + git_status.unstaged_count

            # Log execution end to session
            self.session_logger.log_execution_end(
                session_id,
                exit_code=run_result.exit_code,
                output=run_result.output,
                error=run_result.error,
                tokens_input=run_result.tokens_input,
                tokens_output=run_result.tokens_output,
                tokens_source="parsed" if run_result.tokens_used > 0 else "none",
                orch_tag=run_result.orch_tag,
                orch_summary=self._extract_summary(run_result),
                git_tag_post=git_tag_post,
            )

            # Record token usage
            if run_result.tokens_used > 0:
                self.token_gate.record_usage(
                    input_tokens=run_result.tokens_input,
                    output_tokens=run_result.tokens_output,
                    model="claude",
                    task=task,
                )

            # Record for loop detection
            self.loop_gate.record_run(
                output=run_result.output,
                error=run_result.error,
                files_changed=files_changed,
            )

            # Log run (legacy logger)
            self.logger.log_run_end(
                tokens_used=run_result.tokens_used,
                duration_ms=run_result.duration_ms,
                exit_code=run_result.exit_code,
                orch_tag=run_result.orch_tag,
            )

            return OrchResult(
                blocked=False,
                gate_results=gate_results,
                run_result=run_result,
                semaphore=self.state_manager.load_semaphore(),
            )

        finally:
            # Finalize all logging (handle None values from early exits)
            exit_code = run_result.exit_code if run_result else -1
            tokens = run_result.tokens_used if run_result else 0
            orch_action = run_result.orch_tag if run_result else None

            if session_id:
                self.session_logger.finalize_session(session_id, exit_code=exit_code)
            if audit_entry_id:
                self.audit_logger.log_command_end(
                    entry_id=audit_entry_id,
                    exit_code=exit_code,
                    tokens_used=tokens,
                    files_changed=files_changed,
                    orch_action=orch_action,
                )

            # Always release lock
            self.session_gate.release()
            self.logger.log_session_end()

    def _find_claude_cli(self) -> Optional[str]:
        """Find the Claude CLI executable.

        Returns:
            Path to claude executable, or None if not found
        """
        # Try shutil.which first (checks PATH)
        claude_path = shutil.which("claude")
        if claude_path:
            return claude_path

        # On Windows, check common npm global locations
        if os.name == 'nt':
            npm_paths = [
                Path.home() / "AppData" / "Roaming" / "npm" / "claude.cmd",
                Path.home() / "AppData" / "Roaming" / "npm" / "claude",
            ]
            for p in npm_paths:
                if p.exists():
                    return str(p)

        return None

    def _find_last_session(self) -> Optional[str]:
        """Find the most recent Claude session UUID.

        Scans ~/.claude/projects/ for .jsonl session files and returns
        the UUID of the most recently modified one.
        """
        claude_dir = Path.home() / ".claude" / "projects"
        if not claude_dir.exists():
            return None

        # Find all .jsonl files across all projects
        sessions = []
        for jsonl in claude_dir.glob("*/*.jsonl"):
            # Skip agent sessions (they have 'agent-' prefix)
            if jsonl.stem.startswith("agent-"):
                continue
            try:
                sessions.append((jsonl.stat().st_mtime, jsonl.stem))
            except OSError:
                continue

        if not sessions:
            return None

        # Return the most recent UUID
        sessions.sort(reverse=True)
        return sessions[0][1]

    def _invoke_claude(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_turns: Optional[int] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
        continue_session: bool = False,
        resume_session: Optional[str] = None,
        fallback_model: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        blocked_tools: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        last_session: bool = False,
        yolo: bool = False,
        auto_mode: bool = False,
    ) -> RunResult:
        """Invoke Claude with the given prompt.

        Args:
            prompt: The prompt to send
            model: Optional model override (haiku, sonnet, opus)
            max_turns: Limit agentic turns
            system_prompt: Append to system prompt
            verbose: Enable verbose output
            continue_session: Continue last conversation
            resume_session: Resume specific session by ID
            fallback_model: Fallback model if primary overloaded
            allowed_tools: List of allowed tools
            blocked_tools: List of blocked tools
            timeout: Timeout in seconds
            last_session: Resume the most recent session
            yolo: Auto-approve all permissions
            auto_mode: Full auto mode (yolo + no questions + default max-turns)

        Returns:
            RunResult with output and metadata
        """
        # Auto mode: enable yolo, set default max-turns, add system prompt
        if auto_mode:
            yolo = True
            if max_turns is None:
                max_turns = 50  # Prevent infinite loops
            auto_system = "Do not ask clarifying questions. Make reasonable assumptions and proceed autonomously."
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{auto_system}"
            else:
                system_prompt = auto_system
        # Find claude CLI
        claude_cli = self._find_claude_cli()
        if not claude_cli:
            return RunResult(
                success=False,
                output="",
                error="Claude CLI not found. Is 'claude' installed? Try: npm install -g @anthropic-ai/claude-code",
                exit_code=-1,
                duration_ms=0,
            )

        # Generate prompt hash for logging
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        self.logger.log_run_start(prompt_hash)

        start_time = time.time()

        try:
            # Build command with all parameters
            cmd = [
                claude_cli,
                "-p", prompt,
                "--output-format", "json",
            ]

            # Model selection
            if model:
                cmd.extend(["--model", model])

            # Tier 1 parameters
            if max_turns is not None:
                cmd.extend(["--max-turns", str(max_turns)])
            if system_prompt:
                cmd.extend(["--append-system-prompt", system_prompt])
            if verbose:
                cmd.append("--verbose")
            if continue_session:
                cmd.append("--continue")
            if resume_session:
                cmd.extend(["--resume", resume_session])

            # Tier 2 parameters
            if fallback_model:
                cmd.extend(["--fallback-model", fallback_model])
            if allowed_tools:
                for tool in allowed_tools:
                    cmd.extend(["--allowedTools", tool.strip()])
            if blocked_tools:
                for tool in blocked_tools:
                    cmd.extend(["--disallowedTools", tool.strip()])

            # Session shortcuts
            if last_session and not resume_session and not continue_session:
                # Find and resume the most recent session
                last_id = self._find_last_session()
                if last_id:
                    cmd.extend(["--resume", last_id])

            # YOLO mode - auto-approve all permissions
            if yolo:
                cmd.append("--dangerously-skip-permissions")

            # Determine timeout
            effective_timeout = timeout if timeout else (self.config.session.timeout_minutes * 60)

            # Run claude command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.cwd),
                timeout=effective_timeout,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            # Parse output for tokens (if JSON format)
            token_info = self._parse_token_usage(result.stdout)

            # Parse ORCH tag
            orch_tag = self._parse_orch_tag(result.stdout)

            return RunResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.stderr else None,
                exit_code=result.returncode,
                duration_ms=duration_ms,
                tokens_used=token_info["total"],
                tokens_input=token_info["input_tokens"],
                tokens_output=token_info["output_tokens"],
                orch_tag=orch_tag,
            )

        except subprocess.TimeoutExpired:
            duration_ms = int((time.time() - start_time) * 1000)
            return RunResult(
                success=False,
                output="",
                error="Timeout expired",
                exit_code=-1,
                duration_ms=duration_ms,
            )

        except FileNotFoundError:
            return RunResult(
                success=False,
                output="",
                error="Claude CLI not found. Is 'claude' in PATH?",
                exit_code=-1,
                duration_ms=0,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return RunResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                duration_ms=duration_ms,
            )

    def _parse_token_usage(self, output: str) -> Dict[str, int]:
        """Parse token usage from Claude output.

        Claude CLI with --output-format json returns:
        {
            "result": "...",
            "usage": {
                "input_tokens": N,
                "output_tokens": N,
                "cache_creation_input_tokens": N,
                "cache_read_input_tokens": N
            },
            ...
        }

        Returns dict with:
            input_tokens, output_tokens, cache_creation, cache_read, total
        """
        empty = {"input_tokens": 0, "output_tokens": 0, "cache_creation": 0, "cache_read": 0, "total": 0}

        if not output or not output.strip():
            return empty

        try:
            import json
            data = json.loads(output)
            if isinstance(data, dict):
                usage = data.get("usage", {})
                if usage:
                    result = {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_creation": usage.get("cache_creation_input_tokens", 0),
                        "cache_read": usage.get("cache_read_input_tokens", 0),
                    }
                    result["total"] = sum(result.values())
                    return result
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return empty

    def _parse_orch_tag(self, output: str) -> Optional[str]:
        """Parse ORCH tag from Claude output."""
        final_tag = get_final_action(output)
        if final_tag:
            return final_tag.action.value.upper()
        return None

    def _extract_summary(self, result: RunResult) -> Optional[str]:
        """Extract a summary for git commit from run result."""
        # Try to get summary from ORCH tag
        final_tag = get_final_action(result.output)
        if final_tag and final_tag.summary:
            return final_tag.summary[:50]  # Truncate to 50 chars

        # Use ORCH tag action if present
        if result.orch_tag:
            return f"Prompt {result.orch_tag.lower()}"

        # Default based on success
        return "Completed" if result.success else "Partial/failed"

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        semaphore = self.state_manager.load_semaphore()
        lock_info = self.session_gate.get_lock_info()

        return {
            "state": semaphore.state,
            "reason": semaphore.reason,
            "timestamp": semaphore.timestamp,
            "blocked_by": semaphore.blocked_by,
            "session_active": lock_info is not None,
            "session_info": lock_info,
            "pirqs": semaphore.pirqs,
        }
