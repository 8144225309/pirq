"""PIRQ Session Logger - Detailed per-session logging.

Maintains detailed records of each PIRQ session including:
- Full prompts
- Gate results
- Execution details
- Files touched
- Token usage
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import uuid4


@dataclass
class SessionData:
    """Session data being built during execution."""
    session_id: str
    started_at: str
    ended_at: Optional[str] = None
    duration_ms: int = 0

    prompt: str = ""
    prompt_hash: str = ""
    cwd: str = ""
    model: str = "claude"

    gates_all_clear: bool = True
    gates_results: Dict[str, Any] = field(default_factory=dict)

    prompt_num: Optional[int] = None
    git_tag_pre: Optional[str] = None
    git_tag_post: Optional[str] = None
    exit_code: Optional[int] = None
    orch_tag: Optional[str] = None
    orch_summary: Optional[str] = None

    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    tokens_source: str = "unknown"

    files_touched: List[Dict[str, Any]] = field(default_factory=list)
    output_hash: Optional[str] = None
    error: Optional[str] = None


class SessionLogger:
    """Detailed session logging with full context."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir / "sessions"
        self._ensure_dir()
        self._active_sessions: Dict[str, SessionData] = {}

    def _ensure_dir(self) -> None:
        """Ensure log directory exists."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_file(self, session_id: str) -> Path:
        """Get path to session log file."""
        # Format: YYYYMMDD_HHMMSS_uuid.json
        return self.log_dir / f"{session_id}.json"

    def start_session(
        self,
        prompt: str,
        cwd: Path,
        model: str = "claude",
    ) -> str:
        """Start a new session.

        Args:
            prompt: The prompt being executed
            cwd: Current working directory
            model: Model being used

        Returns:
            session_id for tracking
        """
        now = datetime.utcnow()
        session_id = f"{now:%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        prompt_hash = f"sha256:{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"

        session = SessionData(
            session_id=session_id,
            started_at=now.isoformat() + "Z",
            prompt=prompt,
            prompt_hash=prompt_hash,
            cwd=str(cwd),
            model=model,
        )

        self._active_sessions[session_id] = session

        # Write initial entry immediately
        self._write_session(session)

        return session_id

    def log_gate_results(
        self,
        session_id: str,
        all_clear: bool,
        results: Dict[str, Any],
    ) -> None:
        """Log gate check results."""
        session = self._active_sessions.get(session_id)
        if not session:
            return

        session.gates_all_clear = all_clear
        session.gates_results = {
            name: {
                "status": r.status.value if hasattr(r, 'status') else r.get('status'),
                "message": r.message if hasattr(r, 'message') else r.get('message'),
            }
            for name, r in results.items()
        }

        self._write_session(session)

    def log_execution_start(
        self,
        session_id: str,
        prompt_num: int,
        git_tag_pre: Optional[str] = None,
    ) -> None:
        """Log when Claude invocation begins."""
        session = self._active_sessions.get(session_id)
        if not session:
            return

        session.prompt_num = prompt_num
        session.git_tag_pre = git_tag_pre

        self._write_session(session)

    def log_execution_end(
        self,
        session_id: str,
        exit_code: int,
        output: str,
        error: Optional[str] = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
        tokens_source: str = "unknown",
        orch_tag: Optional[str] = None,
        orch_summary: Optional[str] = None,
        git_tag_post: Optional[str] = None,
    ) -> None:
        """Log when Claude invocation ends."""
        session = self._active_sessions.get(session_id)
        if not session:
            return

        session.exit_code = exit_code
        session.output_hash = f"sha256:{hashlib.sha256(output.encode()).hexdigest()[:16]}"
        session.error = error
        session.tokens_input = tokens_input
        session.tokens_output = tokens_output
        session.tokens_total = tokens_input + tokens_output
        session.tokens_source = tokens_source
        session.orch_tag = orch_tag
        session.orch_summary = orch_summary
        session.git_tag_post = git_tag_post

        self._write_session(session)

    def log_file_change(
        self,
        session_id: str,
        path: str,
        action: str,
        lines_changed: int = 0,
    ) -> None:
        """Log file modifications."""
        session = self._active_sessions.get(session_id)
        if not session:
            return

        session.files_touched.append({
            "path": path,
            "action": action,
            "lines_changed": lines_changed,
        })

        self._write_session(session)

    def finalize_session(
        self,
        session_id: str,
        exit_code: Optional[int] = None,
    ) -> None:
        """Close out the session log."""
        session = self._active_sessions.get(session_id)
        if not session:
            return

        now = datetime.utcnow()
        session.ended_at = now.isoformat() + "Z"

        # Calculate duration
        start = datetime.fromisoformat(session.started_at.replace("Z", "+00:00")).replace(tzinfo=None)
        session.duration_ms = int((now - start).total_seconds() * 1000)

        if exit_code is not None:
            session.exit_code = exit_code

        self._write_session(session)

        # Remove from active sessions
        del self._active_sessions[session_id]

    def _write_session(self, session: SessionData) -> None:
        """Write session data to file."""
        data = {
            "session_id": session.session_id,
            "started_at": session.started_at,
            "ended_at": session.ended_at,
            "duration_ms": session.duration_ms,
            "invocation": {
                "prompt": session.prompt,
                "prompt_hash": session.prompt_hash,
                "cwd": session.cwd,
                "model": session.model,
            },
            "gates": {
                "all_clear": session.gates_all_clear,
                "results": session.gates_results,
            },
            "execution": {
                "prompt_num": session.prompt_num,
                "git_tag_pre": session.git_tag_pre,
                "git_tag_post": session.git_tag_post,
                "exit_code": session.exit_code,
                "orch_tag": session.orch_tag,
                "orch_summary": session.orch_summary,
            },
            "tokens": {
                "input": session.tokens_input,
                "output": session.tokens_output,
                "total": session.tokens_total,
                "source": session.tokens_source,
            },
            "files_touched": session.files_touched,
            "output_hash": session.output_hash,
            "error": session.error,
        }

        try:
            session_file = self._get_session_file(session.session_id)
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except IOError:
            pass  # Best effort

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session log by ID."""
        session_file = self._get_session_file(session_id)
        if not session_file.exists():
            return None

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return None

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent sessions."""
        sessions = []

        try:
            # Get all session files, sorted by modification time
            files = sorted(
                self.log_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            for session_file in files[:limit]:
                try:
                    with open(session_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # Return summary info
                        sessions.append({
                            "session_id": data.get("session_id"),
                            "started_at": data.get("started_at"),
                            "duration_ms": data.get("duration_ms"),
                            "prompt_hash": data.get("invocation", {}).get("prompt_hash"),
                            "exit_code": data.get("execution", {}).get("exit_code"),
                            "orch_tag": data.get("execution", {}).get("orch_tag"),
                            "tokens_total": data.get("tokens", {}).get("total"),
                        })
                except (IOError, json.JSONDecodeError):
                    continue

        except IOError:
            pass

        return sessions

    def get_active_session(self, session_id: str) -> Optional[SessionData]:
        """Get active session data (for in-progress sessions)."""
        return self._active_sessions.get(session_id)
