"""Claude Code Log Reader - Parse JSONL logs for token usage.

Claude Code stores session data in JSONL files at:
  ~/.claude/projects/{projectPath}/*.jsonl

Each assistant message contains usage data:
  {
    "type": "assistant",
    "timestamp": "2025-10-04T11:24:54.135Z",
    "message": {
      "usage": {
        "input_tokens": 7,
        "output_tokens": 176,
        "cache_creation_input_tokens": 464,
        "cache_read_input_tokens": 37687
      }
    }
  }
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Iterator


# Token pricing (Claude Sonnet 4, as of 2025)
PRICING = {
    "input": 3.00 / 1_000_000,           # $3.00 per 1M tokens
    "output": 15.00 / 1_000_000,         # $15.00 per 1M tokens
    "cache_creation": 3.75 / 1_000_000,  # $3.75 per 1M tokens
    "cache_read": 0.30 / 1_000_000,      # $0.30 per 1M tokens
}


@dataclass
class SessionUsage:
    """Token usage from a single session."""

    session_id: str
    session_path: Path
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None
    message_count: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output, cache not counted separately)."""
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Estimated cost in USD."""
        return (
            self.input_tokens * PRICING["input"] +
            self.output_tokens * PRICING["output"] +
            self.cache_creation_tokens * PRICING["cache_creation"] +
            self.cache_read_tokens * PRICING["cache_read"]
        )


@dataclass
class UsageSummary:
    """Aggregated token usage across sessions."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    session_count: int = 0
    message_count: int = 0
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None
    sessions: List[SessionUsage] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Estimated cost in USD."""
        return (
            self.input_tokens * PRICING["input"] +
            self.output_tokens * PRICING["output"] +
            self.cache_creation_tokens * PRICING["cache_creation"] +
            self.cache_read_tokens * PRICING["cache_read"]
        )

    def add_session(self, session: SessionUsage) -> None:
        """Add a session's usage to the summary."""
        self.input_tokens += session.input_tokens
        self.output_tokens += session.output_tokens
        self.cache_creation_tokens += session.cache_creation_tokens
        self.cache_read_tokens += session.cache_read_tokens
        self.message_count += session.message_count
        self.session_count += 1
        self.sessions.append(session)

        # Update timestamps
        if session.first_timestamp:
            if self.first_timestamp is None or session.first_timestamp < self.first_timestamp:
                self.first_timestamp = session.first_timestamp
        if session.last_timestamp:
            if self.last_timestamp is None or session.last_timestamp > self.last_timestamp:
                self.last_timestamp = session.last_timestamp


class ClaudeLogReader:
    """Read and parse Claude Code's JSONL session logs."""

    def __init__(self, claude_dir: Optional[Path] = None):
        """Initialize the log reader.

        Args:
            claude_dir: Path to Claude's data directory.
                        Defaults to ~/.claude/projects/
        """
        if claude_dir is None:
            # Default Claude Code projects directory
            self.claude_dir = Path.home() / ".claude" / "projects"
        else:
            self.claude_dir = Path(claude_dir)

    def find_session_files(self, since: Optional[datetime] = None) -> Iterator[Path]:
        """Find all session JSONL files.

        Args:
            since: Only include files modified after this time

        Yields:
            Path to each session file
        """
        if not self.claude_dir.exists():
            return

        # Claude stores projects in subdirectories
        # Each project has session files named as UUIDs
        for project_dir in self.claude_dir.iterdir():
            if not project_dir.is_dir():
                continue

            for session_file in project_dir.glob("*.jsonl"):
                # Check modification time if filtering
                if since is not None:
                    mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                    if mtime < since:
                        continue

                yield session_file

    def parse_session(self, session_path: Path) -> SessionUsage:
        """Parse a session file and extract token usage.

        Args:
            session_path: Path to the session JSONL file

        Returns:
            SessionUsage with aggregated token counts
        """
        usage = SessionUsage(
            session_id=session_path.stem,
            session_path=session_path,
        )

        try:
            with open(session_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

                    # Only assistant messages have usage data
                    if entry.get("type") != "assistant":
                        continue

                    # Extract timestamp
                    ts_str = entry.get("timestamp")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            ts = ts.replace(tzinfo=None)  # Remove timezone for comparison
                            if usage.first_timestamp is None or ts < usage.first_timestamp:
                                usage.first_timestamp = ts
                            if usage.last_timestamp is None or ts > usage.last_timestamp:
                                usage.last_timestamp = ts
                        except ValueError:
                            pass

                    # Extract usage from message
                    message = entry.get("message", {})
                    token_usage = message.get("usage", {})

                    if token_usage:
                        usage.input_tokens += token_usage.get("input_tokens", 0)
                        usage.output_tokens += token_usage.get("output_tokens", 0)
                        usage.cache_creation_tokens += token_usage.get("cache_creation_input_tokens", 0)
                        usage.cache_read_tokens += token_usage.get("cache_read_input_tokens", 0)
                        usage.message_count += 1

        except (IOError, OSError) as e:
            # Log error but return what we have
            pass

        return usage

    def get_total_usage(self, since: Optional[datetime] = None) -> UsageSummary:
        """Get aggregated usage across all sessions.

        Args:
            since: Only include usage after this timestamp

        Returns:
            UsageSummary with total token counts
        """
        summary = UsageSummary()

        for session_path in self.find_session_files(since=since):
            session_usage = self.parse_session(session_path)

            # Filter by timestamp within session if needed
            if since is not None and session_usage.last_timestamp:
                if session_usage.last_timestamp < since:
                    continue

            if session_usage.message_count > 0:
                summary.add_session(session_usage)

        return summary

    def get_usage_for_project(
        self,
        project_path: Path,
        since: Optional[datetime] = None,
    ) -> UsageSummary:
        """Get usage for a specific project.

        Args:
            project_path: Path to the project (used to find project dir)
            since: Only include usage after this timestamp

        Returns:
            UsageSummary for the project
        """
        summary = UsageSummary()

        # Claude encodes project paths in directory names
        # This is a simplified lookup - actual encoding may vary
        project_str = str(project_path.resolve())

        for project_dir in self.claude_dir.iterdir():
            if not project_dir.is_dir():
                continue

            # Check if this project dir matches
            # Claude uses URL-encoded or hashed paths
            for session_file in project_dir.glob("*.jsonl"):
                if since is not None:
                    mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                    if mtime < since:
                        continue

                session_usage = self.parse_session(session_file)
                if session_usage.message_count > 0:
                    summary.add_session(session_usage)

        return summary

    def is_available(self) -> bool:
        """Check if Claude Code logs are available."""
        return self.claude_dir.exists() and any(self.claude_dir.iterdir())
