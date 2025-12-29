"""PIRQ Audit Logger - Tamper-evident command audit trail.

Maintains a chain-hashed log of all PIRQ invocations.
Each entry links to the previous via cryptographic hash.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import uuid4


@dataclass
class AuditEntry:
    """A single audit log entry."""
    entry_id: int
    uuid: str
    chain_hash: str
    timestamp_start: str
    timestamp_end: Optional[str]
    command: str
    prompt_hash: str
    session_id: str
    gates_passed: bool
    gates_blocked_by: List[str]
    exit_code: Optional[int]
    tokens_used: int
    files_changed: int
    orch_action: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "uuid": self.uuid,
            "chain_hash": self.chain_hash,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "command": self.command,
            "prompt_hash": self.prompt_hash,
            "session_id": self.session_id,
            "gates_passed": self.gates_passed,
            "gates_blocked_by": self.gates_blocked_by,
            "exit_code": self.exit_code,
            "tokens_used": self.tokens_used,
            "files_changed": self.files_changed,
            "orch_action": self.orch_action,
        }


class AuditLogger:
    """Tamper-evident audit trail with chain hashing.

    Each entry contains a hash linking to the previous entry.
    This allows verification that the log hasn't been modified.
    """

    GENESIS_HASH = "0" * 64  # Hash for first entry

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir / "audit"
        self.log_path = self.log_dir / "commands.jsonl"
        self._ensure_dir()
        self._last_hash: Optional[str] = None
        self._next_id: Optional[int] = None

    def _ensure_dir(self) -> None:
        """Ensure log directory exists."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_last_entry(self) -> Optional[Dict[str, Any]]:
        """Get the last entry from the log."""
        if not self.log_path.exists():
            return None

        last_line = None
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last_line = line
        except IOError:
            return None

        if last_line:
            try:
                return json.loads(last_line)
            except json.JSONDecodeError:
                return None
        return None

    def _get_last_hash(self) -> str:
        """Get the chain hash of the last entry."""
        if self._last_hash is not None:
            return self._last_hash

        last_entry = self._get_last_entry()
        if last_entry:
            self._last_hash = last_entry.get("chain_hash", self.GENESIS_HASH)
        else:
            self._last_hash = self.GENESIS_HASH

        return self._last_hash

    def _get_next_id(self) -> int:
        """Get the next entry ID."""
        if self._next_id is not None:
            result = self._next_id
            self._next_id += 1
            return result

        last_entry = self._get_last_entry()
        if last_entry:
            self._next_id = last_entry.get("entry_id", 0) + 1
        else:
            self._next_id = 1

        result = self._next_id
        self._next_id += 1
        return result

    def _compute_chain_hash(self, entry_data: Dict[str, Any]) -> str:
        """Compute hash linking to previous entry."""
        previous_hash = self._get_last_hash()
        # Create deterministic string from entry (excluding chain_hash itself)
        data_copy = {k: v for k, v in entry_data.items() if k != "chain_hash"}
        data_str = f"{previous_hash}|{json.dumps(data_copy, sort_keys=True)}"
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _write_entry(self, entry: Dict[str, Any]) -> None:
        """Append entry to log file."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            # Update cached last hash
            self._last_hash = entry.get("chain_hash")
        except IOError as e:
            # Log error but don't crash
            pass

    def log_command_start(
        self,
        command: str,
        prompt: str,
        session_id: str,
        gates_passed: bool,
        gates_blocked_by: Optional[List[str]] = None,
    ) -> int:
        """Log command start, return entry_id for later completion."""
        entry_id = self._get_next_id()
        prompt_hash = f"sha256:{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"

        entry_data = {
            "entry_id": entry_id,
            "uuid": str(uuid4()),
            "timestamp_start": datetime.utcnow().isoformat() + "Z",
            "timestamp_end": None,
            "command": command,
            "prompt_hash": prompt_hash,
            "session_id": session_id,
            "gates_passed": gates_passed,
            "gates_blocked_by": gates_blocked_by or [],
            "exit_code": None,
            "tokens_used": 0,
            "files_changed": 0,
            "orch_action": None,
        }

        # Compute chain hash
        entry_data["chain_hash"] = self._compute_chain_hash(entry_data)

        self._write_entry(entry_data)
        return entry_id

    def log_command_end(
        self,
        entry_id: int,
        exit_code: int,
        tokens_used: int = 0,
        files_changed: int = 0,
        orch_action: Optional[str] = None,
    ) -> None:
        """Log command completion by appending completion entry."""
        # We append a completion record rather than modifying the start record
        # This preserves the append-only nature of the log
        entry_data = {
            "entry_id": entry_id,
            "uuid": str(uuid4()),
            "timestamp_start": None,  # Indicates this is a completion record
            "timestamp_end": datetime.utcnow().isoformat() + "Z",
            "command": "_complete",
            "prompt_hash": "",
            "session_id": "",
            "gates_passed": True,
            "gates_blocked_by": [],
            "exit_code": exit_code,
            "tokens_used": tokens_used,
            "files_changed": files_changed,
            "orch_action": orch_action,
        }

        entry_data["chain_hash"] = self._compute_chain_hash(entry_data)
        self._write_entry(entry_data)

    def verify_chain(self) -> tuple:
        """Verify entire chain integrity.

        Returns:
            Tuple of (is_valid, broken_entries)
            - is_valid: True if chain is intact
            - broken_entries: List of entry_ids where chain broke
        """
        if not self.log_path.exists():
            return (True, [])

        broken = []
        previous_hash = self.GENESIS_HASH

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        broken.append(line_num)
                        continue

                    # Recompute what the chain hash should be
                    data_copy = {k: v for k, v in entry.items() if k != "chain_hash"}
                    data_str = f"{previous_hash}|{json.dumps(data_copy, sort_keys=True)}"
                    expected_hash = hashlib.sha256(data_str.encode()).hexdigest()

                    if entry.get("chain_hash") != expected_hash:
                        broken.append(entry.get("entry_id", line_num))

                    previous_hash = entry.get("chain_hash", expected_hash)

        except IOError:
            return (False, [-1])

        return (len(broken) == 0, broken)

    def get_recent_invocations(
        self,
        hours: Optional[int] = None,
        minutes: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent invocations within time window.

        Args:
            hours: Time window in hours
            minutes: Time window in minutes (overrides hours)

        Returns:
            List of entry dicts within the time window
        """
        if not self.log_path.exists():
            return []

        if minutes:
            cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        elif hours:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
        else:
            cutoff = datetime.utcnow() - timedelta(hours=1)

        results = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        # Only count start entries (not completion records)
                        if entry.get("timestamp_start"):
                            ts = datetime.fromisoformat(
                                entry["timestamp_start"].replace("Z", "+00:00")
                            ).replace(tzinfo=None)
                            if ts >= cutoff:
                                results.append(entry)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except IOError:
            pass

        return results

    def get_last_invocation(self) -> Optional[Dict[str, Any]]:
        """Get the most recent invocation."""
        recent = self.get_recent_invocations(hours=24)
        if recent:
            return recent[-1]
        return None

    def get_prompt_history(self, prompt_hash: str) -> List[Dict[str, Any]]:
        """Get all invocations with the same prompt hash."""
        if not self.log_path.exists():
            return []

        results = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("prompt_hash") == prompt_hash:
                            results.append(entry)
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass

        return results

    def get_entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent entries (most recent last)."""
        if not self.log_path.exists():
            return []

        entries = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except IOError:
            pass

        return entries[-limit:]
