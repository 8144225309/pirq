"""Session Gate - Prevents concurrent runs with atomic locking.

Uses a lock file with atomic file locking to track active sessions.
Supports multiple concurrent sessions with --force mode.
Handles stale locks from crashed processes.
Includes heartbeat tracking for hung process detection.
"""

import atexit
import json
import os
import sys
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

from .base import Gate, GateResult
from ..config import Config

# Heartbeat interval in seconds
HEARTBEAT_INTERVAL = 5
# Session considered stale if no heartbeat for this many seconds
STALE_THRESHOLD = 30


def _is_process_alive(pid: int) -> bool:
    """Check if a process with given PID is running."""
    if not pid:
        return False

    if sys.platform == "win32":
        import subprocess
        try:
            # tasklist /FI "PID eq <pid>" returns the process if it exists
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # If PID exists, output contains the PID number
            return str(pid) in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    else:
        # Unix-like: try to send signal 0
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class SessionGate(Gate):
    """Prevents concurrent PIRQ sessions with atomic locking."""

    name = "run_complete"

    def __init__(self, config: Config):
        super().__init__(config)
        self.lock_file = self._get_lock_path()
        self._mutex_file = self.lock_file.with_suffix('.lck')
        self._heartbeat_file = self.lock_file.with_suffix('.heartbeat')
        self._my_session_id: Optional[str] = None
        self._fd: Optional[int] = None  # File descriptor for mutex
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()
        atexit.register(self._cleanup_on_exit)

    def _get_lock_path(self) -> Path:
        """Get path to lock file."""
        pirq_dir = self.config.pirq_dir or Path.cwd() / ".pirq"
        return pirq_dir / "runtime" / "sessions.json"

    @contextmanager
    def _file_lock(self, timeout: float = 10.0):
        """Cross-platform atomic file locking context manager.

        Args:
            timeout: Max seconds to wait for lock (default 10)
        """
        self._mutex_file.parent.mkdir(parents=True, exist_ok=True)

        if sys.platform == "win32":
            import msvcrt
            # Windows: use msvcrt.locking with retry loop
            fd = os.open(str(self._mutex_file), os.O_RDWR | os.O_CREAT)
            try:
                # Retry loop for lock acquisition
                start = time.time()
                while True:
                    try:
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                        break  # Lock acquired
                    except OSError:
                        if time.time() - start > timeout:
                            raise TimeoutError(f"Could not acquire lock within {timeout}s")
                        time.sleep(0.1)  # Wait and retry
                try:
                    yield
                finally:
                    try:
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
            finally:
                os.close(fd)
        else:
            import fcntl
            # Unix: use fcntl.flock for exclusive lock
            with open(self._mutex_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _read_lock_data(self) -> Optional[Dict[str, Any]]:
        """Read lock file data."""
        if not self.lock_file.exists():
            return None
        try:
            return json.loads(self.lock_file.read_text())
        except (json.JSONDecodeError, IOError):
            return None

    def _write_lock_data(self, data: Dict[str, Any]) -> None:
        """Write lock file data."""
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file.write_text(json.dumps(data, indent=2))

    def _clean_stale_sessions(self, sessions: List[Dict]) -> List[Dict]:
        """Remove sessions from dead processes or stale heartbeats."""
        active = []
        now = datetime.utcnow()

        for s in sessions:
            pid = s.get("pid", 0)

            # Check if process is alive
            if not _is_process_alive(pid):
                continue

            # Check heartbeat staleness
            last_heartbeat = s.get("last_heartbeat")
            if last_heartbeat:
                try:
                    hb_time = datetime.fromisoformat(last_heartbeat.replace("Z", ""))
                    age = (now - hb_time).total_seconds()
                    if age > STALE_THRESHOLD:
                        # Process alive but heartbeat stale - likely hung
                        continue
                except (ValueError, TypeError):
                    pass  # Invalid timestamp, keep session

            active.append(s)

        return active

    def _clean_lock(self) -> None:
        """Remove the lock file and heartbeat file."""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except IOError:
            pass  # Best effort
        try:
            if self._heartbeat_file.exists():
                self._heartbeat_file.unlink()
        except IOError:
            pass

    def _start_heartbeat(self) -> None:
        """Start background heartbeat thread."""
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="pirq-heartbeat"
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        """Stop heartbeat thread."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)
            self._heartbeat_thread = None

    def _heartbeat_loop(self) -> None:
        """Background loop that updates heartbeat."""
        while not self._heartbeat_stop.wait(timeout=HEARTBEAT_INTERVAL):
            self._update_heartbeat()

    def _update_heartbeat(self) -> None:
        """Update our session's heartbeat timestamp."""
        if not self._my_session_id:
            return

        try:
            with self._file_lock(timeout=2):
                data = self._read_lock_data()
                if not data:
                    return

                now = datetime.utcnow().isoformat() + "Z"

                # Update our session's heartbeat
                for s in data.get("sessions", []):
                    if s.get("id") == self._my_session_id:
                        s["last_heartbeat"] = now
                        break

                data["last_updated"] = now
                self._write_lock_data(data)
        except (TimeoutError, Exception):
            pass  # Best effort - don't crash on heartbeat failure

    def check(self) -> GateResult:
        """Check if another session is running.

        Detection:
        1. Acquire atomic file lock
        2. Check if sessions exist
        3. Clean stale sessions from dead PIDs
        4. Return status
        """
        try:
            with self._file_lock():
                data = self._read_lock_data()
                if not data or not data.get("sessions"):
                    return GateResult.clear(
                        "No active session",
                        {"session_active": False}
                    )

                # Clean stale sessions
                active = self._clean_stale_sessions(data["sessions"])

                # Update file if we cleaned any
                if len(active) != len(data["sessions"]):
                    if active:
                        data["sessions"] = active
                        data["active_count"] = len(active)
                        data["last_updated"] = datetime.utcnow().isoformat() + "Z"
                        self._write_lock_data(data)
                    else:
                        self._clean_lock()

                if active:
                    info = active[0]
                    return GateResult.block(
                        f"Session running (PID {info['pid']}, task: {info.get('task', 'unknown')})",
                        {
                            "session_active": True,
                            "active_count": len(active),
                            "sessions": active,
                        }
                    )

                return GateResult.clear(
                    "No active session",
                    {"session_active": False}
                )
        except Exception as e:
            # On lock failure, assume clear to avoid deadlocks
            return GateResult.clear(
                f"No active session (lock check failed: {e})",
                {"session_active": False, "error": str(e)}
            )

    def acquire(self, task: str = "unknown", force: bool = False) -> bool:
        """Atomically acquire the session lock.

        Args:
            task: Task name for logging
            force: If True, allow multiple sessions

        Returns:
            True if lock acquired, False if blocked
        """
        try:
            with self._file_lock():
                data = self._read_lock_data() or {"sessions": [], "active_count": 0}

                # Clean stale sessions
                data["sessions"] = self._clean_stale_sessions(data["sessions"])

                # Check if blocked (unless force)
                if data["sessions"] and not force:
                    return False

                # Generate unique session ID
                now = datetime.utcnow()
                session_id = f"{os.getpid()}-{now.timestamp()}"
                self._my_session_id = session_id

                # Add our session with heartbeat
                data["sessions"].append({
                    "id": session_id,
                    "pid": os.getpid(),
                    "started": now.isoformat() + "Z",
                    "last_heartbeat": now.isoformat() + "Z",
                    "task": task,
                    "force": force,
                    "hostname": os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "unknown")),
                })
                data["active_count"] = len(data["sessions"])
                data["last_updated"] = now.isoformat() + "Z"

                self._write_lock_data(data)

            # Start heartbeat AFTER releasing file lock
            self._start_heartbeat()
            return True
        except Exception:
            return False

    def release(self) -> None:
        """Release our session from the lock."""
        # Stop heartbeat first
        self._stop_heartbeat()

        if not self._my_session_id:
            return

        try:
            with self._file_lock():
                data = self._read_lock_data()
                if not data:
                    self._my_session_id = None
                    return

                # Remove only OUR session
                original_count = len(data.get("sessions", []))
                data["sessions"] = [
                    s for s in data.get("sessions", [])
                    if s.get("id") != self._my_session_id
                ]

                if data["sessions"]:
                    data["active_count"] = len(data["sessions"])
                    data["last_updated"] = datetime.utcnow().isoformat() + "Z"
                    self._write_lock_data(data)
                else:
                    # No sessions left, clean up
                    self._clean_lock()

                self._my_session_id = None
        except Exception:
            self._my_session_id = None

    def _cleanup_on_exit(self) -> None:
        """Ensure lock and heartbeat are released on process exit."""
        self._stop_heartbeat()
        self.release()

    def get_lock_info(self) -> Optional[Dict[str, Any]]:
        """Get info about current lock, if any."""
        try:
            with self._file_lock():
                data = self._read_lock_data()
                if not data:
                    return None

                # Clean stale and return
                data["sessions"] = self._clean_stale_sessions(data.get("sessions", []))
                data["active_count"] = len(data["sessions"])

                if not data["sessions"]:
                    return None

                return data
        except Exception:
            # Fallback: try to read without lock
            if not self.lock_file.exists():
                return None
            try:
                return json.loads(self.lock_file.read_text())
            except (json.JSONDecodeError, IOError):
                return {"error": "corrupted lock file"}
