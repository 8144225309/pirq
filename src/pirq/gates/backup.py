"""Backup Gate - Primary safety gate.

Checks for version control (git, mercurial, svn) or cloud sync.
Blocks execution if no backup system is detected.

This is the PRIMARY gate - if we can't undo, we shouldn't proceed.

Enhanced features:
- Verify git is actually functional (not just exists)
- Check for uncommitted changes (configurable warn/block)
- Check last commit age (stale repo detection)
"""

import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .base import Gate, GateResult, GateStatus
from ..config import Config


class BackupGate(Gate):
    """Checks for version control or backup system."""

    name = "backup"

    def __init__(self, config: Config, cwd: Path = None):
        super().__init__(config)
        self.cwd = cwd or Path.cwd()

    def check(self) -> GateResult:
        """Check for backup/version control system.

        Detection hierarchy:
        1. Check for version control (.git, .hg, .svn)
        2. Check for cloud sync paths (Dropbox, OneDrive, Google Drive)
        3. Check config override (external, disabled)
        4. Block if nothing found and require=True
        """
        backup_config = self.config.backup

        # Check if user explicitly disabled (requires acknowledgment)
        if backup_config.provider == "disabled":
            if backup_config.i_accept_the_risk:
                return GateResult.warn(
                    "Backup disabled by user (risk acknowledged)",
                    {"provider": "disabled", "acknowledged": True}
                )
            else:
                return GateResult.block(
                    "Backup disabled without risk acknowledgment. "
                    "Set backup.i_accept_the_risk=true to proceed.",
                    {"provider": "disabled", "acknowledged": False}
                )

        # Check if user attests to external backup
        if backup_config.provider == "external":
            return GateResult.clear(
                "External backup (user attestation)",
                {"provider": "external"}
            )

        # Auto-detect version control
        if backup_config.provider in ("auto", "git"):
            git_root = self._find_git_root()
            if git_root:
                # Enhanced: Verify git is functional
                if getattr(backup_config, 'verify_on_check', True):
                    if not self._verify_git_works(git_root):
                        return GateResult.block(
                            "Git repository exists but is not functional. "
                            "Run 'git status' to diagnose.",
                            {"provider": "git", "root": str(git_root), "functional": False}
                        )

                # Enhanced: Check for uncommitted changes
                if getattr(backup_config, 'require_clean', False):
                    uncommitted = self._get_uncommitted_count(git_root)
                    if uncommitted > 0:
                        return GateResult.warn(
                            f"Uncommitted changes: {uncommitted} files. "
                            "Commit or stash before proceeding.",
                            {"provider": "git", "uncommitted_count": uncommitted}
                        )

                # Enhanced: Check last commit age (stale repo)
                warn_stale_days = getattr(backup_config, 'warn_stale_days', 7)
                if warn_stale_days > 0:
                    days_since = self._days_since_last_commit(git_root)
                    if days_since is not None and days_since > warn_stale_days:
                        return GateResult.warn(
                            f"No commits in {days_since} days - stale repo?",
                            {"provider": "git", "days_since_commit": days_since}
                        )

                return GateResult.clear(
                    "Git repository verified",
                    {"provider": "git", "root": str(git_root), "functional": True}
                )

        if backup_config.provider in ("auto", "mercurial"):
            if (self.cwd / ".hg").exists() or self._find_in_parents(".hg"):
                return GateResult.clear(
                    "Mercurial repository detected",
                    {"provider": "mercurial"}
                )

        if backup_config.provider == "auto":
            if (self.cwd / ".svn").exists() or self._find_in_parents(".svn"):
                return GateResult.clear(
                    "SVN repository detected",
                    {"provider": "svn"}
                )

        # Check for cloud sync paths
        cwd_str = self.cwd.as_posix().lower()
        cloud_providers = ["dropbox", "onedrive", "google drive", "icloud"]

        for provider in cloud_providers:
            if provider in cwd_str:
                if backup_config.warn_cloud_only:
                    return GateResult.warn(
                        f"Cloud sync detected ({provider}) - git recommended for better rollback",
                        {"provider": "cloud", "cloud_service": provider}
                    )
                else:
                    return GateResult.clear(
                        f"Cloud sync detected ({provider})",
                        {"provider": "cloud", "cloud_service": provider}
                    )

        # No backup detected
        if backup_config.require:
            return GateResult.block(
                "No backup system detected. Run 'git init' or configure backup settings. "
                "Set backup.require=false to bypass (not recommended).",
                {"provider": None}
            )
        else:
            return GateResult.warn(
                "No backup system detected (backup.require=false)",
                {"provider": None}
            )

    def _find_git_root(self) -> Path | None:
        """Find .git directory in current or parent directories."""
        current = self.cwd.resolve()

        while current != current.parent:
            git_dir = current / ".git"
            if git_dir.exists():
                return current
            current = current.parent

        return None

    def _find_in_parents(self, dirname: str) -> Path | None:
        """Find a directory in current or parent directories."""
        current = self.cwd.resolve()

        while current != current.parent:
            target = current / dirname
            if target.exists():
                return current
            current = current.parent

        return None

    def _verify_git_works(self, git_root: Path) -> bool:
        """Verify git is functional by running a simple command."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(git_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            # If git status works, the repo is functional
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def _get_uncommitted_count(self, git_root: Path) -> int:
        """Get count of uncommitted changes (staged + unstaged + untracked)."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(git_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return 0

            # Count non-empty lines (each represents a changed file)
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            return len(lines)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return 0

    def _days_since_last_commit(self, git_root: Path) -> Optional[int]:
        """Get number of days since the last commit."""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ct"],
                cwd=str(git_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None

            # Parse Unix timestamp
            timestamp = int(result.stdout.strip())
            last_commit = datetime.fromtimestamp(timestamp)
            delta = datetime.now() - last_commit
            return delta.days
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError):
            return None
