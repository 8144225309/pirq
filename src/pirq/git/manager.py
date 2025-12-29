"""PIRQ Git Manager - Per-prompt commit management.

Handles:
- Pre-prompt tagging (save restore point)
- Post-prompt semantic commits
- Rollback to previous prompts
"""

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class GitStatus:
    """Current git repository status."""
    is_repo: bool
    has_changes: bool
    branch: str
    staged_count: int
    unstaged_count: int
    untracked_count: int
    last_commit: Optional[str]
    last_commit_hash: Optional[str]


@dataclass
class PromptTag:
    """A pirq prompt tag."""
    number: int
    hash: str
    timestamp: str
    summary: Optional[str]


class GitManager:
    """Manages git operations for per-prompt commits."""

    TAG_PREFIX = "pirq-prompt-"
    WIP_PREFIX = "wip: pirq checkpoint"

    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path.cwd()
        self._prompt_counter: Optional[int] = None

    def _run_git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command."""
        cmd = ["git", "-C", str(self.repo_path)] + list(args)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
        )

    def is_git_repo(self) -> bool:
        """Check if current directory is a git repository."""
        result = self._run_git("rev-parse", "--git-dir", check=False)
        return result.returncode == 0

    def get_status(self) -> GitStatus:
        """Get current repository status."""
        if not self.is_git_repo():
            return GitStatus(
                is_repo=False,
                has_changes=False,
                branch="",
                staged_count=0,
                unstaged_count=0,
                untracked_count=0,
                last_commit=None,
                last_commit_hash=None,
            )

        # Get branch
        branch_result = self._run_git("branch", "--show-current", check=False)
        branch = branch_result.stdout.strip() or "HEAD"

        # Get status counts
        status_result = self._run_git("status", "--porcelain", check=False)
        lines = status_result.stdout.strip().split("\n") if status_result.stdout.strip() else []

        staged = sum(1 for l in lines if l and l[0] in "MADRCT")
        unstaged = sum(1 for l in lines if l and len(l) > 1 and l[1] in "MADRCT")
        untracked = sum(1 for l in lines if l and l.startswith("??"))

        # Get last commit
        log_result = self._run_git("log", "-1", "--format=%H|%s", check=False)
        if log_result.returncode == 0 and log_result.stdout.strip():
            parts = log_result.stdout.strip().split("|", 1)
            last_hash = parts[0] if parts else None
            last_msg = parts[1] if len(parts) > 1 else None
        else:
            last_hash = None
            last_msg = None

        return GitStatus(
            is_repo=True,
            has_changes=bool(lines),
            branch=branch,
            staged_count=staged,
            unstaged_count=unstaged,
            untracked_count=untracked,
            last_commit=last_msg,
            last_commit_hash=last_hash,
        )

    def get_next_prompt_number(self) -> int:
        """Get the next prompt number based on existing tags."""
        if self._prompt_counter is not None:
            self._prompt_counter += 1
            return self._prompt_counter

        # Find highest existing prompt tag
        result = self._run_git("tag", "-l", f"{self.TAG_PREFIX}*", check=False)
        if result.returncode != 0 or not result.stdout.strip():
            self._prompt_counter = 1
            return 1

        tags = result.stdout.strip().split("\n")
        numbers = []
        for tag in tags:
            try:
                num = int(tag.replace(self.TAG_PREFIX, ""))
                numbers.append(num)
            except ValueError:
                continue

        self._prompt_counter = max(numbers) + 1 if numbers else 1
        return self._prompt_counter

    def on_prompt_start(self) -> Optional[int]:
        """Called before a prompt is executed.

        Creates a tag marking the pre-prompt state for rollback.

        Returns:
            Prompt number if successful, None if failed
        """
        if not self.is_git_repo():
            return None

        prompt_num = self.get_next_prompt_number()

        # Stage any changes and commit if needed (clean slate)
        status = self.get_status()
        if status.has_changes:
            # Create a pre-prompt checkpoint
            self._run_git("add", "-A", check=False)
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._run_git(
                "commit",
                "-m", f"{self.WIP_PREFIX} {timestamp} (pre-prompt {prompt_num})",
                check=False,
            )

        # Tag current state as prompt start point
        tag_name = f"{self.TAG_PREFIX}{prompt_num}-pre"
        self._run_git("tag", "-f", tag_name, check=False)

        return prompt_num

    def on_prompt_complete(
        self,
        prompt_num: int,
        summary: Optional[str] = None,
        success: bool = True,
    ) -> bool:
        """Called after a prompt completes.

        Creates a semantic commit with the prompt results.

        Args:
            prompt_num: The prompt number from on_prompt_start
            summary: Optional summary for commit message
            success: Whether the prompt succeeded

        Returns:
            True if commit was created
        """
        if not self.is_git_repo():
            return False

        status = self.get_status()
        if not status.has_changes:
            # No changes to commit, just tag
            tag_name = f"{self.TAG_PREFIX}{prompt_num}"
            self._run_git("tag", "-f", tag_name, check=False)
            return True

        # Stage all changes
        self._run_git("add", "-A", check=False)

        # Create semantic commit message
        if summary:
            msg = f"AI: {summary}"
        elif success:
            msg = f"AI: Prompt {prompt_num} completed"
        else:
            msg = f"AI: Prompt {prompt_num} (partial/failed)"

        # Commit
        result = self._run_git("commit", "-m", msg, check=False)
        if result.returncode != 0:
            return False

        # Tag the commit
        tag_name = f"{self.TAG_PREFIX}{prompt_num}"
        self._run_git("tag", "-f", tag_name, check=False)

        return True

    def rollback(self, prompt_num: Optional[int] = None) -> bool:
        """Rollback to a previous prompt state.

        Args:
            prompt_num: Prompt number to rollback to (None = previous)

        Returns:
            True if rollback succeeded
        """
        if not self.is_git_repo():
            return False

        if prompt_num is None:
            # Find the most recent prompt tag
            result = self._run_git("tag", "-l", f"{self.TAG_PREFIX}*", "--sort=-v:refname", check=False)
            if result.returncode != 0 or not result.stdout.strip():
                return False

            tags = result.stdout.strip().split("\n")
            # Filter to just numbered tags (not -pre tags)
            numbered = [t for t in tags if not t.endswith("-pre")]
            if len(numbered) < 2:
                return False

            # Get second most recent (rollback target)
            target_tag = numbered[1]
        else:
            target_tag = f"{self.TAG_PREFIX}{prompt_num}"

        # Check tag exists
        result = self._run_git("rev-parse", target_tag, check=False)
        if result.returncode != 0:
            return False

        # Hard reset to that tag
        result = self._run_git("reset", "--hard", target_tag, check=False)
        return result.returncode == 0

    def get_prompt_history(self, limit: int = 10) -> List[PromptTag]:
        """Get history of prompt commits.

        Returns:
            List of PromptTag objects
        """
        if not self.is_git_repo():
            return []

        result = self._run_git(
            "tag", "-l", f"{self.TAG_PREFIX}*",
            "--sort=-v:refname",
            "--format=%(refname:short)|%(objectname:short)|%(creatordate:iso)",
            check=False,
        )

        if result.returncode != 0 or not result.stdout.strip():
            return []

        tags = []
        for line in result.stdout.strip().split("\n")[:limit]:
            parts = line.split("|")
            if len(parts) < 3:
                continue

            tag_name = parts[0]
            if tag_name.endswith("-pre"):
                continue

            try:
                num = int(tag_name.replace(self.TAG_PREFIX, ""))
            except ValueError:
                continue

            # Get commit message for summary
            msg_result = self._run_git(
                "log", "-1", "--format=%s", tag_name,
                check=False,
            )
            summary = msg_result.stdout.strip() if msg_result.returncode == 0 else None

            tags.append(PromptTag(
                number=num,
                hash=parts[1],
                timestamp=parts[2],
                summary=summary,
            ))

        return tags

    def get_changes_since_prompt(self, prompt_num: int) -> Dict[str, Any]:
        """Get changes made since a specific prompt.

        Returns:
            Dict with files changed, insertions, deletions
        """
        if not self.is_git_repo():
            return {"error": "Not a git repository"}

        tag = f"{self.TAG_PREFIX}{prompt_num}"

        # Get diff stats
        result = self._run_git(
            "diff", "--stat", tag, "HEAD",
            check=False,
        )

        if result.returncode != 0:
            return {"error": f"Tag {tag} not found"}

        return {
            "since_prompt": prompt_num,
            "diff": result.stdout.strip(),
        }
