"""PIRQ ORCH Tag Parser.

Parses ORCH tags embedded in Claude output for self-orchestration.

Tag format:
    <!-- ORCH:ACTION key="value" key2="value2" -->

Supported actions:
    CONTINUE - More work needed, continue with same context
    DONE     - Task complete
    WAIT     - Cannot proceed, needs human intervention
    BLOCKED  - Stuck, needs human help
    SPAWN    - Fork a subtask
    ABORT    - Stop execution entirely
    ROLLBACK - Revert to previous state
    PRIORITY - Suggest priority change
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class OrchAction(Enum):
    """ORCH tag action types."""

    CONTINUE = "continue"
    DONE = "done"
    WAIT = "wait"
    BLOCKED = "blocked"
    SPAWN = "spawn"
    ABORT = "abort"
    ROLLBACK = "rollback"
    PRIORITY = "priority"


@dataclass
class OrchTag:
    """Parsed ORCH tag."""

    action: OrchAction
    params: Dict[str, str] = field(default_factory=dict)
    raw: str = ""

    @property
    def summary(self) -> Optional[str]:
        """Get summary parameter if present."""
        return self.params.get("summary")

    @property
    def reason(self) -> Optional[str]:
        """Get reason parameter if present."""
        return self.params.get("reason")

    @property
    def priority(self) -> Optional[int]:
        """Get priority parameter if present."""
        p = self.params.get("priority")
        if p:
            try:
                return int(p)
            except ValueError:
                pass
        return None

    @property
    def task(self) -> Optional[str]:
        """Get task parameter (for SPAWN)."""
        return self.params.get("task")

    @property
    def prompt(self) -> Optional[str]:
        """Get prompt parameter (for SPAWN)."""
        return self.params.get("prompt")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "params": self.params,
        }


# Regex patterns
TAG_PATTERN = re.compile(
    r'<!--\s*ORCH:(\w+)\s*(.*?)\s*-->',
    re.IGNORECASE | re.DOTALL
)

PARAM_PATTERN = re.compile(
    r'(\w+)\s*=\s*["\']([^"\']*)["\']'
)


def _parse_params(param_str: str) -> Dict[str, str]:
    """Parse key="value" parameters from string."""
    params = {}
    for match in PARAM_PATTERN.finditer(param_str):
        key = match.group(1).lower()
        value = match.group(2)
        params[key] = value
    return params


def parse_orch_tags(text: str) -> List[OrchTag]:
    """Parse all ORCH tags from text.

    Args:
        text: Text to parse (Claude output)

    Returns:
        List of OrchTag objects found
    """
    tags = []

    for match in TAG_PATTERN.finditer(text):
        action_str = match.group(1).upper()
        param_str = match.group(2)

        try:
            action = OrchAction[action_str]
        except KeyError:
            continue  # Unknown action, skip

        params = _parse_params(param_str)

        tags.append(OrchTag(
            action=action,
            params=params,
            raw=match.group(0),
        ))

    return tags


def get_final_action(text: str) -> Optional[OrchTag]:
    """Get the final (most relevant) ORCH tag from text.

    If multiple tags exist, returns the last one that indicates
    a terminal state (DONE, ABORT, BLOCKED) or the last CONTINUE.

    Args:
        text: Text to parse

    Returns:
        The most relevant OrchTag, or None if no tags found
    """
    tags = parse_orch_tags(text)
    if not tags:
        return None

    # Priority: terminal states first
    terminal = [OrchAction.DONE, OrchAction.ABORT, OrchAction.BLOCKED, OrchAction.WAIT]

    # Check for terminal state (last one wins)
    for tag in reversed(tags):
        if tag.action in terminal:
            return tag

    # Otherwise return last tag
    return tags[-1]


def should_continue(text: str) -> bool:
    """Check if output indicates more work is needed.

    Args:
        text: Claude output text

    Returns:
        True if CONTINUE tag found (and no terminal state)
    """
    final = get_final_action(text)
    if final is None:
        return False
    return final.action == OrchAction.CONTINUE


def is_complete(text: str) -> bool:
    """Check if output indicates task is complete.

    Args:
        text: Claude output text

    Returns:
        True if DONE tag found
    """
    final = get_final_action(text)
    if final is None:
        return False
    return final.action == OrchAction.DONE


def needs_human(text: str) -> bool:
    """Check if output indicates human intervention needed.

    Args:
        text: Claude output text

    Returns:
        True if WAIT, BLOCKED, or ABORT tag found
    """
    final = get_final_action(text)
    if final is None:
        return False
    return final.action in [OrchAction.WAIT, OrchAction.BLOCKED, OrchAction.ABORT]
