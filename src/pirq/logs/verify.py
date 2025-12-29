"""PIRQ Log Verification - Check integrity of audit logs."""

from pathlib import Path
from typing import Tuple, List

from .audit import AuditLogger


def verify_audit_log(log_dir: Path) -> Tuple[bool, List[int], str]:
    """Verify the audit log chain integrity.

    Args:
        log_dir: Path to .pirq/logs directory

    Returns:
        Tuple of (is_valid, broken_entries, message)
    """
    audit = AuditLogger(log_dir)

    if not audit.log_path.exists():
        return (True, [], "No audit log exists yet")

    is_valid, broken = audit.verify_chain()

    if is_valid:
        entries = audit.get_entries(limit=1000)
        return (True, [], f"Audit log verified: {len(entries)} entries, chain intact")
    else:
        return (False, broken, f"TAMPERING DETECTED: Chain broken at entries {broken}")


def get_audit_summary(log_dir: Path) -> dict:
    """Get summary statistics from audit log.

    Args:
        log_dir: Path to .pirq/logs directory

    Returns:
        Dict with summary statistics
    """
    audit = AuditLogger(log_dir)

    if not audit.log_path.exists():
        return {
            "exists": False,
            "total_entries": 0,
            "chain_valid": True,
        }

    entries = audit.get_entries(limit=10000)
    is_valid, broken = audit.verify_chain()

    # Count by command type
    commands = {}
    total_tokens = 0
    total_files = 0

    for entry in entries:
        cmd = entry.get("command", "unknown")
        if cmd != "_complete":  # Skip completion records
            commands[cmd] = commands.get(cmd, 0) + 1

        total_tokens += entry.get("tokens_used", 0)
        total_files += entry.get("files_changed", 0)

    return {
        "exists": True,
        "total_entries": len(entries),
        "chain_valid": is_valid,
        "broken_entries": broken,
        "commands": commands,
        "total_tokens": total_tokens,
        "total_files_changed": total_files,
    }
