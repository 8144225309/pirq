"""PIRQ Configuration System.

Zero-config defaults with layered overrides:
1. Built-in defaults (this file)
2. User global config (~/.config/pirq/config.json)
3. Project config (.pirq/config.json)
4. Environment variables (PIRQ_*)
5. CLI flags
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any


# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "backup": {
        "provider": "auto",  # auto, git, mercurial, external, disabled
        "require": True,
        "warn_cloud_only": True,
        "i_accept_the_risk": False,
    },
    "tokens": {
        "budget": 50000,
        "budget_type": "monthly",
        "reset_day": 1,
        # Thresholds (first non-zero wins)
        "warn_at_percent_used": 80.0,
        "block_at_percent_used": 95.0,
        # Reserve
        "reserve_percent": 5.0,
        "reserve_mode": "soft",
        # Pacing
        "pacing_enabled": True,
        "pace_warn_threshold": 150.0,
        "pace_block_threshold": 200.0,
    },
    "turbo": {
        "enabled": True,
        "activate_days_before_reset": 3,
        "activate_hours_before_reset": 0,
        "min_remaining_percent": 20.0,
        "allow_reserve_dip": False,
    },
    "session": {
        "stale_lock_action": "inform",  # clean, inform, stop
        "timeout_minutes": 60,
        "heartbeat_enabled": False,
        "heartbeat_interval_seconds": 60,
    },
    "git": {
        "commit_strategy": "per_prompt",  # per_prompt, checkpoint, manual
        "checkpoint_enabled": True,
        "checkpoint_interval_seconds": 30,
        "checkpoint_max_tool_calls": 5,
        "auto_tag_prompts": True,
        "squash_wip_on_complete": True,
        "commit_message_template": "AI: {summary}",
        "tag_prefix": "pirq-prompt-",
    },
    "pirqs": {
        "backup": {"enabled": True, "strictness": "required"},
        "tokens": {"enabled": True},
        "run_complete": {"enabled": True},
        "git_clean": {"enabled": True, "strictness": "warn"},
        "cooldown": {"enabled": False, "min_seconds": 30},
        "external_block": {"enabled": True},
        "validators": {"enabled": False, "run_tests": False, "run_lint": False, "run_build": False},
    },
    "self_manage": {
        "max_consecutive_continues": 10,
        "max_continues_per_hour": 30,
        "abort_on_no_progress": 3,
        "require_progress": True,
    },
    "alerts": {
        "console": True,
        "system_notification": False,
        "sound": False,
    },
}


@dataclass
class BackupConfig:
    provider: str = "auto"
    require: bool = True
    warn_cloud_only: bool = True
    i_accept_the_risk: bool = False


# Plan-based budget presets (tokens per month)
# These are estimates - Claude doesn't publish exact limits
PLAN_PRESETS = {
    "free": 0,           # No budget (blocked)
    "pro": 500_000,      # ~$20/month estimate
    "max": 2_500_000,    # 5x Pro estimate
    "api": 10_000_000,   # API users set their own
    "unlimited": -1,     # No limit (use -1 to indicate)
}


@dataclass
class TokensConfig:
    budget: int = 50000
    budget_type: str = "monthly"
    reset_day: int = 1
    plan: str = "custom"  # free, pro, max, api, unlimited, custom

    # WARN THRESHOLDS - first non-zero wins
    warn_at_percent_used: float = 80.0      # Warn when X% used (default)
    warn_at_percent_remaining: float = 0.0  # OR warn when X% remaining
    warn_at_tokens: int = 0                 # OR warn when X tokens left
    warn_at_dollars: float = 0.0            # OR warn when $X left

    # BLOCK THRESHOLDS - first non-zero wins
    block_at_percent_used: float = 95.0     # Block when X% used (default)
    block_at_percent_remaining: float = 0.0 # OR block when X% remaining
    block_at_tokens: int = 0                # OR block when X tokens left
    block_at_dollars: float = 0.0           # OR block when $X left

    # Legacy compatibility (maps to warn_at_percent_used/block_at_percent_used)
    warn_percent: int = 0   # Deprecated, use warn_at_percent_used
    block_percent: int = 0  # Deprecated, use block_at_percent_used

    # Reserve settings (emergency fund)
    reserve_tokens: int = 0           # Absolute reserve (if > 0, overrides percent)
    reserve_percent: float = 5.0      # % of budget to reserve (default 5%)
    reserve_dollars: float = 0.0      # Reserve in dollars (if > 0, overrides percent)
    reserve_mode: str = "soft"        # "soft" (warn) or "hard" (block)

    # Pacing settings
    pacing_enabled: bool = True       # Enable time-aware pacing
    pace_warn_threshold: float = 150.0   # Warn when pace % exceeds this
    pace_block_threshold: float = 200.0  # Block when pace % exceeds this


@dataclass
class TurboConfig:
    """Turbo mode - burn remaining tokens before reset."""
    enabled: bool = True
    activate_days_before_reset: int = 3     # Activate X days before reset
    activate_hours_before_reset: int = 0    # OR X hours before (first non-zero wins)
    min_remaining_percent: float = 20.0     # Only if >X% remaining
    allow_reserve_dip: bool = False         # Can turbo dip into reserve?


@dataclass
class SessionConfig:
    stale_lock_action: str = "inform"
    timeout_minutes: int = 60
    heartbeat_enabled: bool = False
    heartbeat_interval_seconds: int = 60


@dataclass
class GitConfig:
    commit_strategy: str = "per_prompt"
    checkpoint_enabled: bool = True
    checkpoint_interval_seconds: int = 30
    checkpoint_max_tool_calls: int = 5
    auto_tag_prompts: bool = True
    squash_wip_on_complete: bool = True
    commit_message_template: str = "AI: {summary}"
    tag_prefix: str = "pirq-prompt-"


@dataclass
class PIRQGateConfig:
    enabled: bool = True
    strictness: str = "warn"
    # Additional gate-specific fields (optional)
    min_seconds: int = 30  # For cooldown
    run_tests: bool = False  # For validators
    run_lint: bool = False  # For validators
    run_build: bool = False  # For validators


@dataclass
class PIRQsConfig:
    backup: PIRQGateConfig = field(default_factory=lambda: PIRQGateConfig(strictness="required"))
    tokens: PIRQGateConfig = field(default_factory=PIRQGateConfig)
    run_complete: PIRQGateConfig = field(default_factory=PIRQGateConfig)
    git_clean: PIRQGateConfig = field(default_factory=lambda: PIRQGateConfig(strictness="warn"))
    cooldown: PIRQGateConfig = field(default_factory=lambda: PIRQGateConfig(enabled=False))
    external_block: PIRQGateConfig = field(default_factory=PIRQGateConfig)
    validators: PIRQGateConfig = field(default_factory=lambda: PIRQGateConfig(enabled=False))


@dataclass
class SelfManageConfig:
    max_consecutive_continues: int = 10
    max_continues_per_hour: int = 30
    abort_on_no_progress: int = 3
    require_progress: bool = True


@dataclass
class Config:
    """Main PIRQ configuration."""

    backup: BackupConfig = field(default_factory=BackupConfig)
    tokens: TokensConfig = field(default_factory=TokensConfig)
    turbo: TurboConfig = field(default_factory=TurboConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    git: GitConfig = field(default_factory=GitConfig)
    pirqs: PIRQsConfig = field(default_factory=PIRQsConfig)
    self_manage: SelfManageConfig = field(default_factory=SelfManageConfig)

    # Runtime paths (set during load)
    project_root: Optional[Path] = None
    pirq_dir: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excluding paths)."""
        result = {}
        for key in ["backup", "tokens", "turbo", "session", "git", "self_manage"]:
            result[key] = asdict(getattr(self, key))
        # Handle pirqs specially
        result["pirqs"] = {}
        for gate_name in ["backup", "tokens", "run_complete", "git_clean", "cooldown", "external_block", "validators"]:
            result["pirqs"][gate_name] = asdict(getattr(self.pirqs, gate_name))
        return result


def _deep_merge(base: Dict, overlay: Dict) -> Dict:
    """Deep merge overlay into base, returning new dict."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _env_to_config_key(env_key: str) -> list:
    """Convert PIRQ_TOKENS_BUDGET to ['tokens', 'budget']."""
    if not env_key.startswith("PIRQ_"):
        return []
    parts = env_key[5:].lower().split("_")
    return parts


def _apply_env_vars(config: Dict) -> Dict:
    """Apply PIRQ_* environment variables to config."""
    result = config.copy()

    for key, value in os.environ.items():
        if not key.startswith("PIRQ_"):
            continue

        parts = _env_to_config_key(key)
        if len(parts) < 2:
            continue

        # Navigate to the right place
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value (try to parse as bool/int/float)
        final_value = value
        if value.lower() in ("true", "false"):
            final_value = value.lower() == "true"
        else:
            try:
                final_value = int(value)
            except ValueError:
                try:
                    final_value = float(value)
                except ValueError:
                    pass

        current[parts[-1]] = final_value

    return result


def _dict_to_config(data: Dict) -> Config:
    """Convert dictionary to Config object."""
    config = Config()

    if "backup" in data:
        config.backup = BackupConfig(**data["backup"])
    if "tokens" in data:
        config.tokens = TokensConfig(**data["tokens"])
    if "turbo" in data:
        config.turbo = TurboConfig(**data["turbo"])
    if "session" in data:
        config.session = SessionConfig(**data["session"])
    if "git" in data:
        config.git = GitConfig(**data["git"])
    if "self_manage" in data:
        config.self_manage = SelfManageConfig(**data["self_manage"])
    if "pirqs" in data:
        pirqs = PIRQsConfig()
        for gate_name in ["backup", "tokens", "run_complete", "git_clean", "cooldown", "external_block", "validators"]:
            if gate_name in data["pirqs"]:
                setattr(pirqs, gate_name, PIRQGateConfig(**data["pirqs"][gate_name]))
        config.pirqs = pirqs

    return config


def find_project_root(start: Optional[Path] = None) -> Optional[Path]:
    """Find project root by looking for .pirq or .git directory."""
    current = start or Path.cwd()

    while current != current.parent:
        if (current / ".pirq").exists():
            return current
        if (current / ".git").exists():
            return current
        current = current.parent

    return None


def load_config(
    project_root: Optional[Path] = None,
    user_config_path: Optional[Path] = None,
    project_config_path: Optional[Path] = None,
) -> Config:
    """Load configuration with layered overrides.

    Order (later overrides earlier):
    1. Built-in defaults
    2. User global config
    3. Project config
    4. Environment variables
    """
    # Start with defaults
    config_dict = DEFAULT_CONFIG.copy()

    # Find project root
    if project_root is None:
        project_root = find_project_root()

    # User global config
    user_config = user_config_path or Path.home() / ".config" / "pirq" / "config.json"
    if user_config.exists():
        try:
            user_data = json.loads(user_config.read_text())
            config_dict = _deep_merge(config_dict, user_data)
        except (json.JSONDecodeError, IOError):
            pass  # Ignore invalid config

    # Project config
    if project_root:
        proj_config = project_config_path or project_root / ".pirq" / "config.json"
        if proj_config.exists():
            try:
                proj_data = json.loads(proj_config.read_text())
                config_dict = _deep_merge(config_dict, proj_data)
            except (json.JSONDecodeError, IOError):
                pass  # Ignore invalid config

    # Environment variables
    config_dict = _apply_env_vars(config_dict)

    # Convert to Config object
    config = _dict_to_config(config_dict)

    # Set runtime paths
    config.project_root = project_root or Path.cwd()
    config.pirq_dir = config.project_root / ".pirq"

    return config


def ensure_pirq_dir(config: Config) -> None:
    """Ensure .pirq directory structure exists."""
    pirq_dir = config.pirq_dir
    if pirq_dir is None:
        pirq_dir = Path.cwd() / ".pirq"

    # Create directories
    (pirq_dir / "runtime").mkdir(parents=True, exist_ok=True)
    (pirq_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Create default config if missing
    config_file = pirq_dir / "config.json"
    if not config_file.exists():
        config_file.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
