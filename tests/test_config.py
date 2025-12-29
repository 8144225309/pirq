"""Tests for PIRQ configuration."""

import pytest
from pirq.config import (
    Config,
    TokensConfig,
    TurboConfig,
    load_config,
    DEFAULT_CONFIG,
    PLAN_PRESETS,
)


class TestTokensConfig:
    """Tests for TokensConfig dataclass."""

    def test_default_values(self):
        """Test default token configuration values."""
        config = TokensConfig()
        assert config.budget == 50000
        assert config.warn_at_percent_used == 80.0
        assert config.block_at_percent_used == 95.0
        assert config.reserve_percent == 5.0
        assert config.reserve_mode == "soft"

    def test_custom_values(self):
        """Test custom token configuration."""
        config = TokensConfig(
            budget=1000000,
            warn_at_percent_used=70.0,
            block_at_percent_used=90.0,
        )
        assert config.budget == 1000000
        assert config.warn_at_percent_used == 70.0
        assert config.block_at_percent_used == 90.0


class TestTurboConfig:
    """Tests for TurboConfig dataclass."""

    def test_default_values(self):
        """Test default turbo configuration values."""
        config = TurboConfig()
        assert config.enabled is True
        assert config.activate_days_before_reset == 3
        assert config.min_remaining_percent == 20.0
        assert config.allow_reserve_dip is False

    def test_custom_values(self):
        """Test custom turbo configuration."""
        config = TurboConfig(
            enabled=False,
            activate_days_before_reset=5,
            min_remaining_percent=30.0,
        )
        assert config.enabled is False
        assert config.activate_days_before_reset == 5
        assert config.min_remaining_percent == 30.0


class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self):
        """Test that default config has expected structure."""
        config = Config()
        assert hasattr(config, 'tokens')
        assert hasattr(config, 'turbo')
        assert hasattr(config, 'backup')
        assert hasattr(config, 'session')

    def test_to_dict(self):
        """Test config serialization to dict."""
        config = Config()
        data = config.to_dict()
        assert 'tokens' in data
        assert 'turbo' in data
        assert 'backup' in data


class TestPlanPresets:
    """Tests for plan presets."""

    def test_presets_exist(self):
        """Test that expected plan presets exist."""
        assert 'free' in PLAN_PRESETS
        assert 'pro' in PLAN_PRESETS
        assert 'max' in PLAN_PRESETS
        assert 'api' in PLAN_PRESETS
        assert 'unlimited' in PLAN_PRESETS

    def test_preset_values(self):
        """Test that preset values are reasonable."""
        assert PLAN_PRESETS['free'] >= 0
        assert PLAN_PRESETS['pro'] > PLAN_PRESETS['free']
        assert PLAN_PRESETS['unlimited'] < 0  # -1 means unlimited


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG constant."""

    def test_structure(self):
        """Test that DEFAULT_CONFIG has expected keys."""
        assert 'tokens' in DEFAULT_CONFIG
        assert 'turbo' in DEFAULT_CONFIG
        assert 'backup' in DEFAULT_CONFIG
        assert 'session' in DEFAULT_CONFIG

    def test_tokens_section(self):
        """Test tokens section of DEFAULT_CONFIG."""
        tokens = DEFAULT_CONFIG['tokens']
        assert 'budget' in tokens
        assert 'warn_at_percent_used' in tokens
        assert 'block_at_percent_used' in tokens
