"""Tests for PIRQ CLI."""

import subprocess
import sys
import pytest


class TestCLIBasics:
    """Basic CLI smoke tests."""

    def test_version(self):
        """Test --version flag."""
        result = subprocess.run(
            [sys.executable, "-m", "pirq", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "pirq" in result.stdout.lower()

    def test_help(self):
        """Test --help flag."""
        result = subprocess.run(
            [sys.executable, "-m", "pirq", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()

    def test_status_help(self):
        """Test status subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "pirq", "status", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_check_help(self):
        """Test check subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "pirq", "check", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_tokens_help(self):
        """Test tokens subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "pirq", "tokens", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "status" in result.stdout
        assert "pace" in result.stdout

    def test_turbo_help(self):
        """Test turbo subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "pirq", "turbo", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "status" in result.stdout

    def test_invalid_command(self):
        """Test that invalid commands fail gracefully."""
        result = subprocess.run(
            [sys.executable, "-m", "pirq", "notacommand"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()


class TestRunCommand:
    """Tests for the run command and its options."""

    def test_run_help(self):
        """Test run subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "pirq", "run", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Check for output modes
        assert "--output" in result.stdout
        assert "brief" in result.stdout
        assert "normal" in result.stdout
        assert "full" in result.stdout
        assert "json" in result.stdout
        assert "raw" in result.stdout

    def test_run_has_claude_params(self):
        """Test run command has Claude CLI pass-through params."""
        result = subprocess.run(
            [sys.executable, "-m", "pirq", "run", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Tier 1 params
        assert "--max-turns" in result.stdout
        assert "--system-prompt" in result.stdout
        assert "--verbose" in result.stdout
        assert "--continue" in result.stdout
        assert "--resume" in result.stdout
        # Tier 2 params
        assert "--fallback" in result.stdout
        assert "--tools" in result.stdout
        assert "--no-tools" in result.stdout
        assert "--timeout" in result.stdout

    def test_run_has_quiet_mode(self):
        """Test run command has quiet mode."""
        result = subprocess.run(
            [sys.executable, "-m", "pirq", "run", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--quiet" in result.stdout or "-q" in result.stdout

    def test_run_requires_prompt(self):
        """Test run command requires a prompt."""
        result = subprocess.run(
            [sys.executable, "-m", "pirq", "run"],
            capture_output=True,
            text=True,
        )
        # Should fail because no prompt provided
        assert result.returncode != 0
        assert "prompt required" in result.stdout.lower() or "prompt" in result.stderr.lower()


class TestCLISubcommands:
    """Tests for CLI subcommands (may require .pirq directory)."""

    def test_gates_list(self):
        """Test gates command lists gates."""
        result = subprocess.run(
            [sys.executable, "-m", "pirq", "gates"],
            capture_output=True,
            text=True,
        )
        # Should succeed and show gate names
        assert result.returncode == 0
        assert "gates" in result.stdout.lower() or "backup" in result.stdout.lower()
