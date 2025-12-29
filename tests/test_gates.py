"""Tests for PIRQ gates."""

import pytest
from pirq.config import Config
from pirq.gates.base import Gate, GateResult, GateStatus


class TestGateStatus:
    """Tests for GateStatus enum."""

    def test_status_values(self):
        """Test that expected status values exist."""
        assert GateStatus.CLEAR.value == "clear"
        assert GateStatus.WARN.value == "warn"
        assert GateStatus.BLOCK.value == "block"


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_clear_result(self):
        """Test creating a clear result."""
        result = GateResult(
            status=GateStatus.CLEAR,
            message="All good",
        )
        assert result.status == GateStatus.CLEAR
        assert result.message == "All good"
        assert result.is_blocked is False
        assert result.is_warning is False

    def test_warn_result(self):
        """Test creating a warning result."""
        result = GateResult(
            status=GateStatus.WARN,
            message="Watch out",
        )
        assert result.status == GateStatus.WARN
        assert result.is_blocked is False
        assert result.is_warning is True

    def test_block_result(self):
        """Test creating a block result."""
        result = GateResult(
            status=GateStatus.BLOCK,
            message="Stopped",
        )
        assert result.status == GateStatus.BLOCK
        assert result.is_blocked is True
        assert result.is_warning is False

    def test_result_with_data(self):
        """Test result with additional data."""
        result = GateResult(
            status=GateStatus.CLEAR,
            message="OK",
            data={"tokens": 1000, "budget": 5000},
        )
        assert result.data["tokens"] == 1000
        assert result.data["budget"] == 5000

    def test_to_dict(self):
        """Test serializing result to dict."""
        result = GateResult(
            status=GateStatus.CLEAR,
            message="Test",
            data={"key": "value"},
        )
        d = result.to_dict()
        assert d["status"] == "clear"
        assert d["message"] == "Test"
        assert d["data"]["key"] == "value"


class TestGateBase:
    """Tests for Gate base class."""

    def test_gate_requires_implementation(self):
        """Test that Gate.check() must be implemented."""
        config = Config()

        class IncompleteGate(Gate):
            name = "incomplete"

        gate = IncompleteGate(config)
        with pytest.raises(NotImplementedError):
            gate.check()

    def test_gate_is_enabled_default(self):
        """Test that gates are enabled by default."""
        config = Config()

        class TestGate(Gate):
            name = "test"

            def check(self):
                return GateResult(status=GateStatus.CLEAR, message="OK")

        gate = TestGate(config)
        assert gate.is_enabled() is True
