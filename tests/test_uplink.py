"""Tests for remote uplink functionality."""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from xp.uplink import Uplink


def test_uplink_init_defaults():
    """Test Uplink initialization with default parameters"""
    ul = Uplink("testhost")

    assert ul.host == "testhost"
    assert ul.progbar is False
    assert ul.dry is False
    assert ul.use_M is True


def test_uplink_init_custom():
    """Test Uplink initialization with custom parameters"""
    ul = Uplink("myhost", progbar=True, dry=True, use_M=False)

    assert ul.host == "myhost"
    assert ul.progbar is True
    assert ul.dry is True
    assert ul.use_M is False


def test_uplink_repr():
    """Test Uplink string representation"""
    ul = Uplink("testhost", progbar=True, dry=False)

    repr_str = repr(ul)
    assert "testhost" in repr_str
    assert "progbar=True" in repr_str
    assert "dry=False" in repr_str


def test_uplink_ssh_command_construction_unix():
    """Test SSH command construction on Unix-like systems"""
    with patch("os.name", "posix"):
        ul = Uplink("testhost")

        assert "ssh" in ul.ssh_M
        assert "ControlMaster=auto" in ul.ssh_M
        assert "ControlPath=~/.ssh/" in ul.ssh_M
        assert "ControlPersist=1m" in ul.ssh_M


def test_uplink_ssh_command_construction_windows():
    """Test SSH command construction on Windows"""
    with patch("os.name", "nt"):
        ul = Uplink("testhost")

        assert "ssh" in ul.ssh_M
        assert "USERPROFILE" in ul.ssh_M or "~/.ssh/" in ul.ssh_M


def test_uplink_cmd_string_command(monkeypatch):
    """Test cmd() with string command"""
    ul = Uplink("testhost")

    mock_run = Mock(return_value=Mock(stdout="output", stderr=""))
    monkeypatch.setattr(subprocess, "run", mock_run)

    ul.cmd("echo hello")

    # Check subprocess.run was called
    assert mock_run.called
    call_args = mock_run.call_args[0][0]

    # Should contain ssh command
    assert "ssh" in call_args
    assert "testhost" in call_args
    # Command should be wrapped in bash -l -c
    assert any("bash -l -c" in str(arg) for arg in call_args)


def test_uplink_cmd_list_command(monkeypatch):
    """Test cmd() with list command"""
    ul = Uplink("testhost")

    mock_run = Mock(return_value=Mock(stdout="output", stderr=""))
    monkeypatch.setattr(subprocess, "run", mock_run)

    ul.cmd(["ls", "-la", "/tmp"])

    assert mock_run.called
    # List should be joined into string
    call_args = mock_run.call_args[0][0]
    assert any("ls -la /tmp" in str(arg) for arg in call_args)


def test_uplink_cmd_no_login_shell(monkeypatch):
    """Test cmd() without login shell"""
    ul = Uplink("testhost")

    mock_run = Mock(return_value=Mock(stdout="output", stderr=""))
    monkeypatch.setattr(subprocess, "run", mock_run)

    ul.cmd("pwd", login_shell=False)

    assert mock_run.called
    call_args = mock_run.call_args[0][0]
    # Should NOT have bash -l -c wrapper
    assert not any("bash -l -c" in str(arg) for arg in call_args)


def test_uplink_cmd_error_handling(monkeypatch):
    """Test cmd() error handling"""
    ul = Uplink("testhost")

    error = subprocess.CalledProcessError(1, "cmd", stderr="error message")
    mock_run = Mock(side_effect=error)
    monkeypatch.setattr(subprocess, "run", mock_run)

    with pytest.raises(subprocess.CalledProcessError):
        ul.cmd("false")


def test_uplink_rsync_dry_run():
    """Test rsync in dry-run mode"""
    ul = Uplink("testhost", dry=True)

    cmd = ul.rsync("/local/src", "/remote/dst")

    # In dry mode, should return command string
    assert isinstance(cmd, str)
    assert "rsync" in cmd
    assert "-azhL" in cmd
    assert "/local/src" in cmd
    assert "testhost:/remote/dst" in cmd


def test_uplink_rsync_with_opts():
    """Test rsync with additional options"""
    ul = Uplink("testhost", dry=True)

    cmd = ul.rsync("/src", "/dst", opts="--delete --exclude=*.pyc")

    assert "--delete" in cmd
    assert "--exclude=*.pyc" in cmd


def test_uplink_rsync_with_opts_list():
    """Test rsync with options as list"""
    ul = Uplink("testhost", dry=True)

    cmd = ul.rsync("/src", "/dst", opts=["--delete", "--exclude=*.pyc"])

    assert "--delete" in cmd
    assert "--exclude=*.pyc" in cmd


def test_uplink_rsync_reverse():
    """Test rsync in reverse mode (download)"""
    ul = Uplink("testhost", dry=True)

    cmd = ul.rsync("/local/dst", "/remote/src", reverse=True)

    # Source should be remote, destination local
    assert "testhost:/remote/src" in cmd
    assert "/local/dst" in cmd
    # Remote should come before local in rsync command
    remote_pos = cmd.index("testhost:/remote/src")
    local_pos = cmd.index("/local/dst")
    assert remote_pos < local_pos


def test_uplink_rsync_multiplex_enabled():
    """Test rsync with multiplexing enabled"""
    ul = Uplink("testhost", dry=True, use_M=True)

    cmd = ul.rsync("/src", "/dst")

    assert "-e" in cmd
    assert "ssh" in cmd


def test_uplink_rsync_multiplex_disabled():
    """Test rsync with multiplexing disabled"""
    ul = Uplink("testhost", dry=True, use_M=False)

    cmd = ul.rsync("/src", "/dst")

    # Should not have -e ssh option
    assert cmd.count("-e") == 0 or "ssh" not in cmd.split("-e")[1].split()[0]


def test_uplink_rsync_progress_bar(monkeypatch):
    """Test rsync with progress bar enabled"""
    ul = Uplink("testhost", dry=True, progbar=True)

    # Mock rsync version check
    mock_run = Mock(return_value=Mock(stdout="rsync  version 3.2.3  protocol version 31"))
    monkeypatch.setattr(subprocess, "run", mock_run)

    cmd = ul.rsync("/src", "/dst")

    # Should include progress options for rsync >= 3.1
    assert "--info=progress2" in cmd or "--progress" in cmd


def test_uplink_rsync_executes(monkeypatch):
    """Test that rsync actually executes when dry=False"""
    ul = Uplink("testhost", dry=False)

    mock_run = Mock(return_value=Mock(stdout="rsync  version 3.2.3  protocol version 31"))
    monkeypatch.setattr(subprocess, "run", mock_run)

    result = ul.rsync("/src", "/dst")

    # Should return None in non-dry mode
    assert result is None
    # subprocess.run should be called (at least for version check and rsync)
    assert mock_run.call_count >= 2


def test_uplink_sym_sync_context_manager(monkeypatch):
    """Test sym_sync context manager basic flow"""
    ul = Uplink("testhost", dry=True)

    mock_cmd = Mock()
    mock_rsync = Mock(return_value="rsync command")

    ul.cmd = mock_cmd
    ul.rsync = mock_rsync

    with ul.sym_sync(Path("/remote/target"), Path("/local/source")):
        # Inside context
        pass

    # Should call mkdir on remote
    assert any("mkdir" in str(call) for call in mock_cmd.call_args_list)

    # Should rsync at least twice (upload and download)
    assert mock_rsync.call_count >= 2


def test_uplink_sym_sync_upload_download(monkeypatch):
    """Test sym_sync uploads and downloads correctly"""
    ul = Uplink("testhost", dry=True)

    rsync_calls = []

    def mock_rsync(src, dst, reverse=False):
        rsync_calls.append({"src": src, "dst": dst, "reverse": reverse})
        return "mock rsync"

    ul.cmd = Mock()
    ul.rsync = mock_rsync

    with ul.sym_sync(Path("/remote/target"), Path("/local/source")):
        pass

    # Check upload happened (reverse=False or not specified)
    uploads = [c for c in rsync_calls if not c.get("reverse", False)]
    assert len(uploads) >= 1

    # Check download happened (reverse=True)
    downloads = [c for c in rsync_calls if c.get("reverse", False)]
    assert len(downloads) >= 1


def test_uplink_sym_sync_with_additional_dirs(monkeypatch, tmp_path):
    """Test sym_sync with additional directories to sync"""
    ul = Uplink("testhost", dry=True)

    # Create test directories
    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()

    rsync_calls = []

    def mock_rsync(src, dst, reverse=False):
        rsync_calls.append({"src": str(src), "dst": str(dst), "reverse": reverse})
        return "mock rsync"

    ul.cmd = Mock()
    ul.rsync = mock_rsync

    with ul.sym_sync(Path("/remote/target"), Path("/local/source"), extra_dir):
        pass

    # Should have synced both source and extra_dir
    assert len(rsync_calls) >= 3  # source upload, extra upload, source download


def test_uplink_sym_sync_prevents_home_sync(tmp_path):
    """Test that sym_sync prevents syncing entire home directory"""
    ul = Uplink("testhost", dry=True)

    ul.cmd = Mock()
    ul.rsync = Mock(return_value="mock rsync")

    # Try to sync home directory - should raise ValueError
    with pytest.raises(ValueError, match="home dir"):
        with ul.sym_sync(Path("/remote/target"), Path("/local/source"), Path.home()):
            pass


def test_uplink_sym_sync_downloads_on_exception(monkeypatch):
    """Test that sym_sync downloads even if exception occurs"""
    ul = Uplink("testhost", dry=True)

    rsync_calls = []

    def mock_rsync(src, dst, reverse=False):
        rsync_calls.append({"reverse": reverse})
        return "mock rsync"

    ul.cmd = Mock()
    ul.rsync = mock_rsync

    try:
        with ul.sym_sync(Path("/remote/target"), Path("/local/source")):
            raise RuntimeError("Test error")
    except RuntimeError:
        pass

    # Should still download (reverse=True) even after exception
    downloads = [c for c in rsync_calls if c.get("reverse", False)]
    assert len(downloads) >= 1
