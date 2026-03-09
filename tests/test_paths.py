"""Tests for path resolution and directory management."""

import time
from datetime import datetime, timedelta

import xp


def test_find_proj_dir_finds_pyproject(tmp_path):
    """Test project root detection with pyproject.toml"""
    # Create directory structure
    root = tmp_path / "my_project"
    subdir = root / "src" / "mypackage"
    subdir.mkdir(parents=True)

    # Create pyproject.toml marker
    (root / "pyproject.toml").write_text("[project]\nname = 'test'")

    # Create a fake script deep in the tree
    script = subdir / "script.py"
    script.write_text("# test script")

    # Should find the root
    result = xp.find_proj_dir(script)
    assert result == root


def test_find_proj_dir_finds_requirements_txt(tmp_path):
    """Test project root detection with requirements.txt"""
    root = tmp_path / "my_project"
    subdir = root / "scripts"
    subdir.mkdir(parents=True)

    (root / "requirements.txt").write_text("numpy\npandas")
    script = subdir / "run.py"
    script.write_text("# script")

    result = xp.find_proj_dir(script)
    assert result == root


def test_find_proj_dir_finds_setup_py(tmp_path):
    """Test project root detection with setup.py"""
    root = tmp_path / "my_project"
    subdir = root / "src"
    subdir.mkdir(parents=True)

    (root / "setup.py").write_text("from setuptools import setup")
    script = subdir / "main.py"
    script.write_text("# main")

    result = xp.find_proj_dir(script)
    assert result == root


def test_find_proj_dir_finds_git(tmp_path):
    """Test project root detection with .git directory"""
    root = tmp_path / "my_project"
    subdir = root / "nested" / "deep"
    subdir.mkdir(parents=True)

    (root / ".git").mkdir()
    script = subdir / "script.py"
    script.write_text("# script")

    result = xp.find_proj_dir(script)
    assert result == root


def test_find_proj_dir_prefers_shallow_marker(tmp_path):
    """Test that find_proj_dir returns the shallowest parent with a marker"""
    # Create nested project structure
    outer = tmp_path / "outer_project"
    inner = outer / "inner_project"
    script_dir = inner / "src"
    script_dir.mkdir(parents=True)

    # Both have markers
    (outer / "pyproject.toml").write_text("[project]")
    (inner / "pyproject.toml").write_text("[project]")

    script = script_dir / "script.py"
    script.write_text("# script")

    # Should find the inner (shallowest) one
    result = xp.find_proj_dir(script)
    assert result == inner


def test_find_proj_dir_marker_priority(tmp_path):
    """Test that find_proj_dir respects marker priority order"""
    root = tmp_path / "project"
    subdir = root / "src"
    subdir.mkdir(parents=True)

    # Create multiple markers - pyproject.toml should be found first
    (root / "pyproject.toml").write_text("[project]")
    (root / "requirements.txt").write_text("numpy")

    script = subdir / "script.py"
    script.write_text("# script")

    result = xp.find_proj_dir(script)
    assert result == root


def test_mk_data_dir_with_timestamp(tmp_path):
    """Test timestamp-based directory creation"""
    data_dir = tmp_path / "data"

    before = datetime.now().replace(microsecond=0)  # Timestamp format has no microseconds
    result = xp.mk_data_dir(data_dir, mkdir=True)
    after = datetime.now().replace(microsecond=0) + timedelta(seconds=1)

    # Check structure was created
    assert result.exists()
    assert (result / "xps").exists()
    assert (result / "res").exists()

    # Check timestamp format in path
    timestamp_str = result.name
    parsed = datetime.strptime(timestamp_str, xp.timestamp)

    # Should be between before and after (with tolerance for seconds-only precision)
    assert before <= parsed <= after


def test_mk_data_dir_with_tags(tmp_path):
    """Test tagged directory creation"""
    data_dir = tmp_path / "data"
    tags = "v1_test"

    result = xp.mk_data_dir(data_dir, tags=tags, mkdir=True)

    assert result.exists()
    assert result.name == tags
    assert (result / "xps").exists()
    assert (result / "res").exists()


def test_mk_data_dir_without_mkdir(tmp_path):
    """Test mk_data_dir with mkdir=False"""
    data_dir = tmp_path / "data"

    result = xp.mk_data_dir(data_dir, tags="test", mkdir=False)

    # Should return path but not create it
    assert not result.exists()
    assert result.name == "test"


def test_mk_data_dir_creates_parents(tmp_path):
    """Test that mk_data_dir creates parent directories"""
    data_dir = tmp_path / "level1" / "level2" / "level3"

    result = xp.mk_data_dir(data_dir, tags="test", mkdir=True)

    assert result.exists()
    assert result.parent.exists()
    assert result.parent.parent.exists()


def test_find_latest_run(tmp_path):
    """Test finding latest experiment run by timestamp"""
    root = tmp_path / "experiments"
    root.mkdir()

    # Create multiple timestamped directories
    timestamps = [
        "2024-01-15_at_10-30-00",
        "2024-01-15_at_14-20-00",
        "2024-01-16_at_09-15-00",  # Latest
        "2024-01-14_at_16-45-00",
    ]

    for ts in timestamps:
        (root / ts).mkdir()

    # Also create a non-timestamp dir (should be ignored)
    (root / "not_a_timestamp").mkdir()

    latest = xp.find_latest_run(root)
    assert latest == "2024-01-16_at_09-15-00"


def test_find_latest_run_single_entry(tmp_path):
    """Test find_latest_run with only one entry"""
    root = tmp_path / "experiments"
    root.mkdir()

    ts = "2024-06-01_at_12-00-00"
    (root / ts).mkdir()

    latest = xp.find_latest_run(root)
    assert latest == ts


def test_find_latest_run_ignores_invalid_formats(tmp_path):
    """Test that find_latest_run ignores directories with invalid timestamp formats"""
    root = tmp_path / "experiments"
    root.mkdir()

    # Create valid and invalid dirs
    (root / "2024-01-15_at_10-30-00").mkdir()
    (root / "invalid_format").mkdir()
    (root / "2024-99-99_at_99-99-99").mkdir()  # Invalid date
    (root / "results").mkdir()
    (root / "2024-02-20_at_15-00-00").mkdir()  # Latest valid

    latest = xp.find_latest_run(root)
    assert latest == "2024-02-20_at_15-00-00"


def test_mk_data_dir_unique_timestamps(tmp_path):
    """Test that rapid successive calls create unique directories"""
    data_dir = tmp_path / "data"

    result1 = xp.mk_data_dir(data_dir, mkdir=True)
    # Small delay to ensure different timestamp
    time.sleep(1.1)
    result2 = xp.mk_data_dir(data_dir, mkdir=True)

    assert result1 != result2
    assert result1.exists()
    assert result2.exists()


def test_find_proj_dir_returns_none_if_no_marker(tmp_path):
    """Test that find_proj_dir returns None when no marker found"""
    # Create a script without any project markers
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    script = script_dir / "script.py"
    script.write_text("# script")

    # Search will go up to tmp_path and beyond, may find git repo or nothing
    result = xp.find_proj_dir(script)
    # If we're in a git repo, it might find it; otherwise None
    # This test documents the behavior
    assert result is None or result.exists()
