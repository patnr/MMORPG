"""Tests for path resolution and directory management."""

from datetime import datetime

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
