"""Tests for `results.dicts2index()`: how missing keys, `None`, and NaN values in `xps`
get folded into the `NONE` sentinel string -- and how `dispatch._fill_fun_defaults()` (see
`test_dispatch_defaults.py`) now keeps a bare missing key from reaching this in practice."""

import pytest

from mmorpg.dispatch import _fill_fun_defaults
from mmorpg.results import NONE, dicts2index


def test_missing_key_becomes_sentinel():
    xps = [{"a": 1, "b": 2}, {"a": 3}]  # 2nd dict omits "b"
    index = dicts2index(xps)
    assert index[1] == (3, NONE)


def test_explicit_none_becomes_sentinel():
    index = dicts2index([{"a": 1, "b": None}])
    assert index[0] == (1, NONE)


def test_nan_becomes_sentinel():
    index = dicts2index([{"a": 1, "b": float("nan")}])
    assert index[0] == (1, NONE)


def test_duplicates_are_dropped_with_warning():
    xps = [{"a": 1}, {"a": 1}, {"a": 2}]
    with pytest.warns(UserWarning, match="duplicate"):
        index = dicts2index(xps)
    assert len(index) == 2


def test_fully_specified_xps_have_no_sentinel():
    """Sanity check: when every `xp` shares the full key set with real values, nothing
    gets conflated into `NONE`."""
    index = dicts2index([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert NONE not in index.get_level_values("a")
    assert NONE not in index.get_level_values("b")


def test_fill_fun_defaults_closes_the_missing_key_gap():
    """End-to-end: an `xp` that omits a key, once normalized by `_fill_fun_defaults()` (as
    `dispatch()` does before saving), records the real default in the index -- not the
    `NONE` sentinel a bare missing key would otherwise produce."""

    def experiment(x=1, y=42):
        return x, y

    xps = [{"x": 1}, {"x": 2, "y": 99}]  # 1st omits "y"
    filled = _fill_fun_defaults(experiment, xps)
    index = dicts2index(filled)

    assert NONE not in index.get_level_values("y")
    assert index[0] == (1, 42)  # real default, not "NA"
    assert index[1] == (2, 99)
