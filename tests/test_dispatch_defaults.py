"""Tests for `dispatch._fill_fun_defaults()`: backfilling `fun`'s own default values into
`xp` dicts that omit them, so `results.dicts2index()` records what actually ran instead of
mislabeling it as missing (see `results/__init__.py`'s `dicts2index()` docstring)."""

import pytest

from mmorpg.dispatch import _fill_fun_defaults


def experiment(x=1, y=None, z="default"):
    return x, y, z


def test_fills_missing_keys_with_fun_defaults():
    assert _fill_fun_defaults(experiment, [{}]) == [{"x": 1, "y": None, "z": "default"}]


def test_preserves_explicitly_given_values():
    result = _fill_fun_defaults(experiment, [{"x": 99, "z": "custom"}])
    assert result == [{"x": 99, "y": None, "z": "custom"}]


def test_supports_ragged_inputs():
    """Different `xp` dicts may omit different keys -- each is bound/defaulted independently."""
    inputs = [{"x": 1}, {}, {"y": 3, "z": "c"}]
    result = _fill_fun_defaults(experiment, inputs)
    assert result == [
        {"x": 1, "y": None, "z": "default"},
        {"x": 1, "y": None, "z": "default"},
        {"x": 1, "y": 3, "z": "c"},
    ]


def test_leaves_unmatched_keys_untouched_per_item():
    """A key that doesn't match any parameter (e.g. a typo) is left as-is rather than
    raising -- it'll surface as a per-item crash when `fun(**xp)` is actually called
    (caught by `local_mp.mp(log_errors=True)`), same as before this helper existed."""
    inputs = [{"typo": 1}]
    assert _fill_fun_defaults(experiment, inputs) == inputs


def test_raises_if_fun_accepts_kwargs():
    def fun_kwargs(x=1, **kw):
        return x, kw

    with pytest.raises(TypeError, match="kwargs"):
        _fill_fun_defaults(fun_kwargs, [{"x": 1}])


def test_raises_if_fun_signature_uninspectable():
    # Most builtin/C-implemented callables have no introspectable signature.
    with pytest.raises(TypeError, match="inspect"):
        _fill_fun_defaults(dict.update, [{}])
