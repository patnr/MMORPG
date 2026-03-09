import numpy as np

import xp


def test_dict_prod_basic():
    """Test basic cartesian product of parameters"""
    result = xp.dict_prod(a=[1, 2], b=["x", "y"])
    assert len(result) == 4
    assert {"a": 1, "b": "x"} in result
    assert {"a": 1, "b": "y"} in result
    assert {"a": 2, "b": "x"} in result
    assert {"a": 2, "b": "y"} in result


def test_dict_prod_single_param():
    """Test with single parameter"""
    result = xp.dict_prod(x=[1, 2, 3])
    assert len(result) == 3
    assert result == [{"x": 1}, {"x": 2}, {"x": 3}]


def test_dict_prod_empty():
    """Test with empty inputs"""
    result = xp.dict_prod()
    assert result == [{}]


def test_dict_prod_order():
    """Test that order is preserved: first keys are slowest to increment"""
    result = xp.dict_prod(a=[1, 2], b=[10, 20], c=[100, 200])
    # First key 'a' should be slowest (changes least frequently)
    assert result[0] == {"a": 1, "b": 10, "c": 100}
    assert result[1] == {"a": 1, "b": 10, "c": 200}
    assert result[2] == {"a": 1, "b": 20, "c": 100}
    assert result[3] == {"a": 1, "b": 20, "c": 200}
    assert result[4] == {"a": 2, "b": 10, "c": 100}
    assert len(result) == 8


def test_dict_prod_single_values():
    """Test with single-element lists"""
    result = xp.dict_prod(a=[1], b=[2])
    assert result == [{"a": 1, "b": 2}]


def test_dict_prod_with_numpy_arrays():
    """Ensure dict_prod works with numpy arrays as values (common use case)"""
    result = xp.dict_prod(seed=3000 + np.arange(3), N=[10, 100])
    assert len(result) == 6
    # Check that numpy values are preserved
    assert result[0]["seed"] == 3000
    assert result[1]["seed"] == 3000
    assert result[2]["seed"] == 3001


def test_dict_prod_mixed_types():
    """Test with mixed types in parameter values"""
    result = xp.dict_prod(a=[1, 2.5, "three"], b=[True, None], c=[[1, 2], {"key": "val"}])
    assert len(result) == 12
    assert {"a": 1, "b": True, "c": [1, 2]} in result
    assert {"a": "three", "b": None, "c": {"key": "val"}} in result
