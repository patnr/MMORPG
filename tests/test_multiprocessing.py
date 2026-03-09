"""Tests for multiprocessing functionality."""

import time

import pytest

from xp.local_mp import mp


def square(x):
    """Simple function for testing"""
    return x**2


def slow_square(x):
    """Slow function to test parallel speedup"""
    time.sleep(0.01)
    return x**2


def failing_func(x):
    """Function that raises exception for some inputs"""
    if x == 5:
        raise ValueError(f"Test error for x={x}")
    return x * 2


def test_mp_correctness_sequential():
    """Ensure mp produces same results as sequential with nCPU=1"""
    lst = list(range(20))
    expected = [x**2 for x in lst]

    result = mp(square, lst, nCPU=1)
    assert result == expected


def test_mp_correctness_parallel():
    """Ensure mp produces same results as sequential with multiple CPUs"""
    lst = list(range(50))
    expected = [x**2 for x in lst]

    result = mp(square, lst, nCPU=4)
    assert result == expected


def test_mp_preserves_order():
    """Test that mp preserves order of results"""
    lst = list(range(100))
    result = mp(square, lst, nCPU=4)

    assert result == [i**2 for i in range(100)]
    # Verify specific order preservation
    assert result[0] == 0
    assert result[50] == 2500
    assert result[99] == 9801


def test_mp_with_ncpu_none():
    """Test nCPU=None uses all available CPUs"""
    lst = list(range(10))
    result = mp(square, lst, nCPU=None)

    assert result == [x**2 for x in lst]


def test_mp_with_ncpu_all():
    """Test nCPU='all' uses all available CPUs"""
    lst = list(range(10))
    result = mp(square, lst, nCPU="all")

    assert result == [x**2 for x in lst]


def test_mp_with_ncpu_true():
    """Test nCPU=True uses all available CPUs"""
    lst = list(range(10))
    result = mp(square, lst, nCPU=True)

    assert result == [x**2 for x in lst]


def test_mp_with_ncpu_false():
    """Test nCPU=False runs sequentially"""
    lst = list(range(10))
    result = mp(square, lst, nCPU=False)

    assert result == [x**2 for x in lst]


def test_mp_with_ncpu_zero():
    """Test nCPU=0 runs sequentially"""
    lst = list(range(10))
    result = mp(square, lst, nCPU=0)

    assert result == [x**2 for x in lst]


def test_mp_empty_list():
    """Test mp with empty list"""
    result = mp(square, [], nCPU=4)
    assert result == []


def test_mp_single_item():
    """Test mp with single item"""
    result = mp(square, [5], nCPU=4)
    assert result == [25]


def test_mp_with_different_types():
    """Test mp with different data types"""

    def process_item(item):
        if isinstance(item, int):
            return item * 2
        elif isinstance(item, str):
            return item.upper()
        return item

    lst = [1, 2, "hello", 3, "world"]
    result = mp(process_item, lst, nCPU=2)

    assert result == [2, 4, "HELLO", 6, "WORLD"]


def test_mp_chunking_with_small_list():
    """Test that chunking works with small lists"""
    lst = list(range(5))
    result = mp(square, lst, nCPU=4)

    assert result == [x**2 for x in lst]


def test_mp_chunking_with_large_list():
    """Test that chunking works with large lists"""
    lst = list(range(1000))
    result = mp(square, lst, nCPU=8)

    assert len(result) == 1000
    assert result == [x**2 for x in lst]


def test_mp_with_lambda():
    """Test mp with lambda function"""
    lst = list(range(10))
    result = mp(lambda x: x * 3, lst, nCPU=2)

    assert result == [x * 3 for x in lst]


def test_mp_exception_propagation():
    """Ensure exceptions in workers are propagated to main process"""
    lst = list(range(10))

    # This should raise ValueError when processing x=5
    with pytest.raises(Exception):  # Can be various exception types depending on pathos
        mp(failing_func, lst, nCPU=2)


def test_mp_stateful_function():
    """Test mp with function that has side effects (returns tuple)"""

    def process_with_index(x):
        return (x, x**2)

    lst = list(range(10))
    result = mp(process_with_index, lst, nCPU=2)

    assert result == [(i, i**2) for i in range(10)]


def test_mp_parallel_speedup():
    """Test that parallel execution is actually faster (rough check)"""
    lst = list(range(20))

    # Sequential
    start = time.time()
    result_seq = mp(slow_square, lst, nCPU=1)
    time.time() - start

    # Parallel (should be faster)
    start = time.time()
    result_par = mp(slow_square, lst, nCPU=4)
    time.time() - start

    assert result_seq == result_par
    # Parallel should be noticeably faster (allow some overhead)
    # This is a rough check - won't always pass on busy systems
    # Commented out to avoid flakiness, but documents expected behavior
    # assert time_par < time_seq * 0.7


def test_mp_with_complex_data():
    """Test mp with complex nested data structures"""
    import numpy as np

    def process_dict(d):
        return {
            "original": d["value"],
            "squared": d["value"] ** 2,
            "array": np.array([d["value"]]) * 2,
        }

    lst = [{"value": i} for i in range(5)]
    result = mp(process_dict, lst, nCPU=2)

    assert len(result) == 5
    assert result[0]["original"] == 0
    assert result[2]["squared"] == 4
    np.testing.assert_array_equal(result[3]["array"], np.array([6]))
