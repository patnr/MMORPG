"""Tests for data saving and loading functionality."""

import numpy as np

import xp


def test_save_and_load_basic(tmp_path):
    """Test saving and loading experiment data"""
    xps = [{"seed": i, "N": 100} for i in range(10)]
    data_dir = tmp_path / "test_data"
    (data_dir / "xps").mkdir(parents=True)

    xp.save(xps, data_dir, nBatch=3)
    loaded = xp.load_data(data_dir / "xps", pbar=False)

    assert loaded == xps
    assert len(loaded) == 10


def test_save_creates_batches(tmp_path):
    """Test that save creates correct number of batch files"""
    xps = [{"i": i} for i in range(100)]
    data_dir = tmp_path / "test_data"
    (data_dir / "xps").mkdir(parents=True)

    nBatch = 5
    xp.save(xps, data_dir, nBatch=nBatch)

    # Check that nBatch files were created
    batch_files = list((data_dir / "xps").iterdir())
    assert len(batch_files) == nBatch

    # Check that all data is preserved
    loaded = xp.load_data(data_dir / "xps", pbar=False)
    assert len(loaded) == 100


def test_save_single_batch(tmp_path):
    """Test saving with single batch"""
    xps = [{"x": i} for i in range(5)]
    data_dir = tmp_path / "test_data"
    (data_dir / "xps").mkdir(parents=True)

    xp.save(xps, data_dir, nBatch=1)
    loaded = xp.load_data(data_dir / "xps", pbar=False)

    assert loaded == xps


def test_load_preserves_numpy_arrays(tmp_path):
    """Ensure dill preserves numpy arrays correctly"""
    xps = [
        {"seed": 0, "data": np.array([1, 2, 3])},
        {"seed": 1, "data": np.array([[4, 5], [6, 7]])},
    ]
    data_dir = tmp_path / "test_data"
    (data_dir / "xps").mkdir(parents=True)

    xp.save(xps, data_dir, nBatch=1)
    loaded = xp.load_data(data_dir / "xps", pbar=False)

    assert len(loaded) == 2
    np.testing.assert_array_equal(loaded[0]["data"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(loaded[1]["data"], np.array([[4, 5], [6, 7]]))


def test_load_preserves_complex_types(tmp_path):
    """Ensure dill preserves complex nested structures"""
    xps = [
        {
            "seed": 0,
            "nested": {"a": [1, 2], "b": {"c": 3}},
            "array": np.array([1.5, 2.5]),
            "tuple": (1, "two", 3.0),
            "none": None,
        }
    ]
    data_dir = tmp_path / "test_data"
    (data_dir / "xps").mkdir(parents=True)

    xp.save(xps, data_dir, nBatch=1)
    loaded = xp.load_data(data_dir / "xps", pbar=False)

    assert loaded[0]["nested"] == {"a": [1, 2], "b": {"c": 3}}
    np.testing.assert_array_equal(loaded[0]["array"], np.array([1.5, 2.5]))
    assert loaded[0]["tuple"] == (1, "two", 3.0)
    assert loaded[0]["none"] is None


def test_load_sorts_by_batch_number(tmp_path):
    """Test that load_data sorts batches numerically (not lexicographically)"""
    data_dir = tmp_path / "test_data" / "xps"
    data_dir.mkdir(parents=True)

    # Create batches in non-sequential order
    import dill

    (data_dir / "0").write_bytes(dill.dumps([{"batch": 0, "i": 0}]))
    (data_dir / "10").write_bytes(dill.dumps([{"batch": 10, "i": 10}]))
    (data_dir / "2").write_bytes(dill.dumps([{"batch": 2, "i": 2}]))
    (data_dir / "1").write_bytes(dill.dumps([{"batch": 1, "i": 1}]))

    loaded = xp.load_data(data_dir, pbar=False)

    # Should be sorted: 0, 1, 2, 10 (not 0, 1, 10, 2)
    assert [x["batch"] for x in loaded] == [0, 1, 2, 10]


def test_save_with_uneven_batches(tmp_path):
    """Test that uneven division into batches works correctly"""
    xps = [{"i": i} for i in range(10)]
    data_dir = tmp_path / "test_data"
    (data_dir / "xps").mkdir(parents=True)

    # 10 items divided into 3 batches: should be [4, 4, 2]
    xp.save(xps, data_dir, nBatch=3)
    loaded = xp.load_data(data_dir / "xps", pbar=False)

    assert loaded == xps
    assert len(loaded) == 10


def test_save_empty_list(tmp_path):
    """Test saving empty experiment list"""
    xps = []
    data_dir = tmp_path / "test_data"
    (data_dir / "xps").mkdir(parents=True)

    xp.save(xps, data_dir, nBatch=1)
    loaded = xp.load_data(data_dir / "xps", pbar=False)

    assert loaded == []


def test_load_data_with_progress_bar(tmp_path):
    """Test that progress bar parameter works"""
    xps = [{"i": i} for i in range(5)]
    data_dir = tmp_path / "test_data"
    (data_dir / "xps").mkdir(parents=True)

    xp.save(xps, data_dir, nBatch=2)

    # Should work with pbar=True (default)
    loaded1 = xp.load_data(data_dir / "xps", pbar=True)
    assert loaded1 == xps

    # Should work with pbar=False
    loaded2 = xp.load_data(data_dir / "xps", pbar=False)
    assert loaded2 == xps


def test_save_preserves_order(tmp_path):
    """Test that save/load preserves exact order of experiments"""
    xps = [{"id": f"exp_{i:03d}"} for i in range(50)]
    data_dir = tmp_path / "test_data"
    (data_dir / "xps").mkdir(parents=True)

    xp.save(xps, data_dir, nBatch=7)
    loaded = xp.load_data(data_dir / "xps", pbar=False)

    assert loaded == xps
    assert [x["id"] for x in loaded] == [f"exp_{i:03d}" for i in range(50)]
