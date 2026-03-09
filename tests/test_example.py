import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path to import example.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from example import experiment, list_experiments


class TestExperiment:
    """Test the experiment function for numerical integration."""

    def test_deterministic_method_accuracy(self):
        """Test deterministic method produces accurate integral estimate."""
        result = experiment(seed=None, method="deterministic", N=1000)
        assert "estimate" in result
        assert "true_val" in result
        assert "error" in result
        assert result["true_val"] == 1 / 3
        assert result["error"] < 1e-6
        np.testing.assert_allclose(result["estimate"], 1 / 3, atol=1e-6)

    def test_deterministic_convergence(self):
        """Test that deterministic method error decreases with N."""
        result_10 = experiment(seed=None, method="deterministic", N=10)
        result_100 = experiment(seed=None, method="deterministic", N=100)
        result_1000 = experiment(seed=None, method="deterministic", N=1000)

        assert result_10["error"] > result_100["error"]
        assert result_100["error"] > result_1000["error"]

    def test_stochastic_method_seeded(self):
        """Test stochastic method with seed produces reproducible results."""
        result1 = experiment(seed=3000, method="stochastic", N=100)
        result2 = experiment(seed=3000, method="stochastic", N=100)

        assert result1["estimate"] == result2["estimate"]
        assert result1["error"] == result2["error"]

    def test_stochastic_method_reasonable(self):
        """Test stochastic method produces reasonable estimates."""
        result = experiment(seed=3000, method="stochastic", N=1000)

        assert 0 < result["estimate"] < 1
        assert result["error"] < 0.1
        assert result["true_val"] == 1 / 3

    def test_stochastic_different_seeds(self):
        """Test different seeds produce different estimates."""
        result1 = experiment(seed=3000, method="stochastic", N=100)
        result2 = experiment(seed=3001, method="stochastic", N=100)

        assert result1["estimate"] != result2["estimate"]

    @pytest.mark.parametrize("N", [10, 100, 1000])
    def test_deterministic_n_values(self, N):
        """Test deterministic method works for different N values."""
        result = experiment(seed=None, method="deterministic", N=N)
        assert 0.3 < result["estimate"] < 0.35
        assert result["error"] < 0.01

    def test_known_deterministic_values(self):
        """Test against known deterministic output values."""
        result = experiment(seed=None, method="deterministic", N=10)
        np.testing.assert_allclose(result["estimate"], 0.335391, atol=1e-6)
        np.testing.assert_allclose(result["error"], 2.057613e-03, atol=1e-8)

        result = experiment(seed=None, method="deterministic", N=100)
        np.testing.assert_allclose(result["estimate"], 0.333350, atol=1e-6)
        np.testing.assert_allclose(result["error"], 1.700507e-05, atol=1e-10)

    def test_known_stochastic_values(self):
        """Test against known stochastic output values with fixed seeds."""
        result = experiment(seed=3000, method="stochastic", N=10)
        np.testing.assert_allclose(result["estimate"], 0.332873, atol=1e-6)
        np.testing.assert_allclose(result["error"], 4.603193e-04, atol=1e-8)

        result = experiment(seed=3001, method="stochastic", N=1000)
        np.testing.assert_allclose(result["estimate"], 0.329567, atol=1e-6)
        np.testing.assert_allclose(result["error"], 3.766277e-03, atol=1e-8)


class TestListExperiments:
    """Test the list_experiments function for configuration generation."""

    def test_experiment_count(self):
        """Test that correct number of experiments is generated."""
        xps = list_experiments()
        assert len(xps) == 9

    def test_seed_handling_deterministic(self):
        """Test deterministic experiments have seed=None."""
        xps = list_experiments()
        deterministic = [x for x in xps if x["method"] == "deterministic"]

        assert len(deterministic) == 3
        assert all(x["seed"] is None for x in deterministic)

    def test_no_duplicates(self):
        """Test that there are no duplicate experiments."""
        xps = list_experiments()
        # Convert to tuples for comparison
        xps_tuples = [tuple(sorted(x.items())) for x in xps]
        assert len(xps_tuples) == len(set(xps_tuples))

    def test_all_experiments_valid(self):
        """Test that all generated experiments can be run."""
        xps = list_experiments()
        for kwargs in xps:
            result = experiment(**kwargs)
            assert "estimate" in result
            assert "error" in result

    def test_experiment_keys(self):
        """Test that all experiments have required keys."""
        xps = list_experiments()
        required_keys = {"method", "N", "seed"}

        for xp in xps:
            assert set(xp.keys()) == required_keys


class TestIntegration:
    """Integration tests running the full example workflow."""

    def test_tabulation(self):
        """Checking working input to pandas and formatting"""
        import pandas as pd

        xps = list_experiments()
        res = [experiment(**kwargs) for kwargs in xps]

        # Create DataFrame as done in example.py
        df = pd.DataFrame(xps).set_index(list(xps[0]))
        df = pd.DataFrame.from_records(res, index=df.index)

        # Verify DataFrame was created successfully
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(xps)
        assert all(col in df.columns for col in ["estimate", "true_val", "error"])

    @pytest.fixture(scope="class")
    def reference_results(self):
        """Compute reference results using direct execution."""
        return [experiment(**kwargs) for kwargs in list_experiments()]

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "host",
        [
            None,  # SUBPROCESS (local multiprocessing)
            "localhost",  # SSH to local machine
            "my-gcp-*",  # GCP VM with wildcard
            "cno-0001",  # Specific compute node
            "login-1.hpc.intra.norceresearch.no",  # HPC cluster
        ],
        ids=["subprocess", "localhost", "gcp-vm", "compute-node", "hpc-cluster"],
    )
    def test_host_dispatch(self, host, reference_results):
        """Test that dispatch on a specific host produces same results as direct execution.

        This test runs experiments on a single host and verifies results match
        the reference (direct execution). Each host appears as a separate test.

        To run all host tests: pytest -m slow
        To run specific host: pytest -m slow -k "test_host_dispatch[localhost]"
        """
        import subprocess
        import tempfile
        import warnings
        from pathlib import Path

        from xp import dispatch, load_data

        # Use same experiments as reference
        xps = list_experiments()

        # Run dispatch for this host
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                data_root = Path(tmpdir)
                data_dir = dispatch(
                    experiment,
                    xps,
                    host=host,
                    data_root=data_root,
                    nCPU=2,
                )

                # Load results
                results = load_data(data_dir / "res", pbar=False)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            # Host is unreachable - skip test with warning
            msg = f"Host '{host}' unreachable: {type(e).__name__}"
            warnings.warn(msg, UserWarning)
            pytest.skip(msg)

        # Verify results match reference
        assert len(results) == len(reference_results), (
            f"Host '{host}' produced {len(results)} results, expected {len(reference_results)}"
        )
        for i, (result, reference) in enumerate(zip(results, reference_results)):
            assert result["true_val"] == reference["true_val"], (
                f"Host '{host}' computed different true_val for experiment {i}"
            )
            np.testing.assert_allclose(
                result["estimate"],
                reference["estimate"],
                err_msg=f"Host '{host}' gave different estimate for {xps}",
            )
