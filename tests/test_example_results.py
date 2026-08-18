"""Integration test for `example_results.py`: exercises the `mmorpg.results`/
`mmorpg.results.lineplots` layer against `example.py`'s own `dispatch()` output, which is
deliberately ragged ("deterministic" doesn't span `seed`, unlike "stochastic").
"""

import runpy
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from example import experiment, list_experiments
from mmorpg import dispatch

REPO_ROOT = Path(__file__).parent.parent


@pytest.mark.slow
def test_example_results_pipeline(monkeypatch):
    """Full pipeline: dispatch `example.py`'s own grid, then run `example_results.py` (as a
    script, via its CLI arg) to tabulate/plot it end to end."""
    inputs = list_experiments()
    data_dir = dispatch(experiment, inputs, host=None)
    try:
        monkeypatch.setattr(sys, "argv", ["example_results.py", str(data_dir)])
        ns = runpy.run_path(str(REPO_ROOT / "example_results.py"))

        plotter = ns["plotter"]
        assert len(plotter.fig_registry) == 3  # 2 data figs ("func") + 1 legend fig
        assert set(plotter.handles.index.get_level_values("method")) == {
            "stochastic",
            "deterministic",
        }
        assert (data_dir / "figures").is_dir()
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
