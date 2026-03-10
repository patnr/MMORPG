"""Wrapper for executing a set of simulations/experiments.

Not "just" `xargs -P` since it also does

- Load `kws` dicts (serialized by `dill`).
- Import `fun` by `fun_name` (from `script`).
- Run each one through `fun` using `nCPU` in parallel.
- Save results.
"""

# NOTE: "batch_runner.py" imports `script`.
# We want it to support "standalone" scripts, i.e. run as `python path/to/{script}`
# (instead of `python -m path/to{script}` which forces package structuring on {script}).
# ⇒ must copy into `to/`, or insert `to/` in `sys.path`.
# For remote work, we need to do the copy anyways, let's choose the copy solution.

import sys
from importlib import import_module
from pathlib import Path

import dill

from mmorpg.local_mp import mp

if __name__ == "__main__":
    # Unpack args
    (
        _,  # name of this script
        script,  # e.g. "my_experiments"
        fun_name,  # e.g. "experiment"
        nCPU,  # number of kws to run simultaneously
        inpt,  # e.g. "my_experiments/inputs/0"
    ) = sys.argv

    # Process args
    nCPU = None if nCPU == "None" else int(nCPU)

    # Import fun
    fun = getattr(import_module(script), fun_name)

    inpt = Path(inpt).expanduser()

    # Load parameter sets
    inputs = dill.loads(inpt.read_bytes())

    # results = [fun(**kws) for kws in inputs]  # -- for debugging --
    results = mp(lambda kws: fun(**kws), inputs, nCPU, log_errors=True)

    outp = Path(str(inpt).replace("/inputs/", "/outputs/"))
    outp.write_bytes(dill.dumps(results))
