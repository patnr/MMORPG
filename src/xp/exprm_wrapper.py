"""Wrapper for executing a set of experiments (`xps`).

Not "just" `xargs -P` since it also does

- Load `xps` dicts (serialized by `dill`).
- Import `fun` by `fun_name` (from `script`).
- Run each one through `fun` using `nCPU` in parallel.
- Save results.
"""

# NOTE: "exprm_wrapper.py" imports `script`.
# We want it to support "standalone" scripts, i.e. run as `python path/to/{script}`
# (instead of `python -m path/to{script}` which forces package structuring on {script}).
# ⇒ must copy into `to/`, or insert `to/` in `sys.path`.
# For remote work, we need to do the copy anyways, let's choose the copy solution.

import sys
from importlib import import_module
from pathlib import Path

import dill

from xp.local_mp import mp

if __name__ == "__main__":
    # Unpack args
    (
        _, # name of this script
        script, # e.g. "my_experiments"
        fun_name, # e.g. "experiment"
        nCPU, # number of xps to run simultaneously
        xps_path, # e.g. "my_experiments/xps/0"
    ) = sys.argv

    # Process args
    nCPU = None if nCPU == "None" else int(nCPU)

    # Import fun
    fun = getattr(import_module(script), fun_name)

    xps_path = Path(xps_path).expanduser()

    # Load parameter sets
    xps = dill.loads(xps_path.read_bytes())

    # res = [fun(xp) for xp in xps]  # -- for debugging --
    results = mp(lambda kwargs: fun(**kwargs), xps, nCPU, log_errors=True)

    dir_res = Path(str(xps_path).replace("/xps/", "/res/"))
    dir_res.write_bytes(dill.dumps(results))
