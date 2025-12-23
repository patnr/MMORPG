"""Load `xps` dicts. Run through `fun_name` (from `script`) using `nCPU`. Save results."""
# NOTE: This file *imports* `script` and invokes the `fun` defined therein.
# But want to support "standalone" scripts, i.e. run as `python path/to/{script}`.
# ⇒ This file must get copied into `to/` or insert `to/` in `sys.path`.
# For remote work, we need to do the copy anyways, let's choose the copy solution.

import sys
from importlib import import_module
from pathlib import Path

import dill

from xp.local_mp import mp

if __name__ == "__main__":
    # Unpack arguments
    _, script, fun_name, dir_xps, nCPU = sys.argv
    nCPU = None if nCPU == "None" else int(nCPU)

    fun = getattr(import_module(script), fun_name)

    dir_xps = Path(dir_xps).expanduser()
    xps = dill.loads(dir_xps.read_bytes())

    # res = [fun(xp) for xp in xps]  # -- for debugging --
    results = mp(lambda kwargs: fun(**kwargs), xps, nCPU)

    dir_res = Path(str(dir_xps).replace("/xps/", "/res/"))
    dir_res.write_bytes(dill.dumps(results))
