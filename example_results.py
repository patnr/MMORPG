"""Post-process and plot the results of `example.py`.

Companion to `example.py`: demonstrates the (independent) `mmorpg.results`/
`mmorpg.results.lineplots` layer -- tabulating and plotting a sparse, possibly *ragged*
parameter grid -- on the same `experiment()`. Only needs the `results` extra
(`uv sync --extra results`); unlike `example.py`, it never touches `mmorpg`'s
dispatch/remote-execution machinery.

Run `python example.py` first (takes a couple seconds locally), then this script.

`example.py`'s own grid is deliberately ragged: `"deterministic"` doesn't depend on `seed` or
`antithetic` (6 points total, one per `N`/`func`), while `"stochastic"` is repeated over 1000
seeds per (`N`, `func`, `antithetic`) -- exactly the kind of irregularly-shaped grid
`mmorpg.results` is built to handle without densifying/padding. It also includes one
deliberately-invalid `"unsupported"` method entry, to exercise `find_crashed()`.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from IPython.utils.ipstruct import Struct  # One of many "Bunch" variants

from example import experiment
from mmorpg.results import (
    dicts2index,
    find_crashed,
    get_data_dir,
    load_data,
    pd_sel,
    projection,
    shape_tables,
    validate,
)
from mmorpg.results.lineplots import LinePlots

plt.ion()  # so repeated `%run`s (in ipython) update the figures live, without blocking

data_dir = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else get_data_dir(fun=experiment, tags="latest")[0]
)

print("Loading from", data_dir)
xps, res = load_data(data_dir)
crashed = find_crashed(res)  # warns with crash count/types, if any -- catches the deliberately
# invalid "unsupported" method entry from `example.py`'s `list_experiments()`.

# `error` is already the scalar of interest -- for an experiment producing something
# higher-dimensional you'd reduce it to scalar(s) here instead, guarding on `crashed` so a
# crash doesn't get silently conflated with a legitimately bad (but non-crashed) result.
stats = [{"error": np.nan} if c else {"error": r["error"]} for c, r in zip(crashed, res)]

# Index by `xps`, folding missing/`None` values into the `NONE` sentinel (kept distinct from
# NaN -- see `dicts2index()`).
df = pd.DataFrame.from_records(stats, index=dicts2index(xps))
err = df["error"]

# Row-filter via `pd_sel()`: drop the deliberately-invalid "unsupported" method demo entry
# (already reported above by `find_crashed()`'s warning) before tabulating/plotting -- it's
# not a real method, just a crash-handling demo, so it has no business cluttering the table.
err = pd_sel(err, dict(method=["stochastic", "deterministic"]))

# Sparse xarray: ragged combinations (e.g. "deterministic" not spanning `seed`/`antithetic`)
# don't need to be densified/padded with NaNs.
xa = xr.DataArray.from_series(err, sparse=True)
xa.name = "Error"

# `orient` maps plot/processing roles to data dims.
orient = Struct(
    mean=["seed"],  # average away the (ragged) `seed` dim
    tuned=["antithetic"],  # pick whichever setting has lower error (see `.min()` below)
    fig=["func"],  # separate figure per integrand
    panel_row=[],
    panel_col=[],
    linestyle=[],  # one line per method
    unlabelled=[],
    xaxis="N",
)
validate(orient, xa.dims)

# `skipna=True` -- unlike a dense grid (where `False` would flag a genuine hole): here,
# "deterministic" legitimately only has one value (no seed variation, stored at the `NONE`
# sentinel tick), so the mean correctly reduces to just that value.
skill = xa.mean(orient.mean, skipna=True)
skill.name = "mean " + skill.name
# `.min()`, not the more common `.max()` (e.g. Singleton-Results.py): lower error is better,
# whereas `.max()`/`scale01()` assume a "higher is better" skill by convention.
skill = skill.min(orient.tuned, skipna=True)

# ## Tabulate
pd.set_option("display.precision", 3)
table = shape_tables(skill, orient)
print(table)

# ## Plot
with plt.rc_context({"axes.xmargin": 0.005}):  # tight xlim -- avoids log/symlog flip-flop
    plotter = LinePlots(
        skill,
        orient,
        fig_title=f"{data_dir.parent.name}/{data_dir.name}",
        xscale="log",
    )
for fig in plotter.fig_registry:
    for ax in fig.axes:
        ax.set_yscale("log")

# Reference lines for each method's theoretical convergence rate, anchored to match its
# empirical curve at the smallest `N` -- so the slope can be eyeballed directly. Note this is
# only an *asymptotic* rate: e.g. for `func="oscillatory"` (a smooth periodic function
# sampled over whole periods, endpoints included), the trapezoid rule is spectrally accurate
# -- it converges far faster than the generic `N^-2`, so don't expect a match there.
rates = {
    "stochastic": (0.5, "$N^{-1/2}$"),  # Monte Carlo: error ~ std/sqrt(N)
    "deterministic": (2, "$N^{-2}$"),  # trapezoid rule (generic smooth case): error ~ O(h^2)
}
# `projection()` reproduces the same (fig-dim -> value) order `LinePlots` iterated internally.
figs_by_func = zip(plotter.fig_registry[:-1], projection(skill, orient.fig))  # skip legend fig
for fig, vFig in figs_by_func:
    for ax in fig.axes:
        for method, (p, label) in rates.items():
            series = table.loc[(vFig["func"], method)].sort_index()
            xticks = series.index.to_numpy(dtype=float)
            ref_ys = series.iloc[0] * (xticks[0] / xticks) ** p
            ax.plot(xticks, ref_ys, ls=":", color="k", lw=1.5)
            ax.annotate(
                label,
                (xticks[-1], ref_ys[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
            )

img_dir = data_dir / "figures"
# plotter.save(img_dir=img_dir, data_dir=data_dir)
# print("Saved figures under", img_dir)
