"""Companion to `example.py`: demo `mmorpg.results/` for post-process and plotting.

The benefit of splitting into  `example.py` and `example_results.py`
is that the latter is usually much quicker and we'll want to re-run it
often as part of adjustments to the post-processing and plotting.

`example.py`'s own grid is deliberately ragged: `"deterministic"` doesn't depend on `seed`,
`antithetic`, or `bias` (27 points total, one per `N`/`func`/`nDim`), while `"stochastic"` is
repeated over 10 seeds per (`N`, `func`, `antithetic`, `bias`, `nDim`) -- exactly the kind of
irregularly-shaped grid `mmorpg.results` is built to handle without densifying/padding. It also
includes one deliberately-invalid `"unsupported"` method entry, to exercise `find_crashed()`.
"""

# Debugging guide
# - First make sure the (weirdness of the) results are not due to some strange choice
#   in the aggregations in Results.py. I.e. in the mean() and min() operations.
#   Example questions:
#   - Are you setting skipna appropriately (`False` for mean, `True` for tuned) ?
#   - Are you optimising on the last iteration (good),
#     or for each individual iteration (bad) ?
#   - Have you exploded the `iter` dimension correctly?
#     Did you `fillna(last_or_lowest_value)` after the stopping iteration?
#   - If larger ⇔ better for skill, then did you do max() rather than min() ?
# - Drill down to particular data points (sub-selection) of interest,
#   - Empty `orient.mean` and `.tuned`.
#   - Inspect the raw values. For example,
#       >>> xp = dict(nEns=30, case="quadratic", nDim=2, method="GD", aspect=1)
#       ... print(sparse_to_series(xa.sel(xp, drop=True)).unstack("iter"))
#     Keep adding kws to `xp` until you get just the sub-selection you want.
# - Use that xp here.
# - Check that `objs` values match those from results processing script.
# - Run script in debug mode. I.e. do `breakpoint(); m.experiment(**xp)`

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
    sparse_to_series,
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
    tuned=["antithetic", "bias"],  # grid-search over both & pick the lower-error combo (`.min()`)
    fig=[],
    panel_row=["func"],
    panel_col=["nDim"],  # one panel per dimensionality of the integration domain
    linestyle=[],
    unlabelled=[],
    xaxis="N",
)
# `method` and `func` are left unassigned -- both become "hue": each (method, func) combo gets
# its own colored line, overlaid within each `nDim` panel.
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
# only an *asymptotic* rate (unaffected by `nDim`, since each `func` is separable -- see
# `experiment()`'s docstring): e.g. for `func="oscillatory"` (a smooth periodic function
# sampled over whole periods, endpoints included), the trapezoid rule is spectrally accurate
# -- it converges far faster than the generic `N^-2`, so don't expect a match there.
rates = {
    "stochastic": (0.5, "$N^{-1/2}$"),  # Monte Carlo: error ~ std/sqrt(N)
    "deterministic": (2, "$N^{-2}$"),  # trapezoid rule (generic smooth case): error ~ O(h^2)
}
# Anchor each reference line to the "quadratic" func line -- one dotted line/label per method
# per panel, rather than one per (method, func) combo (which would just overplot near-identical
# slopes at slightly different heights). Only matters when `func` is left as "hue" (i.e. not
# `fig`/`panel_row`/`panel_col`), in which case multiple funcs are overlaid per panel; if `func`
# *is* one of those roles, `combo` below already pins it and this is unused.
ANCHOR_FUNC = "quadratic"

# Look up each reference line's data straight from `skill` (keyed by dimension name) rather than
# from `table` (keyed by `orient`-dependent row/column position) -- this way the loop doesn't
# care which of `fig`/`panel_row`/`panel_col`/hue `func` (or any other dim) has been placed in.
ncols = len(projection(skill, orient.panel_col))
figs_by_vFig = zip(plotter.fig_registry[:-1], projection(skill, orient.fig))  # skip legend fig
for fig, vFig in figs_by_vFig:
    for iRow, uPanel in enumerate(projection(skill, orient.panel_row)):
        for jCol, vPanel in enumerate(projection(skill, orient.panel_col)):
            ax = fig.axes[iRow * ncols + jCol]
            combo = {**vFig, **uPanel, **vPanel}
            func = combo.get("func", ANCHOR_FUNC)
            for method, (p, label) in rates.items():
                xSect = skill.sel({**combo, "func": func, "method": method}, drop=True)
                series = sparse_to_series(xSect).sort_index()
                # `sparse_to_series()` always returns a (here single-level) `MultiIndex`.
                xticks = series.index.get_level_values(orient.xaxis).to_numpy(dtype=float)
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
