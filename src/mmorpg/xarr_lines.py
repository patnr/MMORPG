"""
Rationale
-----
`xarray.plot.line` is great, but has some limitations (examples: cannot allocate
>1 dim to "hue"; cannot re-use figs if supplying kwargs `row` and `col`)
and requires some workarounds that make it easier to just do the processing ourselves.

Meanwhile it's tempting to just call `plt.plot(..., shape_tables())`.
since `shape_tables()` uses a few pandas API calls,
which is seemingly reinvented by the more elaborate `line_plots()` (relying on xarray).
But `line_plots()` nests `.sel()` calls (one per fig/panel/linestyle/unlabelled leaf) and only
unstacks (i.e. *densifies*) `xdim` *within* each already-narrowed leaf.
Meanwhile `shape_tables()` instead unstacks the *entire* `skill` in one shot (less overhead).
But for a ragged grid of experiments (e.g. different `fig`/`panel_row`/... groups cover different `xaxis` ticks)
the single/global unstack pads *every* group with NaNs for ticks only *some* other group has.

NOTE: on pandas vs xarray:
----
- Cleaning and tabulating data is easier with pandas,
  as is loading and iterating on _sparse_ data.
- But xarray greatly facilitates aggregations (min/max/mean). For example, compare
    - `skill.mean(orient.mean)` vs.
    - `skill.groupby(skill.index.names.difference(orient.mean), sort=False).mean()`.
- Similarly, arithmetics w/ broadcasting, as in `scale01`, would be
  much harder with pandas (with multiindex).
"""

import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from IPython.utils.ipstruct import Struct  # One of many "Bunch" variants
from matplotlib.text import OffsetFrom
from matplotlib.ticker import ScalarFormatter, SymmetricalLogLocator
from matplotlib.transforms import Bbox
from pandas import pandas as pd

from .tools import confirm_cold_call
from .tools import yank as yanker

# None-like value that pandas does not convert to NaN. Will show as blank in legend.
# PS: Cannot use actual `None` because gets re-converted to NaN unless `dtype` is object,
# but neither `df.set_index()` nor `pd.MultiIndex.from_frame()` support arg `dtype`.
# Try `set_levels(..., df.astype(object).where(df.notna(), None), ...)` ?
NONE = "NA"


# YAML print w/ indentation _also_ for lists
class IndentedDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


lbox = dict(facecolor="lightyellow", edgecolor="k", alpha=0.5, boxstyle="round,pad=0.11")

legend_mono = dict(
    prop={"family": "monospace"},
    alignment="right",
    title_fontproperties={"family": "monospace", "weight": "bold"},
    frameon=True,
    framealpha=0.8,
    facecolor="white",
    fancybox=True,
    handlelength=4,
    loc="upper left",
)


class SymmetricalLogLocator1(SymmetricalLogLocator):
    """Patched to avoid 0 and 1 overlapping."""

    def __call__(self):
        ticks = super().__call__()
        if len(ticks) >= 4 and (0 in ticks) and (1 in ticks):
            ticks = [tick for tick in ticks if tick != 0]
        return ticks


class ScalarFormatter10(ScalarFormatter):
    def __call__(self, x, pos=None):
        if abs(x) >= 1000:
            if x % 1 == 0:
                return f"10$^{int(np.log10(x))}$"
            else:
                return f"{x:.0e}"
        else:
            return f"{x:.0f}"


def sparse_to_series(a):
    """Convert *sparse* `DataArray` to a `pd.Series`."""
    # Enables round-tripping with xr.DataArray.from_series(., sparse=True).
    # From pending/stale PR #4007.
    # PS: cannot simply use a.coords.to_index() coz yields non-sparse coords
    index = pd.MultiIndex.from_arrays(a.data.coords, names=list(a.coords))
    index = index.set_levels(
        [
            [labels.values[i] for i in ints]
            for (labels, ints) in zip(a.coords.values(), index.levels)
        ]
    )
    return pd.Series(a.data.data, index=index, name=a.name)


def projection(a, dims, sparse=True, dicts=True):
    """Return list of `a.coords` (as dict) projected down onto `dims`."""
    if not dims:
        coords = [["ensure at least one point even though there are no dims"]]
    elif sparse:
        coords = a.data.coords[list(a.get_axis_num(dims))]  # as indices
        coords = np.unique(coords, axis=1).tolist()  # rm dupes (cause: dims ⊆ a.dims)
        coords = [a.coords[d].values[subs].tolist() for d, subs in zip(dims, coords)]  # as labels
    else:
        # Create grid (outer prod) of coord axes.
        axes = [a.coords[d].values.tolist() for d in dims]
        coords = list(itertools.product(*axes))
        # NB: the following (inspired by `sparse_to_series`) is no good since
        # (1) inefficient and (2) `np.unique` sometimes casts to str dtype.
        # coords = [a.coords.to_index().get_level_values(d).to_list() for d in dims]
        # coords = np.unique(coords, axis=1).tolist()

    if dicts:
        # Convert to list of dicts
        coords = [dict(zip(dims, val)) for val in zip(*coords)]
    return coords


def enum_explode(s: pd.Series, idx_name):
    """Explode a pandas Series and enumerate (as new index level)."""
    t = s.explode()
    enums = t.groupby(level=t.index.names).cumcount()
    t.index = pd.MultiIndex.from_tuples(
        [(*idx, enum) for idx, enum in zip(t.index, enums)],
        names=[*t.index.names, idx_name],
    )
    return t


def dicts2index(xps) -> pd.Index:
    """Convert a list of experiment-input dicts (e.g. `load_data(data_dir / "inputs")`) into
    the `pd.MultiIndex` of a to-be-populated results table/`DataArray`.

    - Drops exact duplicates (which don't make sense, and don't fit with xarray).
    - Gives functions/callables a prettier repr (their `__name__`), column-wise for speed.
    - The df instantiation detects missing keys (yay!) and converts them and None's to NaNs (nay!),
      ⇒ {missing keys, None, NaN} are henceforth indistinguishable! But NaNs are generally not supported
      for _sparse_ xarray coords. Workaround: replace by custom string value.
    """
    df = pd.DataFrame(xps)

    dupes = df.duplicated()
    if any(dupes):
        warnings.warn(f"{dupes.sum()} duplicate xp's found & removed.")
        df = df[~dupes.values]

    for col in df:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: x.__name__ if hasattr(x, "__name__") else x)

    df = df.fillna(NONE)
    df = df.set_index(df.columns.tolist())
    return df.index


def pd_sel(s: pd.Series, dct, drop=False, assert_nonempty=True):
    """Similar to `df.loc[]` but applied to a dict.

    Also see
    --------
    Resembles `xarray.DataArray.sel` but singleton levels of resulting index only dropped if `drop`.

    - `df.xs()` is for cross-sectioning, i.e. selecting a single value.
    - `df.iloc[]` is for integer-location based indexing.

    Parameters
    ----------
    drop : bool, optional
        If `True`, drop singleton levels of the _resulting_ index.
        `False` by default because then you don't have to remember to
        also edit `orient` every time you sloppily toggle a filter in the input `dct`.
    assert_nonempty : bool, optional
        If `True`, raise an `AssertionError` if the resulting selection is empty.

    Example:

    >>> dct = dict(
    ...     seed=3002,
    ...     case="quadratic",
    ...     sdev=slice(None),
    ...     method="BFGS",
    ...     iter=[0, 1, *range(7, 15)],
    ... )
    >>> sub_df = pd_sel(df, dct, True)
    """
    idx = tuple(dct.get(k, slice(None)) for k in s.index.names)
    sub = s.loc[idx]

    if assert_nonempty:
        assert len(sub), "Selection appears to be empty!"
    if drop:
        lvls = [k for k in sub.index.names if sub.index.get_level_values(k).nunique() == 1]
        sub = sub.droplevel(lvls)
    return sub


def find_categorical(ds, dim):
    """Find categorical (or nan) values (of `dim`). Add corresponding dimension (index level)."""
    ticks = ds.index.get_level_values(dim)
    numeric = pd.to_numeric(ticks, errors="coerce").notna()
    if any(~numeric):
        cat = "fix_" + dim
        # Split into numeric and non-numeric, rename non-numeric dim
        ds_num = ds[numeric]
        ds_cat = ds[~numeric].rename_axis(index={dim: cat})
        # Re-concatenate -- for the purpose of downstream legend generation
        # (too much of a hassle to work w/ 2 separate DataFrames)
        df = pd.concat([ds_num.reset_index(), ds_cat.reset_index()])
        # sparse xr don't accept nans in index ⇒ assign special values
        df[dim] = df[dim].fillna("_CAT_")
        df[cat] = df[cat].fillna("_VAR_")

        df = df.set_index([cat] + ds.index.names)
        ds = df.squeeze()  # df -> series
        # ticks = ds_num.index.levels[ds_num.index.names.index(dim)].astype(float)

    return ds


def clear_fig(num, figsize=None, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        plt.figure(num=num, figsize=figsize, **kwargs).clear()


def set_panel_col_label(ax, dct, nickname, keys=True):
    if keys:
        txt = "\n".join(f"$\\bf{{{nickname(k)}}}$ {v}" for (k, v) in dct.items())
    else:
        txt = "\n".join(f"{v}" for v in dct.values())
    ax.xaxis.set_label_position("top")
    ax.xaxis.set_label_coords(0.5, 1.03)
    ax.set_xlabel(txt, fontsize=14, bbox=lbox, va="bottom", ha="center")


def set_panel_row_label(ax, dct, nickname, keys=True):
    if keys:
        # m = max(map(len, dct))
        txt = "\n".join(f"$\\bf{{{nickname(k)}}}$ {v}" for (k, v) in dct.items())
    else:
        txt = "\n".join(f"{v}" for v in dct.values())
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_label_coords(1.03, 0.5)
    ax.set_ylabel(txt, fontsize=14, bbox=lbox, va="bottom", rotation=-90, ha="center")


def validate(orient, data_dims):
    """Validate `orient` structure."""
    seen = []
    for role in orient:
        axes = orient[role]
        if role == "xaxis":
            assert isinstance(axes, str)
            axes = [axes]
        else:
            assert isinstance(axes, list)

        for dim in axes:
            assert dim not in seen, f"'{dim}' listed more than once in `orient`."
            seen.append(dim)
            assert dim in data_dims, f"'{dim}' not found among data."


def sharey_recommended(skill_space):
    """Recommend a `sharey` value for `line_plots`, given the (caller-defined) `skill_space`.

    `skill_space` lists the `orient` roles (e.g. `["fig", "panel_row"]`) whose subspaces the
    caller has independently rescaled (e.g. via a 0-to-1 normalization) before plotting --
    typically paired with `scale01()`/`normalize_spaces()`-like preprocessing done by the caller.
    """
    if "panel_row" in skill_space and "panel_col" in skill_space:
        sharey = "all"
    elif "panel_row" in skill_space:
        sharey = "row"
    elif "panel_col" in skill_space:
        sharey = "col"
    else:
        sharey = False
    return sharey


def scale01(x: xr.DataArray, groupby: list, xaxis: str, min_like="min"):
    """Normalize `x`.

    Performs linear (affine) transformations within each cross-section (subspace)
    indexed by `groupby` so that it has range 0 --> 1
    (other ranges result if using `min_like="mean"` or `"median"`).
    """
    dims = set(x.dims) - set(groupby)
    a = x.max(dims)
    __min = getattr(xr.DataArray, min_like)  # e.g. min(), mean(), or median()
    # b = __min(x.sel({xaxis: 0}, drop=True), dims - {xaxis})
    b = x.min(dims)
    skill = (x - b) / (a - b)
    if (a == b).all():
        print("Warning: all values are equal. Normalization will result in NaNs.")
    return skill


def add_lines(ax, xarr, xdim, ls, vLS, line_registry, kws, mark_stop=False):
    """Plot lines (including those flat/constant) onto `ax`."""

    if xarr.data.nnz == 0:
        return

    ds = sparse_to_series(xarr)
    ds = find_categorical(ds, xdim)
    # xarr = xr.DataArray.from_series(ds, sparse=True)

    # Whether `find_categorical()` split off any flat/constant lines (into the "_CAT_" tick).
    has_flat = ("fix_" + xdim) in ds.index.names

    def plot1(s):
        "Do `plot` (with `mark_stop`) or `axhline` for single data series."
        x = s.index

        if has_flat:
            const = s["_CAT_"]
            s = s.drop("_CAT_")
            x = x.drop("_CAT_")
            if not pd.isna(const):
                return ax.axhline(const, alpha=kws["alpha"], lw=kws["lw"], ls=ls)

        line = ax.plot(x, s, **kws, ls=ls)[0]

        # Add stopping markers
        if mark_stop:
            idx_notna = pd.notna(series).values.nonzero()[0]
            if np.any(idx_notna):
                x = series.index[idx_notna[-1]]
                y = series.iloc[idx_notna[-1]]
                line.scatter = ax.scatter(x, y, kws["ms"] ** 2, marker="s", alpha=kws["alpha"])

        return line

    # Ensure 2-level index
    if ds.index.nlevels == 1:
        ds = pd.concat(dict(DUMMY=ds), names=["singleton_hue"])  # similar to expand_dims
    # Move xaxis from index level to column index
    stack = ds.unstack(xdim)  # NB: causes NaNs if xdim coords differ

    # Legend labels
    labls = stack.index.to_frame(index=False)
    # Add column(s) for the linestyle key
    for dim in vLS:
        labls[dim] = vLS[dim]

    # As (list of) of hashable (for registering) tuples
    labls = list(labls.itertuples(index=False, name="Label"))

    # Plot
    for i, (_coord, series) in enumerate(stack.iterrows()):
        # s = ds.loc[_coord]  # series w/o NaNs, possibly caused by `unstack`
        line = plot1(series)
        line_registry.setdefault(labls[i], []).append(line)


def _legend_parts(handles):
    """Derive `(title, labels, line0s)` for `handles`'s legend from its (deduped) index.

    Recomputed fresh from `handles` every time (rather than cached), so this stays correct
    even after `handles` is mutated in place (e.g. by `groupby(...).sum()`), and works
    uniformly regardless of how many entries `handles` has -- including 0 or 1.
    """
    if len(handles) == 0:
        return "", [], []
    title, *labels = handles.index.to_frame(index=False).to_string(index=False).splitlines()
    line0s = [lines[0] for lines in handles.values]
    return title, labels, line0s


class PartialShow:
    """Simple setting of visibility/glow for many lines, figs, and deduplicated legend.

    `handles`, `fig_registry`: as returned by `line_plots()`. The legend figure/axes is
    assumed to be the last entry of `fig_registry` (as arranged by `line_plots`).

    Usage: `only_show = PartialShow(handles, fig_registry).only_these`
    """

    @staticmethod
    # Modified from mpl dhaitz/mplcyberpunk to add kwargs
    def make_lines_glow(
        ax: plt.Axes | None = None,
        n_glow_lines: int = 10,
        diff_linewidth: float = 1.05,
        alpha_line: float = 0.3,
        lines: plt.Line2D | list[plt.Line2D] | None = None,
        **kwargs,
    ) -> None:
        """Add a glow effect to the lines in an axis object.

        Each existing line is redrawn several times with increasing width and low alpha to create the glow effect.
        """
        if not ax:
            ax = plt.gca()

        lines = ax.get_lines() if lines is None else lines
        lines = [lines] if isinstance(lines, plt.Line2D) else lines

        alpha_value = alpha_line / n_glow_lines

        for line in lines:
            data = line.get_data(orig=False)
            linewidth = line.get_linewidth()

            try:
                step_type = line.get_drawstyle().split("-")[1]
            except IndexError:
                step_type = None

            for n in range(1, n_glow_lines + 1):
                if step_type:
                    (glow_line,) = ax.step(*data)
                else:
                    (glow_line,) = ax.plot(*data)
                glow_line.update_from(
                    line
                )  # line properties are copied as seen in this solution: https://stackoverflow.com/a/54688412/3240855

                if kwargs is not None:
                    glow_line.set(**kwargs)

                glow_line.set_alpha(alpha_value)
                glow_line.set_linewidth(linewidth + (diff_linewidth * n))
                # Mark the glow lines, to disregard them in the underglow function.
                glow_line.is_glow_line = True  # ty: ignore[unresolved-attribute]

    @staticmethod
    def set_visibility(lines, visibility=None, alpha=None):
        for line in lines:
            if line:
                if alpha is not None:
                    line.set_alpha(alpha)
                    if sc := getattr(line, "scatter", None):
                        sc.set_alpha(alpha)
                if visibility is not None:
                    line.set_visible(visibility)
                    if sc := getattr(line, "scatter", None):
                        sc.set_visible(visibility)

    def __init__(self, handles, fig_registry, alpha=None):
        """
        Parameters
        ----------
        handles, fig_registry
            As returned by `line_plots()`.
        alpha : float or (float, float), optional
            Line transparency: `a` for entries in `iShow`, else `b` (default 0), where
            `a, b = alpha` if `alpha` is a pair, else `a = alpha`. By default (`None`),
            use visibility (on/off), instead of transparency, to distinguish `iShow`.
        """
        self.handles = handles
        self.fig_registry = fig_registry
        self.alpha = alpha
        self.current_glow = []

    def only_these(self, iShow, iGlow):
        """Show/hide/glow legend entries `iShow`/`iGlow` (both index `handles`, or `True` for all).

        Parameters
        ----------
        iShow : list or True
            Indices (into `handles`) of the entries to show. If `self.alpha` is set, all
            entries are shown, with `alpha` distinguishing between "in `iShow`" or not
            (rather than the "shown"/hidden line visibility that's used if `alpha=None`).
        iGlow : list or True
            Indices (into `handles`) of the entries to add a glow effect to.
        """
        handles = self.handles
        fig_registry = self.fig_registry
        current_glow = self.current_glow
        alpha = self.alpha

        ax0 = fig_registry[-1].axes[0]
        ALL = list(range(len(handles)))
        if iShow is True:
            iShow = ALL
        if iGlow is True:
            iGlow = ALL

        alert = "#EB811B"  # Beamer Moloch theme alert text color

        # ╔═══════════╗
        # ║ main axes ║
        # ╚═══════════╝
        # Rm previous glow
        while current_glow:
            current_glow.pop(0).remove()

        # Disable autoscale for glow (edges for axes can be very sensitive)
        for fig in fig_registry:
            for ax in fig.axes:
                ax.set_autoscale_on(False)  # Disable autoscaling

        for i, (_idx, lines) in enumerate(handles.items()):
            # Visibility
            if alpha is None:
                # Using `visible`
                self.set_visibility(lines, i in iShow)
            else:
                # Using `alpha`
                try:
                    a, b = alpha
                except TypeError:
                    a = alpha
                    b = 0
                self.set_visibility(lines, alpha=a if i in iShow else b)

            # Glow
            if i in iGlow:
                for line in lines:
                    old_lines = line.axes.get_lines()
                    self.make_lines_glow(
                        line.axes,
                        n_glow_lines=10,
                        diff_linewidth=0.5 + 0.4 / len(lines),
                        alpha_line=0.5 + 0.3 / len(lines),
                        lines=line,
                        color=alert,
                        zorder=1,
                        linestyle="-",
                    )
                    # Find glow lines by difference before and after adding glow
                    new_lines = line.axes.get_lines()
                    line.glow_lines = [ln for ln in new_lines if ln not in old_lines]
                    current_glow.extend(line.glow_lines)

        # ╔════════╗
        # ║ legend ║
        # ╚════════╝
        # NB: simply appending legend lines `handles` enables toggle_visibility(). But
        #   * requires using plt.pause(0.1) beforehand
        #   * the text (label) is not toggled
        #   * savefig() misplaces glow (for all data transformation I tried).
        # So instead, we redraw the legend entirely
        title, labels, line0s = _legend_parts(handles)
        lines = line0s[:]
        for i in iGlow:
            # glow = plt.Line2D([], [], linestyle="-", color=alert, linewidth=6, alpha=0.7)
            # lines[i] = (glow, lines[i])
            lines[i] = (*lines[i].glow_lines, lines[i])
        # Sub-select
        labls = labels[:]
        if alpha is None:
            lines = [lines[i] for i in iShow]
            labls = [labls[i] for i in iShow]
        # Draw
        if not lines:
            ax0.legend([], [])
        else:
            ax0.legend(lines, labls, title=title, **legend_mono)


def line_plots(
    skill: xr.DataArray,
    orient: Struct,
    meta={},
    dim_aliases={},
    aliases={},
    fig_title="",
    sharey=False,
    sharex="col",
    xscale="linear",
    axes_labelcolor=None,
    cmap="tab20",
    possible_linestyles=["-", "--", ":", "-."],
    alpha=0.9,
    lw=3,
    ms=7,
    mark_stop=True,
    ls_once=False,
):
    """
    Parameters
    ----------
    skill : xr.DataArray
        Must use sparse underlying data.
    orient : Struct
        Dict mapping dims of plots to list of dims of `skill`.
    meta : dict, optional
        Additional info to list in legend, by default {}.
    dim_aliases : dict, optional
        Nick names, by default {}.
    fig_title : str, optional
        Figure title (prefix), by default "".
    sharey : str, optional
        "all" | False | "row" | "col", by default False.
        Use `sharey_recommended(skill_space)` to compute a recommended value based on which
        `orient` roles you've independently rescaled (e.g. via `scale01`) before plotting.
    sharex : str, optional
        "all" | False | "row" | "col", by default "col".
    axes_labelcolor : str, optional
        Apply to sup-x/y-labels, which go along with xtics and yticks.
        Not to be confused with `axes.labelcolor` (applies to an axes' xlabel and ylabel),
        which we use for panel row/column indicators.
    cmap : str, optional
        "Spectral", "viridis", "tab20c", "tab20c", "jet", "Dark2", "Accent", by default "tab20".
    possible_linestyles : list, optional
        List of possible line styles, by default ["-", "--", ":", "-."].
    alpha : float, optional
        Alpha value for line transparency, by default 0.9.
    lw : int, optional
        Line width, by default 3.
    ms : int, optional
        Marker size, by default 7.
    mark_stop : bool, optional
        Add square stopping marker, by default True.
    """

    # While it would be nice to adapt marker if a panel has shorter x-axis,
    # that would create confusion in legend (even if duplicate items are included).
    marker = "o" if len(skill[orient.xaxis]) < 9 else None
    kws = dict(lw=lw, ms=ms, alpha=alpha, marker=marker)

    def nickname(dim):
        for key in dim_aliases:
            if key in dim:
                idx = dim.index(key)
                dim = dim[:idx] + dim_aliases[key] + dim[idx + len(key) :]
        return dim

    fig_registry = []
    line_registry = {}

    for vFig in projection(skill, orient.fig):
        fign = f"{fig_title} -- {vFig}"
        clear_fig(fign, figsize=(8, 4))
        fig, axs = plt.subplots(
            num=fign,
            nrows=len(projection(skill, orient.panel_row)),
            ncols=len(projection(skill, orient.panel_col)),
            squeeze=False,
            sharex=True if sharex == "fig" else sharex,
            sharey=sharey,
        )
        fig_registry.append(fig)

        for iPanel, uPanel in enumerate(projection(skill, orient.panel_row)):
            for jPanel, vPanel in enumerate(projection(skill, orient.panel_col)):
                ax = axs[iPanel, jPanel]

                for iLS, vLS in enumerate(projection(skill, orient.linestyle)):
                    ls = possible_linestyles[iLS]

                    for vUnlabelled in projection(skill, orient.unlabelled):
                        # Select cross-section of data to be plotted
                        xSect = skill.sel(
                            {
                                **vFig,
                                **uPanel,
                                **vPanel,
                                **vLS,
                                **vUnlabelled,
                            },
                            drop=True,
                        )
                        add_lines(ax, xSect, orient.xaxis, ls, vLS, line_registry, kws, mark_stop)

                # xscale
                # NOTE: it seems necessary to apply this here (rather than outside this function)
                # because otherwise the xlim autoscaling (ref rcParam "xmargin") won't propely apply.
                # NB: it's best not to fiddle with xlim (herein) because arduous to respect `sharex`.
                if xscale == "log" and ax.get_xlim()[0] <= 0:
                    print("Warning: xscale is 'log' but ∃ vals<=0. Switching to 'symlog'.")
                    xscale = "symlog"  # prevents repeat warnings
                if xscale == "symlog":
                    ax.set_xscale(xscale, linthresh=1, linscale=0.1)
                else:
                    ax.set_xscale(xscale)

                # ylabel
                if isinstance(skill.name, str) and "(%)" in skill.name:
                    ax.set_yticks([0, 25, 50, 75, 100])
                    ax.set_yticklabels([None, 25, 50, 75, 100])

                # grid
                ax.grid(True, "both", axis="both")

                # Sup-x/y-label padding
                gs = ax.get_subplotspec()
                if gs.is_first_col():
                    ax.set_ylabel(" ")
                if gs.is_last_row():
                    ax.set_xlabel(" ")

                # Panel-col/row labels
                if gs.is_first_row():
                    set_panel_col_label(ax, vPanel, nickname, gs.is_first_col())
                if gs.is_last_col():
                    set_panel_row_label(ax, uPanel, nickname, gs.is_first_row())

        with plt.rc_context({} if axes_labelcolor is None else {"text.color": axes_labelcolor}):
            fig.supylabel(str(skill.name), x=0.03, y=0.55)
            fig.supxlabel(nickname(orient.xaxis), y=0.04)
        fig.tight_layout(h_pad=0.1, w_pad=0.1, pad=1.3 if axs.size > 1 else 3.0)

    # MultiIndex-ed lists of line handles
    handles = pd.Series(
        data=list(line_registry.values()),
        index=pd.MultiIndex.from_tuples(line_registry.keys(), names=list(line_registry)[0]._fields),
        name="line handles",
    )

    # Color
    # Ignore linestyle for the purpose of coloring
    color_df = handles.unstack(orient.linestyle)
    if isinstance(color_df, pd.Series):
        color_df = color_df.to_frame()
    for i, (_label, lines) in enumerate(color_df.iterrows()):
        for line in np.ravel(list(lines)):
            # Must be listed colormap
            colr = plt.colormaps[cmap](i % plt.colormaps[cmap].N)
            line.set_color(colr)
            if sc := getattr(line, "scatter", None):
                sc.set_facecolor(colr)

    # Legend
    clear_fig("legend", figsize=(4, 4), frameon=False)
    fig, ax0 = plt.subplots(num="legend")
    ax0.axis("off")

    # Easier to work with df than MultiIndex
    labels = handles.index.to_frame(index=False)

    # Drop unnecessary cols -- unless that would drop *all* of them (e.g. len(handles) <= 1,
    # where every col is trivially "all same"), in which case keep them all so there's still
    # something to show in the legend, and `pd.MultiIndex.from_frame` below doesn't choke on
    # a 0-column frame.
    not_all_same = labels.nunique().gt(1)
    if not not_all_same.any():
        not_all_same[:] = True
    # Add to meta
    for col in labels.columns:
        if not not_all_same[col]:
            meta[col] = str(labels[col].iloc[0])
    labels = labels.loc[:, not_all_same]
    # labels = labels.drop(columns="singleton_hue", errors="ignore")  # rm dummy

    # Each ls only gets labelled once
    if ls_once and orient.linestyle:
        vLS0 = projection(skill, orient.linestyle)[0]
        if vLS0 is not None:
            # Set non-ls coords to "*" (thus creating duplicate labels) if vLS != vLS[0]
            once = (labels[list(vLS)] != vLS0).all(axis=1)
            labels.loc[once, labels.columns.difference(vLS)] = "*"

    # Drop categorical columns (starting with `fix_`)
    labels = labels.loc[:, ~labels.columns.str.startswith("fix_")]
    # OR: prettify
    # labels.columns = [ f"({lbl[4:]})" if lbl.startswith("fix_") else lbl for lbl in labels.columns ]
    # labels = labels.replace("_VAR_", "")

    # Alias
    labels = labels.replace(NONE, "N/A")
    labels = labels.rename(columns=dim_aliases)
    for dim in aliases:
        if dim in labels.columns:
            labels[dim] = labels[dim].replace(aliases[dim])

    # Merge equal rows
    handles.index = pd.MultiIndex.from_frame(labels)
    handles = handles.groupby(level=list(range(handles.index.nlevels))[::-1]).sum()

    # Draw legend
    title, labels, line0s = _legend_parts(handles)
    legend = ax0.legend(line0s, labels, title=title, **legend_mono)

    # Add meta for *all* data
    already_labelled = ["fig", "panel_row", "panel_col", "xaxis", "linestyle"]
    meta = {**{k: [nickname(d) for d in v] for (k, v) in orient.items()}, **meta}
    meta = {k: v for (k, v) in meta.items() if k not in already_labelled}
    meta = yaml.dump(meta, indent=4, Dumper=IndentedDumper, sort_keys=False).replace(NONE, "·")
    meta = meta.rstrip("\n")  # trailing newline
    # Place text below legend
    # Ref stackoverflow.com/q/49355810
    meta = ax0.annotate(
        meta,
        xy=(0, 0),
        xytext=(0, 0),
        xycoords="figure fraction",
        textcoords=OffsetFrom(legend, (0, -0.1)),
        horizontalalignment="left",
        verticalalignment="top",
        size="x-small",
    )
    # Get bbox including both legend and meta
    bb = Bbox.union([legend.get_window_extent(), meta.get_window_extent()])
    # bb = bb.transformed(fig.dpi_scale_trans.inverted())
    handles.total_bbox = bb

    fig_registry.append(fig)

    return fig_registry, handles


def flatten_categorical(df: pd.DataFrame, dim):
    """Undo `find_categorical()`'s split, on an *unstacked* (wide) `df`: broadcast each
    flat/constant line's value -- held in its own, otherwise-empty, `"_CAT_"`-tagged `dim`
    column -- across the regular (numeric) `dim` columns, instead of leaving it stranded in
    its own extra column. This mirrors `add_lines()`'s flat-line treatment (an `axhline`
    spanning the whole axis), but in tabular form.

    The `"fix_" + dim` row level added by `find_categorical()` is kept -- unlike
    `line_plots()`'s legend, which drops it (see "Drop categorical columns" therein),
    relying on the plot itself (an axhline vs. a sloped line) to convey flatness for free.
    A table has no such visual, so the level is kept, collapsed to a boolean flagging
    whether that row/line is flat.
    """
    fix_dim = "fix_" + dim
    is_cat_col = df.columns.get_level_values(dim) == "_CAT_"
    if is_cat_col.any():
        cat, real = df.loc[:, is_cat_col], df.loc[:, ~is_cat_col]
        is_multi = isinstance(df.columns, pd.MultiIndex)
        real_key = real.columns.droplevel(dim) if is_multi else pd.Index([0] * real.shape[1])
        cat_key = cat.columns.droplevel(dim) if is_multi else pd.Index([0] * cat.shape[1])
        cat.columns = cat_key
        fill = cat.reindex(columns=real_key)
        fill.columns = real.columns
        df = real.where(real.notna(), fill)

    if fix_dim in df.index.names:
        idx = df.index.to_frame(index=False)
        idx[fix_dim] = idx[fix_dim] != "_VAR_"
        df.index = pd.MultiIndex.from_frame(idx)

    return df


def shape_tables(skill, orient, dim_aliases={}, col_dims=None, find_cat=False):
    """Print `skill` as a single table, instead of plotting it via `line_plots()`.

    `orient.xaxis` and `col_dims` are unstacked into a (`MultiIndex`'d, if
    `col_dims` is non-empty) column index. The row `MultiIndex` is reordered so
    `orient.fig` is the outermost level(s), followed by `orient.panel_row`, then
    everything else (`linestyle`, `unlabelled`, plus whatever's left over for
    "hue") -- mirroring `line_plots()`'s figure-then-row nesting.

    Parameters
    ----------
    skill : xr.DataArray
        Must use sparse underlying data.
    orient : Struct
        As passed to `line_plots()`.
    dim_aliases : dict, optional
        Nick names for dims, used in the row/column index names.
    col_dims : list, optional
        Extra dims (besides `orient.xaxis`) to unstack into (outer levels of) the
        column `MultiIndex`. By default (`None`), uses `orient.panel_col` --
        mirroring `line_plots()`'s use of `panel_col` for (visually) side-by-side
        panels. Pass `[]` for single-level (`xaxis`-only) columns.
    find_cat : bool, optional
        Whether to treat non-numeric `xaxis` ticks (see `find_categorical()`) as
        flat/constant lines, mirroring `line_plots()`/`add_lines()`: rather than
        adding an extra `xaxis` tick/column (NaN for every other line, and NaN
        at every other tick for this one), the value is broadcast across the
        regular (numeric) `xaxis` ticks -- see `flatten_categorical()`. A
        `"fix_" + orient.xaxis` row level then flags (boolean) which lines were
        flattened this way. By default `False`.

    Returns
    -------
    pd.DataFrame
        Row `MultiIndex`: `orient.fig`, then `orient.panel_row`, then the rest
        (including a `"fix_" + orient.xaxis` boolean level if `find_cat`).
        Column index: `col_dims`, then `orient.xaxis`.
    """
    if col_dims is None:
        col_dims = orient.panel_col

    row_dims = [*orient.fig, *orient.panel_row]

    ds = sparse_to_series(skill)
    if find_cat:
        ds = find_categorical(ds, orient.xaxis)
    df = ds.unstack([*col_dims, orient.xaxis])  # NB: causes NaNs if xaxis/col_dims coords differ
    if find_cat:
        df = flatten_categorical(df, orient.xaxis)

    fix_dim = "fix_" + orient.xaxis
    etc_dims = [d for d in df.index.names if d not in row_dims and d != fix_dim]
    if fix_dim in df.index.names:
        etc_dims.append(fix_dim)
    df = df.reorder_levels([*row_dims, *etc_dims], axis=0)
    df = df.sort_index(axis=0).sort_index(axis=1)
    df = df.rename_axis(index=dim_aliases, columns=dim_aliases)

    return df


def save_all(
    figures,
    img_dir,
    data_dir,
    meta,
    orient,
    handles,
    tags=tuple(),
    ext="pdf",
    yank=False,
    script=None,
    seconds=300,
):
    """Save `figures` (as returned by `line_plots()`) to `img_dir`.

    Parameters
    ----------
    figures : list of Figure
        `fig_registry`, as returned by `line_plots()`.
    img_dir : Path
        Directory to save into.
    data_dir : Path
        Its `.name` (e.g. a timestamp) is included in the generated filename.
    meta, orient : dict, Struct
        Included (stringified) in the generated filename as data-processing info.
    handles
        As returned by `line_plots()`; `handles.total_bbox` is used as the legend's bbox.
    tags : tuple, optional
        Custom tags (e.g. "backup", "dirty") appended to the filename, by default ().
    ext : str, optional
        File extension/format passed to `fig.savefig`, by default "pdf".
    yank : bool or "first", optional
        Copy the filename(s) to the clipboard, by default False.
    script : str, optional
        Passed to `confirm_cold_call` (typically the caller's `__file__`).
        Prevents a "cold" re-run (e.g. re-opening the script after a long time)
        from overwriting existing figures unless user confirms, while rapid
        successive calls (e.g. iterating on plot styling) don't nag on every save.
        By default `None`, which skips `confirm_cold_call` and always saves.
    seconds : int, optional
        Passed to `confirm_cold_call`, by default 300.
    """
    for i, fig in enumerate(figures):
        parts = [
            data_dir.name,  # timestamp (≈implies git dir and sha)
            str({**meta, **orient})[1:-1],  # data processing info
            fig.get_label().split(" -- ")[-1],  # fig title/label
            *tags,  # "backup", "dirty", ...,  # custom tags
        ]
        # Sanitize for file-naming.
        # Use sub-dirs to limit filename length (constrained on many systems)
        parts = [
            part.replace(": ", "=").replace("'", "").replace(",", "").replace("/", "-")
            for part in parts
        ]
        rel_path = Path(*parts[:-1], f"{parts[-1]}.{ext}")
        name = str(rel_path)

        # Facilitate importing into slides.tex
        if yank and (yank is True or i == 0):
            print("* " + name)
            yanker(name, append=i)  # copy to clipboard

        bbox = "tight"
        # Keep legend box size constant accross overlays
        if "legend" == fig.get_label():
            bbox = handles.total_bbox.transformed(fig.dpi_scale_trans.inverted())

        out_path = img_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        @confirm_cold_call(script, seconds)
        def save_figure():
            fig.savefig(out_path, bbox_inches=bbox, pad_inches=0.05)
