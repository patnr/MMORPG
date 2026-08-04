"""
Rationale
-----
`xarray.plot.line` is great, but has some limitations (examples: cannot allocate
>1 dim to "hue"; cannot re-use figs if supplying kwargs `row` and `col`)
and requires some workarounds that make it easier to just do the processing ourselves.

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
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from IPython.utils.ipstruct import Struct  # One of many "Bunch" variants
from matplotlib.text import OffsetFrom
from matplotlib.ticker import ScalarFormatter, SymmetricalLogLocator
from matplotlib.transforms import Bbox
from pandas import pandas as pd

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


# Modified from mpl dhaitz/mplcyberpunk to add kwargs
def make_lines_glow(
    ax: Optional[plt.Axes] = None,
    n_glow_lines: int = 10,
    diff_linewidth: float = 1.05,
    alpha_line: float = 0.3,
    lines: Union[plt.Line2D, List[plt.Line2D]] = None,
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
        except:
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
            glow_line.is_glow_line = (
                True  # mark the glow lines, to disregard them in the underglow function.
            )


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


def pd_sel(s: pd.Series, dct, drop=True):
    """Select pandas Series/DataFrame by `.loc[]` applied to a dict.

    Similar to `xarray.DataArray.sel`, but for pandas. Alternatives:

    - `df.xs()` is for cross-sectioning, i.e. selecting a single value.
    - `df.iloc[]` is for integer-location based indexing.

    If `drop` then drop singleton levels of the _resulting_ index.

    Example:

    >>> dct = dict(
    ...     seed=3002,
    ...     case="quadratic",
    ...     sdev=slice(None),
    ...     method="BFGS",
    ...     iter=[0, 1, *range(7, 15)],
    ... )
    >>> sub_df = pd_sel(df, dct, True)

    You may need to do `s = s.sort_index()` beforehand.
    """
    idx = tuple(dct.get(k, slice(None)) for k in s.index.names)
    sub = s.loc[idx]
    if drop:
        lvls = [k for k in sub.index.names if sub.index.get_level_values(k).nunique() == 1]
        sub = sub.droplevel(lvls)
    return sub


def find_categorical(ds, dim):
    """Find categorical (or nan) values (of `dim`). Add corresponding dimension (index level)."""
    ticks = ds.index.get_level_values(dim)
    numeric = pd.to_numeric(ticks, errors="coerce").notna()
    ticks = ticks[numeric].drop_duplicates().astype(float)
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
    if "panel_row" in skill_space and "panel_col" in skill_space:
        sharey = "all"
    elif "panel_row" in skill_space:
        sharey = "row"
    elif "panel_col" in skill_space:
        sharey = "col"
    else:
        sharey = False
    return sharey


def add_lines(ax, xarr, xdim, ls, vLS, line_registry, kws, mark_stop=False):
    """Plot lines (including those flat/constant) onto `ax`."""

    if xarr.data.nnz == 0:
        return

    ds = sparse_to_series(xarr)
    ds = find_categorical(ds, xdim)
    # xarr = xr.DataArray.from_series(ds, sparse=True)

    def plot1(s):
        "Do `plot` (with `mark_stop`) or `axhline` for single data series."
        x = s.index

        # coords[xdim][-1] "_CAT_" is caused by find_categorical()
        if x[-1] == "_CAT_":
            *s, const = s
            *x, _na = x
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


def line_plots(
    skill: xr.DataArray,
    orient: Struct,
    meta={},
    dim_aliases={},
    aliases={},
    fig_title="",
    sharey="auto",
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
        "all" | False | "row" | "col" | "auto", by default "auto".
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
                if "(%)" in skill.name:
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

        with plt.rc_context({"text.color": axes_labelcolor}):
            fig.supylabel(skill.name, x=0.03, y=0.55)
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
    for i, (label, lines) in enumerate(color_df.iterrows()):
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

    if len(handles) <= 1:
        legend = ax0.legend([], [], title="", **legend_mono)
    else:
        # Easier to work with df than MultiIndex
        labels = handles.index.to_frame(index=False)

        # Drop unnecessary cols
        not_all_same = labels.nunique().gt(1)
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

        # Finalize legend
        title, *labels = handles.index.to_frame(index=False).to_string(index=False).splitlines()
        line0s = [lines[0] for lines in handles.values]

        # Draw legend
        legend = ax0.legend(line0s, labels, title=title, **legend_mono)

    current_glow = []

    def partial_show(iShow, iGlow, alpha=None):
        ALL = list(range(len(handles)))
        if iShow is True:
            iShow = ALL
        if iGlow is True:
            iGlow = ALL

        alert = "#EB811B"  # Beamer Moloch theme alert text color

        # ╔═══════════╗
        # ║ main plot ║
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
                set_visibility(lines, i in iShow)
            else:
                # Using `alpha`
                try:
                    a, b = alpha
                except TypeError:
                    a = alpha
                    b = 0
                set_visibility(lines, alpha=a if i in iShow else b)

            # Glow
            if i in iGlow:
                for line in lines:
                    old_lines = line.axes.get_lines()
                    make_lines_glow(
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
        # The following appends legend lines to regular/main plot lines
        # >>> for i, line in enumerate(legend.get_lines()):
        # >>>     handles.iloc[i].append(line)
        # Thus, lagend handles would also get treated by toggle_visibility(). But
        #   * requires using plt.pause(0.1) beforehand
        #   * the text (label) is not toggled
        #   * savefig() misplaces glow (for all data transformation I tried).
        # ⇒ Redraw the legend, using tuple handles
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

    handles.show_partial = partial_show

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
