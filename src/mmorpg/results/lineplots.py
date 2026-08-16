"""Build multi-panel line plots from a sparse `xr.DataArray`.

Rationale
---------
`xarray.plot.line` is great, but has some limitations (examples: cannot allocate
>1 dim to "hue"; cannot re-use figs if supplying kwargs `row` and `col`)
and requires some workarounds that make it easier to just do the processing ourselves
-- hence `LinePlots`.

Meanwhile it's tempting to just call `plt.plot(..., shape_tables())`
since `shape_tables()` uses a few pandas API calls,
which is seemingly reinvented by the more elaborate `LinePlots` (relying on xarray).
But `LinePlots` nests `.sel()` calls (one per fig/panel/linestyle/unlabelled leaf) and only
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

import json
import subprocess
import sys
import time
import warnings
from functools import wraps
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

from . import NONE, find_categorical, projection, sparse_to_series

# Running in iPython?
ip = __import__("IPython").get_ipython() if "IPython" in sys.modules else None


def confirm_cold_call(script: str | None, seconds: int = 300):
    """Run decorated function only if it was last run within `seconds`, or by user confirmation."""

    def decorator(func):
        @wraps(func)
        def wrapper():
            if script is None:
                func()
                return

            # Already cancelled in this ipytho session ⇒ re-cancel
            if ip:
                fkey = (func.__name__, func.__module__)
                if confirm_cold_call.register.get(fkey, None) == ip.execution_count:
                    print(f"Re-ignoring invocation of {func.__name__}.")
                    return

            # ╔═══════════╗
            # ║ timestamp ║
            # ╚═══════════╝
            script_path = Path(script)
            timestamp_file = script_path.parent / ".call_timestamps"

            # write
            def update_timestamp():
                timestamps[str(script_path)] = time.time()
                with open(timestamp_file, "w") as f:
                    json.dump(timestamps, f)

            # read
            if timestamp_file.exists():
                with open(timestamp_file) as f:
                    timestamps = json.load(f)
            else:
                timestamps = {}

            # check
            now = time.time()
            need_confirmation = True
            last_run = timestamps.get(str(script_path))
            if last_run is not None and now - last_run <= seconds:
                need_confirmation = False

            def cancel():
                print("Operation cancelled.")
                if ip:
                    fkey = (func.__name__, func.__module__)
                    confirm_cold_call.register[fkey] = ip.execution_count

            def call():
                func()
                update_timestamp()

            # ╔══════════════════════╗
            # ║ ask for confirmation ║
            # ╚══════════════════════╝
            try:
                if need_confirmation:
                    print(
                        f"It's been more than {seconds // 60}m since confirmed invocation."
                        f" You sure you want to {func.__name__}?"
                    )
                    try:
                        if input("Confirm [y/N]: ").strip().lower() == "y":
                            call()
                        else:
                            cancel()
                    except KeyboardInterrupt:
                        print()  # To move to a new line after Ctrl-C
                        cancel()
                else:
                    call()
            finally:
                pass

        wrapper()

        # return wrapper
        return lambda *a, **b: print("Function already called (in decorator).")

    return decorator


confirm_cold_call.register = {}


def yank(txt, append=False):
    "Copy to clipboard (mac/darwin)."

    if append:
        old = subprocess.check_output("pbpaste", env={"LANG": "en_US.UTF-8"}).decode("utf-8")
        txt = old + "\n" + txt

    process = subprocess.Popen("pbcopy", env={"LANG": "en_US.UTF-8"}, stdin=subprocess.PIPE)
    process.communicate(txt.encode("utf-8"))


# Alias: `LinePlots.save`/`save_all` take `yank` as a bool parameter, shadowing the function.
yanker = yank



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


class LinePlots:
    """Build multi-panel line plots from a sparse `xr.DataArray`, with legend dedup and
    partial-show/glow controls for progressive reveal, plus figure saving.

    Consolidates what used to be free functions `line_plots()` + `add_lines()` +
    `_legend_parts()` + `set_panel_col_label()`/`set_panel_row_label()`/`clear_fig()` plus the
    `PartialShow` class: the post-construction state (`fig_registry`, `handles`) that used to
    be threaded by hand across `line_plots()` -> `PartialShow(...)` -> `save_all(...,
    handles=...)` now just lives on `self`. `line_registry` (the raw, un-deduped accumulator
    `_add_lines()` fills in across the fig/panel/linestyle/hue loop) stays a local variable in
    `_plot()` -- it's only needed to build `handles`, never read afterward.

    `only_these()` and `save()` are the two methods meant to be called after construction;
    everything else (prefixed `_`) is internal wiring for `__init__`.

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
        Use `LinePlots.sharey_recommended(skill_space)` to compute a recommended value based
        on which `orient` roles you've independently rescaled (e.g. via `scale01`) before
        plotting.
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
    ls_once : bool, optional
        Each linestyle only gets labelled once, by default False.
    show_alpha : float or (float, float), optional
        Used by `only_these()`: line transparency `a` for entries in `iShow`, else `b`
        (default 0), where `a, b = show_alpha` if `show_alpha` is a pair, else `a = show_alpha`.
        By default (`None`), `only_these()` uses visibility (on/off), instead of transparency,
        to distinguish `iShow`.
    """

    @staticmethod
    def sharey_recommended(skill_space):
        """Recommend a `sharey` value, given the (caller-defined) `skill_space`.

        `skill_space` lists the `orient` roles (e.g. `["fig", "panel_row"]`) whose subspaces
        the caller has independently rescaled (e.g. via a 0-to-1 normalization) before
        plotting -- typically paired with `scale01()`/`normalize_spaces()`-like preprocessing
        done by the caller.
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

    def __init__(
        self,
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
        show_alpha=None,
    ):
        self.orient = orient
        self.dim_aliases = dim_aliases
        self.meta = meta
        self.show_alpha = show_alpha
        self.current_glow = []
        self.fig_registry = []
        self.handles = None

        self._plot(
            skill,
            dim_aliases,
            aliases,
            fig_title,
            sharey,
            sharex,
            xscale,
            axes_labelcolor,
            cmap,
            possible_linestyles,
            alpha,
            lw,
            ms,
            mark_stop,
            ls_once,
        )

    # ── Internal wiring for __init__ ──

    def _nickname(self, dim):
        for key in self.dim_aliases:
            if key in dim:
                idx = dim.index(key)
                dim = dim[:idx] + self.dim_aliases[key] + dim[idx + len(key) :]
        return dim

    @staticmethod
    def _clear_fig(num, figsize=None, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            plt.figure(num=num, figsize=figsize, **kwargs).clear()

    @staticmethod
    def _set_panel_col_label(ax, dct, nickname, keys=True):
        if keys:
            txt = "\n".join(f"$\\bf{{{nickname(k)}}}$ {v}" for (k, v) in dct.items())
        else:
            txt = "\n".join(f"{v}" for v in dct.values())
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_label_coords(0.5, 1.03)
        ax.set_xlabel(txt, fontsize=14, bbox=lbox, va="bottom", ha="center")

    @staticmethod
    def _set_panel_row_label(ax, dct, nickname, keys=True):
        if keys:
            # m = max(map(len, dct))
            txt = "\n".join(f"$\\bf{{{nickname(k)}}}$ {v}" for (k, v) in dct.items())
        else:
            txt = "\n".join(f"{v}" for v in dct.values())
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_label_coords(1.03, 0.5)
        ax.set_ylabel(txt, fontsize=14, bbox=lbox, va="bottom", rotation=-90, ha="center")

    @staticmethod
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

    @staticmethod
    def _add_lines(ax, xarr, xdim, ls, vLS, line_registry, kws, mark_stop=False):
        """Plot lines (including those flat/constant) onto `ax`, registering them into
        `line_registry` (mutated in place)."""

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

    def _plot(
        self,
        skill,
        dim_aliases,
        aliases,
        fig_title,
        sharey,
        sharex,
        xscale,
        axes_labelcolor,
        cmap,
        possible_linestyles,
        alpha,
        lw,
        ms,
        mark_stop,
        ls_once,
    ):
        orient = self.orient
        nickname = self._nickname

        # While it would be nice to adapt marker if a panel has shorter x-axis,
        # that would create confusion in legend (even if duplicate items are included).
        marker = "o" if len(skill[orient.xaxis]) < 9 else None
        kws = dict(lw=lw, ms=ms, alpha=alpha, marker=marker)

        fig_registry = self.fig_registry
        line_registry = {}

        for vFig in projection(skill, orient.fig):
            fign = f"{fig_title} -- {vFig}"
            self._clear_fig(fign, figsize=(8, 4))
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
                            self._add_lines(
                                ax, xSect, orient.xaxis, ls, vLS, line_registry, kws, mark_stop
                            )

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
                        self._set_panel_col_label(ax, vPanel, nickname, gs.is_first_col())
                    if gs.is_last_col():
                        self._set_panel_row_label(ax, uPanel, nickname, gs.is_first_row())

            with plt.rc_context({} if axes_labelcolor is None else {"text.color": axes_labelcolor}):
                fig.supylabel(str(skill.name), x=0.03, y=0.55)
                fig.supxlabel(nickname(orient.xaxis), y=0.04)
            fig.tight_layout(h_pad=0.1, w_pad=0.1, pad=1.3 if axs.size > 1 else 3.0)

        # MultiIndex-ed lists of line handles
        handles = pd.Series(
            data=list(line_registry.values()),
            index=pd.MultiIndex.from_tuples(
                line_registry.keys(), names=list(line_registry)[0]._fields
            ),
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
        self._clear_fig("legend", figsize=(4, 4), frameon=False)
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
                self.meta[col] = str(labels[col].iloc[0])
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
        title, labels, line0s = self._legend_parts(handles)
        legend = ax0.legend(line0s, labels, title=title, **legend_mono)

        # Add meta for *all* data
        already_labelled = ["fig", "panel_row", "panel_col", "xaxis", "linestyle"]
        full_meta = {**{k: [nickname(d) for d in v] for (k, v) in orient.items()}, **self.meta}
        full_meta = {k: v for (k, v) in full_meta.items() if k not in already_labelled}
        full_meta = yaml.dump(full_meta, indent=4, Dumper=IndentedDumper, sort_keys=False).replace(
            NONE, "·"
        )
        full_meta = full_meta.rstrip("\n")  # trailing newline
        # Place text below legend
        # Ref stackoverflow.com/q/49355810
        meta_text = ax0.annotate(
            full_meta,
            xy=(0, 0),
            xytext=(0, 0),
            xycoords="figure fraction",
            textcoords=OffsetFrom(legend, (0, -0.1)),
            horizontalalignment="left",
            verticalalignment="top",
            size="x-small",
        )
        # Get bbox including both legend and meta
        bb = Bbox.union([legend.get_window_extent(), meta_text.get_window_extent()])
        # bb = bb.transformed(fig.dpi_scale_trans.inverted())
        handles.total_bbox = bb

        fig_registry.append(fig)

        self.handles = handles

    # ── Post-construction: partial show/hide/glow ──

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

    def only_these(self, iShow, iGlow):
        """Show/hide/glow legend entries `iShow`/`iGlow` (both index `self.handles`, or `True` for all).

        Parameters
        ----------
        iShow : list or True
            Indices (into `self.handles`) of the entries to show. If `self.show_alpha` is set,
            all entries are shown, with alpha distinguishing between "in `iShow`" or not
            (rather than the "shown"/hidden line visibility that's used if `show_alpha=None`).
        iGlow : list or True
            Indices (into `self.handles`) of the entries to add a glow effect to.
        """
        handles = self.handles
        fig_registry = self.fig_registry
        current_glow = self.current_glow
        show_alpha = self.show_alpha

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
            if show_alpha is None:
                # Using `visible`
                self.set_visibility(lines, i in iShow)
            else:
                # Using `alpha`
                try:
                    a, b = show_alpha
                except TypeError:
                    a = show_alpha
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
        title, labels, line0s = self._legend_parts(handles)
        lines = line0s[:]
        for i in iGlow:
            # glow = plt.Line2D([], [], linestyle="-", color=alert, linewidth=6, alpha=0.7)
            # lines[i] = (glow, lines[i])
            lines[i] = (*lines[i].glow_lines, lines[i])
        # Sub-select
        labls = labels[:]
        if show_alpha is None:
            lines = [lines[i] for i in iShow]
            labls = [labls[i] for i in iShow]
        # Draw
        if not lines:
            ax0.legend([], [])
        else:
            ax0.legend(lines, labls, title=title, **legend_mono)

    # ── Post-construction: saving figures ──

    def save(self, img_dir, data_dir, tags=tuple(), ext="pdf", yank=False, script=None, seconds=300):
        """Save `self.fig_registry` to `img_dir`. See `save_all()` for parameter docs."""
        self.save_all(
            self.fig_registry,
            img_dir,
            data_dir,
            self.meta,
            self.orient,
            self.handles,
            tags=tags,
            ext=ext,
            yank=yank,
            script=script,
            seconds=seconds,
        )

    @staticmethod
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
        """Save `figures` (e.g. `LinePlots.fig_registry`) to `img_dir`.

        Usually called via `save()`, which fills in `figures`/`meta`/`orient`/`handles` from
        `self`. Kept callable directly too (`LinePlots.save_all(...)`), for saving figures
        built some other way.

        Parameters
        ----------
        figures : list of Figure
            `LinePlots.fig_registry` (i.e. `.fig_registry` of a `LinePlots` instance).
        img_dir : Path
            Directory to save into.
        data_dir : Path
            Its `.name` (e.g. a timestamp) is included in the generated filename.
        meta, orient : dict, Struct
            Included (stringified) in the generated filename as data-processing info.
        handles
            `LinePlots.handles`; `handles.total_bbox` is used as the legend's bbox.
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
