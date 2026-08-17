"""Build/filter/reshape results tables (pandas/xarray only -- no plotting).

`shape_tables()` is the non-plotting sibling of `LinePlots` (see `lineplots.py`): both
consume a sparse `skill: xr.DataArray` shaped by an `orient` `Struct`.
"""

import itertools
import warnings

import numpy as np
import xarray as xr
from pandas import pandas as pd

# Also re-exports `get_data_dir`/`load_data` from `mmorpg.dispatch` since results-processing
# scripts invariably need them to locate/load the data they then feed into `shape_tables()`
# or `LinePlots`.
from mmorpg.dispatch.paths import get_data_dir, load_data

__all__ = [
    "NONE",
    "dicts2index",
    "enum_explode",
    "find_categorical",
    "find_crashed",
    "flatten_categorical",
    "get_data_dir",
    "is_crashed",
    "load_data",
    "pd_sel",
    "projection",
    "scale01",
    "shape_tables",
    "sparse_to_series",
    "validate",
]

# None-like value that pandas does not convert to NaN. Will show as blank in legend.
# PS: Cannot use actual `None` because gets re-converted to NaN unless `dtype` is object,
# but neither `df.set_index()` nor `pd.MultiIndex.from_frame()` support arg `dtype`.
# Try `set_levels(..., df.astype(object).where(df.notna(), None), ...)` ?
NONE = "NA"


def dicts2index(xps) -> pd.Index:
    """Convert a list of experiment-input dicts (e.g. `xps` from `load_data(data_dir)`) into
    the `pd.MultiIndex` of a to-be-populated results table/`DataArray`.

    - Drops exact duplicates (which don't make sense, and don't fit with xarray).
    - Gives functions/callables a prettier repr (their `__name__`), column-wise for speed.
    - The df instantiation converts missing keys and `None`s to NaN, which would make
      {missing key, `None`, NaN} indistinguishable -- except `dispatch()` fills each `xp` with
      `fun`'s own defaults for any key it omits before ever saving `xps` (see
      `dispatch._fill_fun_defaults()`), so a *bare* missing key shouldn't reach here in
      practice. What's still conflated is explicit `None` vs. actual NaN values *within* `xps`:
      both become NaN, and since NaN isn't supported in _sparse_ xarray coords, both get
      replaced below by the sentinel string `NONE`.
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


def is_crashed(r) -> bool:
    """Whether `r` (a single entry of `res`, e.g. from `load_data(data_dir)`) is a crashed
    result: the `(exception, traceback_str)` tuple that `local_mp.mp(..., log_errors=True)`
    substitutes for a per-item exception, instead of raising and killing the whole batch.
    """
    return isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], Exception) and isinstance(r[1], str)


def find_crashed(res, warn=True) -> np.ndarray:
    """Boolean mask over `res` (e.g. from `load_data(data_dir)`) flagging crashed entries --
    see `is_crashed()`. Downstream code otherwise has to hand-roll this same check per script,
    which tends to fold a *crash* into the same NaN as a legitimately bad result, discarding
    the crash count/type in the process. `warn` surfaces both instead, e.g.:

        crashed = find_crashed(res)
        stats = [nan_stats if c else result2stats(r) for c, r in zip(crashed, res)]
    """
    crashed = np.fromiter((is_crashed(r) for r in res), dtype=bool, count=len(res))
    if warn and crashed.any():
        kinds = pd.Series([type(res[i][0]).__name__ for i in np.flatnonzero(crashed)])
        warnings.warn(f"{crashed.sum()}/{len(res)} results crashed: {kinds.value_counts().to_dict()}")
    return crashed


def enum_explode(s: pd.Series, idx_name):
    """Explode a pandas Series and enumerate (as new index level)."""
    t = s.explode()
    enums = t.groupby(level=t.index.names).cumcount()
    t.index = pd.MultiIndex.from_tuples(
        [(*idx, enum) for idx, enum in zip(t.index, enums)],
        names=[*t.index.names, idx_name],
    )
    return t


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


# ── Preparing `skill`/`orient` for `LinePlots`/`shape_tables` ──


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


def find_categorical(ds, dim):
    """Find categorical (or nan) values (of `dim`). Add corresponding dimension (index level).

    Also see `flatten_categorical()`, which undoes this split on an unstacked (wide) `df`.
    """
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


def flatten_categorical(df: pd.DataFrame, dim):
    """Undo `find_categorical()`'s split, on an *unstacked* (wide) `df`: broadcast each
    flat/constant line's value -- held in its own, otherwise-empty, `"_CAT_"`-tagged `dim`
    column -- across the regular (numeric) `dim` columns, instead of leaving it stranded in
    its own extra column. This mirrors `LinePlots`'s flat-line treatment (an `axhline`
    spanning the whole axis), but in tabular form.

    The `"fix_" + dim` row level added by `find_categorical()` is kept -- unlike
    `LinePlots`'s legend, which drops it (see "Drop categorical columns" therein),
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
    """Print `skill` as a single table, instead of plotting it via `LinePlots`.

    `orient.xaxis` and `col_dims` are unstacked into a (`MultiIndex`'d, if
    `col_dims` is non-empty) column index. The row `MultiIndex` is reordered so
    `orient.fig` is the outermost level(s), followed by `orient.panel_row`, then
    everything else (`linestyle`, `unlabelled`, plus whatever's left over for
    "hue") -- mirroring `LinePlots`'s figure-then-row nesting.

    Parameters
    ----------
    skill : xr.DataArray
        Must use sparse underlying data.
    orient : Struct
        As passed to `LinePlots`.
    dim_aliases : dict, optional
        Nick names for dims, used in the row/column index names.
    col_dims : list, optional
        Extra dims (besides `orient.xaxis`) to unstack into (outer levels of) the
        column `MultiIndex`. By default (`None`), uses `orient.panel_col` --
        mirroring `LinePlots`'s use of `panel_col` for (visually) side-by-side
        panels. Pass `[]` for single-level (`xaxis`-only) columns.
    find_cat : bool, optional
        Whether to treat non-numeric `xaxis` ticks (see `find_categorical()`) as
        flat/constant lines, mirroring `LinePlots`: rather than
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
