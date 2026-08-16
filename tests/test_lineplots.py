"""Tests for `mmorpg.results`/`mmorpg.results.lineplots`, focused on `shape_tables()` and its
relationship to `line_plots()`.

The workflow mirrored here (aggregate over `seed`, tune/pick-best over a regularization-like
dim, then either tabulate via `shape_tables()` or plot via `line_plots()`) is a simplified
version of the pattern used in DPhil/EnOpt/Hessian/Singleton-Results.py.
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
import xarray as xr
from IPython.utils.ipstruct import Struct

from mmorpg.results import NONE, shape_tables
from mmorpg.results.lineplots import LinePlots


def build_skill(rows, index_names, mean_dims, tuned_dims):
    """Build a `skill` `xr.DataArray`, mirroring Singleton-Results.py: load raw
    per-(seed, reg, ...) results into a sparse `DataArray`, then average over
    `mean_dims` (e.g. `seed`) and take the best (`max`) over `tuned_dims`
    (e.g. regularization strength).
    """
    df = pd.DataFrame(rows).set_index(index_names)["skill"]
    xa = xr.DataArray.from_series(df, sparse=True)
    xa.name = "skill"
    skill = xa.mean(mean_dims, skipna=False)
    skill = skill.max(tuned_dims, skipna=True)
    return skill


class TestShapeTablesGolden:
    """Full aggregate-then-tabulate pipeline, checked against hand-computed golden values.

    `method="flat"` lines don't depend on `nEns` at all (stored at the sentinel tick
    `NONE`), exercising `find_cat=True`'s flattening on top of the aggregation.
    """

    CASES = ["A", "B"]
    NDIMS = [2, 4]
    ASPECTS = [1, 2]
    METHODS = ["varies", "flat"]
    REGS = [0.1, 1.0]
    SEEDS = [10, 20, 30]  # mean-invariant (no seed-dependent term) -- kept for realism
    NENSS = [10, 20, 40]

    @staticmethod
    def case_n(case):
        return {"A": 0, "B": 1}[case]

    def base(self, case, nDim, aspect):
        return 1.0 + 0.1 * self.case_n(case) + 0.01 * nDim + 0.001 * aspect

    def base_flat(self, case, nDim, aspect):
        return 0.5 + 0.1 * self.case_n(case) + 0.01 * nDim + 0.001 * aspect

    def make_rows(self):
        rows = []
        for case in self.CASES:
            for nDim in self.NDIMS:
                for aspect in self.ASPECTS:
                    for reg in self.REGS:
                        for seed in self.SEEDS:
                            # method="varies": value depends on nEns.
                            for nEns in self.NENSS:
                                val = self.base(case, nDim, aspect) + 0.0001 * reg + 1.0 / nEns
                                rows.append(
                                    dict(
                                        case=case,
                                        nDim=nDim,
                                        aspect=aspect,
                                        method="varies",
                                        reg=reg,
                                        seed=seed,
                                        nEns=nEns,
                                        skill=val,
                                    )
                                )
                            # method="flat": constant over nEns -> stored at nEns=NONE.
                            val = self.base_flat(case, nDim, aspect) + 0.0001 * reg
                            rows.append(
                                dict(
                                    case=case,
                                    nDim=nDim,
                                    aspect=aspect,
                                    method="flat",
                                    reg=reg,
                                    seed=seed,
                                    nEns=NONE,
                                    skill=val,
                                )
                            )
        return rows

    @pytest.fixture(scope="class")
    def skill(self):
        rows = self.make_rows()
        index_names = ["case", "nDim", "aspect", "method", "reg", "seed", "nEns"]
        return build_skill(rows, index_names, mean_dims=["seed"], tuned_dims=["reg"])

    @pytest.fixture(scope="class")
    def orient(self):
        return Struct(
            fig=["case"],
            panel_row=[],
            panel_col=["nDim"],
            linestyle=["aspect"],
            unlabelled=[],
            xaxis="nEns",
        )

    def golden_table(self):
        """Expected table, computed independently of `shape_tables()`'s own pivoting logic.

        `seed` doesn't affect the value (mean is a no-op), and `reg=1.0` always wins the
        max (the reg term is monotonically increasing in `reg`).
        """
        rows = {}
        for case in self.CASES:
            for aspect in self.ASPECTS:
                for method in self.METHODS:
                    row_key = (case, aspect, method, method == "flat")
                    row = {}
                    for nDim in self.NDIMS:
                        for nEns in self.NENSS:
                            if method == "varies":
                                val = self.base(case, nDim, aspect) + 0.0001 + 1.0 / nEns
                            else:
                                val = self.base_flat(case, nDim, aspect) + 0.0001
                            row[(nDim, nEns)] = val
                    rows[row_key] = row
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index = pd.MultiIndex.from_tuples(
            df.index, names=["case", "aspect", "method", "fix_nEns"]
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["nDim", "nEns"])
        return df.sort_index(axis=0).sort_index(axis=1)

    def test_golden_values(self, skill, orient):
        table = shape_tables(skill, orient, find_cat=True)
        golden = self.golden_table()
        pd.testing.assert_frame_equal(
            table,
            golden,
            check_exact=False,
            rtol=1e-9,
            check_dtype=False,
            check_index_type=False,
            check_column_type=False,
        )


class TestLinePlotsMatchesShapeTables:
    """Cross-check: the line data `line_plots()` actually draws should match the
    corresponding row of `shape_tables()`'s output, for both a varying and a flat line.
    """

    ASPECTS = [1, 2]
    METHODS = ["varies", "flat"]
    REGS = [0.1, 1.0]
    SEEDS = [10, 20, 30]
    NENSS = [10, 20, 40]

    def base(self, aspect):
        return 1.0 + 0.01 * aspect

    def base_flat(self, aspect):
        return 0.5 + 0.01 * aspect

    def make_rows(self):
        rows = []
        for aspect in self.ASPECTS:
            for reg in self.REGS:
                for seed in self.SEEDS:
                    for nEns in self.NENSS:
                        val = self.base(aspect) + 0.0001 * reg + 1.0 / nEns
                        rows.append(
                            dict(
                                aspect=aspect,
                                method="varies",
                                reg=reg,
                                seed=seed,
                                nEns=nEns,
                                skill=val,
                            )
                        )
                    val = self.base_flat(aspect) + 0.0001 * reg
                    rows.append(
                        dict(
                            aspect=aspect,
                            method="flat",
                            reg=reg,
                            seed=seed,
                            nEns=NONE,
                            skill=val,
                        )
                    )
        return rows

    @pytest.fixture(scope="class")
    def skill(self):
        rows = self.make_rows()
        index_names = ["aspect", "method", "reg", "seed", "nEns"]
        return build_skill(rows, index_names, mean_dims=["seed"], tuned_dims=["reg"])

    @pytest.fixture(scope="class")
    def orient(self):
        # `method` is deliberately unlisted -> it becomes a "hue" dim, giving each
        # (aspect, method) combo its own line/row.
        return Struct(
            fig=[],
            panel_row=[],
            panel_col=[],
            linestyle=["aspect"],
            unlabelled=[],
            xaxis="nEns",
        )

    def test_varying_line_matches_table_row(self, skill, orient):
        table = shape_tables(skill, orient, find_cat=True)
        handles = LinePlots(skill, orient, mark_stop=False).handles

        for aspect in self.ASPECTS:
            (line,) = handles.loc[(aspect, "varies")]
            x, y = line.get_data()
            plotted = dict(zip(x, y))

            expected = table.loc[(aspect, "varies", False)].to_dict()

            assert plotted.keys() == expected.keys()
            for tick, val in expected.items():
                assert plotted[tick] == pytest.approx(val, rel=1e-9)

    def test_flat_line_matches_table_row(self, skill, orient):
        table = shape_tables(skill, orient, find_cat=True)
        handles = LinePlots(skill, orient, mark_stop=False).handles

        for aspect in self.ASPECTS:
            (line,) = handles.loc[(aspect, "flat")]
            const = line.get_ydata()[0]

            expected = table.loc[(aspect, "flat", True)]
            assert (expected == const).all()
            assert const == pytest.approx(self.base_flat(aspect) + 0.0001, rel=1e-9)
