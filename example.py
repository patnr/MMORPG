import numpy as np
import numpy.random as rnd

from mmorpg import dict_prod, dispatch, load_data


def experiment(seed=None, method=None, N=None, func="quadratic", antithetic=False):
    """The main (entry point) function of the experiment: numerically integrate `func` over
    [0, 1] via `method`. `antithetic` (only meaningful for `method="stochastic"`) toggles the
    classic antithetic-variates variance-reduction trick: pairing each sample `x` with `1 - x`
    (at the same total sample count `N`, so it's a fair comparison)."""

    if func == "quadratic":

        def f(x):
            return x**2

        exact = 1 / 3
    elif func == "oscillatory":

        def f(x):
            k = 4.3 # non-integer freq since trapezoid too good for purpose of illustration
            return np.sin(2 * np.pi * k * x) ** 2

        exact = 1 / 2 - np.sin(4 * np.pi * k) / (8 * np.pi * k)
    else:
        raise ValueError("Unknown func")

    rnd.seed(seed)

    if method == "stochastic":
        if antithetic: # a variance-reduction technique
            half = rnd.rand(N // 2 + N % 2)
            x = np.concatenate([half, 1 - half])[:N]
        else:
            x = rnd.rand(N)
        estimate = np.mean(f(x))
    elif method == "deterministic":
        x = np.linspace(0, 1, N)
        y = f(x)
        estimate = np.trapezoid(y, x)
    else:
        raise ValueError("Unknown method")

    error = abs(estimate - exact)
    return {"estimate": estimate, "error": error}


def list_experiments():
    """Setup a `list` of `dicts` of `experiment`'s args as `kwargs`."""
    dcts = []
    # Use a loop with clauses for fine-grained control parameter config
    for method in ["stochastic", "deterministic"]:
        for func in ["quadratic", "oscillatory"]:
            kws = {}  # overrule `common` params to create dupes that will be removed
            if method == "deterministic":
                kws["seed"] = None
                kws["antithetic"] = None  # unused by "deterministic" -- fix to avoid dupes
            dcts.append(dict(method=method, func=func, **kws))

    # Convenience function to re-do each experiment for a list of common parameters.
    common = dict_prod(
        N=[10, 100, 1000],
        seed=42 + np.arange(10**1),
        antithetic=[False, True],
    )
    # Combine: each `dcts` item gets all combinations in `common`
    dcts = [{**c, **d} for d in dcts for c in common]  # latter `for` is "inner/faster"
    dcts = [dict(t) for t in {tuple(d.items()): None for d in dcts}]  # rm dupes (preserve order)

    # Deliberately invalid entry -- demonstrates that a single bad param combo doesn't crash
    # the whole batch (see `find_crashed()`/`is_crashed()`, used downstream in
    # `example_results.py`), just that one result.
    dcts.append(dict(method="unsupported", N=10))

    return dcts


if __name__ == "__main__":
    inputs = list_experiments()
    # outputs = [experiment(**kwargs) for kwargs in inputs]

    host = None  # or "SUBPROCESS" # Run locally
    # host = "localhost"           # Run locally, but via ssh (NB: may be blocked by sysadmin)
    # host = "my-gcp-*"            # Example GCP server configured for ssh
    # host = "cno-0001"            # NORCE-DAO workstation
    # host = "login-1.hpc.intra.norceresearch.no" # NORCE HPC
    data_dir = dispatch(experiment, inputs, host)
    xps, outputs = load_data(data_dir)

    # Print table of results. `list_experiments()` includes one deliberately-invalid entry
    # (see above), whose crashed result is a `(exception, traceback_str)` tuple rather than a
    # dict -- swap it for a NaN row here (a minimal inline check, to avoid pulling in
    # `mmorpg.results.find_crashed()`/the `results` extra just for this quick look; see
    # `example_results.py` for the fuller crash-handling treatment).
    import pandas as pd

    outputs = [o if isinstance(o, dict) else {"estimate": np.nan, "error": np.nan} for o in outputs]
    df = pd.DataFrame(xps).set_index(list(xps[0]))
    df = pd.DataFrame.from_records(outputs, index=df.index)
    print(df)
