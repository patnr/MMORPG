import numpy as np
import numpy.random as rnd

from xp import dict_prod, dispatch, load_data


def experiment(seed=None, method=None, N=None):
    """The main experiment of interest: Integrate f(x) = x^2 over [0, 1]."""

    def f(x):
        return x**2

    rnd.seed(seed)

    if method == "stochastic":
        x = rnd.rand(N)
        estimate = np.mean(f(x))
    elif method == "deterministic":
        x = np.linspace(0, 1, N)
        y = f(x)
        estimate = np.trapezoid(y, x)
    else:
        raise ValueError("Unknown method")

    error = abs(estimate - 1 / 3)
    return {"estimate": estimate, "error": error}


def list_experiments():
    """Setup a `list` of `dicts` of `experiment`'s args as `kwargs`."""
    xps = []
    # Use a loop with clauses for fine-grained control parameter config
    for method in ["stochastic", "deterministic"]:
        kws = {}  # overrule `common` params to create dupes that will be removed
        if method == "deterministic":
            kws["seed"] = None
        xps.append(dict(method=method, **kws))

    # Convenience function to re-do each experiment for a list of common parameters.
    common = dict_prod(
        N=[10, 100, 1000],
        seed=42 + np.arange(10**5),
    )
    # Combine: each `xps` item gets all combinations in `common`
    xps = [{**c, **d} for d in xps for c in common]  # latter `for` is "inner/faster"
    xps = [dict(t) for t in {tuple(d.items()): None for d in xps}]  # remove dupes (preserve order)
    return xps


if __name__ == "__main__":
    xps = list_experiments()
    # res = [experiment(**kwargs) for kwargs in xps]

    host = None                                 # a.k.a. "SUBPROCESS" i.e. run locally
    # host = "localhost"                          # Run locally, but via ssh (NB: may be blocked by sysadmin)
    # host = "my-gcp-*"                           # Example GCP server configured for ssh
    # host = "cno-0001"                           # NORCE-DAO workstation
    # host = "login-1.hpc.intra.norceresearch.no" # NORCE HPC
    dir = dispatch(experiment, xps, host)
    res = load_data(dir / "res")

    # Print table of results
    import pandas as pd

    df = pd.DataFrame(xps).set_index(list(xps[0]))
    df = pd.DataFrame.from_records(res, index=df.index)
    print(df)
