"""Measure total duration for jobs running on different servers.

Each individual job hogs 1 CPU and a given amount of memory, for a given amount of time.
"""

import time

import numpy as np
import pandas as pd

from xp import dict_prod, dispatch, load_data


def experiment(MB=1, seconds=10, job_nr=3000):
    """Hog computing resources."""
    start_time = time.time()
    np.random.seed(job_nr)

    # Memory
    data = np.random.rand(int(MB * 1024**2 / 8))

    # if job_nr == 4:
    #     raise RuntimeError("Simulate a failed job")

    # Compute
    nIter = 0
    y = 0.0
    while time.time() < (start_time + seconds):
        nIter += 1
        x = np.random.rand(1000)
        y += np.sum(np.sin(x) * np.cos(x))

    return {"nIter": nIter, "mval": np.mean(data)}


if __name__ == "__main__":
    xps = dict_prod(
        MB=[1],
        seconds=[10],
        job_nr=list(range(3 * 1024)),
    )

    # host = None
    # host = "cno-0001" # has 128 CPUs, runs in 3:20

    # With all CPUs available, and fine-tuned {nBatch,nCPU} running in 0:17 is achievable:
    host = "login-1.hpc.intra.norceresearch.no"

    dir = dispatch(experiment, xps, host)
    res = load_data(dir / "res")

    df = pd.concat([pd.DataFrame(xps), pd.DataFrame.from_records(res)], axis=1)
    print(df)
