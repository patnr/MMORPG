"""Thin wrapper (for convenience) on top of `pathos.multiprocessing`."""

import traceback

import pathos.multiprocessing as MP
import threadpoolctl

threadpoolctl.threadpool_limits(1)  # make np use only 1 core


def mp(f, lst, nCPU=None, log_errors=False):
    """Multiprocessing map with progress bar."""
    from xp import progbar


    def g(*a, **b):
        """Same as `f` but catch any/all exceptions and 'log' them instead of raise."""
        try:
            return f0(*a, **b)
        except Exception as e:
            return e, traceback.format_exc()

    f0 = f
    if log_errors:
        f = g


    if nCPU in [None, "all"] or nCPU is True:
        nCPU = MP.cpu_count()

    if nCPU in [0, 1, False]:
        # Use this for debugging
        jobs = map(f, lst)
    else:
        # Chunking is important for speed, but not done automatically by imap.
        D = 1 + len(lst) // nCPU // 10  # heuristic chunksize
        with MP.ProcessPool(nCPU) as pool:
            jobs = pool.imap(f, lst, chunksize=D)
    return list(progbar(jobs, total=len(lst)))
