# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MMORPG (Manage Multitudes of Online Runs in Python and Graph them) runs independent
simulation/experiment functions over a parameter grid, in parallel, either locally
(subprocess/multiprocessing) or remotely via SSH (a single host or a NORCE SLURM HPC cluster).
It automates push/pull/track/save/load of parameters and results. The entire public API is
essentially one function: `dispatch(fun, inputs, host)`, which replaces
`[fun(**kwargs) for kwargs in inputs]`.

Read `README.md` for the motivation and QoL details; read `example.py` for the canonical usage
pattern (also used as a live test fixture — see `tests/test_example.py`).

## Commands

Refer "Development" section of `README.md`

Dependencies are managed with `uv` (`uv.lock` present); `pyproject.toml` targets Python 3.12+
(`.python-version` pins 3.13 locally). `pytest` markers `slow`/`integration` are defined in
`pyproject.toml`; default `addopts = "-m 'not slow'"` excludes them.

## Architecture

`src/mmorpg/` splits into two subpackages: `dispatch/` is the dispatch/remote-execution engine
(everything below), and `results/` (table-shaping in `__init__.py`, plotting in `lineplots.py`)
is an independent, non-re-exported results-shaping/plotting layer — see its own module
docstrings.

`dispatch()` in `src/mmorpg/dispatch/__init__.py` is the orchestrator.
`mmorpg/__init__.py` (the top-level package) is just a thin re-export layer.
Understanding `dispatch()` requires tracing through several files together:

1. **Project/script resolution** — `find_proj_dir(script)` (`src/mmorpg/dispatch/paths.py`) walks up
   from the calling script to find the nearest `pyproject.toml`/`requirements.txt`/`setup.py`/`.git`
   marker. This `proj_dir` gets copied wholesale into `data_dir` (and, for remote runs,
   uploaded) so the remote side has everything it needs to import `fun` by name — `fun` itself
   is never pickled, only referenced by `(script, fun_name)`, since deep references in `fun`
   closures are often unpicklable/huge.

2. **Data layout** — everything lives under `data_root/proj_dir.stem/script.stem/<tag-or-timestamp>/`:
   - `inputs/` — `inputs` list, batched into `nBatch` dill-pickled files (see `save()` in
     `src/mmorpg/dispatch/paths.py`)
   - `outputs/` — one dill-pickled results file per input batch, written by `batch_runner.py`
   - a copy of `proj_dir`, plus `batch_runner.py` and `slurm_job_array.sbatch`

   `load_data()` (`src/mmorpg/dispatch/paths.py`) reconstitutes results by reading & concatenating
   all `outputs/*` in numeric order.

3. **Execution paths**, chosen by `host`:
   - `"SUBPROCESS"`/`None` (default): runs `batch_runner.py` once per input batch via
     `subprocess.run`, locally. Deliberately mirrors the remote code path (rather than calling
     `fun` in-process) so that local runs exercise the same logic as remote ones, for easier
     debugging.
   - any other hostname/alias (optionally with `*` glob, resolved via `resolve_host_glob()`
     against `~/.ssh/config`): opens an `Uplink` (`src/mmorpg/dispatch/uplink.py`), which wraps a
     multiplexed SSH connection (`ControlMaster`/`ControlPath`, so many small commands don't
     each pay connection setup cost) plus `rsync`. `Uplink.sym_sync()` is a context manager:
     upload `data_dir` on enter, download it back on exit/exception — so results always sync
     back even on failure or Ctrl-C.
   - if the host contains `"hpc.intra.norceresearch"`: after upload, jobs are submitted as a
     SLURM job array (`submit_and_monitor_slurm()` in `src/mmorpg/dispatch/remote_ops.py`), which
     polls `squeue` for progress and, on `KeyboardInterrupt`, runs `scancel` before re-raising.
     `slurm_job_array.sbatch` is a minimal wrapper (`"$@"/$SLURM_ARRAY_TASK_ID`) — the
     array-index-to-input-file mapping happens there, not in Python. Errors are pulled from
     `sacct`/`error/<task>` and re-raised.
   - otherwise (bare remote host): each input batch is run directly over SSH via `remote.cmd()`,
     without SLURM.

4. **`batch_runner.py`** is the actual entry point run on the target machine
   (`python path/to/batch_runner.py <script> <fun_name> <nCPU> <input_file>`), always invoked as
   a standalone script (not `-m`) so it doesn't require the target project to have package
   structure around the user's script — this is why it gets copied alongside the script rather
   than imported directly. It imports `fun` by name from `script`, loads one input batch, and
   fans it out via `local_mp.mp()`.

5. **`local_mp.mp()`** (`src/mmorpg/dispatch/local_mp.py`) wraps `pathos.multiprocessing` with a
   progress bar, per-item exception catching (`log_errors=True` returns `(exception, traceback)`
   tuples instead of raising, so one bad parameter set doesn't kill the whole batch), and a
   chunksize heuristic. It also calls `threadpoolctl.threadpool_limits(1)` at import time so
   NumPy doesn't oversubscribe CPUs when combined with process-level parallelism.

6. **Dependency installation on remote** — `install_deps()` (`src/mmorpg/dispatch/remote_ops.py`)
   runs a list of shell commands (`setup` param, default `"uv"`) against a `venv` path, with
   `{proj_name}`/`{venv}` placeholder substitution. `setups.py` holds example command lists for
   `uv`/`conda`/`pip`/HPC module systems. The venv is intentionally a *central* cache location
   (`~/.cache/venvs/{proj_name}`), not `{proj_dir}/.venv`, so re-uploads don't force
   re-creating the environment each time.

7. **Nested parallelism (`nBatch` × `nCPU`)** — for SLURM, `inputs` is split into `nBatch`
   batches (one SLURM array task each), and each array task further fans out over `nCPU` local
   processes. See the extensive docstring on `dispatch()` for the tuning tradeoffs (queue limits,
   SLURM dispatch overhead, load balancing vs. node count). NORCE HPC gets auto-tuned defaults
   (`nBatch=55` capped at 1000, `nCPU=64`); other hosts default to `nBatch=1`.

## Notes for making changes

- `dispatch()`'s docstring is the source of truth for parameter semantics — keep it in sync
  with any signature changes.
- Local (`"SUBPROCESS"`) execution intentionally reuses the same `batch_runner.py` subprocess
  path as remote execution rather than calling `fun` in-process; don't "simplify" this away, it
  is what makes local runs a faithful debug proxy for remote runs.
- Editable local development of `mmorpg` itself alongside a consuming project requires
  symlinking (`ln -s path/to/src/mmorpg your_project/mmorpg`) plus `RSYNC_OPTS="-L"` in the
  environment so the symlink gets dereferenced on upload (see README "Development" section).
- `dispatch/` (dispatch engine) and `results/` (tables/plotting) are independent subpackages —
  don't re-export `results/` symbols from the top-level `__init__.py`; consumers import
  `mmorpg.results`/`.lineplots` directly. The one intentional exception is `results/__init__.py`
  importing `get_data_dir`/`load_data` from `dispatch.paths` for re-export (both are
  lightweight — `dill`/`tqdm` only — so this doesn't pull the heavy `results` deps into
  `dispatch`, only the reverse, which is fine); don't add further `dispatch/` → `results/`
  imports beyond that without similar justification.
- The `dispatch()` function lives in `dispatch/__init__.py` itself, not in a `dispatch.py`
  submodule — that submodule name would collide with the function name and the package name.
  Keep it that way; if `dispatch/__init__.py` ever needs to shrink, split its *internals* into
  a differently-named submodule (e.g. `engine.py`) and re-export, rather than reintroducing a
  `dispatch/dispatch.py`.
- `Uplink.rsync()` reads `RSYNC_OPTS` from the environment and prepends it to its own opts.
