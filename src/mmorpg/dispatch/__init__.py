import shutil
import subprocess
import sys
from pathlib import Path

from .paths import DATA_ROOT, get_data_dir, save
from .remote_ops import install_deps, submit_and_monitor_slurm
from .uplink import Uplink, resolve_host_glob

responsive = {"check": True, "capture_output": True, "text": True}


def dispatch(
    fun: callable,
    inputs: list,
    host: str = "SUBPROCESS",
    script: Path = None,
    nCPU: int = None,
    nBatch: int = None,
    proj_dir: Path = None,
    tags: list | str = None,
    data_root_on_remote: Path = None,
    slurm_kws: dict = None,
    setup: list[str] | str = "uv",
    venv: str = None,
):
    """
    Execute function over parameter sets on remote hosts/clusters (or locally).

    Essentially: `[fun(**kwargs) for kwargs in inputs]`.

    Parameters
    ----------
    fun : callable
        Function to apply to each experiment.
    inputs : list
        Job array, i.e. list of (parameter) dictionaries to pass to `fun`.
    host : str, optional
        Remote server, e.g. "cno-006".
        Can also be an `ssh/.config` alias, and supports wildcards, e.g., "my-gcp*".
        See `setup-compute-node.sh` for instructions on setting up a Google cloud VM.
        Default is `"SUBPROCESS"`, i.e. local execution.
        Another value commonly used for testing is `"localhost"`.
    script : Path, optional
        Path to script containing `fun`, auto-detected if `None`.
        Used to import "by name" and thus avoid pickling `fun`, which often contains deep references,
        and would consume excessive storage/bandwidth (especially if saved with each experiment).
    nCPU : int, optional
        Number of CPUs used by python's multiprocessing (locally, on a given server, or cluster node).
        Defaults to `None` ⇒ auto-detect.
    nBatch : int, optional
        Number of batches to split `inputs` job array into. Useful for SLURM clusters.
        Note: this enables *nested* multiprocessing (SLURM + python).
        * Let `N` be the total available CPUs, and suppose `len(inputs) >> N` for simplicity.
          Example: NORCE HPC cluster has 3584 CPUs distributed as 14 nodes * 256 CPUs/node.
        * Maybe don't want to hog all available CPUs? Not an important consideration if using `--nice`.
        * Want `nBatch * nCPU == n N` for some integer `n > 0` to make use of all CPUs.
          If instead `n` is slightly above integer, e.g. 5.01,
          then only a single batch will be running towards the end of the total job
          (assuming uniformity of experiment duration and nodes).
        * It might seem that you could set `nCPU=1` and use `nBatch=N`, however
          - Must keep `nBatch < 1000` due to queue system limit.
          - SLURM is significantly slower in distributing jobs than py multiprocessing.
          - Saving many `inputs` is slow (even though total data is same), even w/ multiprocessing.
        * Still, want at least `nBatch > 4x nNodes`, to get some load balancing by SLURM.

        Defaults: `56` for NORCE HPC, `1` for local/other.
        Also see: `get_cluster_resources`
    proj_dir : Path, optional
        Project root directory.
        Gets copied into (and so uploaded with) `data_dir`.
        Does not actually have to be the root of a python package,
        but must be parent of `script` (for example, its basename).
        Auto-detected via git if `None`.
        - NOTE: using "." may seem reasonable, but is bad practice since it promotes dependence
            on whatever happens to be `cwd`.
            Instead, resources (and imports) should be absolute or relative to `script`.
        - NOTE: if you need to access resources outside of `proj_dir` then you should refer to them
            with absolute paths and upload them manually, since our auto-push/pull mechanism is intended
            for allowing fast testing of your code, not all manner of other resources
            (which may be reliant on all manner of further resources and ecosystems).
    tags: list, optional
        By default the data gets stamped with the current datetime.
        You can chose to replace this with your custom tags, for example: ["v1"].
    data_root_on_remote : Path, optional
        Remote root for data. Auto-set: `${USERWORK}` (NORCE HPC) or `${HOME}/data` (other).
    setup : list[str], optional
        Commands to run on remote before all of the jobs to setup environment and install dependencies.
        See `setups.py` for examples with uv, poetry, conda, and pip.

    venv : str or list of str, optional
        Path to virtual environment directory.
        Defaults is the central location "~/.cache/venvs/{proj_dir.stem}"
        (rather than `{proj_dir}/.venv` or a hash location as used by poetry)
        which avoids re-creating the venv for every upload.
        Use {proj_name} and {venv} placeholders in setup.

    Returns
    -------
    Path
        Path to local data directory containing experiment inputs and results.

    Examples
    --------
    See `example.py`

    Notes
    -----
    This is all largely an exercise in path management!
    """
    # Validate inputs before expensive operations
    if not callable(fun):
        raise TypeError(f"fun must be callable, got {type(fun)}")
    if not inputs:
        raise ValueError("inputs list cannot be empty")

    # Make data_dir (working dir for current job)
    data_dir, script, proj_dir = get_data_dir(script, fun, proj_dir, tags)
    data_dir.mkdir(parents=True)
    (data_dir / "inputs").mkdir()
    (data_dir / "outputs").mkdir()

    # Copy resources to data_dir
    ignores = shutil.ignore_patterns("*.pyc", "__pycache__")
    # Follow symlinks during copy (they'll be regular dirs/files in data_dir)
    shutil.copytree(proj_dir, data_dir / proj_dir.stem, ignore=ignores, symlinks=False)
    shutil.copy(Path(__file__).parent / "slurm_job_array.sbatch", data_dir)
    shutil.copy(Path(__file__).parent / "batch_runner.py", data_dir / script.parent)

    # Save inputs -- partitioned for node distribution
    if host and "hpc.intra.norceresearch" in host:
        if nBatch is None:
            nBatch = 55
        nBatch = min(1000, nBatch)  # formal queue limit
        if nCPU is None:
            nCPU = 64
    elif nBatch is None:
        nBatch = 1
    save(inputs, data_dir, nBatch)

    def concat_cmd(python, scrpt):
        args = [python, scrpt.parent / "batch_runner.py", scrpt.stem, fun.__name__, nCPU]
        args = [str(x) for x in args]
        return args

    # Run locally
    if host in ["SUBPROCESS", None]:
        # subprocessing is unecessary, but using a similar code path (as remote) facilitates debugging.
        cmd = concat_cmd(sys.executable, data_dir / script)
        for inpt in (data_dir / "inputs").iterdir():  # or sorted(--"--, key=lambda p: int(p.name)):
            try:
                subprocess.run(cmd + [inpt], check=True, cwd=Path.cwd())
            except subprocess.CalledProcessError:
                raise

    # Run remotely
    else:
        # Connect
        if host.endswith("*"):
            host = resolve_host_glob(host)
        remote = Uplink(host)

        # Get remote_dir
        if data_root_on_remote is None:
            data_root_on_remote = (
                "${USERWORK}" if "hpc.intra.norceresearch" in host else "${HOME}/data"
            )
        data_root_on_remote = remote.shell_expand(data_root_on_remote)
        remote_dir = Path(data_root_on_remote) / data_dir.relative_to(DATA_ROOT)

        with remote.sym_sync(data_dir, remote_dir):  # up- & download
            py = install_deps(remote, remote_dir / proj_dir.stem, setup, venv)
            cmd = concat_cmd(py, remote_dir / script)

            if "hpc.intra.norceresearch" in host:
                # Run on NORCE HPC cluster via SLURM queueing system
                submit_and_monitor_slurm(remote, cmd, remote_dir, slurm_kws)

            else:
                # Run directly (on remote host)
                for inpt in (data_dir / "inputs").iterdir():
                    remote.cmd(cmd + [str(remote_dir / "inputs" / inpt.name)], capture_output=False)
    return data_dir
