import itertools
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import dill
from tqdm.auto import tqdm

from .uplink import Uplink, resolve_host_glob

timestamp = "%Y-%m-%d_at_%H-%M-%S"
bar_frmt = "{l_bar}|{bar}| {n_fmt}/{total_fmt}, ⏱️ {elapsed} ⏳{remaining}, {rate_fmt}{postfix}"
responsive = {"check": True, "capture_output": True, "text": True}


def dict_prod(**kwargs):
    """Product of `kwargs` values."""
    # PS: the first keys in `kwargs` are the slowest to increment.
    return [dict(zip(kwargs, x, strict=True)) for x in itertools.product(*kwargs.values())]


def progbar(*args, **kwargs):
    return tqdm(*args, bar_format=bar_frmt, **kwargs)


def load_data(pth, pbar=True):
    pbar = progbar if pbar else (lambda x: x)
    data = []
    for r in pbar(sorted(pth.iterdir(), key=lambda p: int(p.name))):
        try:
            data.extend(dill.loads(r.read_bytes()))
        except Exception as e:
            print(f"Warning: Failed to load {r}: {e}")
    return data


def find_latest_run(root: Path):
    """Find the latest experiment (dir containing many)"""
    lst = []
    for f in root.iterdir():
        try:
            f = datetime.strptime(f.name, timestamp)
        except ValueError:
            pass
        else:
            lst.append(f)
    f = max(lst)
    f = datetime.strftime(f, timestamp)
    return f


def git_dir():
    """Get project (.git) root dir and HEAD 'sha'."""
    git_dir = subprocess.run(["git", "rev-parse", "--show-toplevel"], **responsive).stdout.strip()
    return Path(git_dir)


def git_sha():
    """Get project HEAD 'sha'."""
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], **responsive).stdout.strip()


def find_proj_dir(script: Path):
    """Find python project's root dir.

    Returns the (shallowest) parent below `script`
    of first found among some common root markers.
    """
    markers = ["pyproject.toml", "requirements.txt", "setup.py", ".git"]
    for d in script.resolve().parents:
        for marker in markers:
            candidate = d / marker
            if candidate.exists():
                return d


def save(xps, data_dir, nBatch):
    print(f"Saving {len(xps)} xp's to", data_dir)
    ceil_division = lambda a, b: (a + b - 1) // b  # noqa: E731
    batch_size = ceil_division(len(xps), nBatch)
    nBatch = ceil_division(len(xps), batch_size)

    def save_batch(i):
        xp_batch = xps[i * batch_size : (i + 1) * batch_size]
        (data_dir / "xps" / str(i)).write_bytes(dill.dumps(xp_batch))

    # saving can be slow ⇒ mp
    # from .local_mp import mp
    # mp(save_batch, range(nBatch))
    for i in tqdm(list(range(nBatch))):
        save_batch(i)


def get_cluster_resources(remote: Uplink):
    # SLURM
    # Columns: [Partition, CPUS(A/I/O/T), NODES(A/I)]
    resources = remote.cmd('sinfo -o "%P %C %A"').stdout
    for line in resources.strip().splitlines()[1:]:  # skip header
        partition, nCPUS, nNODES = line.split()
        if partition.startswith("comp"):
            cpus = map(int, nCPUS.split("/"))
            nodes = map(int, nNODES.split("/"))
            cpus = dict(zip(["allocated", "idle", "other", "total"], cpus))
            nodes = dict(zip(["allocated", "idle"], nodes))
            return cpus, nodes


def install_deps(remote: Uplink, remote_dir: Path, proj_dir: Path):
    # Install (potentially outdated) deps (from lockfile)
    # PS: Pre-install `uv` using `wget -qO- https://astral.sh/uv/install.sh | sh`
    venv = f"~/.cache/venvs/{proj_dir.stem}"
    remote.cmd(
        f"command cd {remote_dir / proj_dir.stem}; UV_PROJECT_ENVIRONMENT={venv} uv sync",
        capture_output=False,  # simply print
    )
    venv = remote.cmd("echo " + venv).stdout.splitlines()[0]  # eval ~ $HOME etc
    py = f"{venv}/bin/python"
    return py


def submit_and_monitor_slurm(remote, cmd, remote_dir, slurm_kws):
    # Unpack
    nCPU = cmd[-1]
    nJobs = int(remote.cmd(f"ls {remote_dir}/xps | wc -l").stdout.strip())

    defaults = {
        # These CLI options take precedence over #SBATCH directives
        # Also see https://documentation.sigma2.no/software/userinstallsw/conda.html
        "account": "energytech",            # Not necessary?
        "partition": "comp",                # Type of nodes?
        # "job_name": script.name,
        "qos": "normal",                    # Only one available I think
        "nice": 1000,                       # High value ⇒ low priority in queue
        "array": f"0-{nJobs-1}",            # list of job/batch indices
        "output": "output/%a",              # StdOut (separate files per array task)
        "error": "error/%a",                # StdErr
        "mem-per-cpu": "200M",              # Max memory (per array task)
        "time": "01:00:00",                 # Max runtime (HH:MM:SS)
        "cpus-per-task": nCPU               # Max CPUs (per array task)
        # Relevant only for MPI jobs (we ony handle embarrasingly parallelisable jobs):
        # "ntasks": ???
        # "nodes": ???
        # If venv not found, or other issues arise that might be due to file system, perhaps try:
        # "requeue": True
        # "max-requeue": 3
    }  # fmt: skip
    slurm_kws = {**defaults, **(slurm_kws or {})}
    slurm_opts = {
        "--" + k.replace("_", "-") + ("" if v is True else f"={v}"): v for k, v in slurm_kws.items()
    }

    # Submit
    job_id = remote.cmd(
        ["sbatch", *slurm_opts, "slurm_job_array.sbatch", *cmd, str(remote_dir / "xps")],
        cwd=remote_dir,
    )
    print(job_id.stdout, end="")
    job_id = int(re.search(r"job (\d*)", job_id.stdout).group(1))

    # Monitor job progress
    try:
        with tqdm(total=nJobs, desc="Jobs") as pbar:
            unfinished = nJobs
            while unfinished:
                time.sleep(1)  # dont clog the ssh uplink
                new = f"squeue -j {job_id} -r -h -t pending,running,completing | wc -l"
                new = int(remote.cmd(new).stdout)
                inc = unfinished - new
                pbar.update(inc)
                unfinished = new
    except KeyboardInterrupt:
        print(f"\nCancelling job {job_id}...")
        remote.cmd(f"scancel {job_id}")
        raise

    # Provide error summary
    # NOTE: Most errors will be caught (and logged) already by `local_mp.py`
    failed = f"sacct -j {job_id} --format=JobID,State,ExitCode,NodeList | grep -E FAILED"
    failed = remote.cmd(failed, check=False).stdout.splitlines()
    if failed:
        regex = r"_(\d+).*(node-\d+) *$"
        nodes = {int((m := re.search(regex, ln)).group(1)): m.group(2) for ln in failed}
        for task in nodes:
            print(f" Error for job {job_id}_{task} on {nodes[task]} ".center(70, "="))
            print(remote.cmd(f"cat {remote_dir}/error/{task}").stdout)
        raise RuntimeError(f"Task(s) {list(nodes)} had errors, see printout above.")


def dispatch(
    fun: callable,
    xps: list,
    host: str = "SUBPROCESS",
    script: Path = None,
    nCPU: int = None,
    nBatch: int = None,
    proj_dir: Path = None,
    tags: list | str = None,
    data_root: Path = Path.home() / "data",
    data_root_on_remote: Path = None,
    slurm_kws: dict = None,
):
    """
    Execute function over parameter sets on remote hosts/clusters (or locally).

    Essentially: `[fun(**kwargs) for kwargs in xps]`.

    Parameters
    ----------
    fun : callable
        Function to apply to each experiment.
    xps : list
        Job array, i.e. list of (parameter) dictionaries to pass to `fun`.
    host : str, optional
        Remote server, e.g. "cno-006".
        Can also be an `ssh/.config` alias, and supports wildcards, e.g., "my-gcp*".
        See `xp/setup-compute-node.sh` for instructions on setting up a Google cloud VM.
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
        Number of batches to split `xps` job array into. Useful for SLURM clusters.
        Note: this enables *nested* multiprocessing (SLURM + python).
        * Let `N` be the total available CPUs, and suppose `len(xps) >> N` for simplicity.
          Example: NORCE HPC cluster has 3584 CPUs distributed as 14 nodes * 256 CPUs/node.
        * Maybe don't want to hog all available CPUs? Not an important consideration if using `--nice`.
        * Want `nBatch * nCPU == n N` for some integer `n > 0` to make use of all CPUs.
          If instead `n` is slightly above integer, e.g. 5.01,
          then only a single batch will be running towards the end of the total job
          (assuming uniformity of experiment duration and nodes).
        * It might seem that you could set `nCPU=1` and use `nBatch=N`, however
          - Must keep `nBatch < 1000` due to queue system limit.
          - SLURM is significantly slower in distributing jobs than py multiprocessing.
          - Saving many `xps` is slow (even though total data is same), even w/ multiprocessing.
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
    data_root : Path, optional
        Local root for experiment data. Default: `~/data`
        Gets populated by `xps/`, `res/`, the `proj_dir`, and `slurm_job_array.sbatch`.
    data_root_on_remote : Path, optional
        Remote root for data. Auto-set: `${USERWORK}` (NORCE HPC) or `${HOME}/data` (other).

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
    if not xps:
        raise ValueError("xps list cannot be empty")

    # Get path to `script`
    if script is None:
        # Use `co_filename` because `fun.__module__` is sometimes "__main__" and sometimes relative
        script = fun.__code__.co_filename
    script = Path(script)

    # Find proj_dir (code to upload)
    if proj_dir is None:
        proj_dir = find_proj_dir(script)
    if len(proj_dir.relative_to(Path.home()).parts) <= 2:
        msg = f"The `proj_dir` ({proj_dir}) should be uploaded, but is too close to home dir."
        raise RuntimeError(msg)

    # Save to data_dir (root archive & working dir for current job)
    data_dir = data_root / proj_dir.stem / script.stem  # ⇒ ~/data/proj/script [usually]
    if tags:
        data_dir /= tags
    else:
        data_dir /= datetime.now().strftime(timestamp)
    data_dir.mkdir(parents=True)
    (data_dir / "xps").mkdir()
    (data_dir / "res").mkdir()

    # Make relative
    script = proj_dir.stem / script.relative_to(proj_dir)

    # Copy resources to data_dir
    ignores = shutil.ignore_patterns("*.pyc", "__pycache__")
    shutil.copytree(proj_dir, data_dir / proj_dir.stem, ignore=ignores)
    shutil.copy(Path(__file__).parent / "slurm_job_array.sbatch", data_dir)
    shutil.copy(Path(__file__).parent / "exprm_wrapper.py", data_dir / script.parent)

    # Save xps -- partitioned for node distribution
    if host and "hpc.intra.norceresearch" in host:
        if nBatch is None:
            nBatch = 55
        nBatch = min(1000, nBatch)  # formal queue limit
        if nCPU is None:
            nCPU = 64
    elif nBatch is None:
        nBatch = 1
    save(xps, data_dir, nBatch)

    def concat_cmd(python, scrpt):
        args = [python, scrpt.parent / "exprm_wrapper.py", scrpt.stem, fun.__name__, nCPU]
        args = [str(x) for x in args]
        return args

    # Run locally
    if host in ["SUBPROCESS", None]:
        # subprocessing is unecessary, but using a similar code path (as remote) facilitates debugging.
        cmd = concat_cmd(sys.executable, data_dir / script)
        for xp in (data_dir / "xps").iterdir():  # or sorted(--"--, key=lambda p: int(p.name)):
            try:
                subprocess.run(cmd + [xp], check=True, cwd=Path.cwd())
            except subprocess.CalledProcessError:
                raise

    # Run remotely
    else:
        if host.endswith("*"):
            host = resolve_host_glob(host)
        remote = Uplink(host)

        # data_root_on_remote
        if data_root_on_remote is None:
            data_root_on_remote = (
                "${USERWORK}" if "hpc.intra.norceresearch" in host else "${HOME}/data"
            )
        # Evaluate. Maybe unecessary since ${some_envar} could work below as-is
        data_root_on_remote = remote.cmd("echo " + data_root_on_remote).stdout.splitlines()[0]
        # remote_dir
        remote_dir = Path(data_root_on_remote) / data_dir.relative_to(data_root)

        # Sync on enter & exit
        with remote.sym_sync(remote_dir, data_dir, proj_dir):
            py = install_deps(remote, remote_dir, proj_dir)
            cmd = concat_cmd(py, remote_dir / script)

            if "hpc.intra.norceresearch" in host:
                # Run on NORCE HPC cluster via SLURM queueing system
                submit_and_monitor_slurm(remote, cmd, remote_dir, slurm_kws)

            else:
                # Run directly (on remote host)
                for xp in (data_dir / "xps").iterdir():
                    remote.cmd(cmd + [str(remote_dir / "xps" / xp.name)], capture_output=False)
    return data_dir
