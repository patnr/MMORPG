import itertools
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import dill
from tqdm.auto import tqdm

from . import uplink
from .local_mp import mp

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


def mk_data_dir(
    data_dir,
    tags=(),  # Whatever you want, e.g. "v1"
    mkdir=True,  # Make dirs, including xps/ and res/
):
    """Add timestamp/tag and mkdir for data storage."""
    if tags:
        data_dir /= tags
    else:
        data_dir /= datetime.now().strftime(timestamp)

    if mkdir:
        data_dir.mkdir(parents=True)
        (data_dir / "xps").mkdir()
        (data_dir / "res").mkdir()

    return data_dir


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
    batch_size = 1 + len(xps) // nBatch

    def save_batch(i):
        xp_batch = xps[i * batch_size : (i + 1) * batch_size]
        (data_dir / "xps" / str(i)).write_bytes(dill.dumps(xp_batch))

    # saving can be slow ⇒ mp
    mp(save_batch, range(nBatch))


# Note on development
# Would like to develop `dispatch` through an *editable* install,
# but this does not easily work when transposed to *remote*,
# because the local path is of course invalid (and hard to replicate).
# Flagging the package as `--dev` or optional (and not installing it on remote) does not work,
# because `uv` will still check if it's present.
# Solutions:
#
# - Sync it via github. Seems overkill, and has complications:
#   * Use `uv sync --upgrade-package xp` to upgrade in lockfile.
#   * May need to clear cache to pick up latest commit.
# - Copy `src/xp/` dir to PWD. I.e. no install.
#   ⇒ must disable `xp` among dependencies, but copy over sub-dependencies.
# - The above duplicate is quite likely to cause confusion.
#   ⇒ Symlink instead. Requires `-L` flag to `rsync` (append to `-azh`).


def dispatch(
    fun: callable,
    xps: list,
    host: str = None,  # Server alias
    script: Path = None,  # Path to script containing `fun`
    nCPU: int = None,  # number of CPUs to engage
    nBatch: int = None,  # number of batches (splits) of xps
    # NB: `multiprocessing` module already does "chunking",
    # so this is intended to be used on clusters with queue systems.
    # For efficiency, the resulting batch_size should be >= nCPU (per node) * 100
    proj_dir: Path = None,  # e.g. Path(__file__).parents[0]
    data_root: Path = Path.home() / "data",
    data_root_on_remote: Path = None,  # e.g. "${HOME}/data" or "${USERWORK}"
):
    """
    Do `[fun(**kwargs) for kwargs in xps]`, but on various remote hosts/servers.

    The `proj_dir` must be a parent to `script`,
    and gets copied into (and so uploaded with) `data_dir` (which also mirrors path of `proj_dir`!).
    To promote independence of the uploaded code "environment" vs. whatever
    "happens to be" the `cwd` (less headaches!), the `proj_dir` should NOT be the `cwd`.
    Still, if possible (if subpath to `proj_dir`), the `cwd` is "preserved" on remote,
    such that resources specified relative to it (sloppy!) may be found.
    """
    # Validate inputs before expensive operations
    if not callable(fun):
        raise TypeError(f"fun must be callable, got {type(fun)}")
    if not xps:
        raise ValueError("xps list cannot be empty")

    # Don't want to pickle `fun`, because it often contains very deep references,
    # and take up a lot of storage (especially if saved with each xp).
    # ⇒ Ensure we know the script from which we can import it.
    if script is None:
        # Use `co_filename` because `fun.__module__` is sometimes "__main__" and sometimes relative
        script = fun.__code__.co_filename
    script = Path(script)

    # Place launch script in same dir as script
    shutil.copy(Path(__file__).parent / "launch_xps.py", script.parent)

    # proj_dir
    if proj_dir is None:
        proj_dir = find_proj_dir(script)
    if len(proj_dir.relative_to(Path.home()).parts) <= 2:
        msg = f"The `proj_dir` ({proj_dir}) should be uploaded, but is too close to home dir."
        raise RuntimeError(msg)

    # data_dir
    data_dir = data_root / proj_dir.stem / script.relative_to(proj_dir).stem
    data_dir = mk_data_dir(data_dir)

    # Host alias "globbing"
    if host is None:
        host = "SUBPROCESS"
    elif host.endswith("*"):
        host = uplink.resolve_host_glob(host)

    # Save xps -- partitioned (for node distribution)
    if nBatch is None:
        nBatch = 40 if host.startswith("login-") else 1
    save(xps, data_dir, nBatch)

    # List resulting paths
    paths_xps = sorted((data_dir / "xps").iterdir(), key=lambda p: int(p.name))
    assert paths_xps, f"No files found in {data_dir}"

    # Run locally
    if host == "SUBPROCESS":
        for xp in paths_xps:
            try:
                # current_interpreter = "python" # requires active venv
                current_interpreter = sys.executable
                subprocess.run(
                    [
                        current_interpreter,
                        script.parent / "launch_xps.py",
                        script.stem,
                        fun.__name__,
                        xp,
                        str(nCPU),
                    ],
                    check=True,
                    cwd=Path.cwd(),
                )
            except subprocess.CalledProcessError:
                raise

    # Run remotely
    # NOTE:
    # - See xp/setup-compute-node.sh for instructions on setting up a GCP VM.
    # - Use "localhost" for testing/debugging w/o actual server.
    else:
        remote = uplink.Uplink(host)

        # Eval data_root_on_remote
        # PS: not strictly necessary, since cmd/rsync evaluates it on its own,
        # but seems more robust and future-proof (for complex commands)
        if data_root_on_remote is None:
            if "hpc.intra.norceresearch" in host:
                data_root_on_remote = "${USERWORK}"
            else:
                data_root_on_remote = "${HOME}/data"
        data_root_on_remote = remote.cmd("echo " + data_root_on_remote).stdout.splitlines()[0]

        data_dir_remote = Path(data_root_on_remote) / data_dir.relative_to(data_root)
        paths_xps = [data_dir_remote / xp.relative_to(data_dir) for xp in paths_xps]

        # Make (try!) cwd such that the relative path of the script is same as locally
        try:
            cwd = Path.cwd().relative_to(proj_dir)
        except ValueError:
            print("Warning: The cwd is outside of project path ⇒ should not be relied on.")
            cwd = Path(".")
        finally:
            cwd = data_dir_remote / proj_dir.stem / cwd
        script = data_dir_remote / proj_dir.stem / script.relative_to(proj_dir)

        with remote.sym_sync(data_dir_remote, data_dir, proj_dir):
            # Install (potentially outdated) deps (from lockfile)
            # PS: Pre-install `uv` using `wget -qO- https://astral.sh/uv/install.sh | sh`
            venv = f"~/.cache/venvs/{proj_dir.stem}"
            remote.cmd(
                f"cd {data_dir_remote / proj_dir.stem}; UV_PROJECT_ENVIRONMENT={venv} uv sync",
                capture_output=False,  # simply print
            )

            # Run on NORCE HPC cluster with SLURM queueing system
            if "hpc.intra.norceresearch" in host:
                # Send job submission script
                with NamedTemporaryFile(mode="w+t", delete_on_close=False) as sbatch:
                    txt = (Path(__file__).parent / "slurm_script.sbatch").read_text()
                    txt = eval(f"f'''{txt}'''", {}, locals())  # interpolate f-strings inside {txt}
                    sbatch.write(txt)
                    sbatch.close()
                    remote.rsync(sbatch.name, data_dir_remote / "job_script.sbatch")

                # Submit
                job_id = remote.cmd(f"command cd {data_dir_remote}; sbatch job_script.sbatch")
                print(job_id.stdout, end="")
                job_id = int(re.search(r"job (\d*)", job_id.stdout).group(1))

                # Monitor job progress
                nJobs = len(paths_xps)
                with tqdm(total=nJobs, desc="Jobs") as pbar:
                    unfinished = nJobs
                    while unfinished:
                        time.sleep(1)  # dont clog the ssh uplink
                        new = f"squeue -j {job_id} -h -t pending,running -r | wc -l"
                        new = int(remote.cmd(new).stdout)
                        inc = unfinished - new
                        pbar.update(inc)
                        unfinished = new

                # Provide error summary
                failed = (
                    f"sacct -j {job_id} --format=JobID,State,ExitCode,NodeList | grep -E FAILED"
                )
                failed = remote.cmd(failed, check=False).stdout.splitlines()
                if failed:
                    regex = r"_(\d+).*(node-\d+) *$"
                    nodes = {int((m := re.search(regex, ln)).group(1)): m.group(2) for ln in failed}
                    for task in nodes:
                        print(f" Error for job {job_id}_{task} on {nodes[task]} ".center(70, "="))
                        print(remote.cmd(f"cat {data_dir_remote}/error/{task}").stdout)
                    raise RuntimeError(f"Task(s) {list(nodes)} had errors, see printout above.")

            else:
                # Run (`launch_xps.py` uses `mp` ⇒ no point parallelising this loop)
                for xp in paths_xps:
                    remote.cmd(
                        [
                            # PS: A well-crafted script should be independend of cwd ...
                            f"cd {cwd};",  # ... so should ideally be able to comment out this line.
                            f"{venv}/bin/python",
                            script.parent / "launch_xps.py",
                            script.stem,
                            fun.__name__,
                            xp,
                            nCPU,
                        ],
                        capture_output=False,  # simply print
                    )
    return data_dir
