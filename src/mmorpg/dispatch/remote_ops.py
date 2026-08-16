"""Operations run against a connected `Uplink`: dependency install and SLURM submission."""

import re
import time
from pathlib import Path

from tqdm.auto import tqdm

from . import setups
from .uplink import Uplink


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


def install_deps(
    remote: Uplink,
    proj_on_remote: Path,
    setup: list[str] | str,
    venv: str = None,
):
    """Install dependencies on remote using provided setup commands.

    Parameters
    ----------
    remote : Uplink
        Remote connection.
    proj_on_remote : Path
        Local project directory.
    setup : list[str], optional
        Commands to install dependencies and return python path.
        Use {proj_name} and {venv} placeholders which get replaced automatically.
    venv : str, optional
        Path to virtual environment directory. Defaults to "~/.cache/venvs/{proj_on_remote.stem}".

    Returns
    -------
    str
        Path to python executable in the created environment.
    """
    # Set defaults for venv and setup
    if venv is None:
        venv = f"~/.cache/venvs/{proj_on_remote.stem}"
    if isinstance(setup, str):
        setup = getattr(setups, setup)

    # Replace placeholders in setup
    def interp(cmd):
        return cmd.replace("{proj_name}", proj_on_remote.stem).replace("{venv}", venv)

    setup = [interp(cmd) for cmd in setup]

    # Run installation commands
    remote.cmd(
        f"command cd {proj_on_remote}; " + " && ".join(setup),
        capture_output=False,  # simply print
    )

    return f"{remote.shell_expand(venv)}/bin/python"


def submit_and_monitor_slurm(remote, cmd, remote_dir, slurm_kws):
    # Unpack
    nCPU = cmd[-1]
    nJobs = int(remote.cmd(f"ls {remote_dir}/inputs | wc -l").stdout.strip())

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
        ["sbatch", *slurm_opts, "slurm_job_array.sbatch", *cmd, str(remote_dir / "inputs")],
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
