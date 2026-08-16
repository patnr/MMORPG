import itertools
from datetime import datetime
from pathlib import Path

import dill
from tqdm.auto import tqdm

timestamp = "%Y-%m-%d_at_%H-%M-%S"
bar_frmt = "{l_bar}|{bar}| {n_fmt}/{total_fmt}, ⏱️ {elapsed} ⏳{remaining}, {rate_fmt}{postfix}"

DATA_ROOT: Path = Path.home() / "data"


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


def save(inputs, data_dir, nBatch):
    print(f"Saving {len(inputs)} inputs to", data_dir)
    ceil_division = lambda a, b: (a + b - 1) // b  # noqa: E731
    batch_size = ceil_division(len(inputs), nBatch)
    nBatch = ceil_division(len(inputs), batch_size)

    def save_batch(i):
        xp_batch = inputs[i * batch_size : (i + 1) * batch_size]
        (data_dir / "inputs" / str(i)).write_bytes(dill.dumps(xp_batch))

    # saving can be slow ⇒ mp
    # from .local_mp import mp
    # mp(save_batch, range(nBatch))
    for i in tqdm(list(range(nBatch))):
        save_batch(i)


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

def get_data_dir(script=None, fun=None, proj_dir=None, tags=None):
    """
    Generate (and maybe make) data_dir.

    This is the working dir for current job, and synched to remote.
    Populated by `inputs/`, `outputs/`, the `proj_dir`, and `slurm_job_array.sbatch`.
    """
    # Get path to `script`
    if script is None:
        assert fun is not None, "Either `script` or `fun` must be provided."
        # Use `co_filename` because `fun.__module__` is sometimes "__main__" and sometimes relative
        script = fun.__code__.co_filename
    script = Path(script)

    # Find proj_dir (code to upload)
    if proj_dir is None:
        proj_dir = find_proj_dir(script)
    if len(proj_dir.relative_to(Path.home()).parts) <= 2:
        msg = f"The `proj_dir` ({proj_dir}) should be uploaded, but is too close to home dir."
        raise RuntimeError(msg)

    data_dir = DATA_ROOT / proj_dir.stem / script.stem  # ⇒ ~/data/proj/script [usually]

    if tags == "latest":
        data_dir /= find_latest_run(data_dir)
    elif tags:
        data_dir /= tags
    else:
        data_dir /= datetime.now().strftime(timestamp)

    # Make relative
    script = proj_dir.stem / script.relative_to(proj_dir)

    return data_dir, script, proj_dir
