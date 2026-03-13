"""
Examples of setup configurations for different dependency managers.

These can be passed to dispatch() to configure how dependencies are installed
on remote hosts. The {proj_name} and {venv} placeholders get replaced automatically.
"""

# uv
# Use `--no-install-project` to avoid installing project itself
uv = [
    "UV_PROJECT_ENVIRONMENT={venv} uv sync",
]

# conda
conda = [
    "conda env update -p {venv} -f environment.yml --prune",
]

# pip (with requirements.txt)
pip = [
    "python -m venv {venv}",
    "source {venv}/bin/activate",
    "pip install -r requirements.txt",
]

# HPC with module system (example using uv)
hpc = [
    "module load python/3.11",
    "UV_PROJECT_ENVIRONMENT={venv} uv sync --no-install-project",
]
