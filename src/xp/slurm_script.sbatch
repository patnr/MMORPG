#!/usr/bin/env bash
#
#SBATCH --qos=normal                  # Only one available I think
#SBATCH --account=energytech          # Unnecessary it seems
#SBATCH --job-name={script.stem}      # Job name
#SBATCH --output=output/%a     # StdOut (separate files per array task)
#SBATCH --error=error/%a       # StdErr
#SBATCH --time=00:05:00               # Maximum runtime (HH:MM:SS)
#SBATCH --partition=comp              # Type of nodes?
#SBATCH --ntasks=1                    # Only useful with `srun` ?
#SBATCH --nodes=1                     # Only useful with `srun` ?
#SBATCH --cpus-per-task=255           # max(batch_size, actual_cpu_count)
#SBATCH --array=0-{len(paths_xps)-1}  # job/batch indices

# NB: don't include f option, which disables filename expansion (globbing)
set -eu -o pipefail

# ╔═══════╗
# ║ Notes ║
# ╚═══════╝
# - The contents of this file within braces get interpolated by python f-strings.
#   Double braces {{}} escape this (become single braces after interpolation).
# - TODO: Invesigate "If the running time of an individual job is about 10 minutes or
#   less, however, using a job array may introduce unnecessary
#   overhead; instead, you can loop through files manually"
# - Try the following if sourcing fails (complaining that some file doesn't exist,
#   which is very strange, since every node is supposed to be connected to necessary file system)
#   #SBATCH --requeue --max-requeue=3

# # Also see https://documentation.sigma2.no/software/userinstallsw/conda.html
# module load Miniforge3/24.3.0-0
# source $EBROOTMINIFORGE3/bin/activate
# source $EBROOTMINIFORGE3/etc/profile.d/mamba.sh
# mamba activate /cluster/projects/para-miniforge/envs/Hessian

source {venv}/bin/activate

DATASETS=({data_dir_remote}/xps/*)
INPUT=${{DATASETS[$SLURM_ARRAY_TASK_ID]}}

cd "{str(cwd)}"
python {script.parent / "launch_xps.py"} {script.stem} {fun.__name__} "$INPUT" "{str(nCPU)}"
