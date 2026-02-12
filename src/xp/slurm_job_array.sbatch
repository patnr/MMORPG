#!/usr/bin/env bash

# Why not merge exprm_wrapper.py and slurm script, as per
# #the-slurm-script-does-not-have-to-be-written-in-bash of
# https://documentation.sigma2.no/jobs/job_scripts.html.
# - Modularity: may want to support other HPC systems than SLURM
# - Must insert shebang into exprm_wrapper.py after python runtime
# - Must fix sys.path[0] since SLURM copies script to /var/spool/slurmd/job1297113

set -eu -o pipefail # NB: no -f (need filename expansion/globbing)
module --quiet reset  # Reset the modules to the system default

# DATASETS=(xps/*)
# INPUT=${DATASETS[$SLURM_ARRAY_TASK_ID]}

"$@"/$SLURM_ARRAY_TASK_ID
