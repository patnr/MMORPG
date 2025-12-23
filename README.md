# `MMORPG` - Manage *large numbers* of python simulations/experiments

<img src="icon.png" alt="icon" width="200" align="left">

MMORPG (Massively Multiple Online Runs in Python with Graphs)
runs independent simulation experiments in parallel and remotely
(on a single host or a SLURM cluster), automating the work of push, pull, track, save
and load for parameters and results, with minimal code (mental) overhead:
```py
data_dir = dispatch(f, params, host)
```

MMORPG also helps in the post-processing, analysis and presentation
of high-dimensional parameter and results arrays,
including its tabulation/plotting, despite them being irregularly shaped or missing values.

It leverages battle-tested libraries:
`ssh/rsync` for remote execution,
`pathos` for multiprocessing and storage serialisation,
`pandas` for tabulation,
and `sparse` `xarray` for post-processing.

QoL details:

- SSH multiplexing to limit connection overhead
- Progress bars everywhere
- Error logging, not raising
- `threadpoolctl.threadpool_limits(1)`

## Motivation

When developing and testing numerical methods and algorithms, it is common practice to evaluate them on prototype/toy problems which trade off simplicity (speed and interpretability) with representativeness of the real-world scenarios being targeted. Despite the simplicity, one quickly runs up a large number of parameters to consider; each of the following considerations introduces *at least* one additional parameter *dimension*:

1. **Algorithms/methods (context):** Different algorithms or methods should be compared.
2. **Tuning parameters (fairness):** Each method should be allowed some tuning.
3. **Problem selection (relevance):** A variety of problems should be considered.
4. **Problem parameters (generality):** Variations on the problems should be considered.
5. **Random seed (reliability):** Results should be averaged over multiple random seeds.

Thus a great number of experiments must[^1] be run and analysed,
with the complication that many parameters only apply for some methods/experiments.
As mentioned [here](https://www.youtube.com/watch?v=EeqhOSvNX-A)
hand crafted solutions are error prone, and often an after-thought.

[^1]: Unfortunately, the scientific principle of only varying one parameter at a time
is not all that useful wanting to optimize for a set of methods and tuning parameters.
Moreover, the problem and seed parameter space should be thoroughly sampled
(jointly, rather than marginally) in order to provide reliable and generalizable averages.

## Alternatives

- **MMORPG**:
  Python-native, manages parameters and results, minimal code overhead, supports remote/HPC execution (via rsync/ssh), no dashboard, low-medium setup.
- **Weights & Biases / MLflow / Neptune.ai**:
  Advanced experiment/parameter tracking, rich dashboards, Python APIs (extra config needed), remote execution for ML (not HPC), medium-high setup.
- **Ray / Dask**:
  Distributed parameter sweeps, web dashboards, Python-native, remote/cloud/cluster execution, high setup complexity.
- **Snakemake / Nextflow**:
  Workflow-based param management (config files), some visualization, Snakemake is Pythonic, strong HPC/remote support, medium-high setup.
- **Sacred / Hydra**:
  Hierarchical config management, minimal dashboard (Sacred only), Python-native, no remote/HPC, low-medium setup.
- **Joblib**:
  Simple param mapping, no dashboard, Python-native, local parallel only, low setup.
- **GNU Parallel / xargs + tmux/screen**:
  Manual param management, no dashboard, not Python-native, remote via manual SSH, low setup, shell skills needed.

Also see: [comparison with ML-domain tools](https://dagshub.com/blog/how-to-compare-ml-experiment-tracking-tools-to-fit-your-data-science-workflow/)

## Development

When developing MMORPG alongside some different project
then you essentially need an "editable" install of MMORPG,
but this does not transpose to remote hosts.
The solution is to symlink `src/mmorpg` into your project and make sure `rsync` is run with `-L` option.

### Testing

```bash
pytest # most tests
pytest -m slow # slow tests (integration tests, with SSH/remote execution)
pytest -m slow -k "localhost" -v # specific parameter
pytest -m "slow or not slow" # all tests
pytest -m "" # idem
pytest tests/test_example.py::TestIntegration::test_host_dispatch[subprocess] -v -m "" # by full path
pytest --collect-only -m "" # list tests
```

### Linting and Formatting

```bash

ruff format --check . # Check format
ruff format . # Format
ruff check . # Check lint
ruff check --fix . # fix lint
```
