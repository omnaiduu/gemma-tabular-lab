# Running the experiment

## Local verification, with no ML dependencies

From the repository root:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/run_modal.py digits --dry-run
```

The PyTorch head tests are skipped if PyTorch is unavailable. A pass on the core
tests is not evidence that the Gemma checkpoint loads or trains successfully.

## Data

Obtain `train.tsv` from the [Mercari data page](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data)
under its applicable terms. Extract it to `data/raw/train.tsv`, which Git ignores.
Prepare the allowlisted inputs and fixed duplicate-grouped splits:

```bash
PYTHONPATH=src python3 -m gemma_tabular.cli prepare --source data/raw/train.tsv --train-limit 10000 --holdout-limit 1000
```

This step uses only standard Python. Raw data, processed data, row-level
predictions, and model weights stay out of Git. Aggregate result reports belong
in `results/` and findings belong in `docs/findings.md`.

## Modal authentication

The task-specific profile is named `gemma-tabular-lab`. Credentials are stored
in the user's Modal configuration outside this repository. The launcher selects
this profile, and its authentication preflight must succeed before it reserves
or launches a job.

```bash
MODAL_PROFILE=gemma-tabular-lab modal token info
```

Do not paste credentials into this runbook, Git, shell examples, or logs.

## Bounded launches

Modal builds only the image needed for the selected job. The remote GPU is an
L40S, jobs have explicit timeouts and no retries, and the local reservation
ledger retains worst-case allocations after completion or ambiguous failure.

```bash
PYTHONPATH=src python3 scripts/run_modal.py baselines --timeout 1200
PYTHONPATH=src python3 scripts/run_modal.py digits --adaptation frozen --steps 20 --timeout 600
PYTHONPATH=src python3 scripts/run_modal.py regression --adaptation frozen --steps 20 --timeout 600
```

The 20-step runs are smoke tests. They do not establish competitive accuracy.
After verifying model loading, gradients, target handling, and resource use,
select matched LoRA jobs that fit the remaining budget:

```bash
PYTHONPATH=src python3 scripts/run_modal.py digits --adaptation lora --steps 1000 --timeout 3600
PYTHONPATH=src python3 scripts/run_modal.py regression --adaptation lora --steps 1000 --timeout 3600
```

Record how many rows were actually processed. A step bound can stop training
before an epoch finishes; dataset size alone does not prove training coverage.
Pin `--revision` to the resolved checkpoint commit for matched runs.

Each successful run downloads its artifacts locally and writes a summary in
`results/`. Validation selects checkpoints. Test results are final evaluation,
not parameters to optimize. Larger samples, numerical-loss ablations, repeated
seeds, and a serving API are subsequent stages, subject to evidence and budget.

## Current environment limitation

At project initialization, this host could not resolve `api.modal.com`,
`modal.com`, or `pypi.org`. Authentication could not be verified and no Modal job
was launched. A network-enabled execution environment is needed to progress
from the prepared runner to actual data and model experiments.
