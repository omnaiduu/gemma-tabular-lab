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

The pilot uses the published 100K Mercari research subset from
[OpenML dataset 46660](https://www.openml.org/search?type=data&id=46660), retaining
the original `price` column. This is a documented subset of the original Kaggle
data, not the full competition training set. OpenML declares its licence as
`Kaggle`; the original usage conditions apply and source rows remain outside Git.
Acquisition verifies the row count, schema, and agreement between `price` and
the published `log_price = log1p(price)` column, and records both file checksums.

```bash
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python 'pandas>=2.2,<3' pyarrow 'modal>=1.5,<2'
.venv/bin/python scripts/fetch_mercari.py
```

Alternatively, obtain the full `train.tsv` from the
[Mercari data page](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data)
under its applicable terms and extract it to `data/raw/train.tsv`, which Git ignores.
Prepare the allowlisted inputs and fixed duplicate-grouped splits:

```bash
PYTHONPATH=src python3 -m gemma_tabular.cli prepare --source data/raw/train.tsv --train-limit 10000 --holdout-limit 1000
```

This step uses only standard Python. Raw data, processed data, row-level
predictions, and model weights stay out of Git. Aggregate result reports belong
in `results/` and findings belong in `docs/findings.md`.

## Modal authentication

GPU access additionally requires a valid payment method on the Modal workspace,
even with free credits. On 2026-10-15 Modal rejected our L40S allocation for this
reason. [Modal documents the requirement for all GPU types](https://modal.com/docs/guide/gpu).
Do not repeatedly submit GPU jobs until this account setting is resolved.

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
PYTHONPATH=src .venv/bin/python scripts/run_modal.py digits --adaptation lora --steps 1250 --batch-size 8 --timeout 3600 --revision 411aa17b749aa952df1359d2dcea73917a544d9a
PYTHONPATH=src .venv/bin/python scripts/run_modal.py regression --adaptation lora --steps 1250 --batch-size 8 --timeout 3600 --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

Record how many rows were actually processed. A step bound can stop training
before an epoch finishes; dataset size alone does not prove training coverage.
Pin `--revision` to the resolved checkpoint commit for matched runs.
With 10,000 rows, 1,250 completed steps at batch size eight cover one epoch;
the training report records both processed examples and unique rows actually seen.

Each successful run downloads its artifacts locally and writes a summary in
`results/`. Validation selects checkpoints. Test results are final evaluation,
not parameters to optimize. Larger samples, numerical-loss ablations, repeated
seeds, and a serving API are subsequent stages, subject to evidence and budget.

## Current environment limitation

Resolved on 2026-10-14: command networking and the dedicated Modal profile now
work. The following notes describe the original failure and recovery procedure.

At project initialization, the agent shell could not resolve `api.modal.com`,
`modal.com`, or `pypi.org`. Authentication could not be verified and no Modal job
was launched. A network-enabled execution environment is needed to progress
from the prepared runner to actual data and model experiments.

### Start a session with network access

The current agent session has restricted command networking. Its live policy
cannot be changed from inside the sandbox. The user-level Codex configuration
has no network override, and this session makes that configuration read-only.
Assuming this is the cause, start a new session from a normal terminal:

```bash
bash /home/omnaidu/gemma-tabular-lab/scripts/start_networked_codex.sh
```

The launcher passes `sandbox_workspace_write.network_access=true` while keeping
the `workspace-write` filesystem policy. It changes no global configuration,
approval settings, or WSL DNS settings. Managed restrictions may still prevent
network access; this setting is a request for the new session, not an override
of administrator policy.

In the new session, first verify DNS and Modal authentication:

```bash
python3 -c "import socket; print(socket.getaddrinfo('api.modal.com', 443))"
MODAL_PROFILE=gemma-tabular-lab modal token info
```

Then continue the Data and Bounded launches steps above. If DNS still fails in
the new session, the execution-environment assumption remains unconfirmed;
compare with the normal terminal before changing host networking.

Reference: [OpenAI agent approvals and security](https://learn.chatgpt.com/docs/agent-approvals-security#network-access).
