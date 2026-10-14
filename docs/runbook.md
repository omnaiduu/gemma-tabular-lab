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
