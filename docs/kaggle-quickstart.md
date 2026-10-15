# Start the Gemma experiment on Kaggle

## Autonomous execution

The user has authorized the agent to connect and run Kaggle from this workspace.
The official Kaggle CLI is installed in the project environment. Its browser
login stores credentials outside Git, under `~/.kaggle/credentials.json` with
owner-only file permissions. A browser sign-in and any account verification
still have to be completed by the user if no authenticated session is available.

After sign-in, the agent can create a private T4 check, monitor the run, and
download its JSON without asking the user to import or execute cells:

```bash
.venv/bin/kaggle auth login
.venv/bin/python scripts/run_kaggle_preflight.py
```

`run_kaggle_preflight.py --user EXAMPLE --dry-run` prepares files without any
network call or GPU allocation. An actual submission requests T4 x2 explicitly,
sets the notebook private, disables TPU, enables Internet, requests a maximum
600-second runtime, and submits only once. It does not retry or switch to a
different accelerator automatically. The notebook only checks the environment;
it does not load weights or train. Credentials and private API logs are excluded
from the source bundle and Git. Actual Kaggle GPU allocation remains unverified
until the runtime check passes. The connected account reports 30 hours of free
GPU quota, but the first batch checks executed on CPU and failed DNS despite
correctly saved GPU/Internet settings. See `results/kaggle-access-check.json`;
the cause is still unresolved.

The CLI uses its official OAuth scopes. Review the Kaggle authorization screen
when connecting. No password, refresh token, or API token needs to be sent in
chat. Reference: https://www.kaggle.com/docs/api and
https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md.

## What to provide first

Only the notebook preflight output and the remaining GPU quota displayed by
Kaggle are needed for the next step. Do not send a password, browser cookie,
one-time login code, or API token in chat. Manual notebook execution does not
require sharing Kaggle API credentials with the agent.

## First notebook

1. Sign in at https://www.kaggle.com/code and create a notebook. If Kaggle asks
   for account verification to enable accelerators or Internet, complete it in
   Kaggle's own interface.
2. Import `Gemma_E2B_Kaggle_Check.ipynb` using the notebook file import/upload
   action. Keep the notebook private.
3. In Settings / Session options, turn Internet on. Select GPU T4 x2 if offered;
   otherwise choose P100. Do not select TPU for the current PyTorch runner.
4. Run the check cell. It prints Python/library versions, GPU memory and CUDA
   compatibility, and connection checks for public model/dataset metadata. No
   model weights are downloaded, packages installed, or training started.
5. Send the printed JSON or `/kaggle/working/kaggle-preflight.json`, and the
   remaining GPU quota shown in Kaggle. Stop the session after the check.

The runtime startup and tiny CUDA kernel check can use a small amount of Kaggle
quota. No Modal resource is created and no Modal billing allocation is needed.

## Prepared source bundle

Generate the notebook and `Gemma_Tabular_Kaggle_Source.zip` locally with:

```bash
.venv/bin/python scripts/build_kaggle_bundle.py
```

The zip contains an explicit list of source files, the experiment plan, and
aggregate provenance/results. It excludes dataset rows, credentials, weights,
checkpoints, and the local environment. A SHA256 manifest accompanies its files.
The check notebook is standalone; attaching the zip is not required for this
first check.

The later training step will acquire the published OpenML Mercari 100K subset
from its original URL and recreate the fixed 10K/1K/1K partitions. The original
Kaggle dataset terms still apply. Neither this bundle nor its notebook
redistributes the data.

## Training status

The source bundle is not a claim that full Gemma training works on T4/P100.
The existing runner supports FP16 on a single GPU, gradient accumulation,
configurable evaluation batches, and validation-only smoke tests. FP16 LoRA/head
gradients pass tiny CPU integration tests; full-model GPU memory fit still
needs verification. QLoRA remains unimplemented. T4 and P100 lack native BF16,
and two T4s do not automatically pool memory. Choose package and
precision settings after checking the actual runtime; do not blindly replace
Kaggle's CUDA PyTorch installation or assume that all embeddings become 4-bit.

After that test, complete the two E2B heads on matched data/settings before
considering E4B. Do not use final test metrics to tune or schedule later jobs.

Official setup reference: https://www.kaggle.com/docs/notebooks.
