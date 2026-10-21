# Findings and run journal

Append findings as the work proceeds. Separate observations, hypotheses, and
measurements. Every result should cite an artifact or reproducible command.

## 2026-10-15T08:39:26+00:00 — Kaggle interface confirms selected accelerator

- The user's screenshot shows Kaggle's accelerator menu with GPU T4 x2 checked;
  that option and the TPU option appear greyed. The screenshot does not establish
  why they are greyed or whether GPU hardware has been allocated.
- A read-only API recheck preserves GPU=True, Internet=True, and NvidiaTeslaT4,
  while the saved notebook still names the CPU Docker image. The saved batch
  run is complete. GPU quota remains 30 hours allowed, zero used, zero reserved,
  and has_ever_run=False. These API observations describe the saved run/quota;
  they do not directly inspect the browser's draft Python process.
- Evidence: `results/kaggle-ui-recheck.json`. No new session was created, no
  model weights were downloaded, and no training was started by this recheck.
- Next discriminator: run the existing environment-check cell in the browser's
  draft session and inspect its CUDA/device output. Repeating unchanged API
  submissions has not been justified by this screenshot alone.

### 2026-10-15T08:43:39+00:00 — Browser Run All reported by user

- The user clicked Run All in Kaggle. A read-only API check still returns saved
  version 1 and the earlier preflight timestamp 07:34:04 UTC, with CUDA=False.
  This is old saved output, not evidence that the newly requested draft execution
  has completed or failed. Quota still reports zero GPU time used/reserved.
- Evidence: `results/kaggle-run-all-recheck.json`. No new remote session was
  created. The browser cell output is needed to observe this draft execution.

### 2026-10-15T08:43:14+00:00 — Browser preflight confirms the same failure

- The user pasted the newly executed draft report. It independently confirms
  torch 2.11.0+cpu, CUDA=False, no NVIDIA device files, no nvidia-smi, and DNS
  failures for both Hugging Face and OpenML. The duplicated dictionary in chat
  is the same result, not a second independent execution.
- Saved the report as `results/kaggle-browser-preflight.json`. The selected
  accelerator in the interface therefore did not establish actual hardware
  availability. Replacing PyTorch alone cannot create the missing GPU devices.
- The underlying cause remains unresolved; these observations do not prove an
  account-verification requirement, exhausted quota, or a global Kaggle outage.
- Consulted Kaggle's official notebook documentation through Exa. It documents
  original/latest environment selection, an Internet setting, and session
  restart when selecting a GPU: https://www.kaggle.com/docs/notebooks.
- Submitted one changed diagnostic on the existing private notebook: version 2,
  latest environment, no explicit Docker image, explicit SAVE_AND_RUN_ALL,
  NvidiaTeslaT4, Internet enabled, and a 180-second execution cap. No training
  or model download is part of this diagnostic. Its report is recorded in
  `results/kaggle-latest-environment-check.json`.
- Version 2 completed and reproduced the same CPU/no-device/DNS failures. The
  latest-environment request did not fix execution. No further automatic retry
  is justified without changed provider/session state. Next inspect/reselect
  the browser draft's environment and Internet settings after stopping that
  session. This is a diagnostic step, not an established remedy.

### 2026-10-15 — Windows browser control established

- The user asked the agent to operate Kaggle's interface. Linux agent-browser
  could not auto-connect to the user's normal Windows Chrome. Headed Linux
  launches initially failed before providing DevToolsActivePort; a later launch
  reached Kaggle, but the user could not see its window. No exact Linux crash
  cause was established. Exa surfaced similar project issue reports, including
  https://github.com/vercel-labs/agent-browser/issues/1428; the Snap-specific
  explanation in that report does not match our Chrome for Testing install.
- Downloaded official agent-browser v0.32.3 for Windows into the dedicated
  GemmaTabularLab tools directory. Verified the release SHA256 digest:
  d27f77cb4ed7120a25f0b60030b306fe6788be0997f5a6baad74f74a2c0ec627.
- Opened installed Windows Chrome with a dedicated GemmaTabularLab profile and
  loopback DevTools port 9224. The native Windows agent-browser successfully
  retrieved Kaggle's sign-in page accessibility snapshot. Existing user Chrome
  windows/profiles were not closed or copied. The unused Linux session was
  closed afterward.
- Browser control now works, but this dedicated browser still needs the user's
  Kaggle web sign-in. Existing API OAuth credentials do not provide web cookies.
  No Kaggle settings were changed through this browser yet. No GPU job, model
  download, or training was started during the browser setup.

### 2026-10-15 — Signed-in browser exposes an account-side Internet restriction

- The user completed Kaggle sign-in in the dedicated Windows Chrome window.
  The agent opened the private notebook editor and used More settings -> Stop
  session. The interface now reports Draft Session off (run a cell to start).
- Expanded Session options. The environment already says Always use latest
  environment. Internet is off, its checkbox is disabled, and the interface
  offers Get phone verified and Verify identity links.
- This is new direct interface evidence. It explains why saved API flags cannot
  be treated as proof of usable Internet. Account verification is the next
  provider-side condition to inspect. It is not yet proven that verification
  alone will fix GPU allocation. No verification was completed by the agent.
- Evidence: `results/kaggle-browser-session-options.json`. No new runtime,
  model download, or training was started after stopping the old draft.
- The user requested a very detailed handoff in Downloads at this point. The
  handoff must preserve this stopped-session state and the pending verification
  step rather than claim that the experiment is running.

## 2026-10-14 — Project started

- Observation: the user's numerical representation uses categorical signs and
  digits to reconstruct a regression output.
- Design decision: the main experiment uses parallel digit classifiers after
  one Gemma pass. A direct regression head is the assistant-suggested control.
- Budget: $30 maximum Modal expenditure was authorized.
- Repository: local-only; pushing is deferred.
- Status: documentation created before starting model training.
- No performance or spending result has been measured yet.

## Evidence needed for the engineering blog

- Dataset provenance, permitted usage, cleaning and split counts.
- Exact checkpoint and dependency versions.
- Architecture and losses, including zero and rounding handling.
- Pilot results, failures, and resulting design changes.
- Matched baseline results with accuracy, latency and hardware costs.
- Spending ledger and actual job coverage.
- Limitations and experiments that would change the conclusion.

## 2026-10-14 — Implementation and first local verification

- Created a new local Git repository at `gemma-tabular-lab`; no remote is configured.
- Stored the supplied Modal credentials in a dedicated profile outside Git,
  preserving existing profiles. Authentication has not been verified.
- The host cannot resolve Modal or package-server domains. The authentication
  preflight timed out. No image build, GPU allocation, or training job was submitted.
- Implemented data allowlisting, duplicate-grouped splits, independent holdout
  sampling, a four-significant-digit codec, numerical metrics, a conservative
  budget ledger, CPU baseline code, and a bounded Modal launcher.
- Implemented parallel digit heads and a direct-regression control with frozen
  or LoRA adaptation. These GPU paths are prepared but not yet executed.
- Zero handling expands the first mantissa digit's choices to include zero.
  The eight groups therefore contain 64 logits, rather than the original
  nonzero encoding's 63. A predicted leading zero canonicalizes the whole output.
- Core tests: 16 passed. Two PyTorch-dependent tests were skipped because this
  environment lacks PyTorch. Python syntax compilation succeeded.
- The GPU dry run estimated a $3.17 conservative reservation for one 3600-second
  L40S job, including requested CPU/memory, startup allowance, and margin.
  This is an estimate, not a billed amount. The ledger contains no reservations.
- The Mercari dataset has not been downloaded; no real-data baseline or Gemma
  accuracy measurement exists yet. Obtaining the data and restoring network
  access are required before the first remote run.

### Questions to resolve in actual runs

- Does the checkpoint expose and load the expected Gemma text backbone with the
  recorded Transformers version?
- What memory and per-step latency do this row length and LoRA configuration use?
- Does the digit head's exponent accuracy translate into useful price error?
- How much improvement comes from text information versus the output encoding?
- Are the observed gains reproducible across seeds and price/category slices?

## 2026-10-14T10:46:00+00:00 — Modal access recheck

- User requested another access check and execution attempt.
- The dedicated Modal profile contains both credential fields; values were not logged.
- DNS resolution failed for `api.modal.com`, `pypi.org`, and `huggingface.co` with a temporary name-resolution error.
- The authenticated CLI check timed out after 20 seconds; credential validity remains unverified.
- The Mercari raw TSV and prepared splits are still absent.
- No compute job was submitted and no project budget reservation was created.

## 2026-10-14T10:48:52+00:00 — Connectivity diagnosis

- DNS lookup fails for all four tested domains: Modal API, Modal website, PyPI, and example.com. This is broader than a Modal-specific authentication error.
- The shell cannot obtain an address for the API, so it has not reached the point of accepting or rejecting the supplied credentials.
- The local resolver configuration is generated by WSL. The available checks cannot distinguish an execution restriction from a host/WSL DNS failure. A check in the user's normal terminal is needed to isolate the cause.
- Modal's public status page reports all services online: https://status.modal.com/. This does not establish reachability from this host.

## 2026-10-14T10:53:16+00:00 — Exa search connected and DNS evidence

- Exa authentication was accepted and the retried search succeeded. Search connectivity is separate from the local Modal CLI network path.
- Official Microsoft documentation identifies `10.255.255.254` as the default WSL DNS-tunneling nameserver. This matches the local resolver configuration observed earlier.
- Microsoft describes VPN, firewall, and host networking interactions that can cause WSL DNS failures. These are possible causes; the exact cause on this machine has not been established.
- Sources: https://learn.microsoft.com/en-us/windows/wsl/wsl-config and https://learn.microsoft.com/en-us/windows/wsl/troubleshooting.
- A comparison with DNS resolution in a normal user terminal remains necessary before changing WSL or network settings. No compute job was launched.

## 2026-10-14T11:08:23+00:00 — Agent execution network configuration

- User requested treating the failure as an agent execution-environment problem and fixing that configuration.
- Observed: the active session has restricted command networking; user-level Codex configuration contains no network override and is read-only in this session. No local managed-configuration file was found at the inspected standard paths; that does not rule out host-injected policy.
- Prepared `scripts/start_networked_codex.sh`, which requests command network access for a new session and retains the workspace-write filesystem policy. It does not modify the current session, global configuration, approval rules, or WSL settings.
- Reference: https://learn.chatgpt.com/docs/agent-approvals-security#network-access. OpenAI documents that workspace-write disables network by default unless enabled, and higher-level managed restrictions remain separate.
- Status: next-session launch configuration prepared, but live connectivity remains unresolved. Modal authentication is still unverified, the dataset is absent, and no remote job has been submitted.
- Validation: `bash -n scripts/start_networked_codex.sh` and `git diff --check` passed. The launcher was not executed inside the restricted session.

## 2026-10-14T11:36:09+00:00 — Execution access restored

- The new turn uses unrestricted filesystem access and enabled command networking. DNS resolution succeeded for Modal, PyPI, and Hugging Face.
- The dedicated Modal profile passed `modal token info` with exit code 0. Credentials and account identifiers were not copied into the repository. No compute job was submitted by this check.
- The exact proposed checkpoint `google/gemma-4-E4B` is publicly accessible: its metadata reports `gated: false`, and `config.json` downloads without authentication. The resolved model revision is `411aa17b749aa952df1359d2dcea73917a544d9a`; the text hidden size is 2560. Source: https://huggingface.co/google/gemma-4-E4B.
- The original Mercari TSV remains absent. Dataset provenance and a legitimate acquisition path must be established before the first real-data run.

## 2026-10-15T04:41:02+00:00 — Pilot resumed and data acquired

- Reverified networking and dedicated Modal authentication. Existing budget ledger has zero allocations before the first launch.
- Acquired OpenML dataset 46660 version 1, the published Mercari 100K research subset. It includes the original price field, so no inverse target transformation is needed. The published log_price agrees with log1p(price) within 8.89e-16. Provenance and checksums: `results/data-provenance.json`. Data stays excluded from Git; original Kaggle terms apply.
- Prepared 10,000 training rows and independent 1,000-row validation/test sets. The source has 100,000 valid target rows; duplicate-feature hashes determine split membership before sampling.
- Rechecked Modal pricing on 2026-10-15; the implemented L40S, CPU, and memory rates still match. CPU/memory hard caps now match budget assumptions.
- Offline Modal registration exposed two unexecuted-launcher bugs: nested functions require serialized=True, and scaledown_window=0 is invalid. Fixed both, added a registration test using the real SDK, and enabled visible cloud logs. No cloud job was needed to find these errors.

## 2026-10-15T04:41:12+00:00 — Run event

Reserved $0.24 worst-case job charges for `20261015T044104-baselines-3d322c05`. Actual billed cost is not yet known.

## 2026-10-15T04:41:32+00:00 — Run event

Run `20261015T044104-baselines-3d322c05` failed; allocation retained. Inspect Modal logs and billing before another launch.

- First cloud baseline launch built the CPU image, then failed before function execution because an environment image layer followed a local source mount. Moved the environment layer before add_local_dir, as required by Modal. The failed launch retains its $0.24 conservative allocation; image-build billing remains unreconciled.

## 2026-10-15T04:41:57+00:00 — Run event

Reserved $0.24 worst-case job charges for `20261015T044155-baselines-1967a0ed`. Actual billed cost is not yet known.

## 2026-10-15T04:42:01+00:00 — Run event

Run `20261015T044155-baselines-1967a0ed` failed; allocation retained. Inspect Modal logs and billing before another launch.

- The cloud API additionally enforces a two-second minimum scaledown window, stricter than the SDK local check. The second launch failed before a worker ran; its $0.24 allocation is retained. Set scaledown_window=2 before retrying.

## 2026-10-15T04:42:50+00:00 — Run event

Reserved $0.24 worst-case job charges for `20261015T044248-baselines-21e3bb99`. Actual billed cost is not yet known.

## 2026-10-15T04:43:28+00:00 — Run event

Completed `20261015T044248-baselines-21e3bb99`. Metrics: `results/20261015T044248-baselines-21e3bb99.json`. Worst-case allocation remains $0.24; reconcile with actual billing.

## 2026-10-15T04:44:32+00:00 — Run event

Reserved $0.77 worst-case job charges for `20261015T044429-digits-2e766268`. Actual billed cost is not yet known.

## 2026-10-15T04:45:22+00:00 — First real-data baselines completed

- Run report: `results/20261015T044248-baselines-21e3bb99.json`; 10K training examples and a fixed 1K final test set. Two validation-selected trials per trained model family.
- median: test RMSLE 0.758479, MAE $16.54, factor-two error rate 0.351.
- structured_tree: test RMSLE 0.611434, MAE $14.04, factor-two error rate 0.265.
- text_ridge: test RMSLE 0.566214, MAE $12.87, factor-two error rate 0.222.
- text_tree: test RMSLE 0.606705, MAE $14.06, factor-two error rate 0.269.
- Text Ridge is the strongest baseline in this first configuration. No Gemma accuracy comparison is available yet.
- All 19 local tests now pass, with no PyTorch skips. A tiny actual Gemma4 text model with per-layer embeddings also completed digit-head forward/backward on CPU; this validates API wiring, not the full checkpoint or accuracy.

## 2026-10-15T04:45:41+00:00 — Run event

Run `20261015T044429-digits-2e766268` failed; allocation retained. Inspect Modal logs and billing before another launch.

## 2026-10-15T04:50:52+00:00 — GPU account requirement and pilot analysis

- Modal built the GPU dependency image but rejected L40S function creation: a payment method is required. No GPU worker ran and no full Gemma weights were loaded. Official documentation confirms this applies to every GPU type: https://modal.com/docs/guide/gpu. The user was asked to update that account setting; no further GPU attempts were made.
- Added deterministic minibatches and coverage reporting so the next matched pilot can cover 10K rows in 1,250 steps at batch size eight.
- All 23 local tests pass. The real Gemma4 integration test includes shared KV layers, per-layer embeddings, non-reentrant checkpointing, and nonzero LoRA gradients for both output heads.
- The four-significant-digit oracle has zero reconstruction error on the 1K test sample. Several training target slots are nearly constant, so slot accuracy alone would be misleading. Details: `results/pilot-analysis.json`.
- Saved conventional-model metrics and uncertainty intervals in `results/progress-2026-10-15.md`, a CSV comparison, and a preliminary blog draft. All Gemma accuracy conclusions remain pending.
- Conservative allocations total $1.49, including failed launches. Actual project billing is not reconciled; workspace billing summaries can lag and are not a reliable project invoice.

- The actual pinned Gemma tokenizer also loads successfully on CPU. Median test input length is 84 tokens; 1.96% of train rows and 1.9% of test rows exceed the fixed 256-token limit. Saved `results/input-length-audit.json` and a public aggregate data manifest.

## 2026-10-15T06:10:42+00:00 — Cheaper GPU eligibility checked directly

- At the user's request, checked actual function registration for both T4 and L4 using the dedicated experiment profile. Modal rejected both before any GPU worker started.
- T4: `Please add a payment method to use T4 GPU functions.`
- L4: `Please add a payment method to use L4 GPU functions.`
- Exact responses and conservative probe allocations are saved in `results/gpu-access-recheck.json`. No model weights were loaded and no training ran during these probes.
- Rechecked official documentation, which continues to require a valid payment method for GPU use: https://modal.com/docs/guide/gpu. The tested cheaper hardware does not avoid this account requirement.

## 2026-10-15 — E2B-first model-size comparison planned

- The user asked whether both E2B and E4B should be trained. Documented a four-run
  matrix: each backbone with the direct regression control and the user's
  parallel digit head. E4B tests the benefit of size; it is not required by the
  proposed output representation.
- E2B config access was verified: public `google/gemma-4-E2B`, revision
  `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`, hidden size 1536, 35 text
  layers. E4B remains pinned to its previously recorded revision, hidden size
  2560, 42 text layers. Neither full checkpoint has been trained.
- Prioritize a bounded memory smoke test and the two E2B runs. Reserve E4B jobs
  only if access and budget permit both heads. Keep data, example exposure,
  validation protocol, and per-model head settings matched; record any precision
  or hardware differences. Do not use final test results to schedule or tune
  later experiments.
- The current runner is single-GPU BF16. Free-GPU FP16/QLoRA support and memory
  fit remain unverified and require actual implementation/testing. No GPU job or
  additional compute reservation was made while documenting this plan.

## 2026-10-15 — Kaggle onboarding artifacts prepared

- The user chose Kaggle. Prepared a standalone check notebook and an explicit
  source-only zip using `scripts/build_kaggle_bundle.py`; no remote repository,
  notebook, or dataset was published. Copies are saved in both Downloads folders.
- The check reports library versions, GPU memory/CUDA kernel compatibility, and
  connection checks for pinned E2B config and Mercari metadata. It does not
  install packages, download model weights, or train. Runtime startup and the
  tiny CUDA probe may consume a small amount of the user's free GPU quota.
- Requested the available accelerator through an optional question. Actual
  Kaggle GPU allocation, account eligibility, and remaining quota are unverified.
  The next required evidence is the notebook JSON and quota from Kaggle's UI;
  manual execution requires no password or API-token disclosure.
- Ran the script locally on Python 3.12.13 / CPU PyTorch 2.14.1. It correctly
  reported no local CUDA and successfully fetched both public metadata sources.
  Validated notebook/source syntax, archive paths, inclusion scope, all SHA256
  manifest entries, and absence of Modal credential patterns. This is a local
  setup check, not a real Kaggle or full-model training test.
- The source bundle excludes credentials, source rows, weights, and checkpoints.
  Low-memory FP16/QLoRA training remains to be implemented after the actual
  hardware/software report; the bundle does not claim T4/P100 training readiness.
- Setup instructions and next steps: `docs/kaggle-quickstart.md`. Official Kaggle
  notebook settings were rechecked: https://www.kaggle.com/docs/notebooks.

## 2026-10-15 — Autonomous Kaggle execution prepared; sign-in pending

- The user explicitly requested connecting and executing from the agent. Installed
  the official Kaggle CLI 2.2.4 in the project environment and added an optional
  `kaggle` dependency group.
- Checked standard Linux/Windows credential locations and relevant environment
  variables without printing secret values. No Kaggle token/OAuth credential was
  present. No running browser with a signed-in Kaggle session was available.
- Used the agent-browser skill to open Kaggle, confirmed the sign-in screen,
  and started the official CLI OAuth connection. Opened its authorization URL
  in the user's normal Windows browser as well. Authentication is still pending;
  a browser sign-in is required before the user's free GPU quota can be used.
- Prepared `scripts/run_kaggle_preflight.py` to create a unique private notebook,
  request T4 x2 and a maximum 600-second runtime, poll its status, and download
  the JSON report. The runner makes one submission, without retries or switching
  accelerator types. The check does not download weights or train.
- A local, single-use background watcher waits up to 30 minutes for the official
  OAuth credentials, then invokes that runner once. Its PID/log and the OAuth
  login log are stored under `~/.local/state/gemma-tabular-lab`, outside Git.
  Credentials stay in the official CLI's owner-only credential file. No Kaggle
  GPU job has been submitted while sign-in is pending.
- Dry-run and fake-client checks verified private/T4/timeout settings, rejection
  without retry, and refusal to submit a public notebook. This validates the
  local launcher logic, not actual Kaggle GPU availability or full Gemma training.
- Source references: https://www.kaggle.com/docs/api and
  https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md.

## 2026-10-15 — Kaggle connected; actual runtime differs from saved settings

- The user completed the official browser connection. OAuth credentials were
  saved outside Git with mode 0600. Private notebook creation, execution,
  monitoring, report download, and quota queries now work through the API.
- The account reports 30 free GPU hours, pay-to-scale disabled, and a reset at
  2026-10-17 00:00 UTC. The initial quota check reported zero GPU time used or
  reserved and no past GPU session; later session diagnostics are separate.
- Three bounded private T4 checks completed as notebooks, but their runtime
  reports show CPU-only PyTorch 2.11.0, no CUDA, no NVIDIA device files, and no
  `nvidia-smi`. Both public metadata fetches fail with temporary DNS-resolution
  errors. Saved GPU/Internet flags are true and the saved machine shape is T4.
  This establishes a runtime mismatch; it does not establish that the user
  needs phone verification or that the saved Internet setting is disabled.
- The second check added explicit notebook accelerator metadata. The third also
  requested a pinned official Kaggle GPU image, resolved from its GCR manifest.
  The API still returned the CPU image in kernel metadata. Reports and requested
  settings are summarized in `results/kaggle-access-check.json` without account
  identifiers or credentials. No full model weights were downloaded.
- Kaggle's documented direct Jupyter session API accepts a T4 request and
  returns a session ID and tokenized Jupyter URL. Readiness operations complete,
  but the first direct URL/API checks returned HTTP 404. Each inspected session
  was cancelled in a finally block. A bounded follow-up checks proxy startup
  delay and the provided Jupyter token parameter; no model training is involved.
- Opened the private notebook in the user's normal browser and asked what the
  accelerator control shows. This requests missing provider-side evidence,
  rather than asking the user to upload or execute the experiment manually.

## 2026-10-15 — E2B FP16 runner prepared and locally verified

- Added native precision selection: T4/P100 use FP16; newer compatible GPUs may
  use BF16. FP16 training uses autocast and a gradient scaler, with FP32 LoRA
  parameters and output heads. The base checkpoint remains frozen.
- Added gradient accumulation, configurable evaluation batch size, validation
  subset size, and a validation-only mode that does not read the test split.
  Reports include actual data exposure, skipped optimizer updates, precision,
  gradient checks, and effective batch size. Quantization remains unimplemented.
- All 25 local tests pass, including both heads on a tiny real Gemma4 model
  with an FP16 frozen backbone, FP32 LoRA/head parameters, shared KV, PLE, and
  non-reentrant checkpointing. This is an implementation test on CPU, not a
  full E2B GPU memory or accuracy result.
- Saved SHA256 hashes of the original prepared splits in
  `results/processed-checksums.json` for exact input verification before GPU
  training. The fixed 10K/1K/1K split and original baseline results remain the
  reference. Full Gemma training has not started.

## 2026-10-15 — Direct Kaggle code execution verified; GPU still absent

- The direct Jupyter connection now executes code and retrieves the probe JSON
  from this workspace. A 404 at its UI landing page was not sufficient to
  diagnose API failure: its API status endpoint and kernel-creation endpoint
  work with the provided scoped Jupyter URL token. No Kaggle OAuth token is
  sent to the Jupyter proxy or embedded in notebook source.
- The executed direct-session probe still reports CPU-only PyTorch, no NVIDIA
  devices, and temporary DNS-resolution failures for the fixed public metadata
  URLs. This rules out the batch notebook wrapper as the only issue. The saved
  request is T4 x2 with Internet enabled, but physical GPU allocation remains
  unverified and the cause is unresolved.
- All diagnostic sessions were cancelled after inspection, including the one
  that successfully executed code. The three batch notebooks are complete; no
  model training was launched. Saved the direct report in
  `results/kaggle-direct-session-check.json` and refreshed quota/status evidence
  in `results/kaggle-access-check.json`.
- Further GPU attempts wait for the user's report of Kaggle's own accelerator
  control. A verification requirement has not been established, so the user was
  asked what the UI actually shows rather than told to repeat authentication.
- Official connection/protocol references:
  https://www.kaggle.com/docs/notebooks,
  https://jupyter-server.readthedocs.io/en/latest/developers/rest-api.html, and
  https://jupyter-server.readthedocs.io/en/latest/developers/websocket-protocols.html.

- Loaded the actual pinned E2B tokenizer locally and audited the fixed splits.
  Test median is 84 tokens; 1.9% exceed 256, matching the previously observed
  E4B length summary. Saved `results/input-length-audit-e2b.json`. This downloads
  tokenizer assets only, without full model weights or GPU training.
- The last quota query still reports zero GPU time used/reserved and all 30
  hours available. No diagnostic session remains intentionally running.
- 15 October, later: the digit head no longer uses eight scientific-notation
  slots. It predicts four whole-dollar digits (thousands, hundreds, tens, ones),
  covering $0–$9999. Training prices reach $1100; two training prices are $10.50
  and round half-up to $11. Test prices are already whole dollars. Slot balance
  is in `results/dollar-digit-codec.json`.
- 15 October, 10:20 UTC: Kaggle CLI smoke on `ommnaidu/gemma-e2b-regression-smoke`
  version 4. Two Tesla T4 GPUs were visible. E2B LoRA regression used one T4,
  peak memory 9.51 GiB, 20 steps, finite head and LoRA gradients. Validation was
  8 rows only, so its MAE is not a test result. Report:
  `results/e2b-regression-smoke.json`. The matched 1,250-step pilot was then
  pushed as private kernel `ommnaidu/gemma-e2b-matched-pilot`.

## 2026-10-16 — Digit head dropped

The four-box digit head lost to the direct price head ($14.81 versus $11.92
average error) and also lost to Ridge. The user decided to stop that output.
New runs use the regression head only. The E4B job started the same day is
regression-only: `ommnaidu/gemma-e4b-price-head`. It finished later the same
day and did not beat E2B. See the E4B section below.

## 2026-10-16 — Both E2B heads finished

The question was whether LoRA on Gemma, with either a direct price head or
parallel dollar-digit heads, beats a cheap text model on Mercari prices. E2B
was chosen before E4B because the output design does not need the larger
model, and one T4 has about 15 GiB. E4B was left for later. Modal was not used
for the GPU runs: it still requires a payment method. Kaggle was used after
the account was phone-verified. The path that worked was the Kaggle CLI from
WSL, not the browser.

### Why the output changed

The user's original encoding was eight scientific-notation slots, as in
`+1.800e+05`. On these prices most of those slots never change. The assistant
kept the user's parallel-digit idea and changed the slots to four whole
dollars: thousands, hundreds, tens, ones. The regression head was the
assistant's control, not part of the original digit proposal. Half-up rounding
is used. Two training prices are $10.50 and become $11. Test prices are
already whole dollars from $3 to $400.

### What ran

| Run | Kernel | Status | Report |
| --- | --- | --- | --- |
| 20-step regression smoke | `ommnaidu/gemma-e2b-regression-smoke` version 4 | Complete | `results/e2b-regression-smoke.json` |
| Regression, 1,250 steps | `ommnaidu/gemma-e2b-matched-pilot` | Cancelled after this head finished | `results/kaggle-pilot-output/e2b-regression-report.json` |
| Digits, 1,250 steps | `ommnaidu/gemma-e2b-digit-head` | Complete on 16 October | `results/kaggle-digit-output/e2b-digits-report.json` |

Shared settings: `google/gemma-4-E2B` revision
`d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`, LoRA r=8, alpha=16, dropout=0.05
on q/k/v/o, FP16 backbone, FP32 adapter and head, batch 1 with 8 accumulation
steps, 256 tokens, seed 42, validation checkpointing on 256 rows by RMSLE,
then one test pass on 1,000 rows. Torch 2.11.0+cu128, transformers 5.16.1,
peft 0.20.0. Peak memory 9.51 GiB on one Tesla T4. The second T4 was visible
and unused. Trainable parameters: regression 2,680,321; digits 2,740,264.

The first pilot was stopped during the digit head, around step 180, because
the user asked to continue the next day. That session is
`KernelWorkerStatus.CANCEL_ACKNOWLEDGED`. Its saved digit adapter is only the
best checkpoint from the interrupted run. It is not the finished digit result.
The finished digit run is the separate kernel above. It was started fresh
because the interrupted run did not save optimizer state.

### Test result, 1,000 rows

| Model | RMSLE | MAE, USD | Median AE | p90 AE | Factor-two error | Warm batch-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + Ridge | 0.5662 | 12.87 | — | — | 22.2% | 5.41 ms, CPU |
| E2B regression | 0.5058 | 11.92 | $6.67 | $24.30 | 18.3% | 110 ms, T4 |
| E2B four digits | 0.6469 | 14.81 | $7.00 | $30.00 | 24.8% | 113 ms, T4 |

Regression's best validation RMSLE was step 1,250 (0.5303). Digits' best
validation RMSLE was step 500 (0.6458, MAE $12.14). Test scores use those
checkpoints. Digit validation got worse again late in training; step 1,250
validation RMSLE was 0.8826.

Digit test slot accuracy was thousands 100%, hundreds 97.1%, tens 40.1%, ones
16.3%. On the test split the majority digits are already 100%, 97.0%, 36.8%,
and 16.1% (`results/dollar-digit-codec.json`). The head is barely ahead of
always guessing the common digit. Dollar error is the score that matters, and
on that score the digit head loses to both Ridge and the regression head.

### What this does and does not say

The regression head is better than Ridge on this split and seed, by about
$0.95 MAE and 0.060 RMSLE. It is also about twenty times slower, on a GPU
rather than a CPU. The digit encoding did not help. One seed, a 256-row
validation subset, and a public dataset that may have appeared in pretraining
are all limits. Classical text models saw the full description; 1.9% of Gemma
test rows are cut at 256 tokens. E4B was trained later the same day, as a
regression head only, and did not beat this E2B result. Do not launch another
E4B or digit run from this table.

## 2026-10-21 — Made-up sentence shop, both models trained

1,000 train rows, 200 validation rows, 500 test rows. The test is half
wording from training and half wording whose content words were held out.
Ridge and Gemma 4 E2B were both trained. On seen wording the average errors
are $10.21 and $8.63. On new wording they are $80.00 and $9.21. Ridge's gap
on the new wording collapses to about $0. Gemma keeps about $144 of a $159
true gap. The plain conclusion, with the Mercari context and what the run
does and does not prove, is in `docs/blog-synthetic-sentence.md` under
Conclusion. Short record: `docs/learnings-synthetic-sentence.md`. Notebook:
https://www.kaggle.com/code/ommnaidu/gemma-e2b-synthetic-price.

## 2026-10-21 — Made-up sentences, same Ridge

`scripts/synthetic_sentence_gap.py` builds listings that are not Mercari. The
brand and title stay the same. The price is about $200 when the screen is
fine and about $40 when it is broken. Ridge uses the same word-and-word-pair
setup as the real run. Gemma was not trained on these rows. Report:
`results/synthetic-sentence-gap.json`.

On wording Ridge has seen, “no cracks on the screen” is guessed at $184
against a true $187, and “has cracks on the screen” at $41 against a true
$24. On wording Ridge has not seen, “the display is unmarked” and “the
display is smashed” are both guessed at $74. The true prices are $199 and
$37. The gap the real Mercari average hides is this one: a new sentence with
the same product.

## 2026-10-21 — Same test set, split by the sentence

The 1,000 scored Mercari test rows were labeled from the text only, then
Ridge was fit again with the original recipe. Penalty 1. Overall MAE
returned at $12.87, matching the saved baseline. E2B guesses were the saved
file `results/kaggle-pilot-output/e2b-regression/predictions.jsonl`. Report:
`results/mercari-sentence-slice.json`.

Ordinary rows, 733: Ridge $12.47, E2B $11.50. Any sentence rule, 267: Ridge
$13.96, E2B $13.07. The mixed hard pile does not widen the gap. Condition
sentences, 101 rows: Ridge $13.49, E2B $11.58. Bundles, 132 rows: Ridge
$13.55, E2B $13.98. Long descriptions and unseen words make both models
worse. The rules overlap. Do not treat the condition-sentence lead as a
second seed or a hand-labeled test.

### Engineering notes for the next agent

- Do not rerun the Ridge baselines or these two E2B heads unless the user asks
  for another seed or a changed method.
- Do not push a new version of `gemma-e2b-matched-pilot`,
  `gemma-e2b-digit-head`, or `gemma-e4b-price-head`. A push starts training
  again.
- Do not rerun the E4B regression unless the method changes. Its test result
  is already in `results/kaggle-e4b-output/e4b-regression-report.json`.
- Before any new LoRA run, `pip uninstall -y torchao`. Peft 0.20 fails against
  the installed torchao 0.10. Do not upgrade Torch to fix that.
- Kaggle unpacks an uploaded tar. The entry script must find
  `processed-checksums.json` and copy that source tree. It must not expect one
  `payload.tar` file.
- The private dataset is `ommnaidu/gemma-e2b-smoke-files`. Internet must stay
  on so the notebook can fetch Gemma and OpenML.
- The pushed kernel id `ommnaidu/gemma-e2b-pilot` is not the live slug. The
  live regression notebook is `ommnaidu/gemma-e2b-matched-pilot`.
- `kaggle kernels get` does not return `kernel_session_id`. Cancelling through
  that field failed once. The pilot later showed `CANCEL_ACKNOWLEDGED`. Do not
  delete either notebook.
- Credentials stay in `~/.kaggle/credentials.json`. Do not copy them into the
  repo or chat. There is no Git remote. Do not push the repo unless asked.
- `results/kaggle-pilot-output/` also contains a partial Hugging Face cache and
  a copy of the data. The regression report in that folder is complete. The
  digit adapter there is the interrupted run.
