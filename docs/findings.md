# Findings and run journal

Append findings as the work proceeds. Separate observations, hypotheses, and
measurements. Every result should cite an artifact or reproducible command.

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
