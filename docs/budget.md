# Modal compute budget

Total authorized spending is **$30**, including CPU, GPU, memory, and other
billable project resources. Do not rely on an account having exactly $30 in
remaining credits; credentials do not convey billing state.

The working compute target is at most **$25**, leaving **$5** for overhead and
uncertainty. Reserve a worst-case charge before each remote job. A reservation
must cover the configured timeout, GPU count, requested CPU, and requested
memory at the documented rates. Do not launch parallel GPU jobs. Disable retries
and autoscaled replicas for these bounded experiment jobs.

Track each reservation, job ID, elapsed time, outcome, and charge estimate in a
local ledger. Reconcile against the Modal dashboard when accessible. An app-local
ledger is not a billing-enforced account limit and does not account for unrelated
workloads; document this limitation.

Before a larger run, prefer a workspace usage/spend cap if the account can set
one without changing unrelated workloads. Never raise an account cap above the
user's authorized amount. The public pricing page and budget documentation are
the authoritative references:

- https://modal.com/pricing
- https://modal.com/docs/guide/budgets

Stages must fit within the remaining project budget. Model downloads and image
builds can cost time and resources. Do not describe pricing estimates as an
actual invoice or spend measurement.

Credentials are stored outside Git and must never appear in commands logged by
the experiment, metrics, exception dumps, documentation, or artifacts.
