# Testing Gemma's numerical heads against classical price predictors

*Working draft. The conventional baselines are measured; the Gemma comparison
is pending GPU account access. This is not a finished results article.*

The starting idea came from a practical question: if pretrained language models
are widely useful, can a small one become a useful table predictor? The task
stays familiar. A marketplace listing supplies a title, description, category,
brand, condition, and shipping information. The predictor estimates its price.

The user's proposed output method turns the number into small choices.
For example, `180000` becomes `+1.800e+05`. Eight categorical decisions choose
the signs and digits; code supplies the punctuation and reconstructs the value.
The follow-up proposal was to make all eight decisions after one backbone pass.

## One backbone, two outputs

Our implementation serializes the allowed fields, passes the text through
Gemma, and pools the last non-padding hidden vector. A linear layer produces
64 scores split across eight classification groups. The leading mantissa digit
includes zero so the representation can handle zero consistently.

With the selected Gemma hidden size of 2,560, that head has 163,904 parameters.
It bypasses vocabulary generation. It still pays for running the backbone.

The assistant suggested a direct regression head as a control immediately after
the user asked about predicting the digits in one pass. That control maps the
same representation to one standardized log-price value and has 2,561 head
parameters. LoRA adapts the backbone separately from this choice of output.

This control helps distinguish useful pretrained representations from a useful
numerical encoding. The digit proposal is the user's; the regression-head
control is the assistant's suggestion. Neither method is claimed novel without
a separate literature review.

## A real dataset and a limited budget

The pilot uses the published Mercari 100K subset from
[OpenML](https://www.openml.org/search?type=data&id=46660), originating in the
Mercari price-suggestion competition. It retains both `price` and `log_price`.
We verified their agreement with `log_price = log1p(price)` and use the original
price directly. Checksums and provenance are saved; source rows are excluded
from Git and remain subject to the original Kaggle terms.

Normalized feature hashes place duplicate inputs in the same partition before
sampling 10,000 training rows and 1,000 rows for each holdout. Listing IDs and
the target are excluded from the input. Validation selects hyperparameters;
the final test set is for reporting.

The authorized Modal budget is $30. A local ledger reserves conservative
maximum allocations, retains failures, and leaves $5 for overhead. This is
project accounting, not an account-wide billing cap. Remote jobs have explicit
timeouts, no retries, one container, and CPU/memory limits.

## The first measurements

On the held-out test sample, the median-price baseline reaches 0.7585 RMSLE.
Structured XGBoost reaches 0.6114. TF-IDF with Ridge reaches 0.5662 and a $12.87
mean absolute error, outperforming the tested tree configurations. Its median
warm single-row prediction time is about 5.4 milliseconds on the remote CPU.

These are preliminary baseline results, not a comparison with Gemma. They
provide a useful threshold: a larger system must justify its additional cost
against a cheap predictor that already uses the text.

The full measured table and artifacts are in
[the initial results report](../results/progress-2026-10-15.md).

## A trap in digit accuracy

The codec's oracle reconstruction error is zero on all 1,000 test prices. Four
significant digits are sufficient for these observed values. Prediction remains
the hard part.

Several categorical targets are constant or almost constant. Both signs are
positive, the first exponent digit is always zero, and the later mantissa
digits are overwhelmingly zero. A model could obtain high slot accuracy while
making expensive numerical mistakes. We therefore evaluate price error and
severe mistakes alongside slot accuracy.

## Failures that changed the implementation

The initial execution environment blocked networking, Windows interop, and
access to local audio sockets. A fresh session with changed permissions restored
those command paths and allowed Modal authentication to succeed.

Actual cloud registration then exposed untested launcher assumptions. Modal
requires explicit serialization for nested functions, local source mounts must
follow image build steps, and the server requires a scaledown window of at least
two seconds. These failures were documented rather than silently removed from
the budget record.

The next blocker came from the account: Modal rejected the L40S GPU function
because no payment method was present. Its documentation requires a payment
method for all GPU types, even when compute credits are available. No full
Gemma checkpoint has therefore run in this experiment yet.

Meanwhile, local tests exercised both heads on small actual Gemma4 models,
including per-layer embeddings, shared KV layers, checkpointing, and LoRA
gradients. These tests validate the wiring without pretending that random tiny
weights establish anything about full-model prediction quality.

We then connected Kaggle through its official CLI OAuth flow and completed
private notebook submissions, status monitoring, and report downloads. The
account reported 30 hours of free GPU quota. Yet the saved T4/Internet settings
did not match execution: three batch probes used a CPU image, had no NVIDIA
devices, and failed external DNS resolution. Explicit notebook metadata and a
requested GPU image did not resolve it.

Kaggle's documented direct Jupyter connection also worked for executing our
probe remotely, after preserving its scoped URL token for API calls. Its
runtime still lacked a GPU and external DNS. This is an unresolved execution
constraint, not evidence about Gemma accuracy. We cancelled the diagnostic
sessions and requested the accelerator control's actual UI state before another
attempt. The experiment journal preserves these failures separately from model
results.

## The comparison still to run

After account setup, the next stage is a bounded full-checkpoint smoke test,
followed by matched LoRA runs that fit the remaining budget. Mini-batching now
records processed examples and unique rows, so a step count cannot masquerade
as an epoch. An effective batch size of eight needs 1,250 completed updates to
visit all 10,000 training rows once. On smaller GPUs this can use one-row
microbatches with eight accumulation steps. The runner records skipped FP16
scaler updates as well as examples processed; data exposure and optimizer
updates are distinct.

Both Gemma heads will use the same checkpoint revision, inputs, and token limit.
The classical text baselines use the full allowed fields, while the initial
Gemma limit is 256 tokens. That resource-budget difference must be disclosed.
An audit using the real pinned tokenizer found that 1.9% of test rows exceed
this limit; the median test row uses 84 tokens.
The public historical dataset may also have appeared in pretraining; it does
not establish generalization to future private company transactions.

The missing results are numerical error, exponent failures, latency, memory,
and cost for the trained Gemma variants. Until those measurements exist, the
experiment supports no claim that Gemma beats classical ML or that categorical
digits beat a direct numerical head.
