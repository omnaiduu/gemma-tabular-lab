# What, why, and how

## What

Build an offline predictor of a marketplace listing's actual sale price from
information available before the sale. The initial source is Mercari's public
competition training data. The target is `price`; inputs are `name`,
`item_description`, `category_name`, `brand_name`, `item_condition_id`, and
`shipping`. Listing IDs and the target are not features.

The data source has competition-specific usage terms. Dataset access and its
availability must be verified. Do not redistribute the underlying data in Git.
If it cannot be obtained, record the blocker and any proposed replacement; a
different price dataset is not silently interchangeable with actual sale prices.

## Why

Test three hypotheses:

1. Adapted Gemma provides useful additional information for price prediction.
2. Parallel categorical digits improve numerical prediction relative to a direct
   regression output on the same Gemma backbone.
3. Any improvement justifies the added training, memory, and inference costs.

We seek evidence, including negative results, rather than a demonstration with
an intentionally weak classical baseline.

## Architecture

`serialized listing -> Gemma text backbone -> final non-padding hidden vector -> output head`

The digit head predicts four whole-dollar digits in parallel: thousands,
hundreds, tens, and ones. Each digit is a choice from 0 to 9, so the head has
40 scores. The price is `1000*thousands + 100*hundreds + 10*tens + ones`,
which covers $0 to $9999. Prepared training prices reach $1100, so three digits
would clip two training rows. Signs and scientific-notation exponents were
removed because these prices are non-negative and sit in a small range. Values
are rounded half-up to the nearest dollar. The held-out test prices are already
whole dollars from $3 to $400, so that rounding does not change them. The
encoding's own quantization error must still be reported.

The direct-regression control predicts a scaled log-price from the same hidden
vector. Evaluation converts all outputs back to the original currency units.
Head-only adaptation and LoRA adaptation are distinct experimental stages.

## Baselines and fairness

- Training-set median, as a sanity reference.
- A tuned tree regressor with structured features.
- A traditional regressor with text features as well as structured information.
- Frozen Gemma representations with the two output designs.
- Matched LoRA Gemma runs with the two output designs, if budget permits.

Identical held-out splits, allowed input fields, training-row subsets, and
documented tuning budgets are required. Both Gemma heads use the same 256-token
input limit. Conventional text baselines receive the full allowed text fields;
this practical resource-budget difference must be disclosed, and a strictly
matched text-exposure comparison would need another ablation. Fit vocabulary, scaling,
dimensionality reduction, and category preprocessing only on training data.
Keep exact duplicate feature records in the same split. Report the cleaning and
sampling rules. Do not tune using the final test set.

## Stages

1. Verify data access, Modal authentication, model loading, and spend controls.
2. Validate the codec, loss, row serialization, and split isolation locally.
3. Run CPU baselines and a small GPU smoke test with saved metrics.
4. Pilot on approximately 10000 training rows. Cache frozen representations when useful.
5. Use pilot validation results to select bounded LoRA runs. Increase sample size
   only if the measured runtime and available budget support it.
6. Export predictions, metrics, artifacts, and a readable comparison. A production
   API and evaluation on fresh company data follow evidence from the offline study.

The $30 budget may support a useful pilot rather than a full-scale, multi-seed
study. Record actual coverage and do not describe unrun stages as completed.

## Measurements

- MAE in currency units.
- RMSLE for relative price error.
- Median and 90th-percentile absolute error.
- Rate of predictions more than a factor of two from the actual positive price.
- Error by category, price range, missingness, and description length.
- Digit and exponent accuracy, alongside numerical error, for the proposed head.
- Warm batch-1 and batched inference latency and throughput.
- Peak GPU memory, training wall time, and compute charge estimates.
- Model-load and cold-start costs separately from warm inference.

Different CPU/GPU hardware is disclosed. Model names alone are not fair cost
comparisons. A public historical dataset may have appeared in model pretraining;
results do not establish generalization to future company transactions.

## E2B and E4B comparison

E4B was the user's original checkpoint choice. The numerical-output design does
not require that size. E2B was run first to establish memory use and training
cost. Both E2B heads and the E4B regression head have now finished one seed.
The E4B digit head was not run: on 16 October the digit output was dropped.

| Run | Backbone | Output | Purpose |
| --- | --- | --- | --- |
| E2B regression | Gemma 4 E2B | Direct scaled log-price | Smaller-model control |
| E2B digits | Gemma 4 E2B | Four parallel dollar digits | Test the dollar-digit head on E2B |
| E4B regression | Gemma 4 E4B | Direct scaled log-price | Measure size effects with a direct head |
| E4B digits | Gemma 4 E4B | Four parallel dollar digits | Not run. The digit head was dropped after it lost on E2B |

Each run starts independently from its corresponding pinned pretrained
checkpoint. Train LoRA adapters and the output head; do not pretrain the full
backbone. The existing loader accepts a model ID and derives the head input size
from the checkpoint config. Config access was verified for both models:

- E2B: `google/gemma-4-E2B`, revision
  `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`, hidden size 1536.
- E4B: `google/gemma-4-E4B`, revision
  `411aa17b749aa952df1359d2dcea73917a544d9a`, hidden size 2560.

Keep the same 10K training rows, validation/test partitions, row ordering seed,
256-token limit, LoRA configuration, and training-example exposure. Match
precision/quantization across sizes when feasible. If precision, hardware, or
batching must differ, disclose that difference and keep each model's two heads
matched. A wall-time cutoff that stops one run early prevents claiming equal
training coverage. Use validation for checkpoint selection; do not select
whether to run E4B based on final E2B test results.

First perform a bounded 20-step validation-only memory/gradient smoke test.
Then complete the two E2B runs before reserving compute for the two E4B runs.
Report MAE/RMSLE, per-digit accuracy, peak GPU memory, wall time, and inference
latency. Compare within each size to test the head design, and within each head
to test whether a larger backbone adds useful accuracy. E2B failure does not
establish E4B failure, and E4B success alone does not establish a digit-head gain.

The runner chooses native FP16 on a T4. One GPU is used; the second T4 does not
add memory. QLoRA is not implemented. Both E2B heads completed 1,250 optimizer
steps on seed 42, seeing all 10,000 training rows once. Reports:

- Regression: `results/kaggle-pilot-output/e2b-regression-report.json`
- Digits: `results/kaggle-digit-output/e2b-digits-report.json`

On the 1,000 test rows, regression MAE is $11.92 and RMSLE is 0.5058. Digits
MAE is $14.81 and RMSLE is 0.6469. Ridge remains $12.87 and 0.5662. The digit
head did not beat the regression head or Ridge. Peak memory was 9.51 GiB on one
T4. This is one seed.

On 16 October the user dropped the four-box digit head. Later runs use the
direct regression head only. E4B regression is the private kernel
`ommnaidu/gemma-e4b-price-head`. Its text weights are about 14 GiB, so the
5.25 GiB per-layer embedding table stays on CPU and the transformer stays on
the one T4. That is the same lookup, with the table kept off the GPU so the
model fits. The first push died because Kaggle runs only the one notebook
script and did not place `train.py` beside it. The retry packed that source
into the script. Do not start another digit-head run, and do not push the
finished E4B notebook again.

E4B regression finished the same day. The test checkpoint is step 1,100, the
best validation RMSLE (0.5508). Step 1,250 was worse on the checking rows
(RMSLE 0.794) and was not scored. On the 1,000 test rows, MAE is $12.65 and
RMSLE is 0.5363. That beats Ridge by about $0.22 and loses to E2B regression
($11.92, RMSLE 0.5058). Peak GPU memory was 8.87 GiB. Load took 85 seconds,
training plus validation 6,887 seconds, and a warm single-row guess 165 ms.
The report is `results/kaggle-e4b-output/e4b-regression-report.json`. One
seed, with the embedding table on CPU, does not by itself prove E4B is a
worse price model. It does show that this larger run did not beat E2B.

Model references: https://ai.google.dev/gemma/docs/core/model_card_4,
https://huggingface.co/google/gemma-4-E2B, and
https://huggingface.co/google/gemma-4-E4B.

## Improvements to test after the pilot

- Slot cross-entropy compared with a numerical-distance-aware loss.
- Supplying scaled numeric features directly alongside the Gemma representation.
- More significant digits if encoding error is material.
- A small conditional digit decoder if parallel outputs produce inconsistent choices.

Keep these as recorded ablations, not simultaneous changes to the main comparison.
