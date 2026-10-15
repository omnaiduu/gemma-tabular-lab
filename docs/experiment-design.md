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

The main head produces separate categorical logits for sign, four mantissa
digits, exponent sign, and two exponent digits. The formatter reconstructs the
prediction. A canonical zero representation is required. Four significant
digits impose rounding; the encoding's own quantization error must be reported.
For positive prices, sign constraints can be applied consistently.

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

## E2B and E4B comparison (planned, not yet run)

E4B was the user's original checkpoint choice. The numerical-output design does
not require that size. Start with E2B to establish memory use and training cost;
add E4B as a model-size comparison only if GPU access and the remaining budget
support completing both heads.

| Run | Backbone | Output | Purpose |
| --- | --- | --- | --- |
| E2B regression | Gemma 4 E2B | Direct scaled log-price | Smaller-model control |
| E2B digits | Gemma 4 E2B | Eight parallel categorical slots | Test the user's design on E2B |
| E4B regression | Gemma 4 E4B | Direct scaled log-price | Measure size effects with a direct head |
| E4B digits | Gemma 4 E4B | Eight parallel categorical slots | Measure size effects with the digit head |

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
Report MAE/RMSLE, exponent mistakes, peak GPU memory, wall time, and inference
latency. Compare within each size to test the head design, and within each head
to test whether a larger backbone adds useful accuracy. E2B failure does not
establish E4B failure, and E4B success alone does not establish a digit-head gain.

The runner now chooses native FP16 or BF16 on one GPU and supports gradient
accumulation, configurable evaluation batches, and validation-only smoke runs.
Its FP16 frozen backbone with FP32 LoRA/head gradients passes tiny real-Gemma4
CPU tests. Full-model GPU memory fit remains unverified. QLoRA and multi-GPU
execution still require implementation. Neither full E2B nor E4B training has
run yet. The initial four runs use one seed and remain a pilot; repeat training
seeds only if resources permit before making robust claims.

Model references: https://ai.google.dev/gemma/docs/core/model_card_4,
https://huggingface.co/google/gemma-4-E2B, and
https://huggingface.co/google/gemma-4-E4B.

## Improvements to test after the pilot

- Slot cross-entropy compared with a numerical-distance-aware loss.
- Supplying scaled numeric features directly alongside the Gemma representation.
- More significant digits if encoding error is material.
- A small conditional digit decoder if parallel outputs produce inconsistent choices.

Keep these as recorded ablations, not simultaneous changes to the main comparison.
