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

## Improvements to test after the pilot

- Slot cross-entropy compared with a numerical-distance-aware loss.
- Supplying scaled numeric features directly alongside the Gemma representation.
- More significant digits if encoding error is material.
- A small conditional digit decoder if parallel outputs produce inconsistent choices.

Keep these as recorded ablations, not simultaneous changes to the main comparison.
