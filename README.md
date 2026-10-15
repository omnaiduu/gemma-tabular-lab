# Gemma tabular numerical prediction experiment

Status: conventional baselines measured on the Mercari 100K research subset.
Full Gemma training has not run. Modal requires a payment method. Kaggle is now
authenticated, but the [runtime checks](results/kaggle-access-check.json) have
returned CPU-only execution and DNS failures despite saved GPU/Internet settings.
See the [Kaggle quickstart](docs/kaggle-quickstart.md) and
[initial results](results/progress-2026-10-15.md).

This project tests a proposal from the user: predict a number by selecting its
sign, mantissa digits, and exponent digits as small categorical decisions. The
main experimental version uses one Gemma backbone pass and parallel output
heads. A deterministic formatter reconstructs the numerical prediction.
E2B is the first planned backbone, with E4B as a resource-dependent size comparison.

The production task is marketplace listing-price estimation. Given a listing's
title, description, category, brand, condition, and shipping information, predict
its actual sale price. The first planned dataset is Mercari's historical price
suggestion dataset, using the published OpenML 100K subset for this pilot.
It is an offline benchmark, not a production deployment.

## Read first

- [Conversation and attribution](docs/conversation-summary.md)
- [What, why, and how](docs/experiment-design.md)
- [Findings and run journal](docs/findings.md)
- [Compute budget](docs/budget.md)
- [Run commands and environment requirements](docs/runbook.md)
- [Kaggle quickstart and what to share](docs/kaggle-quickstart.md)

## Comparison

1. Structured-feature tree model.
2. A traditional model that also receives text features.
3. Gemma E2B with a direct numerical regression head, a control suggested by the assistant.
4. Gemma E2B with the user's categorical digit representation, the main proposal.
5. The same two heads on E4B if access and remaining resources permit both runs.

Claims will be tied to measured results and recorded limitations. We do not
assume Gemma or the digit representation wins. We will not claim the architecture
is new to the research literature without a separate literature review.

## Constraints

- Total authorized Modal spending: **$30 maximum**.
- Execution uses bounded jobs and a conservative local reservation ledger.
- Credentials stay outside this repository; conversation summaries omit them.
- This is a new local Git repository. A remote and push are deferred until requested.

## Sources

- [Mercari data description](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data)
- [Gemma 4 E2B checkpoint](https://huggingface.co/google/gemma-4-E2B)
- [Gemma 4 E4B checkpoint](https://huggingface.co/google/gemma-4-E4B)
- [Gemma implementation](https://huggingface.co/docs/transformers/model_doc/gemma4)
- [Numerical losses for language models](https://arxiv.org/abs/2411.02083)
- [Language-model adaptation for tabular prediction](https://arxiv.org/abs/2406.12031)

The engineering blog will be written from this repository's design decisions,
run records, results, failed attempts, and limitations. It is not a predetermined
success story.
