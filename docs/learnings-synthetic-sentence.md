# Learnings — made-up sentence shop, 21 October 2026

The listings are generated. They are not Mercari. The full narrative is
`docs/blog-synthetic-sentence.md`. This note is the measured record.

## What was built

`scripts/build_synthetic_sentence.py` wrote:

- 1,000 train rows
- 200 validation rows, same wording as train, used only to choose settings
- 500 test rows: 250 seen wording, 250 new wording

A fine item is about $200. A broken item is about $40. Brand, product,
condition, and shipping do not differ between the two. The new-wording
content words were checked to be absent from the training sentences. Files:
`data/synthetic-sentence/processed/`. Do not rebuild them. The script now
refuses to overwrite the scored split.

The generated `row_id` values were reused across the fine and broken halves,
so there are 500 distinct id strings for 1,000 training rows. The trainer
walks positions. It processed 4,000 examples, which is four passes over the
1,000 rows. Scores were joined by file order, and the join checked that each
saved `actual` matched the test-file price. A later rebuild would issue
unique ids. Do not rebuild in order to fix the ids. That would be a new
dataset.

## Ridge

Same recipe as Mercari: TF-IDF words and neighboring pairs, max 20,000
features, penalty chosen from 1 and 10 on the validation rows. Penalty 1
won (validation RMSLE 0.117 against 0.231). Script:
`scripts/score_synthetic_ridge.py`. Report:
`results/synthetic-sentence-ridge.json`.

| Slice | Average error | RMSLE | Model gap | True gap |
| --- | ---: | ---: | ---: | ---: |
| Seen wording | $10.21 | 0.152 | $155 | $159 |
| New wording | $80.00 | 0.824 | −$0.65 | $159 |

On new wording Ridge guessed about $81 for both the fine and the broken
items. Factor-two error on that slice was 81%.

## Gemma 4 E2B

Kernel: https://www.kaggle.com/code/ommnaidu/gemma-e2b-synthetic-price

Revision `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`. LoRA r=8, alpha=16,
dropout=0.05, on q/k/v/o. Regression head only. FP16 on one T4. Batch 1,
accumulation 8, 500 steps, max length 128, seed 42, validation limit 200.
`torchao` was uninstalled before LoRA, as on the earlier runs. Trainable
parameters 2,680,321. Peak GPU memory 9.51 GiB. Load 63 seconds. Training
plus validation 2,039 seconds.

The scored checkpoint is step 400, validation RMSLE 0.126 and average error
$7.29. Step 500 validation RMSLE was 0.126 as well but slightly higher, so
it was not used. No optimizer step was skipped.

| Slice | Average error | RMSLE | Model gap | True gap |
| --- | ---: | ---: | ---: | ---: |
| Seen wording | $8.63 | 0.131 | $148 | $159 |
| New wording | $9.21 | 0.138 | $144 | $159 |

New-wording guesses were about $187 for fine items and $42 for broken ones.
No test guess was off by more than a factor of two. Report:
`results/synthetic-sentence-gemma.json`. Raw training report:
`results/kaggle-synthetic-output/synthetic-sentence-report.json`.

## Conclusion, in plain words

Context. On real Mercari listings, Gemma's average miss was $11.92 and
Ridge's was $12.87. Splitting that same test by damage sentences and
bundles kept the gap near one dollar. The useful words, the brand and the
product, were already in the training file, so a word counter was enough.

What we did. We made a shop of 1,000 training listings, 200 checking
listings, and 500 test listings. Same brand, same product, condition 3,
shipping 0. A fine screen is about $200. A broken screen is about $40.
Training used "no cracks" and "cracked screen." Half the test repeats those
words. Half says the same fact as "unmarked" and "smashed." We trained
Ridge on the CPU and Gemma 4 E2B on one Kaggle T4 for 500 steps, and scored
both on the 500 test rows. The notebook is
https://www.kaggle.com/code/ommnaidu/gemma-e2b-synthetic-price.

What we found. On seen words the misses are $10.21 for Ridge and $8.63 for
Gemma. On new words they are $80.00 and $9.21. Ridge says about $81 for
both phones. Gemma says about $187 and $42, and keeps about $144 of the
$159 gap.

What we learned. Ridge can price a word only after it has counted that word
in the training file. Gemma already knew English, so a new word for the
same fact still moves the price.

What this proves. The Mercari one-dollar lead is what you get when the
price words have already been seen. The big gap appears when the sentence
is the only clue and the test uses new words. It does not replace the
Mercari score, and it does not say a real broken phone is always $160
cheaper. The long version is `docs/blog-synthetic-sentence.md`. The Mercari
article is `docs/blog-draft.md`. Lab notes are `docs/findings.md`.

## What changed our mind

On seen wording the two models are within about $2 of each other. On new
wording Ridge loses the entire $159 gap and Gemma keeps $144 of it. The
Mercari one-dollar lead was the case where the useful words had already
been seen. This shop is the case where they have not.

## What not to do next

- Do not push a new version of `ommnaidu/gemma-e2b-synthetic-price`. A push
  starts training again.
- Do not treat these numbers as a Mercari result.
- Do not rebuild `data/synthetic-sentence/processed/` and compare the old
  guesses to the new file.
