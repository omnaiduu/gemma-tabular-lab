# I compared Gemma and Ridge on a listing price

I wanted to guess the selling price of a second-hand listing from the words, and I wanted to see when a language model beats a normal word-count model.

The listing has a title, a description, a category, a brand, a condition number, and who pays the shipping. The price is hidden. I train on old listings and score listings the model did not train on. Average error means the real price minus the guess, averaged. A real price of $200 and a guess of $190 is an error of $10.

## The two models

Ridge is a normal machine learning model. I count each word, and each pair of words that sit next to each other. Each counted piece gets a dollar weight. The guess is the sum of those weights. A word that never appeared in the training file has weight zero, so it adds nothing.

Gemma is Google's language model, Gemma 4 E2B. It reads the whole sentence. I did not retrain the whole model. I froze it, and I trained a small add-on, LoRA, plus one new layer that turns the last real word of the listing into one price. That run was on one Kaggle Tesla T4, in 16-bit, seed 42.

## The real listings

The listings are from Mercari, a second-hand app in the US. I used the public 100,000-row set. I kept 10,000 rows to train, 1,000 to choose settings, and 1,000 to test. Every model saw that same split. A typical test item costs about $18. The Mercari file is not in this repository. The download command is in [the runbook](runbook.md).

| Model | Average error | RMSLE |
| --- | ---: | ---: |
| Gemma 4 E2B, one price | $11.92 | 0.506 |
| Ridge, word counts | $12.87 | 0.566 |
| Gemma 4 E4B, one price | $12.65 | 0.536 |
| Gemma 4 E2B, four digit boxes | $14.81 | 0.647 |
| Tree on brand, category, condition | $14.04 | 0.611 |
| Tree on compressed word counts | $14.06 | 0.607 |
| Always guess the middle training price | $16.54 | 0.758 |

RMSLE scores the miss on the log of the price, so a cheap item and a costly item are compared more fairly. I used it to pick Gemma's saved checkpoint. The number I talk about below is the average error in dollars.

On this data, the smaller Gemma with one price was the best model I trained. It beat Ridge by about one dollar. The bigger Gemma did not beat the smaller one. Four digit boxes, thousands through ones, were worse than Ridge, so I dropped that head.

A warm single guess takes about 5 milliseconds for Ridge on CPU, and about 110 milliseconds for Gemma on the T4. The one dollar costs a GPU.

The training settings for the Mercari Gemma runs were LoRA rank 8, 1,250 steps, and a 256-token cutoff. The median test listing is 84 tokens. About 1.9% of test listings are cut.

## Why the gap stayed about one dollar

Most of the price follows the brand and the product name. Those words are already in the training file. Ridge has a dollar weight for them. Gemma can read them too. So the two models stay close. An average miss of about $12 on an $18 item also means neither model learned the prices perfectly. They learned the usual price of the words. The photo and the bargaining are not in the text.

I then labelled the same 1,000 test rows from the text, before looking at the errors.

| Group | Rows | Ridge | Gemma |
| --- | ---: | ---: | ---: |
| All test rows | 1,000 | $12.87 | $11.92 |
| Ordinary listings | 733 | $12.47 | $11.50 |
| Any sentence rule | 267 | $13.96 | $13.07 |
| The sentence talks about damage | 101 | $13.49 | $11.58 |
| The listing is a bundle | 132 | $13.55 | $13.98 |

The mixed hard group is still about a one-dollar gap, because the bundles go the other way. On damage sentences, Gemma's lead grows to about $1.90. Rows can match more than one rule.

## A shop I wrote myself

I wanted a test where the brand cannot decide the price. Only the sentence can, and the test must use words that were absent from training.

I wrote 1,000 training listings, 200 checking listings, and 500 test listings. They are not Mercari sales. Condition is 3 on every row. Shipping is 0 on every row. A fine screen costs about $200. A broken screen costs about $40. The true gap on the test is about $159.

Training used sentences like "no cracks on the screen" and "the screen is cracked." Half the test repeats those words. The other half says the same fact in new words: "the display is unmarked" and "the display is smashed." I trained Ridge on CPU and Gemma on the same 1,000 rows, 500 steps, and I scored the checkpoint with the lowest checking error, step 400.

| Test | Rows | Ridge | Gemma |
| --- | ---: | ---: | ---: |
| Words already seen | 250 | $10.21 | $8.63 |
| New words | 250 | $80.00 | $9.21 |

On words it had seen, Ridge says about $196 and $41. On the new words it says about $81 for the fine phone and about $81 for the broken phone. $81 is the middle guess it gives when it cannot tell the two listings apart. Gemma says about $187 and $42, and it keeps about $144 of the $159 gap.

The reason is the training table. "Cracked" was in the file, so Ridge learned the cheap price. "Smashed" was not in the file, so its weight is zero. The brand and the product are the same on both phones, so they cannot separate the price. Gemma had already learned English before this shop, so "smashed" still means broken.

## What I would expect on a real shop

The Mercari case is the one that shows up on most listings. 733 of the 1,000 real test rows were ordinary. Listings with five or more words missing from the training vocabulary were 35 rows, and there both models got worse together: Ridge $18.83, Gemma $18.02.

The $80 collapse needs a stricter setup. The brand, the product, the condition, and the shipping must all be the same, and the only clue must be a sentence written in words the training file never had. That happens, and it is a small slice. A production score on a shop like Mercari will keep showing the one-dollar gap.

## The limit of this result

The Mercari scores stay $11.92 and $12.87. The made-up shop does not replace them. Neither model saw a photo. I trained one seed. A real broken phone is not always about $160 cheaper than a fine one. Gemma may have met the words "smashed" and "unmarked" while it was learning English. Ridge started from these listings only.

## Where the rest of the notes are

This page is the short account. The lab notes behind it:

- [How the question started](conversation-summary.md)
- [What I set out to measure](experiment-design.md)
- [The $30 Modal cap. Training actually ran on Kaggle](budget.md)
- [Commands to fetch the data and run the models](runbook.md)
- [How the Kaggle notebooks were started](kaggle-quickstart.md)
- [The score notebook, including the runs I should not repeat](findings.md)
- [The longer Mercari write-up](blog-draft.md)
- [The outline of that write-up](blog-outline.md)
- [The longer write-up of the made-up shop](blog-synthetic-sentence.md)
- [The short measured record of that shop](learnings-synthetic-sentence.md)
- [Progress on 15 October](../results/progress-2026-10-15.md)
- [Progress on 16 October](../results/progress-2026-10-16.md)

The score files are `results/kaggle-pilot-output/e2b-regression-report.json`, `results/kaggle-e4b-output/e4b-regression-report.json`, `results/kaggle-digit-output/e2b-digits-report.json`, `results/mercari-sentence-slice.json`, `results/synthetic-sentence-ridge.json`, and `results/synthetic-sentence-gemma.json`. The made-up listings are in `data/synthetic-sentence/processed/`.
