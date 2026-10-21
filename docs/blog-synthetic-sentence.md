# A made-up shop where the sentence is the price

*21 October 2026. These listings were generated. They are not Mercari sales.
Both models were trained on them and scored on a held-out test.*

## The gap we were trying to see

On the real Mercari test, Gemma's direct price head beat Ridge by about one
dollar, $11.92 against $12.87. Splitting that test by hand-written rules
widened the lead to about $1.90 on condition sentences and reversed it on
bundles. The average stayed close because most listings are priced by the
brand and the product words, which both models can see.

The question left over was what happens when those words are not enough.
Same brand, same product, same condition number. The only thing that changes
the price is the sentence, and the test says it in words the model did not
train on.

## The shop

Every listing is a brand and a product: Apple, Samsung, Google, or Sony,
and a phone, laptop, camera, headphones, or watch. The condition number is
3 on every row. Shipping is 0 on every row. A fine item and a broken item
with the same name are identical except for one sentence.

A fine item costs about $200. A broken item costs about $40. The draws are
noisy, so the labels are not a single repeated number. The true gap between
the two kinds of test item is about $159.

The training sentences are things like "no cracks on the screen" and "the
screen is broken." The new test sentences are things like "the display is
unmarked" and "the display is smashed." The content words in the new
sentences do not appear in the training sentences. Every row ends with the
same filler, "No box included," so the word "no" cannot by itself tell the
classes apart.

| Split | Rows | What the model is allowed to do with it |
| --- | ---: | --- |
| Train | 1,000 | Learn |
| Validation | 200 | Pick Ridge's penalty and Gemma's checkpoint. Same wording as the train set |
| Test, seen wording | 250 | Score. Same sentences as training, new prices |
| Test, new wording | 250 | Score. Sentences that were not in training or validation |

The files are `data/synthetic-sentence/processed/`. The generator is
`scripts/build_synthetic_sentence.py`.

## Ridge

Ridge is the same recipe as the Mercari run. Word and neighboring-word
counts, at most 20,000 per text field, plus the brand, the product, the
condition, and shipping. The fit is on `log(1 + price)`. Penalty 1 beat
penalty 10 on the 200 validation rows (RMSLE 0.117 against 0.231), so
penalty 1 was kept. The fit took a few seconds on CPU.

| Test slice | Rows | Average error | RMSLE | True gap | Ridge's gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seen wording | 250 | $10.21 | 0.152 | $159 | $155 |
| New wording | 250 | $80.00 | 0.824 | $159 | −$0.65 |
| Both | 500 | $45.11 | 0.593 | $159 | $77 |

On wording it had seen, Ridge guessed the fine items at about $196 and the
broken items at about $41. On the new wording it guessed both at about $81.
The gap closed. About 81% of those new-wording guesses were more than
double, or less than half, the true price. The report is
`results/synthetic-sentence-ridge.json`.

## Gemma

Gemma 4 E2B, revision `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`, was trained
on the same 1,000 rows. The backbone stayed frozen. LoRA rank 8, alpha 16,
dropout 0.05, on the query, key, value, and output projections. The new
price layer maps the last real token to one number. Trainable parameters:
2,680,321, of which the price layer is 1,537.

The run was one Tesla T4, FP16, batch 1, accumulation 8, 500 optimizer
steps, max length 128, seed 42. Four hundred validation rows were not
needed; all 200 were used. The checkpoint is the lowest validation RMSLE,
step 400 (RMSLE 0.126, average error $7.29). Step 500 was slightly worse on
the validation rows and was not the scored checkpoint. No optimizer step was
skipped. Load took 63 seconds. Training and validation took 2,039 seconds.
Peak memory was 9.51 GiB. A warm single-row guess took about 106 ms.

The laptop has no GPU. The kernel is
https://www.kaggle.com/code/ommnaidu/gemma-e2b-synthetic-price.
The saved guesses are
`results/kaggle-synthetic-output/synthetic-sentence-predictions.jsonl`.
They were joined to the test file in file order, because the generated row
ids were reused across the fine and broken halves. The batcher walks
positions, not ids, and 4,000 examples were processed over 1,000 rows, so
the rows were seen. The id field is a bad counter. It is not the join key.

| Test slice | Rows | Average error | RMSLE | True gap | Gemma's gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seen wording | 250 | $8.63 | 0.131 | $159 | $148 |
| New wording | 250 | $9.21 | 0.138 | $159 | $144 |
| Both | 500 | $8.92 | 0.135 | $159 | $146 |

On the new wording, Gemma guessed the fine items at about $187 and the
broken items at about $42. None of the 500 test guesses was off by more
than a factor of two. The slice table is
`results/synthetic-sentence-gemma.json`.

## What this shows

When the sentence has been seen, Ridge and Gemma are close. Ridge's average
error is $10.21 and Gemma's is $8.63. The word counter is enough.

When the same fact is written in new words, Ridge's two guesses meet in the
middle, about $81 for both, and the average error goes to $80. Gemma's
average error stays $9.21, and it keeps $144 of the $159 gap. That is the
gap the Mercari average was hiding. It appears here because the shop was
built so that nothing except the sentence carries the price, and because
the test words were held out.

## Conclusion

This is the same result, written in plain words. The tables above are the
measured numbers. This section is the story around them.

### What we were trying to understand

We first trained both models on real Mercari listings. The full write-up of
that run is `docs/blog-draft.md`. Gemma's average miss on the 1,000 test
listings was $11.92. Ridge's was $12.87. So Gemma was closer by about one
dollar.

We then split those same 1,000 listings into ordinary listings and listings
where the sentence talks about damage, a bundle, a long description, or
words the training file had not used. The report is
`results/mercari-sentence-slice.json`. The gap stayed about one dollar.
On the 101 damage sentences it grew to about $1.90. On bundles, Ridge was
closer. The mixed pile still looked like a one-dollar gap.

That happened because most Mercari prices follow the brand and the product
name. Those words were already in the training file. Both models can use a
word they have already seen. Ridge gives the word a dollar weight. Gemma
reads it too. So on that shop they stay close.

The question left over was this. Take the same phone. Keep the brand, the
condition number, and the shipping the same. Let only the sentence change
the price. Then write that sentence in words the training file never used.
Does Ridge still get the price? Does Gemma?

### What we did

We built a made-up shop. It is not Mercari. The generator is
`scripts/build_synthetic_sentence.py`. The files are
`data/synthetic-sentence/processed/`.

- 1,000 listings to learn from.
- 200 listings to choose Ridge's penalty and Gemma's checkpoint. These use
  the same sentences as the training file.
- 500 test listings. We did not use them to choose anything.

A fine screen costs about $200. A broken screen costs about $40. On the
test, the true gap between those two is about $159. Condition is 3 on every
row. Shipping is 0 on every row. So those fields cannot explain the price.

The training sentences look like "no cracks on the screen" and "the screen
is cracked." Half the test, 250 rows, repeats those sentences with new
prices. The other half, 250 rows, says the same fact in new words: "the
display is unmarked" and "the display is smashed." The content words in
those new sentences do not appear in the training sentences. Every listing
also ends with "No box included," so the word "no" is on both the fine and
the broken rows.

Ridge was trained on the 1,000 rows on the CPU. It is the same word-count
recipe as the Mercari run. It gives each word, and each pair of neighboring
words, a dollar weight, then fits `log(1 + price)`. Penalty 1 beat penalty
10 on the 200 checking rows. The fit took a few seconds. The script is
`scripts/score_synthetic_ridge.py`.

Gemma 4 E2B was trained on the same 1,000 rows. The laptop has no GPU, so
this ran on one Kaggle Tesla T4. The notebook is
https://www.kaggle.com/code/ommnaidu/gemma-e2b-synthetic-price.
The old English model stayed frozen. A small LoRA adapter and one new price
layer were trained, for 500 steps. The checkpoint we scored is step 400,
because that step had the lowest error on the 200 checking rows. Training
and checking took about 34 minutes. A warm single guess took about 106 ms.

### What we found

| Test | Rows | Ridge's miss | Gemma's miss | What the guesses did |
| --- | ---: | ---: | ---: | --- |
| Words already seen | 250 | $10.21 | $8.63 | Both still separate $200 from $40 |
| New words | 250 | $80.00 | $9.21 | Ridge says about $81 for both. Gemma still says about $187 and $42 |
| Both halves | 500 | $45.11 | $8.92 | The new-word half pulls Ridge's average up |

On words it had seen, Ridge said about $196 for a fine item and about $41
for a broken one. On the new words it said about $81 for both. Its gap fell
from about $159 to about $0. About 81% of those new-word guesses were more
than double the true price, or less than half of it.

Gemma's miss stayed near $9 on both halves. On the new words it said about
$187 and $42. It kept about $144 of the $159 gap. None of its 500 test
guesses was off by more than a factor of two.

### What this proves

Ridge only knows the words it counted in these 1,000 rows. "Cracked" was in
that file, so it learned that "cracked" goes with the cheap price.
"Smashed" was not in that file, so "smashed" has no dollar weight. With no
word to hold onto, Ridge lands in the middle and says about $81 for both
phones.

Gemma had already learned English before this shop. It can read "smashed"
as the same kind of fact as "cracked," and "unmarked" as the same kind of
fact as "no cracks." That is why the new words do not knock it off the
price.

So the one-dollar Mercari lead was the case where the useful words were
already in the training file. This shop is the case where they are not.
When the sentence is the only clue, and the test writes it in new words,
the word counter loses the price and the reader keeps it.

### What this does not prove

- These are made-up listings. A real broken phone is not always about $160
  cheaper than a fine one. The photo and the seller still matter, and
  neither model saw a photo.
- Gemma may have met "smashed" and "unmarked" while it was learning English,
  before this shop. Ridge started from these 1,000 rows only. This compares
  a reader that already knew English with a word counter fit on this shop.
- We used one seed, one checkpoint rule, and one sentence template. A
  different way of saying "fine" and "broken" could move the numbers.
- The Mercari result stays as it was. On that data Gemma's lead is still
  about one dollar. This shop does not replace that score.

### Where to read it

- This blog: `docs/blog-synthetic-sentence.md`
- The short measured record: `docs/learnings-synthetic-sentence.md`
- The Mercari article: `docs/blog-draft.md`
- The lab notes: `docs/findings.md`
- Ridge's scores: `results/synthetic-sentence-ridge.json`
- Gemma's scores: `results/synthetic-sentence-gemma.json`
- Gemma's training report: `results/kaggle-synthetic-output/synthetic-sentence-report.json`
- The training notebook: https://www.kaggle.com/code/ommnaidu/gemma-e2b-synthetic-price
