# I compared Gemma and Ridge on a listing price

I guessed the sale price of a used listing from the text. The listing has a title, a description, a category, a brand, a condition, and a flag for who pays the postage. The price is hidden.

Average error is the real price minus the guess, then the average. RMSLE is that miss on the log of the price. I used RMSLE only to choose the saved checkpoint. The figures below are dollars.

## The models

Every model saw the same Mercari split: 10,000 listings to learn from, 1,000 to choose settings, and 1,000 to test. Mercari is a second-hand app in the US. I used its public 100,000-row set. A typical test item costs about 18 dollars. The test rows were not used to choose settings.

The median guess always says the middle training price. It does not read the listing.

A tree asks yes-or-no questions about the columns, then gives a price. One tree sees the brand, the category, the condition, and shipping. The other also sees the words, after they have been squeezed into 16 numbers.

Ridge gives each word, and each pair of neighboring words, a dollar weight, then adds the weights. A word that never appeared in training has weight zero. The fit is on the log of the price. I tried penalties 1 and 10. The penalty pulls weights toward zero. Penalty 1 won on the checking rows.

Gemma 4 is a language model from Google. It already knows English. E2B is the smaller size and E4B is the larger size. I left the original weights frozen and trained a small add-on, LoRA, at rank 8, plus one new last layer. That layer is the head. It reads the last real token of the listing. A token is one piece of text.

## The two heads

The two E2B runs share the model, the 10,000 rows, and 1,250 steps. Only the head changes.

The regression head outputs one number. That number is a scaled log of the price, which I turn back into dollars.

The four-digit head outputs four digits, each from 0 to 9: thousands, hundreds, tens, and ones. 180 dollars is 0, 1, 8, 0. I join them as `1000*thousands + 100*hundreds + 10*tens + ones`, which covers 0 to 9,999 dollars. I tried this because a price is made of digits.

On E4B I trained only the regression head. The runs used one T4, 16-bit, seed 42. Text is cut at 256 tokens. The median test listing is 84 tokens.

## Scores

| Model | Average error | RMSLE |
| --- | ---: | ---: |
| Gemma 4 E2B, regression head | 11.92 | 0.506 |
| Ridge | 12.87 | 0.566 |
| Gemma 4 E4B, regression head | 12.65 | 0.536 |
| Gemma 4 E2B, four-digit head | 14.81 | 0.647 |
| Tree on brand, category, condition, shipping | 14.04 | 0.611 |
| Tree on 16 word numbers | 14.06 | 0.607 |
| Always the training median | 16.54 | 0.758 |

The regression head on the smaller Gemma is the best of these. It is about one dollar closer than Ridge. The larger Gemma, with the same head, does not improve on it. A warm guess takes about 5 milliseconds for Ridge on CPU and about 110 milliseconds for E2B on the T4.

The four-digit head loses to Ridge. Its accuracy by place is 100 percent, 97.1 percent, 40.1 percent, and 16.3 percent. Always guessing the common digit in each place is already 100, 97.0, 36.8, and 16.1 percent. The thousands digit is almost always 0. The tens and the ones move the price, and those places stay weak. I kept the regression checkpoint from step 1,250 and the digit checkpoint from step 500, the best checking scores.

The lead stays near one dollar because the brand and the product name are already in the training file, so Ridge has a weight for them too. On 101 listings where the sentence describes damage, the lead grows to 1.90 dollars. On bundles, Ridge is closer.

## When the words are new

I then wrote a shop of 1,000 training listings and 500 test listings. Brand, condition, and shipping are the same on every row, so only the sentence can set the price. A fine screen is about 200 dollars. A broken screen is about 40. Half the test repeats training sentences such as "cracked". Half uses new words such as "smashed".

| Test | Ridge | Gemma regression |
| --- | ---: | ---: |
| Words already seen | 10.21 | 8.63 |
| New words | 80.00 | 9.21 |

On the new words Ridge says about 81 dollars for both phones, because "smashed" has no weight and the brand is shared. Gemma says about 187 and 42. It already knew that "smashed" means broken.

On a shop like Mercari, the one-dollar gap is the usual case. The 80-dollar gap appears when a new sentence is the only price clue.

These Mercari numbers stand: 11.92 and 12.87 dollars. One seed, and no photos. The written shop is a separate test.

## Run it

```bash
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python scripts/fetch_mercari.py
PYTHONPATH=src .venv/bin/python -m gemma_tabular.cli prepare --source data/raw/train.tsv --train-limit 10000 --holdout-limit 1000
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

The Mercari file stays out of Git. It is OpenML 46660. The written listings are in `data/synthetic-sentence/processed/`. The score files are the JSON reports under `results/`. Model weights are not in Git.
