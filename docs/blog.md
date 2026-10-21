# I compared Gemma and Ridge on a listing price

I wanted to know if a language model prices a used listing better than a model that only adds word weights. On real listings the language model wins by about one dollar. On a written shop, where the sentence is the only clue, it wins by a much wider gap.

A listing has a title, a description, a category, a brand, a condition, and a shipping flag. Condition runs from 1, new, to 5, poor. Shipping 0 means the buyer pays postage. Shipping 1 means the seller pays. The price is hidden until I score the guess.

Average error is the real price minus the guess, then the average of those misses. A few bad misses pull that average up, so the middle miss matters too. RMSLE is the miss on the log of the price. I used RMSLE only to choose which saved checkpoint to keep. The figures below are dollars.

## The models

Every model saw the same Mercari split: 10,000 listings to learn from, 1,000 to choose settings, and 1,000 to test. Mercari is a second-hand app in the US. I used its public set, OpenML 46660. A typical test item costs about 18 dollars. Test prices run from 3 to 400. The test rows were not used to choose settings. The same listing cannot sit in two splits.

The median guess always says the middle training price. It does not read the listing.

A tree asks yes-or-no questions about the columns, then gives a price. One tree sees the brand, the category, the condition, and shipping. The other also sees the words, after they have been squeezed into 16 numbers. That squeeze is a handicap: the tree never reads the words themselves.

Ridge gives each word, and each pair of neighboring words, a weight, then adds the weights. A word that never appeared in training has weight zero. The fit is on the log of the price, then turned back into dollars. I tried penalties 1 and 10. The penalty pulls large weights toward zero. Penalty 1 won on the checking rows, RMSLE 0.552 against 0.556.

Gemma 4 is a language model from Google. It already knows English. E2B is the smaller size and E4B is the larger size. I left the original weights frozen and trained a small add-on, LoRA, at rank 8, plus one new last layer. That layer is the head. It reads the last real token of the listing. A token is one piece of text. The median test listing is 84 tokens. Text is cut at 256 tokens, which drops the tail of 1.9 percent of the test rows.

## The two heads

The two E2B runs share the model, the 10,000 rows, and the training recipe. Only the head changes.

The regression head outputs one number. That number is a scaled log of the price, which I turn back into dollars. This is a direct price guess.

The four-digit head outputs four digits, each from 0 to 9: thousands, hundreds, tens, and ones. 180 dollars is the digits 0, 1, 8, 0. I join them as 1000 times the thousands digit, plus 100 times the hundreds, plus 10 times the tens, plus the ones. That covers 0 to 9,999 dollars. I tried this because a price is made of digits. While it trains, the head is not scored on the dollar error. It only learns to pick the right digit in each place.

On E4B I trained only the regression head. Each run used one T4, 16-bit numbers, and seed 42.

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

The regression head on the smaller Gemma is the best of these. It is about one dollar closer than Ridge. The middle miss is 6.67 dollars for that Gemma run and 6.55 for Ridge, so the average near 12 dollars is the bad guesses pulling the mean up. About 18.3 percent of Gemma's guesses, and 22.2 percent of Ridge's, are off by more than double. A warm guess takes about 5 milliseconds for Ridge on CPU and about 110 milliseconds for E2B on the T4.

The larger Gemma, with the same regression head, does not beat the smaller one.

The four-digit head loses to Ridge. Its accuracy by place is 100 percent, 97.1 percent, 40.1 percent, and 16.3 percent. Always guessing the common digit in each place is already 100, 97.0, 36.8, and 16.1 percent. The thousands digit is almost always 0, because almost no listing costs 1,000 dollars or more. The tens and the ones are what move the price, and those places stay near that common-digit line. I kept the regression checkpoint from step 1,250 and the digit checkpoint from step 500. Those were the best checking scores. A later digit checkpoint was worse, so extra steps did not repair the weak places.

The lead stays near one dollar because the brand and the product name already appear in training, so Ridge has a weight for them too. I regrouped the same 1,000 test guesses. On 733 ordinary listings the lead is still about one dollar. On 101 listings whose sentence describes damage, Gemma averages 11.58 and Ridge 13.49, a lead of 1.90 dollars. On 132 bundles, Ridge is closer: 13.55 against 13.98. The groups overlap, so they do not add up to 1,000. Sorting them does not change the overall average.

## When the words are new

Mercari never takes the brand away, so it cannot show Ridge failing on a new sentence. I wrote a shop of 1,000 training listings, 200 for the checkpoint, and 500 to test. Brand, condition, and shipping are the same on every row. Only the sentence can set the price. A fine screen is about 200 dollars. A broken screen is about 40. The true gap is about 159 dollars. Half the test repeats training sentences such as "cracked". Half uses new words such as "smashed", and those sentences share no content word with training once the small words are removed.

| Test | Ridge | Gemma regression |
| --- | ---: | ---: |
| Words already seen | 10.21 | 8.63 |
| New words | 80.00 | 9.21 |

On wording the model has seen, both stay close. On the new words, Ridge says about 81 dollars for the fine phone and about 81 for the broken phone. "Smashed" has no weight, and the brand is shared, so Ridge cannot tell them apart. Gemma says about 187 and 42. It already knew that "smashed" means broken, and it keeps about 144 dollars of the 159-dollar gap. None of its 500 guesses is off by more than double.

On a shop like Mercari, the one-dollar gap is the usual case. Most of the test, 733 of 1,000 listings, is ordinary. The 80-dollar gap appears only when a new sentence is the only price clue. That is rare on Mercari: brand words repeat, "cracked" and "broken" repeat, and condition is also its own column. Thirty-five test listings had five or more unseen words. Both models got worse there, and the lead did not jump to 80 dollars.

The Mercari numbers that stand are 11.92 and 12.87 dollars. One seed, and no photos, so the leftover miss is the photo, the seller, and the bargaining. The written shop is a separate test. Gemma may have met words like "smashed" before this experiment. Ridge started from these 1,000 shop rows only.

The code and the score files are at https://github.com/omnaiduu/gemma-tabular-lab.
