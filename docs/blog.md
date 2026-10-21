# I compared Gemma and Ridge on a listing price

I wanted to guess the selling price of a used listing from the text alone.

A listing has a title, a description, a category, a brand, a condition number, and a shipping flag. Shipping 0 means the buyer pays the postage. Shipping 1 means the seller pays. The price is hidden. The model has to guess it.

I learn from old listings. I then score listings that were held out of training. Average error means the real price minus the guess, averaged. A real price of 200 dollars and a guess of 190 dollars is an error of 10 dollars.

I also report RMSLE. That score looks at the log of the price, so a miss on a cheap item and a miss on a costly item can be compared. I used RMSLE to choose which saved Gemma checkpoint to keep. The numbers I discuss below are the average error in dollars.

## What I tried

I trained seven guesses on the same split. A baseline is a simple guess the others have to beat.

The median guess always says the middle training price. It does not read the listing.

A tree asks yes-or-no questions about the columns and then gives a price. I trained two. One sees the brand, the category, the condition, and shipping. The other also sees the words, but those words are first squeezed into 16 numbers, so it does not get the full word list.

Ridge counts words. Each word, and each pair of neighboring words, gets a dollar weight. The guess is the sum of the weights. I fit the weights on the log of the price, then convert back to dollars. A word that never appeared in training has weight zero. I tried two penalties, 1 and 10. The penalty pulls the weights toward zero. Penalty 1 won on the checking rows, so that is the Ridge I report.

Gemma is a language model from Google. It already knows English. I used two sizes of Gemma 4. E2B is the smaller one. E4B is the larger one. Those names are Google's size labels. The original weights stayed frozen. I trained a small add-on, called LoRA, on the attention layers. Rank 8 means each added matrix is narrow. I also trained one new last layer, called the head. The head is the piece that turns the sentence into a price. Both heads read the same vector: the hidden state of the last real token. A token is one piece of text. The last real token is the last piece that is part of the listing, not the blank padding used to make a batch the same length.

## The two heads

The head is the only difference between the two Gemma E2B runs. Same model, same 10,000 training rows, same 1,250 steps, same 256-token limit, seed 42.

The regression head outputs one number. That number is a scaled log of the price. I convert it back to dollars. This is a direct price guess.

The four-digit head does not output one number. A price such as 180 dollars is four digits: 0 thousands, 1 hundred, 8 tens, 0 ones. The head makes four separate choices, each a digit from 0 to 9. Training pushes each digit toward the right choice. At guess time I put the four digits back together: `1000*thousands + 100*hundreds + 10*tens + ones`. That covers 0 to 9,999 dollars. I tried it because a price is made of digits, and I wanted to know if guessing the digits beats guessing one log-price.

On the larger E4B model I trained only the regression head.

The runs were on one Kaggle T4 GPU, in 16-bit. This laptop has no GPU.

## The real listings

The listings are from Mercari, a second-hand app in the US. I used the public 100,000-row research set. The split is 10,000 train rows, 1,000 checking rows, and 1,000 test rows. Gemma's saved checkpoint uses 256 of the checking rows. Ridge's penalty uses the checking rows. The 1,000 test rows were not used to choose anything. A typical test item costs about 18 dollars. The Mercari file is not in the repository. The commands are in the [lab notes](notes.md).

| Model | What comes out | Average error, dollars | RMSLE |
| --- | --- | ---: | ---: |
| Gemma 4 E2B, regression head | one price | 11.92 | 0.506 |
| Ridge, word weights | one price | 12.87 | 0.566 |
| Gemma 4 E4B, regression head | one price | 12.65 | 0.536 |
| Gemma 4 E2B, four-digit head | four digits, then one price | 14.81 | 0.647 |
| Tree on brand, category, condition, shipping | one price | 14.04 | 0.611 |
| Tree on 16 compressed word numbers | one price | 14.06 | 0.607 |
| Always the training median | one price | 16.54 | 0.758 |

The regression head on the smaller Gemma is the best model I trained. It beats Ridge by about one dollar. The bigger Gemma, with the same regression head, does not beat the smaller one.

The four-digit head is worse than Ridge, so I dropped it. On the test rows its digit accuracy is 100 percent for thousands, 97.1 percent for hundreds, 40.1 percent for tens, and 16.3 percent for ones. Always guessing the most common digit in each place is already 100, 97.0, 36.8, and 16.1 percent. The head is barely ahead of that. The thousands digit is almost always 0 on these cheap items, so that place is easy. The tens and the ones are where the price actually moves, and those places stay weak. Both heads were trained for 1,250 steps. I kept the regression checkpoint from step 1,250. I kept the digit checkpoint from step 500, because later checking scores got worse.

A warm single guess takes about 5 milliseconds for Ridge on CPU, and about 110 milliseconds for the E2B regression head on the T4.

Listings are cut at 256 tokens. The median test listing is 84 tokens. About 1.9 percent of test listings are cut.

## Why the gap stayed about one dollar

Most of the price follows the brand and the product name. Those words are already in the training file. Ridge has a weight for them. Gemma can read them too. An average miss of about 12 dollars on an 18-dollar item means both models learned the usual price of the words. The photo and the bargaining are not in the text.

I then labelled the same 1,000 test rows from the text, before looking at the errors.

| Group | Rows | Ridge, dollars | Gemma regression, dollars |
| --- | ---: | ---: | ---: |
| All test rows | 1,000 | 12.87 | 11.92 |
| Ordinary listings | 733 | 12.47 | 11.50 |
| Any sentence rule | 267 | 13.96 | 13.07 |
| The sentence talks about damage | 101 | 13.49 | 11.58 |
| The listing is a bundle | 132 | 13.55 | 13.98 |

The mixed hard group is still about a one-dollar gap, because the bundles go the other way. On damage sentences, Gemma's lead grows to about 1.90 dollars. A row can match more than one rule.

## A shop I wrote, so the sentence is the only clue

I wrote 1,000 training listings, 200 checking listings, and 500 test listings. They are not Mercari sales. Condition is 3 on every row. Shipping is 0 on every row, so those fields cannot explain the price. A fine screen costs about 200 dollars. A broken screen costs about 40 dollars. The true gap on the test is about 159 dollars.

Training used sentences such as "no cracks on the screen" and "the screen is cracked." Half the test repeats those words. The other half says the same fact in new words: "the display is unmarked" and "the display is smashed." Those new words were absent from training. I trained Ridge and the E2B regression head on the same 1,000 rows. The Gemma run was 500 steps. I kept step 400, the best checking score.

| Test | Rows | Ridge, dollars | Gemma regression, dollars |
| --- | ---: | ---: | ---: |
| Words already seen | 250 | 10.21 | 8.63 |
| New words | 250 | 80.00 | 9.21 |

On words it had seen, Ridge says about 196 dollars and 41 dollars. On the new words it says about 81 dollars for the fine phone and about 81 dollars for the broken phone. 81 dollars is the middle guess it gives when the new words have no weight and the brand is the same on both phones. Gemma says about 187 dollars and 42 dollars. It keeps about 144 dollars of the 159-dollar gap. Gemma had already learned English, so "smashed" still means broken.

## What I would expect on a real shop

The Mercari case is the common one. 733 of the 1,000 real test rows were ordinary. Rows with five or more words missing from the training vocabulary were 35, and there both models got worse together: Ridge 18.83 dollars, Gemma 18.02 dollars.

The 80-dollar collapse needs a stricter setup. Brand, product, condition, and shipping must all be the same, and the only clue must be a sentence written in words the training file never had. That slice is real, and it is small.

## The limit of this result

The Mercari scores stay 11.92 and 12.87 dollars. The made-up shop does not replace them. Neither model saw a photo. I trained one seed. A real broken phone is not always about 160 dollars cheaper than a fine one. Gemma may have met "smashed" and "unmarked" while learning English. Ridge started from these listings only.

The commands, the checkpoints, and the score files are in the [lab notes](notes.md).
