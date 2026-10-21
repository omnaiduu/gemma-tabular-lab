# What we learned by asking Gemma to price a marketplace listing

*Engineering note, 16 October 2026. One seed, one public dataset, three Gemma
runs and four classical models. The numbers below are copied from the saved
reports, not from memory.*

## The question

The practical question was small. A listing has a title, a description, a
category, a brand, a condition, and a shipping flag. Can a model guess the
sale price, and is a language model worth using for that guess?

Under that question there were two comparisons.

1. Does a pretrained Gemma, adapted with a small LoRA, beat a cheap text
   model that only counts words?
2. If we do use Gemma, should the price come out as one number, or as four
   digit boxes filled in parallel?

The digit boxes were the original idea: turn a number into a few small
choices, the way `180000` can be written `+1.800e+05`. The single-number
output was added as a control, so a win could be attributed to Gemma itself
or to the digit encoding. The digit idea is the user's. The regression
control is the assistant's. LoRA is only the adaptation method. It is not
an output format.

## The data

The rows are the published Mercari 100,000-listing research subset, OpenML
dataset 46660, version 1. It comes from the old Mercari price-suggestion
competition. Each row has `price` and `log_price`. We checked that
`log_price` matches `log(1 + price)` within `8.89e-16`, then trained on the
original price. Source files stay out of Git. The original Kaggle terms
still apply. Provenance is in `results/data-provenance.json`.

We did not train on all 100,000 rows. Duplicate listings, judged by a hash
of the normalized input fields, stay in the same split. From the remaining
pool we drew:

| Split | Rows | Role |
| --- | ---: | --- |
| Train | 10,000 | The only rows the models learn from |
| Validation | 1,000 | Picks the Ridge penalty, the tree depth, and the Gemma checkpoint |
| Test | 1,000 | Scored once, after those choices |

The seed is 42. Test prices run from $3 to $400. The middle price is about
$18. Training prices run from $0 to $1,100. Two training prices are $10.50
and round half-up to $11. Test prices are already whole dollars, so that
rounding does not change the test labels. Checksums of the three JSONL
files are in `results/processed-checksums.json`. Every GPU run rebuilds the
splits and refuses to train if the checksums differ.

The model is not allowed to see `price`, `log_price`, or the listing id.
The text fields are `name` and `item_description`. Category and brand are
categories. Condition and shipping are numbers.

## What "training" means here

A listing goes in. A price comes out. Training means looking at the 10,000
old listings whose prices we know, and changing the model until those
guesses get closer. The test listings are not used to decide anything.
They only answer, at the end, how wrong the frozen model is.

The error we quote is the mean absolute error: add up the dollar misses and
divide by 1,000. We also report RMSLE, which scores the miss on
`log(1 + price)`, so a bad guess on a cheap item is not hidden by a bad
guess on a $400 item. "Factor-two" is the share of guesses that are more
than double, or less than half, the real price.

None of these models find a law of prices. Two listings with the same words
can still sell for different amounts, because the photo, the bargaining,
the day, and the seller's reputation are not in the columns. The formula is
a summary of the usual price. That is why the best average miss is still
about $12 on items that often cost about $18.

## The classical models, and why Ridge won them

Before any Gemma weights were loaded, four cheap models were fit on a CPU.
The code is `src/gemma_tabular/baselines.py`. Both Ridge trials together
took 1.3 seconds.

| Model | What it saw | Test RMSLE | Average error | Factor-two | Warm guess |
| --- | --- | ---: | ---: | ---: | ---: |
| Training median | Nothing about the listing | 0.7585 | $16.54 | 35.1% | — |
| XGBoost on the table | Condition, shipping, category, brand | 0.6114 | $14.04 | 26.5% | 4.52 ms |
| TF-IDF + Ridge | Those columns, plus up to 20,000 word and word-pair marks for the title and 20,000 for the description | **0.5662** | **$12.87** | **22.2%** | 5.41 ms |
| TF-IDF squeezed to 16 numbers, then XGBoost | The same text, after throwing most of the word identity away | 0.6067 | $14.06 | 26.9% | 6.34 ms |

Ridge is not a tree and not a neural net. Each clue has a weight. The guess
is a sum. We fit that sum on `log(1 + price)` and convert back with
`exp(guess) - 1`, then clip at zero. The penalty that keeps weights small
was tried at 1 and at 10. On the 1,000 checking rows, penalty 1 had RMSLE
0.552 and penalty 10 had 0.556, so penalty 1 was kept. The solver is
`lsqr`. Word marks are TF-IDF unigrams and bigrams, `min_df=1`, at most
20,000 features per text column.

XGBoost used 300 trees, learning rate 0.05, subsample 0.8, column sample
0.8, histogram splits, and a depth of 4 or 8 chosen on the checking rows.
The text XGBoost did not see the 20,000 words. Those marks were compressed
with SVD to 16 numbers first. That is a large part of why it lost. This
result says this Ridge setup beat these two tree setups. It does not say a
tree cannot use raw word counts.

Ridge fails when the meaning is in the order of the words. "No cracks" and
"has cracks" share the word "cracks". A two-word pair helps a little. A
longer sentence does not. It also fails on a word that never appeared in
the 10,000 training listings, and when the price is a bundle rather than a
pile of independent clues. The weight of "pro" is the same on a phone and
on a shampoo bottle.

Bootstrap intervals for this one trained Ridge, on this one split, are
RMSLE 0.5365–0.5966 and average error $11.45–$14.39. They do not measure a
second seed or next month's market. The report is
`results/20261015T044248-baselines-21e3bb99.json`.

## Why Gemma, and why the small one first

Gemma 4 is Google's open language-model family. The "E" means effective
parameters. E2B is about 2.3 billion parameters doing the work, and about
5.1 billion once the embedding tables are counted. E4B is about 4.5 billion
effective and about 8 billion with embeddings. E4B was the original
checkpoint of interest. The digit design does not need that size, and a
free T4 has 15 GiB, so E2B was the first real run. E4B was the size check
after E2B had a number.

We did not train the whole network. The pretrained backbone stays frozen.
LoRA adds small low-rank matrices on the query, key, value, and output
projections. The new price layer is also trained. That is the whole
trainable set.

The listing is serialized to text, tokenized, and cut at 256 tokens. The
real E2B tokenizer puts the median test listing at 84 tokens. 1.9% of test
listings are longer than 256 and get cut. The classical models saw the full
description. That difference has to stay in the comparison. The audit is
`results/input-length-audit-e2b.json`.

One forward pass produces a hidden vector at every token. We keep the last
token that is not padding. That vector is the whole summary of the listing.

- The regression head is one linear layer from that vector to one number.
  The number is a standardized `log(1 + price)`. On E2B the hidden size is
  1,536, so the head has 1,537 parameters. On E4B the hidden size is 2,560,
  so the head has 2,561.
- The digit head is four classifiers, each with 10 scores, 40 scores in
  all. The price is `1000*thousands + 100*hundreds + 10*tens + ones`. On
  E2B that head has 61,480 parameters. Signs and an exponent were removed.

## Why the digits are four dollars, not scientific notation

The first sketch used eight slots: two signs, four mantissa digits, and two
exponent digits. On these prices most of those slots never change. Both
signs are positive. The exponent is almost constant. A model could look
accurate on slots and still be bad at dollars. The slot balance for that
old codec is in `results/pilot-analysis.json`.

The prices we actually have fit in four whole-dollar digits, from $0 to
$9,999. Training only reaches $1,100, so three digits would clip two
training rows. Four digits do not. Values are rounded half-up with
`Decimal`, not with binary `round`, because $10.50 must become $11.
Reconstruction error of this codec on the test prices is zero. The hard
part is predicting the digits, not packing them. The balance is in
`results/dollar-digit-codec.json`.

## The training recipe, the same on every Gemma run

All three finished Gemma runs used this recipe. The code is
`src/gemma_tabular/train.py`.

| Setting | Value | Why |
| --- | --- | --- |
| Seed | 42 | Same shuffle as the data split |
| Steps | 1,250 | 10,000 rows divided by an effective batch of 8, so each row is seen once |
| Microbatch | 1 | A T4 cannot hold a larger batch of this model |
| Accumulation | 8 | Eight microbatches become one optimizer step |
| Max length | 256 | Covers the median listing; 1.9% are truncated |
| Precision | FP16, with a gradient scaler starting at 128 | The T4 is compute capability 7.5 and has no native BF16 |
| LoRA rank, alpha, dropout | 8, 16, 0.05 | Small adapter, on `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| LoRA learning rate | `2e-4` | Adapter updates |
| Head learning rate | `5e-4` | The new layer starts from scratch |
| Optimizer | AdamW, weight decay 0.01 | Standard small-adapter fit |
| Gradient clip | 1.0 | Stops one bad batch from jumping the weights |
| Checkpointing | On, non-reentrant | Recomputes activations so the backward pass fits |
| Validation | 256 rows, every 100 steps | Full 1,000 would dominate the wall clock |
| Selection | Lowest validation RMSLE | The test set is not allowed to pick the checkpoint |
| Test | 1,000 rows, batch 4 | One shot, after selection |
| Quantization | None | 4-bit loading was not implemented |

No optimizer step was skipped by the FP16 scaler in any finished run.
Trainable parameter counts, including LoRA: E2B regression 2,680,321; E2B
digits 2,740,264; E4B regression 4,540,929.

The loss for the regression head is mean squared error on the standardized
log price. The loss for the digit head is the average cross-entropy of the
four digits. A numerical-distance term exists in the code and was left at
weight 0, so it did not affect these runs.

## Libraries and machines

The GPU runs used the Kaggle image, not our laptop.

| Library | Version on the GPU runs | Job |
| --- | --- | --- |
| Python | 3.13.15 | Notebook runtime |
| PyTorch | 2.11.0+cu128 | FP16 training, one GPU |
| Transformers | 5.16.1 | Load Gemma 4 and tokenize |
| PEFT | 0.20.0 | Attach LoRA |
| NumPy | 2.1.3 | Metrics and target scaling |
| safetensors | the image default | Read the E4B weights without loading the whole file at once |
| huggingface_hub | the image default | Download the pinned checkpoint |

The CPU baselines used scikit-learn's `Ridge`, `TfidfVectorizer`, and
`OneHotEncoder`, plus XGBoost. Pandas held the frames. The Kaggle CLI 2.2.4
submitted the notebooks. Modal was installed earlier and was not used for
any successful GPU run.

Pinned checkpoints:

- E2B: `google/gemma-4-E2B`, revision `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`, 35 text layers, hidden size 1,536.
- E4B: `google/gemma-4-E4B`, revision `411aa17b749aa952df1359d2dcea73917a544d9a`, 42 text layers, hidden size 2,560.

## Why the job ran on Kaggle, and the failures that changed the code

Modal was the first plan, with a $30 ceiling. It rejected every GPU,
including T4 and L4, because the account has no payment method. No Modal
GPU worker ever loaded Gemma. The failed launches are recorded in
`docs/findings.md` and were not deleted from the budget ledger.

Kaggle was the free GPU. The first notebooks asked for a T4 and got a CPU
image, no NVIDIA device, and DNS failures. The account had to be phone
verified before Internet and a real T4 appeared. After that, the official
CLI from WSL was the path that worked. Browser automation was stopped.

Three implementation facts matter for the next run.

1. Kaggle's image has `torchao` 0.10. PEFT 0.20 refuses to build LoRA
   unless `torchao` is absent or newer than 0.16. Every training notebook
   runs `pip uninstall -y torchao` first. We do not upgrade PyTorch to fix
   that, because the image's CUDA build is the one that works.
2. An uploaded tar is unpacked. The notebook finds
   `processed-checksums.json` and copies that source tree. It must not
   expect a single `payload.tar`.
3. A notebook push ships one script. Extra `.py` files beside it were not
   placed next to that script. The E4B notebook therefore failed in a few
   seconds with `FileNotFoundError` for `/kaggle/src/train.py`. The retry
   packed `train.py` and `cli.py` inside the one script as text, wrote them
   out on the worker, and only then started training.

The machine is one Tesla T4, 15,360 MiB. A second T4 is visible on the
notebook and is not used. The two cards do not combine their memory for one
model. Attention is SDPA. The device map puts the model on `cuda:0`.

## What E2B did

Both heads saw all 10,000 training rows once. Peak memory was 9.51 GiB.
Regression training plus validation took 5,092 seconds. The digit run took
5,385 seconds. A warm single-row guess took about 110 ms for regression and
113 ms for digits. Ridge takes 5.4 ms on a CPU.

| Model | Test RMSLE | Average error | Median miss | 90th-percentile miss | Factor-two |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | 0.5662 | $12.87 | $6.55 | $26.80 | 22.2% |
| E2B regression | 0.5058 | $11.92 | $6.67 | $24.30 | 18.3% |
| E2B four digits | 0.6469 | $14.81 | $7.00 | $30.00 | 24.8% |

The regression checkpoint used for that test is step 1,250, the best
validation RMSLE of the run (0.530 on 256 rows). The digit checkpoint is
step 500, validation RMSLE 0.646 and average error $12.14. The digit head's
own step 1,250 was worse on the checking rows (RMSLE 0.883), so the last
step was not the one scored.

Digit test accuracy by box was 100%, 97.1%, 40.1%, and 16.3% for thousands,
hundreds, tens, and ones. Always guessing the common digit on that same
test split is already right 100%, 97.0%, 36.8%, and 16.1% of the time.
The head is barely ahead of a constant guess. Thousands is almost always 0
because almost nothing costs $1,000. Hundreds is usually 0 because most
items cost under $100. The tens and ones boxes are the ones that move the
dollars, and those are the boxes the model does not get right. Dollar error
is the score that matters, and on that score the digit head loses to Ridge
and to the regression head.

Reports:

- `results/kaggle-pilot-output/e2b-regression-report.json`
- `results/kaggle-digit-output/e2b-digits-report.json`

The first digit attempt was cancelled around step 180 on 15 October, when
the session was stopped for the day. That partial adapter is still under
`results/kaggle-pilot-output/e2b-digits/`. It is not the test result. The
16 October digit run started again from scratch, because the cancelled run
did not save optimizer state. Its notebook is
`ommnaidu/gemma-e2b-digit-head`.

## Why the digit head was dropped

After that test, the four-box output was dropped. Later runs use the
regression head only. The digit encoding was a reasonable experiment. On
this price range it did not earn its complexity. High slot accuracy was a
trap, exactly the trap the scientific-notation design had already shown in
a more extreme form.

## What E4B required, and what it scored

E4B's downloaded weight file is 15,992,595,884 bytes, about 14.89 GiB. The
text backbone inside it is 14.00 GiB. Vision is 0.31 GiB and audio is 0.57
GiB. We do not need vision or audio for a text listing, so the loader
builds only `Gemma4TextModel`.

Even the text part does not fit on the T4 together with the math of a
training step. Of the 14.00 GiB:

| Piece | Size | Where it sat |
| --- | ---: | --- |
| Transformer layers | 7.45 GiB | GPU |
| Token embedding | 1.25 GiB | GPU |
| Per-layer embedding table | 5.25 GiB | CPU |
| Small projection and norms | under 0.1 GiB | GPU |

The per-layer table is one row per vocabulary token, packed across 42
layers (`262144` by `42 * 256`, FP16). A listing only needs the rows for
the tokens in that listing. The lookup runs on CPU, and the few megabytes
of selected rows are copied to the GPU. The rest of the forward pass is the
same arithmetic. This is recorded as `per_layer_embeddings_device: "cpu"`.
Peak GPU memory on the run was 8.87 GiB, which is the layers plus the token
embedding plus activations, not the 5.25 GiB table.

Rotary frequencies are not in the checkpoint. They are rebuilt on the GPU
after the empty weights are allocated. `layer_scalar` buffers are in the
checkpoint and are copied. The non-persistent `embed_scale` buffers are
restored from the config, because emptying the module would otherwise leave
them uninitialized. The embedding module's `_apply` is replaced so a later
`.to("cuda")` does not drag the 5.25 GiB table onto the GPU and run out of
memory.

The first E4B push died before any of that, on the missing-file error
above. The second push is `ommnaidu/gemma-e4b-price-head`. It used the same
1,250 steps, the same seed, and the same regression head. Model load took
85 seconds. Training and validation took 6,887 seconds. The whole job took
7,064 seconds, about two hours. Evaluation ran at 11.4 rows per second. A
warm single-row guess took 165 ms.

The checking rows at the last step were bad: average error $18.79, RMSLE
0.794. The best checking point was step 1,100, average error $10.48, RMSLE
0.551. The test score uses that checkpoint.

| Model | Test RMSLE | Average error | Factor-two | Warm guess | Peak GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | 0.5662 | $12.87 | 22.2% | 5.4 ms, CPU | — |
| E2B regression | **0.5058** | **$11.92** | **18.3%** | 110 ms, T4 | 9.51 GiB |
| E4B regression | 0.5363 | $12.65 | 19.1% | 165 ms, T4 | 8.87 GiB |
| E2B four digits | 0.6469 | $14.81 | 24.8% | 113 ms, T4 | 9.51 GiB |

The report is `results/kaggle-e4b-output/e4b-regression-report.json`.

E4B beat Ridge by about $0.22 and lost to E2B by about $0.73. A larger
network was not a better price model on this split. It was also slower.
The CPU embedding table is a faithful lookup, so this is not an obvious
bug in the offload, but it is a different placement than the E2B run, where
the whole text model sat on the GPU. One seed cannot separate "E4B is worse
at this task" from "this seed and this offload happened to be worse."

## Why we did not move the job to Colab

Free Colab does not promise a larger GPU. When it gives one, it is often
another T4, and the session can stop if the browser is idle. Paid Colab
sometimes offers a larger chip, and Google still does not reserve a
particular one. This E4B run was already on Kaggle. Moving it would have
restarted the download. Kaggle's published allowance is about 30 GPU hours
a week, and the notebook keeps running after the page is closed.

## The model did not learn the prices too well

On 21 October we asked whether Ridge and Gemma look close because they
learned this dataset too well, and whether the rows are too far from a real
marketplace to show where Ridge fails.

The listings are real. They are old public Mercari sales, with a real title,
description, category, brand, condition, and shipping flag, and a real sold
price. They are a weak place to watch Ridge fail. The titles are short. The
median listing is 84 tokens. Once the brand and the product words are known,
the rest of the sentence rarely moves the dollar. Ridge counts those words.
Gemma reads the sentence. On this split that extra reading was worth about
one dollar.

That one dollar is a small gap between two rough guesses. The typical miss
is about $7. The average miss is about $12, because some guesses are off by
about $25 or more. About one guess in five is more than double, or less than
half, the sold price. A model that had learned the prices too well would land
on the sold price. These models land on the usual price of similar words.

The remaining miss is mostly information neither model was given: the photo,
the seller's reputation, the day, and the bargaining. A new dataset shows
Ridge's downside only when the price actually depends on the sentence and
Ridge cannot see that dependence. The rows to look for:

- Negation and damage written as a sentence, such as "no cracks" against
  "has cracks," or "works, but the battery dies in an hour."
- Bundles, where "phone plus watch" is not the phone price plus the watch price.
- New product names and new slang that never appeared in the training rows.
- Long descriptions, where the useful fact is not in the first words. Gemma
  also cuts 1.9% of these test listings at 256 tokens, so a longer-text test
  has to keep the full description for both models.
- Fresh private sales from after the model's pretraining, with the photo and
  the seller attached, so the result is about a future marketplace and not
  about old public text.

On 21 October those groups were scored on the same 1,000 test rows. The
labels came from the text and from the training vocabulary. Neither model's
error was used to pick the rows. Ridge was fit again with the original
recipe, penalty 1, and its overall average error came back at $12.87. The
saved E2B guesses were used as they were. The report is
`results/mercari-sentence-slice.json`.

| Group | Rows | Ridge | E2B | E2B minus Ridge |
| --- | ---: | ---: | ---: | ---: |
| All test rows | 1,000 | $12.87 | $11.92 | −$0.95 |
| Ordinary, none of the rules | 733 | $12.47 | $11.50 | −$0.97 |
| Any sentence rule | 267 | $13.96 | $13.07 | −$0.89 |
| Condition sentence | 101 | $13.49 | $11.58 | −$1.90 |
| Bundle | 132 | $13.55 | $13.98 | +$0.43 |
| Long description, over 80 words | 67 | $15.09 | $13.80 | −$1.30 |
| Five or more unseen words | 35 | $18.83 | $18.02 | −$0.81 |

A negative gap means E2B was closer. Putting every hard rule into one pile
does not widen the gap. It stays about one dollar, because the bundles go
the other way and cancel the condition sentences. On the 101 listings whose
wording is a condition sentence, E2B's lead grows to about $1.90, and its
RMSLE falls to 0.451 against Ridge's 0.575. On bundles, Ridge is closer by
$0.43. New words make both models worse, and the lead does not grow. A row
can match more than one rule, so the four rule counts add up to more than 267.

## Conclusions

1. On this dataset and this seed, a direct price head on Gemma 4 E2B is the
   best model we trained. Average error $11.92, against $12.87 for Ridge
   and $14.81 for the digit head.
2. The gain over Ridge is about one dollar. The cost is a GPU and about
   twenty times the latency, 110 ms against 5 ms, on different hardware.
3. Four digit boxes did not help. They mostly rediscovered that the
   thousands and hundreds digits are usually zero. The useful digits, tens
   and ones, stayed near a constant guess.
4. E4B did not improve on E2B. Bigger was not better for this short-text
   price task, at least once, with the embedding table on CPU.
5. The world is not matching a formula. The remaining $12 is the part of
   the sale that is not in the title, the brand, or the condition.
6. The models did not learn this dataset too well. They learned the usual
   price of the words. Splitting the same test set shows the gap growing
   only on condition sentences, to about $1.90. Bundles favor Ridge. The
   mixed hard pile still looks like a one-dollar gap.
7. Ridge remains the sensible default if the goal is a cheap, fast guess.
   Gemma is worth it only if that extra dollar matters more than the GPU
   and the latency.
8. On a made-up shop where the sentence is the only price clue, the picture
   changes. We trained both models again: 1,000 train rows, 500 test rows.
   On words they had seen, the misses are $10.21 and $8.63. On new words
   for the same fact, Ridge's miss goes to $80 and it guesses about $81
   for both the fine and the broken item. Gemma's miss stays $9.21, and it
   keeps about $144 of a $159 gap. The plain write-up is
   `docs/learnings-synthetic-sentence.md`. The full one is
   `docs/blog-synthetic-sentence.md`. The notebook is
   https://www.kaggle.com/code/ommnaidu/gemma-e2b-synthetic-price.
   This does not change the Mercari scores above.

## What this does not prove

- A second seed might reorder E2B and E4B. We trained one seed.
- The validation set used for Gemma checkpoints is 256 rows, not the full
  1,000. The test set is the full 1,000 and was not used to pick the
  checkpoint.
- Classical models read the whole description. Gemma cuts 1.9% of test
  listings at 256 tokens.
- These are old public listings. They may have appeared in pretraining.
  They are not a company's future private sales.
- The text XGBoost was handicapped by the 16-number compression. A tree on
  the raw word marks was not trained.
- QLoRA, a second seed, and E4B with the embedding table on a larger GPU
  were not run. Free Colab was not used, because it does not guarantee
  that larger GPU.
- The sentence slices are keyword rules, not a hand label of every listing.
  A row can match more than one rule. The condition-sentence lead is 101
  rows from one seed.

## What the next session should not redo

Do not rerun Ridge, the E2B regression head, the E2B digit head, or this
E4B regression unless the method changes. Do not push a new version of
`ommnaidu/gemma-e2b-matched-pilot`, `ommnaidu/gemma-e2b-digit-head`, or
`ommnaidu/gemma-e4b-price-head`. A push starts training again. Do not start
another digit-head run. The decision on 16 October was to keep the
regression head only.

Credentials stay in `~/.kaggle/credentials.json`. They are not in Git.
There is no Git remote.
