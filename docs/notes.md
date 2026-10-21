# Lab notes

This is the single record behind the post. The post is `docs/blog.md` and [omnaidu.com/blog/listing-price-gemma-and-ridge](https://omnaidu.com/blog/listing-price-gemma-and-ridge).

## What was compared

The task is the sale price of a Mercari listing. Inputs are name, item description, category, brand, condition, and shipping. The price is the target. Listing ids are not features.

The four-digit head was the original proposal: four parallel choices, thousands through ones, each a digit from 0 to 9, then reassembled into dollars. The regression head was the control on the same frozen Gemma backbone: one scaled log-price. LoRA rank 8, alpha 16, dropout 0.05, on the query, key, value, and output projections. The scored Mercari runs use seed 42, 1,250 steps, batch 1, accumulation 8, max length 256, FP16, on one T4. `torchao` was uninstalled before LoRA. Gemma checkpoints use 256 checking rows. Ridge penalty 1 beat penalty 10.

E2B regression test: average error 11.92 dollars, RMSLE 0.5058, checkpoint step 1,250. E2B digits: 14.81 dollars, RMSLE 0.6469, checkpoint step 500. Digit accuracy on the test: thousands 100 percent, hundreds 97.1, tens 40.1, ones 16.3. E4B regression: 12.65 dollars, RMSLE 0.5363. It did not beat E2B. Ridge: 12.87 dollars, RMSLE 0.5662. Structured tree: 14.04 dollars. Compressed-text tree: 14.06 dollars. Median: 16.54 dollars.

The made-up shop is 1,000 train, 200 check, 500 test, seed 42. Fine is about 200 dollars, broken is about 40. New-word content words were checked to be absent from the training sentences. Ridge new-word error is 80.00 dollars. E2B regression new-word error is 9.21 dollars, checkpoint step 400 of 500.

## Commands

```bash
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python scripts/fetch_mercari.py
PYTHONPATH=src .venv/bin/python -m gemma_tabular.cli prepare --source data/raw/train.tsv --train-limit 10000 --holdout-limit 1000
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
PYTHONPATH=src .venv/bin/python scripts/build_synthetic_sentence.py
PYTHONPATH=src .venv/bin/python scripts/score_synthetic_ridge.py
```

Mercari rows stay out of Git. OpenML 46660, the public 100,000-row subset. The made-up listings are in `data/synthetic-sentence/processed/`. Do not rebuild that folder. The generated row ids collide across the fine and broken halves. Scores were joined by file order. A rebuild would be a new dataset.

## Score files

- `results/kaggle-pilot-output/e2b-regression-report.json`
- `results/kaggle-digit-output/e2b-digits-report.json`
- `results/kaggle-e4b-output/e4b-regression-report.json`
- `results/20261015T044248-baselines-21e3bb99.json`
- `results/mercari-sentence-slice.json`
- `results/synthetic-sentence-ridge.json`
- `results/synthetic-sentence-gemma.json`
- `results/dollar-digit-codec.json`

Model weights are not in Git. Revisions: E2B `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`, E4B `411aa17b749aa952df1359d2dcea73917a544d9a`.

## Do not start these again

A push of a finished Kaggle notebook starts training again. Leave these kernels as they are:

- `ommnaidu/gemma-e2b-matched-pilot`
- `ommnaidu/gemma-e2b-digit-head`
- `ommnaidu/gemma-e4b-price-head`
- `ommnaidu/gemma-e2b-synthetic-price`

Do not rerun Ridge, either E2B head, or the E4B regression unless the method changes. Credentials stay in `~/.kaggle/credentials.json`.

Modal was authorized up to 30 dollars. The account had no payment method, so the GPU runs went to Kaggle instead. That cap was not a Kaggle bill.
