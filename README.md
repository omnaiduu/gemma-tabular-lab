# Listing price: Ridge and Gemma

I compared a word-count model with Gemma on used-listing prices. On Mercari, the regression head averages 11.92 dollars of error and Ridge averages 12.87. The four-digit head averages 14.81.

The write-up is [docs/blog.md](docs/blog.md) and [the same post](https://omnaidu.com/blog/listing-price-gemma-and-ridge).

`src/gemma_tabular/` has the split, the baselines, and both heads. `results/` has the scores. `data/synthetic-sentence/` is the written shop. The Mercari file stays outside Git.

```bash
python -m venv .venv
.venv/bin/pip install -e .
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```
