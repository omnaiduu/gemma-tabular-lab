# Listing price: Ridge and Gemma

I compared word-count models with Gemma on second-hand listing prices. On real Mercari listings the regression head on Gemma 4 E2B has average error 11.92 dollars. Ridge has 12.87 dollars. The four-digit head is worse, at 14.81 dollars.

The post is [docs/blog.md](docs/blog.md), also on [omnaidu.com](https://omnaidu.com/blog/listing-price-gemma-and-ridge). The commands and the score files are in [docs/notes.md](docs/notes.md).

## Layout

- `src/gemma_tabular/` has the split, the Ridge and tree baselines, and both Gemma heads.
- `scripts/` downloads Mercari, builds the made-up shop, and scores the models.
- `tests/` checks serialization, the split, and the digit codec.
- `results/` holds the measured scores. Model weights are not in Git.
- `data/synthetic-sentence/` is the made-up shop. The Mercari file stays outside Git.

## Local checks

```bash
python -m venv .venv
.venv/bin/pip install -e .
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

Credentials stay in `~/.kaggle/credentials.json`. They are not in this repository.
