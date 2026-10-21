# Listing price: Ridge and Gemma

I compared a word-count model with Gemma on second-hand listing prices. On real Mercari listings, Gemma's average error was $11.92 and Ridge's was $12.87. On a shop I wrote, where a new sentence is the only price clue, Ridge's error goes to $80 and Gemma's stays about $9.

The short account is [docs/blog.md](docs/blog.md). The same post is on [omnaidu.com](https://omnaidu.com/blog/listing-price-gemma-and-ridge).

## Read next

- [How the question started](docs/conversation-summary.md)
- [What I set out to measure](docs/experiment-design.md)
- [Compute budget](docs/budget.md)
- [Commands](docs/runbook.md)
- [Kaggle setup](docs/kaggle-quickstart.md)
- [Score notebook](docs/findings.md)
- [Longer Mercari write-up](docs/blog-draft.md)
- [Outline](docs/blog-outline.md)
- [Longer made-up shop write-up](docs/blog-synthetic-sentence.md)
- [Short record of that shop](docs/learnings-synthetic-sentence.md)

## Layout

- `src/gemma_tabular/` has the split, the Ridge and tree baselines, and the Gemma training loop.
- `scripts/` downloads Mercari, builds the made-up shop, scores both models, and submits the Kaggle runs.
- `tests/` checks the serialization, the split, and the digit codec.
- `results/` holds the measured scores. Model weights are not in Git.
- `data/synthetic-sentence/` is the made-up shop. The Mercari file stays outside Git. Download it with `scripts/fetch_mercari.py`, as written in the runbook.

## Run the local checks

```bash
python -m venv .venv
.venv/bin/pip install -e .
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

There is one branch, `main`. Credentials stay in `~/.kaggle/credentials.json` and are not in this repository.
