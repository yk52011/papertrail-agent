# Contributing

Thanks for helping improve PaperTrail. Issues and pull requests are welcome.

## Run locally

```bash
git clone https://github.com/yk52011/papertrail-agent.git
cd papertrail-agent
uv sync --frozen --python 3.11 --extra dev
uv run ruff check .
uv run pytest -q
```

Tests run without external API credentials. The HTTP integration test uses a local protocol stub.
Use only the bundled, inspectable workloads for local execution tests. External code should run
in an environment you control, preferably with the Docker backend.

## What to include in an issue

- OS, Python version, installation method, and PaperTrail version.
- The command used and the relevant error, with credentials and private paths removed.
- A minimal public reproducer or a synthetic fixture.
- Whether you used `rules`, `retry`, `none`, or `llm`.

Never attach `.env`, API keys, private repositories, raw checkpoint databases, or private training data.

## Changes and evaluation

Keep patches focused. Include a regression case when changing recovery behavior or verification.
For evaluation claims, publish per-case results, corrected/corrupted IDs, budget, model and data
versions, and limitations. Label synthetic fixtures separately from real-world evaluation.

Useful contributions include new failure fixtures, independently verifiable output contracts,
container tests, real-repository adapters, and model comparisons with fixed budgets.
