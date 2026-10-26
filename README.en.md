# PaperTrail

[中文说明](README.md) · [Contributing](CONTRIBUTING.md)

**Persistent execution and bounded recovery for NLP experiments, grounded in repository evidence.**

PaperTrail runs a declared experiment in a disposable repository copy, verifies fresh prediction
artifacts, retrieves documentation related to failures, and applies only explicitly permitted
configuration updates before rerunning. A LangGraph workflow with SQLite checkpoints supports
pausing and resuming from another process.

This is an engineering project, not a claim of general autonomous paper reproduction.

## Quick start

Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/) are required.

```bash
git clone https://github.com/yk52011/papertrail-agent.git
cd papertrail-agent
uv sync --frozen --python 3.11 --extra dev
uv run streamlit run app.py
```

Open `http://127.0.0.1:8501`. The default workload uses inspectable bundled code and runs without API keys.
The UI is currently in Chinese. The CLI and JSON output can be used independently.

```bash
uv run papertrail reproduce examples/sentiment-lab --backend local --policy rules
```

The workload trains a real TF-IDF + LogisticRegression classifier on 48 handcrafted training sentences
and evaluates 20 handcrafted test sentences. Two deliberately incorrect settings trigger a missing-file
failure followed by a missing-column failure. The recovery workflow fixes the permitted configuration
fields and reruns the experiment. These are synthetic engineering fixtures, not a real-world benchmark.

## Architecture

`inspect → execute → verify → retrieve → diagnose → checkpoint → patch → execute`

- **LangGraph / SQLiteSaver:** conditional recovery loop, durable node state, native interrupts.
- **LangChain / Pydantic:** structured model recovery decisions and server-side validation.
- **RAG:** failure-driven BM25 retrieval with file/line citations; optional embedding + RRF retrieval
  is available separately in the read-only `ask --semantic` path.
- **Bounded actions:** only exact key/value candidates declared in `task.json` may be changed.
- **Independent evaluator:** checks prediction ID coverage and recomputes accuracy/macro F1 against
  reference labels loaded before execution, rejecting stale artifacts or dishonest reported metrics.
- **Streamlit:** task history, execution logs, evidence, config changes and evaluation reports.

## Use a model

Copy `.env.example` to `.env`, set `OPENAI_API_KEY` and `PAPERTRAIL_MODEL`, and optionally set
`OPENAI_BASE_URL` for an OpenAI-compatible provider. The model must support function calling.

```bash
uv run papertrail reproduce examples/sentiment-lab --backend local --policy llm
```

The model receives failure logs, configuration and retrieved evidence. Remove private material before
using an external provider. Never commit credentials. The `rules` policy is deterministic and makes
no LLM calls; its performance must not be presented as model performance.

## Pause and resume

```bash
uv run papertrail reproduce examples/sentiment-lab --backend local --step
uv run papertrail jobs
uv run papertrail resume <job_id>
uv run papertrail resume <job_id> --cancel
```

Consecutive failures can pause multiple times. Checkpoints survive process restarts. Execution is
at-least-once; arbitrary external actions are not exactly-once. Concurrent workers must not resume
the same job.

## Evaluation

```bash
uv run ruff check .
uv run pytest -q
uv run python scripts/evaluate_recovery.py
```

On 10 handcrafted fault-injection cases, single execution and unchanged retries each passed 1/10;
rule-based recovery passed 7/10. None falsely accepted the deliberately dishonest metrics or stale
artifact cases. [Per-case results](eval/results/recovery.json) include corrected/corrupted IDs and
execution counts. These results do not establish model quality, RAG gains or real-repository performance.

The real LangChain HTTP/parsing path is tested against a local protocol stub. External model services,
external embedding services and Docker have not been validated in the current development environment.

## External projects and Docker

Provide a config JSON, `task.json`, `experiment.json`, compatible prediction output and a prepared
runtime. JSON schemas are in `schemas/`. The [adaptation guide](docs/adapt-your-project.md) documents
the contract and current limits. Arbitrary code repair and dependency installation are not implemented.

```bash
docker build -f Dockerfile.nlp -t papertrail-nlp:local .
uv run papertrail reproduce examples/sentiment-lab --policy rules
```

Docker disables networking and caps memory, CPU and process count. `local` is **not OS-isolated** and
is only for trusted code. Independent metric checking does not validate scientific methodology,
training/test leakage or a paper's results. `paper_metric_status` remains `not_verified`.

MIT licensed. Contributions are welcome.
