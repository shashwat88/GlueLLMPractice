# GlueLLM Interview Project

This repo contains an assessment-style, runnable multi-agent project built with the
[`gluellm`](https://github.com/Bioto/glue-llm) SDK.

## Setup

### Install

```bash
cd /Users/shashwatsingh/Documents/GlueLLMPractice
uv pip install -e ".[dev]"
```

### Environment variables

GlueLLM uses your LLM provider keys via environment variables (for example `OPENAI_API_KEY`).
Set the provider keys required by your chosen model.

#### SEC EDGAR User-Agent

The SEC requires a descriptive `User-Agent` header for automated scraping.
Set:

```bash
export GLUELLM_EDGAR_USER_AGENT="gluellm-interview (contact: your-email@example.com)"
```

#### EODHD API Key

The EODHD real-time endpoint requires an API token.
Set:

```bash
export EODHD_API_KEY="your-eodhd-api-key"
```

#### Wikipedia User-Agent (recommended)

Wikipedia may return `403 Forbidden` for requests missing a descriptive `User-Agent`.
Set:

```bash
export WIKIPEDIA_USER_AGENT="gluellm-interview (contact: your-email@example.com)"
```

#### Project logging

This repo includes an optional, GlueLLM-inspired logging setup. Logs are written to `./logs/` by default (rotating file handler).

What gets logged:
- **Agent/workflow events**: workflow start/parse/fallback events and per-turn events for the interactive agents.
- **LLM request/response**: every LLM call logs:
  - `LLM request: ...` (the prompt/query sent to the model)
  - `LLM response: ...` (the model’s returned text, or structured output rendered as a string)

Notes:
- **Truncation**: request/response bodies are truncated at **2000 chars** by default to keep logs readable; the log line includes the total original length.
- **Security**: prompts/responses may include sensitive content. Disable logging or avoid running with secrets in prompts if you don’t want them written to disk.

Common env vars:

```bash
export PROJECT_LOG_CONSOLE_OUTPUT="true"   # show logs in console (default: false)
export PROJECT_LOG_LEVEL="INFO"            # console log level
export PROJECT_LOG_FILE_LEVEL="DEBUG"      # file log level
export PROJECT_LOG_DIR="logs"              # directory for log files (default: ./logs)
export PROJECT_LOG_FILE_NAME="gluellm_practice.log" # log file name
export PROJECT_LOG_JSON_FORMAT="false"     # set true for JSON logs
export PROJECT_LOG_MAX_BYTES="10485760"    # max log file size before rotation (default: 10MB)
export PROJECT_LOG_BACKUP_COUNT="5"        # number of rotated backups to keep
export PROJECT_DISABLE_LOGGING="false"     # set true to disable logging setup
```

By default, logs go to `logs/gluellm_practice.log`.

## Project Layout

```text
agents/  # runnable agents (each as `python -m agents.<name>`)
core/    # shared loop/controller helpers
tools/   # deterministic tool functions used by GlueLLM tool-calling
tests/   # unit tests (external calls mocked)
.cursor/ # editor rules (Cursor .mdc format)
```

## Running

Each agent is runnable as a module from the project root:

```bash
python -m agents.<agent_module>
```

The three interactive agents in Part 1.3 / Part 2 run in a loop:
- You enter an initial question at runtime.
- Then you can enter follow-up questions.
- Type `quit` to exit the loop.

If the tool evidence is insufficient to answer, the agent responds gracefully with a “cannot answer” message (no hallucinated citations).

### Rock Paper Scissors

```bash
python -m agents.rock_paper_scissors --rounds 5
```

What it does:
- Two agents (“Player A” and “Player B”) choose moves using structured output (`rock|paper|scissors`).
- The controller computes the winner deterministically.
- After each round, it prints: A move, B move, and the outcome.

### Poem Writer + Critic

```bash
python -m agents.poem_loop --topic "winter sunrise" --threshold 8 --max-iters 10
```

What it does:
- Writer generates an initial poem for the given topic.
- Critic evaluates the poem and returns structured feedback (including a score out of 10).
- The loop continues until `score >= threshold` or `max-iters` is reached.
- Each iteration logs the poem version and the critic feedback.

### SEC EDGAR Research Agent

```bash
python -m agents.sec_research
```

What it does:
- Uses tool functions to search filings, retrieve filing metadata, and fetch filing text.
- Produces a grounded answer with citations (form type, filing date, accession number, and the primary document URL when available).

At runtime you will be prompted for:
- an initial SEC research question
- follow-up questions until you type `quit`



Notes:
- If the tools do not return relevant evidence for your question, the agent will explain that it cannot answer it from retrieved SEC evidence.
- You control the exact questions at runtime; no questions are hard-coded.

### Directory Crawler Agent

```bash
python -m agents.directory_crawler
```

What it does:
- Scans the target directory tree exactly once into an in-memory report.
- Answers questions using only filesystem metadata (file names/extensions, sizes, and nesting depth).
- It will not read file contents.

At runtime you will be prompted for:
- a root directory path
- a question (examples: “how many Python files are there?”, “what file type is most common?”, “is there a .docx file?”)
- follow-up questions until you type `quit`

Notes:
- Queries about “file types / counts / largest file / nesting depth” are supported from the cached scan.
- If you ask for file contents, it will respond gracefully with `cannot answer` (because it does not read file text).

### Basic Research Agent

```bash
python -m agents.basic_research
```

What it does:
- Uses an external tool layer based on Wikipedia (search + page summaries).
- Synthesizes a structured summary with key points and sources.

At runtime you will be prompted for:
- a research query
- follow-up questions until you type `quit`

Notes:
- If Wikipedia evidence is insufficient, it will respond gracefully that it cannot answer.
- No questions are hard-coded; everything comes from your runtime input.

### EODHD Real-Time Stock Agent

```bash
python -m agents.eodhd_stock_agent
```

Environment variable:

```bash
export EODHD_API_KEY="your-eodhd-api-key"
```

At runtime you will be prompted for:
- a stock symbol (e.g., `AAPL`)
- a stock-related question (price/high/low/volume/etc.)
- follow-up questions until you type `quit`

If your question is not about real-time stock quote facts (or the fetched evidence is missing), the agent responds gracefully that it cannot answer.

## Tests

Run unit tests with:

```bash
pytest
```

## Evaluation (SDK-native GlueLLM recording)

This repo uses GlueLLM's official eval recording APIs directly (no custom eval framework):

- `gluellm.eval.JSONLFileStore`
- `GlueLLM(..., eval_store=store)`
- automatic `EvalRecord` JSONL rows on each `complete()` call

### Run eval recording

Requires provider env vars such as `OPENAI_API_KEY`.

```bash
uv run python -m eval.run_eval_recording
```

Optional flags:

```bash
uv run python -m eval.run_eval_recording \
  --model openai:gpt-4o-mini \
  --output logs/eval_records.jsonl \
  --prompt "What is 2 + 2?" \
  --prompt "Explain deterministic algorithms in one sentence."
```

### Inspect recorded EvalRecord rows

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("logs/eval_records.jsonl")
for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    rec = json.loads(line)
    print(
        f"{i}: id={rec.get('id')} success={rec.get('success')} "
        f"latency_ms={rec.get('latency_ms')} cost={rec.get('estimated_cost_usd')}"
    )
    if i >= 5:
        break
PY
```

The runner also prints a compact preview (`id`, `latency_ms`, `estimated_cost_usd`, `success`) after recording completes.

## Docker

### 1) Prepare env file

Create a local `.env` from the template:

```bash
cp .env.example .env
```

Fill in the keys you need (for example `OPENAI_API_KEY`, `EODHD_API_KEY`).

### 2) Build image with Compose

```bash
docker compose build
```

### 3) Run agents (utility-image style)

The container is intentionally generic. Pass the full Python module command each run:

```bash
docker compose run --rm app python -m agents.basic_research
```

More examples:

```bash
docker compose run --rm app python -m agents.rock_paper_scissors --rounds 5
docker compose run --rm app python -m agents.poem_loop --topic "winter sunrise" --threshold 8 --max-iters 10
docker compose run --rm app python -m agents.sec_research
docker compose run --rm app python -m agents.directory_crawler
docker compose run --rm app python -m agents.eodhd_stock_agent
```

Notes:
- Compose enables TTY/stdin for interactive agents automatically.
- Source is mounted into the container (`.:/app`) for fast local iteration.
- Logs are written under `logs/` (or your configured `PROJECT_LOG_DIR`).

### Optional hardening

For reproducible dependency resolution across environments, you can introduce a lockfile workflow with `uv`:

```bash
uv lock
uv sync --frozen
```

You can then update the Docker build to install from the lockfile in a later pass.

Testing notes:
- Tests mock external HTTP calls (SEC + Wikipedia) so you don’t need API keys to run the test suite.
- The agent loop tests stub the LLM responses to validate termination and fallback behavior.

## Linting and type-checking

Run linting:

```bash
ruff check .
```

Auto-format:

```bash
ruff format .
```

Type-check:

```bash
mypy .
```

### Pre-commit (recommended)

Install hooks:

```bash
pre-commit install
```

Run on all files:

```bash
pre-commit run --all-files
```

