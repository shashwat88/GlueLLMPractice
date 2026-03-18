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

Testing notes:
- Tests mock external HTTP calls (SEC + Wikipedia) so you don’t need API keys to run the test suite.
- The agent loop tests stub the LLM responses to validate termination and fallback behavior.

