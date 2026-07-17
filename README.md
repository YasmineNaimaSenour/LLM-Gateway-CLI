# LLM Gateway CLI

Provider-agnostic CLI gateway over **Ollama** (local) and **Groq** (hosted),
built as the Phase 1 mini-project for the AI Engineering curriculum.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY if you plan to use Groq
```

Ollama path requires a local server: `ollama serve` + `ollama pull llama3.2`.
Groq path requires `GROQ_API_KEY` (env var or `.env`).

## Usage

```bash
# Non-streaming
python -m src.cli --provider ollama --prompt "Explain TCP handshakes"

# Streaming, with a system prompt and sampling controls
python -m src.cli --provider groq --model llama-3.1-8b-instant \
    --system "You are terse." --prompt "Explain TCP handshakes" \
    --temperature 0.2 --max-tokens 200 --stream
```

## What it does

- Switches between providers behind one `BaseProvider` interface (`src/providers/`)
- Streams or blocks on a full response
- Counts tokens in (via tiktoken, falling back to a heuristic) before every request
- Measures wall-clock latency per request
- Never crashes: every failure is classified into `rate_limit | context | format | model | unknown`
  and logged, with a friendly message on stderr and a non-zero exit code
- Appends one structured JSON record per request to `logs/requests.jsonl`

## Repository layout

```
src/
├── providers/       # base.py (interface) + ollama_provider.py + groq_provider.py
├── core/             # errors.py (taxonomy), logger.py (JSONL), telemetry.py (timing)
├── token_utils.py     # pre-request token counting
└── cli.py             # entry point / orchestration
experiments/          # sampling variance, context behavior, and failure-case logs
tests/                 # pytest suite (providers mocked, no network required)
```

## Tests

```bash
python -m pytest -q
```
