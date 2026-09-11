# LLM Gateway CLI

A provider-agnostic command-line gateway for LLM experimentation. One tool, two backends — **Ollama** (local, free, offline) and **Groq** (hosted, fast) — with uniform behavior on top: chat (streaming or not), structured output from a JSON Schema, tool calling, multi-turn sessions, and a structured log line for every request.

Why it exists: experimenting with LLMs usually means rewriting the same plumbing per provider (different HTTP shapes, error dialects, streaming formats, structured-output support). This project centralizes that plumbing once, so switching providers is a `--provider` flag and every request is measurable and logged the same way.

**Key features**

- **Chat** with any registered provider — streaming or non-streaming
- **Structured output:** text + JSON Schema → validated JSON (`structured` command, or `chat --schema`), with automatic validate-and-retry
- **Tool calling:** expose registered Python functions to the model (`--tools`), with a bounded execution loop
- **Multi-turn sessions:** continue one conversation across CLI invocations (`--session`)
- **Observability:** one JSONL record per request (latency, tokens, temperature, status, error taxonomy, tool-call counts) in `logs/requests.jsonl`
- **Reliability:** transient-failure retry with backoff, a five-category error taxonomy, and a CLI that never crashes
- **Extensible:** add a provider or a tool in one file — no CLI edits

## Requirements

- Python **3.10+**
- For the `ollama` provider: a running [Ollama](https://ollama.com) server (native or Docker) with a pulled model (default: `llama3.2`)
- For the `groq` provider: a Groq API key

## Installation

```bash
pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # development (adds pytest)
cp .env.example .env                   # then edit
```

### Ollama setup

Native: `ollama serve`, then `ollama pull llama3.2` — or Docker:

```bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull llama3.2
```

(No compatible GPU? Drop `--gpus=all`. The CLI talks to the same `http://localhost:11434` either way.)

### Groq setup

Set `GROQ_API_KEY=your_api_key_here` in `.env` (or export it as an environment variable).

### Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(required for Groq)* | Groq API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_TIMEOUT` | `60` | Per-request timeout (seconds) |
| `GROQ_TIMEOUT` | `60` | Per-request timeout (seconds) |
| `GROQ_API_URL` | official endpoint | Point Groq at a compatible proxy |
| `LLM_GATEWAY_LOG_PATH` | `logs/requests.jsonl` | Request log location |

Any per-request value can also be set ad hoc with `--timeout` on either subcommand.

> `LLM_GATEWAY_LOG_PATH` is read from the process environment at import time (before `.env` is loaded) — export it in your shell rather than putting it in `.env`.

## Usage

### Chat

```bash
python -m src.cli chat --provider ollama --prompt "Explain TCP handshakes"

# streaming, system prompt, sampling controls
python -m src.cli chat --provider groq --model openai/gpt-oss-20b \
    --system "You are terse." --prompt "Explain TCP handshakes" \
    --temperature 0.2 --max-tokens 200 --stream
```

`--provider` is required; `--model` defaults per provider (`llama3.2` / `openai/gpt-oss-20b`).

> The old flat form (`python -m src.cli --provider ... --prompt ...`) still works but is **deprecated** and prints a warning; say `chat` explicitly.

### Multi-turn sessions

```bash
python -m src.cli chat --provider ollama --session chats/demo.jsonl --prompt "Hi, I'm Bob."
python -m src.cli chat --provider ollama --session chats/demo.jsonl --prompt "What's my name?"
```

The session file is append-only JSONL holding the full transcript (system messages, tool calls and results included), so it's inspectable with `cat`/`jq` and crash-safe. Failed turns are never persisted. On a continuation turn, `--system` and `--schema` are not re-injected.

### Structured output during chat

```bash
python -m src.cli chat --provider groq \
    --prompt "Ada Lovelace, 36, mathematician and writer, based in London, England." \
    --schema examples/structured/person_schema.json
```

Prints the validated JSON. The schema is also passed to the provider as a native hint when possible (Ollama grammar-constrained decoding, Groq `json_object` mode) to cut retries — gateway-side validation always runs either way, so behavior is identical across providers.

### Structured extraction

```bash
python -m src.cli structured \
    --provider ollama \
    --input examples/structured/person_input.txt \
    --schema examples/structured/person_schema.json
```

Extracts from a text file and prints validated JSON (or writes it with `--output result.json`). Key flags: `--input` (required), `--schema` (required), `--output`, `--max-retries` (default 2), `--temperature` (default **0.0** — extraction wants determinism), `--max-tokens`.

More example schema/input pairs in `examples/structured/` (each demonstrates different features; guarded by tests):

| Example | Demonstrates |
|---|---|
| `person_schema.json` | nested objects, arrays, nullable fields |
| `release_notes_schema.json` | nested objects, arrays of objects, string enums, patterns |
| `ticket_schema.json` | enums, nullable strings with patterns, booleans |
| `weather_schema.json` | nullable numbers, enums, nested location |

### Tool calling

```bash
python -m src.cli chat --provider ollama \
    --prompt "What's 40 + 2? Also, what time is it in Tokyo?" \
    --tools calculator,current_time
```

The gateway offers the named tools to the model, executes the calls it makes, and feeds results back until a final answer (or `--max-tool-iterations`, default 8). Built-in tools: `calculator`, `current_time`. Tool failures (bad arguments, runtime errors) are fed back to the model as error messages rather than aborting the run.

Notes:

- `--tools` and `--schema` can be combined: tools first, schema-validated final answer.
- `--stream` is ignored (with a stderr notice) whenever `--tools` or `--schema` is set — those turns are always non-streaming, with progress shown on stderr instead.
- Tool-loop progress (`… thinking (round 1)`, `… calling tool: calculator`) goes to stderr; stdout carries only the answer.

### Supported JSON Schema subset

Schemas are converted to an internal Pydantic model, so a deliberate subset of JSON Schema is supported:

| Feature | Supported |
|---|:---:|
| object, string, integer, number, boolean, array | ✅ |
| Nested objects / arrays (arbitrary depth) | ✅ |
| `properties`, `required` | ✅ |
| `additionalProperties` (boolean only) | ✅ |
| `enum` (on any type) | ✅ |
| `description` (string, validated at every level) | ✅ |
| Nullable types via `"type": [<type>, "null"]` | ✅ |
| String `minLength` / `maxLength` / `pattern` | ✅ |
| Number/integer `minimum` / `maximum` / `exclusiveMinimum` / `exclusiveMaximum` | ✅ |
| Array `items` (single schema), `minItems`, `maxItems` | ✅ |
| `$ref` / `$defs` / `definitions` | ❌ |
| `oneOf` / `anyOf` / `allOf` / `not` / `if`-`then`-`else` | ❌ |
| `const`, `multipleOf` | ❌ |
| `patternProperties`, schema-valued `additionalProperties` | ❌ |
| Tuple-style `items` | ❌ |
| Root schema that isn't `"type": "object"` | ❌ |

Three distinct schema-related errors, all reported as `[format:<Class>]` on stderr: `SchemaError` (not valid JSON Schema), `UnsupportedSchemaError` (valid but uses an unsupported feature), `ExtractionError` (model never produced valid JSON matching the schema within `--max-retries`).

## Project structure

```text
src/
├── cli.py             # entry point: argument parsing + I/O only
├── token_utils.py     # pre-request token counting (tiktoken → heuristic fallback)
├── core/              # orchestrator (tool loop + schema coercion), sessions,
│                      # error taxonomy, JSONL logging, telemetry, shared types
├── providers/         # BaseProvider interface, registry, shared retry transport,
│                      # ollama_provider.py, groq_provider.py
├── structured/        # JSON Schema → Pydantic → validate-and-retry extraction
└── tools/             # in-repo tool registry + executor (calculator, current_time)
examples/structured/   # four schema + input example pairs (test-guarded)
experiments/           # measurement templates (sampling variance, context, failures)
tests/                 # pytest suite — fully offline, providers mocked
```

## Testing

```bash
python -m pytest -q    # 227 tests, no network or API keys required
```

## Documentation

- **`REPORT.md`** — the full technical report: architecture, design rationale, decisions and trade-offs, limitations, and roadmap. Read this before modifying the system.
- **`context/`** — condensed, topic-organized notes (architecture, conventions, decisions, gotchas, current state), optimized as working context for AI assistants.
- Source docstrings — per-module design rationale, kept next to the code.

## Adding a provider

Implement `BaseProvider`, decorate the class, and import the module once — the CLI picks it up automatically (argparse choices, model defaulting, instantiation):

```python
# src/providers/my_provider.py
from .base import BaseProvider
from .registry import register_provider

@register_provider("myprovider", default_model="my-model-7b")
class MyProvider(BaseProvider):
    name = "myprovider"
    # implement chat() and chat_stream()
```

Then add `from . import my_provider as _my_provider  # noqa: F401` to `src/providers/__init__.py`. See `REPORT.md` §15 for the complete recipe and contracts.
