# LLM Gateway CLI

Provider-agnostic CLI gateway over **Ollama** (local) and **Groq** (hosted),
built as a provider-agnostic LLM gateway for experimentation, structured
generation, tool calling, and eventually evaluation of LLM workloads.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
cp .env.example .env
```

Add `GROQ_API_KEY` to `.env` if you plan to use Groq.

### Ollama

The Ollama provider requires an Ollama server running locally and the `llama3.2` model to be available.

#### Option 1: Native Ollama installation

Install Ollama for your operating system, then start the server:

```bash
ollama serve
```

In another terminal, download the model:

```bash
ollama pull llama3.2
```

The default Ollama endpoint is:

```text
http://localhost:11434
```

Keep the Ollama server running while using the CLI.

#### Option 2: Ollama with Docker

Docker can be used to run Ollama in an isolated container.

Start the container:

```bash
docker run -d \
  --gpus=all \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  ollama/ollama
```

Then download the model inside the container:

```bash
docker exec -it ollama ollama pull llama3.2
```

The model is stored in the Docker volume `ollama`, so it remains available when the container is stopped and started again.

Check that the container is running:

```bash
docker ps
```

To stop Ollama:

```bash
docker stop ollama
```

To start the existing container again:

```bash
docker start ollama
```

To check its logs:

```bash
docker logs ollama
```

To remove the container:

```bash
docker rm ollama
```

**Note:** Removing the container does not remove the `ollama` Docker volume, so downloaded models remain available. To remove the models as well, remove the volume:

```bash
docker volume rm ollama
```

### GPU support with Docker

The Docker setup above uses:

```bash
--gpus=all
```

This allows Ollama to access a compatible NVIDIA GPU from inside the container. NVIDIA Container Toolkit must be installed and configured on the host.

For systems without a compatible GPU, remove `--gpus=all`:

```bash
docker run -d \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  ollama/ollama
```

The CLI itself does not need to know whether Ollama is running natively or inside Docker. Both expose the same HTTP API at:

```text
http://localhost:11434
```

If using another Ollama host or port, set it in `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

### Groq

The Groq provider requires a `GROQ_API_KEY`.

Set it in `.env`:

```env
GROQ_API_KEY=your_api_key_here
```

Alternatively, it can be provided as an environment variable:

```bash
export GROQ_API_KEY="your_api_key_here"
```

## Usage

### Chat

The `chat` subcommand can be omitted — `python -m src.cli --provider ... --prompt ...`
still works for backward compatibility, but `chat` is the explicit form. The
implicit form is **deprecated**: each use prints a stderr warning, and it will
be removed in a future release. Update scripts to say `chat` explicitly.

#### Non-streaming

```bash
python -m src.cli chat --provider ollama --prompt "Explain TCP handshakes"
```

See [Multi-turn chat (sessions)](#multi-turn-chat-sessions) for continuing a
conversation across CLI invocations with `--session`.

#### Streaming

```bash
python -m src.cli chat --provider groq --model openai/gpt-oss-20b \
    --system "You are terse." --prompt "Explain TCP handshakes" \
    --temperature 0.2 --max-tokens 200 --stream
```

### Multi-turn chat (sessions)

`chat` is stateless by default: one prompt in, one response out. Add
`--session <file>` to make consecutive CLI calls continue one conversation:

```bash
python -m src.cli chat --provider ollama --session chats/demo.jsonl --prompt "Hi, I'm Bob."
python -m src.cli chat --provider ollama --session chats/demo.jsonl --prompt "What's my name?"
```

The session file is append-only JSONL — one line per `ChatMessage`, exactly
the transcript the orchestrator returns for a turn — so it is inspectable
with `cat`/`jq` and crash-safe (a process dying mid-write loses at most the
torn trailing line, which is skipped on load). Everything is persisted at
full fidelity: system messages, assistant tool calls, and tool results, so
a tool-calling conversation resumes with its complete context.

Notes:

* A failed turn is never persisted — the file only grows on success, so
  retrying the same command resumes cleanly.
* On a continuation turn `--system` and `--schema` are not re-injected
  (the session already carries the system/schema-instruction messages from
  the turn that created it); passing them again just prints a note on
  stderr and is otherwise ignored.

### Structured output during chat

Add `--schema` to a normal `chat` call to force the final answer to
conform to a JSON Schema. Reuses the same schema file format (and the same
validate-and-retry loop) as the `structured` command below, so the schemas
in `examples/structured/` work here too:

```bash
python -m src.cli chat --provider groq \
    --prompt "Ada Lovelace, 36, mathematician and writer, based in London, England." \
    --schema examples/structured/person_schema.json
```

Where possible the schema is also passed to the provider as a native hint
(Ollama gets grammar-constrained decoding via its `format` field; Groq gets
`response_format: json_object`) to cut down on retries — but the result is
always validated and retried gateway-side regardless, so behavior is
identical across providers even though reliability under the hood differs.

### Tool calling

Add `--tools` (comma-separated names) to let the model call functions
mid-conversation. The gateway runs the request/response loop — sending the
tools, executing whichever ones the model calls, feeding results back —
until the model gives a final, tool-free answer or `--max-tool-iterations`
(default `8`) is hit:

```bash
python -m src.cli chat --provider ollama \
    --prompt "What's 40 + 2? Also, what time is it in Tokyo?" \
    --tools calculator,current_time
```

Built-in tools live in `src/tools/registry.py` (currently `calculator` and
`current_time`) — it's a small, explicit, in-repo registry rather than a
plugin system; add a tool by writing a Pydantic model for its arguments and
decorating a function with `@register(...)`.

`--tools` and `--schema` can be combined: the model uses tools as needed,
then its final answer is validated against the schema. `--stream` is
ignored (with a one-line notice on stderr) whenever `--tools` or `--schema`
are set — tool-bearing and schema-coerced turns are always non-streaming.

### Structured extraction

`gateway structured` extracts structured data from arbitrary text using a
JSON Schema you supply — no code, just a schema file. Unlike `chat
--schema` above, it takes input from a file rather than a live prompt and
never uses tools; internally both share the same validation/retry core
(`structured/extractor.py`):

```text
input text + JSON Schema → validate schema → convert schema to a Pydantic model
    → structured LLM generation → validated result
```

```bash
python -m src.cli structured \
    --provider ollama \
    --input examples/structured/sample_input.txt \
    --schema examples/structured/person_schema.json
```

```json
{
  "name": "Ada Lovelace",
  "age": 36,
  "occupation": "mathematician and writer",
  "role": null,
  "skills": ["mathematics", "analytical reasoning", "algorithm design"],
  "address": {
    "city": "London",
    "country": "England"
  }
}
```

Useful flags:

| Flag             | Meaning                                                              |
|------------------|-----------------------------------------------------------------------|
| `--input`        | Path to a text file to extract from (required)                       |
| `--schema`       | Path to a JSON Schema file describing the target shape (required)     |
| `--provider`     | `ollama` or `groq` (required)                                        |
| `--model`        | Model name (defaults per-provider, same as `chat`)                   |
| `--output`       | Write the result to a file instead of stdout                         |
| `--max-retries`  | Retries on unparsable/invalid model output before giving up (default `2`) |
| `--temperature`  | Defaults to `0.0` (extraction wants determinism, not creativity)     |
| `--max-tokens`   | Same as `chat` (default `512`)                                       |

Save the result to a file:

```bash
python -m src.cli structured \
    --provider groq --model openai/gpt-oss-20b \
    --input examples/structured/sample_input.txt \
    --schema examples/structured/person_schema.json \
    --output result.json --max-retries 3
```

#### Supported JSON Schema subset

The gateway converts your schema into an internal Pydantic model, so it
supports a deliberate initial subset of JSON Schema rather than the full
specification:

| Feature                       | Supported |
|--------------------------------|:---------:|
| `type`: object, string, integer, number, boolean, array | Yes |
| Nested objects / arrays (arbitrary depth)                | Yes |
| `properties`, `required`                                 | Yes |
| `additionalProperties` (boolean only)                    | Yes |
| `enum` (on any type)                                      | Yes |
| `description` (must be a string, validated at every level)  | Yes |
| Nullable types via `"type": [<type>, "null"]`             | Yes |
| String: `minLength`, `maxLength`, `pattern`               | Yes |
| Number/integer: `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` | Yes |
| Array: `items` (single schema), `minItems`, `maxItems`    | Yes |
| `$ref` / `$defs` / `definitions`                          | No |
| `oneOf` / `anyOf` / `allOf` / `not` / `if`-`then`-`else`   | No |
| `const`, `multipleOf`                                     | No |
| `patternProperties`, schema-valued `additionalProperties` | No |
| Tuple-style `items` (a list of schemas)                   | No |
| Root schema that isn't `"type": "object"`                 | No |

The gateway distinguishes two different ways a `--schema` file can fail,
both surfaced as `[format:<ClassName>]` on stderr:

* **`SchemaError`** — the file isn't valid JSON Schema at all (bad JSON,
  malformed keywords, etc.), checked via the `jsonschema` library's
  meta-schema validation.
* **`UnsupportedSchemaError`** — it *is* valid JSON Schema, but uses a
  feature outside the table above (e.g. `$ref`, `oneOf`).
* **`ExtractionError`** — the schema was fine, but the model's output never
  became valid JSON matching it, even after `--max-retries` attempts.

All three are `FormatError` subclasses (see `src/core/errors.py`), so they
still log under the same five-category taxonomy (`rate_limit | context |
format | model | unknown`) as everything else — they're just distinguishable
by exception type for callers that care.

## Adding a provider

Providers are discovered through a registry (`src/providers/registry.py`), not
hardcoded in the CLI. The registry mirrors the tool registry (`src/tools/registry.py`):
a provider registers itself at import time and becomes a first-class `--provider`
choice — argparse choices, model defaulting, instantiation — with zero CLI edits:

```python
# src/providers/my_provider.py
from .base import BaseProvider
from .registry import register_provider


@register_provider("myprovider", default_model="my-model-7b")
class MyProvider(BaseProvider):
    name = "myprovider"
    ...
```

Then make sure the module is imported once (list it in
`src/providers/__init__.py`, as the built-ins do) and `--provider myprovider`
works, with `--model` defaulting to `my-model-7b` when not given.

## What it does

* Switches between providers behind one `BaseProvider` interface (`src/providers/`);
  new backends self-register via `@register_provider` (see "Adding a provider" below)
* Supports both streaming and non-streaming chat responses
* Extracts structured data from text against a user-supplied JSON Schema
  (`gateway structured`), or enforces a schema on a live `chat` answer
  (`chat --schema`) — both go through the same validate-and-retry core
  (`structured/extractor.py`), not two separate implementations
* Runs a tool-calling loop (`chat --tools`): offers tools to the model,
  executes whichever ones it calls via an in-repo registry
  (`src/tools/`), and feeds results back until it gets a final answer
* Supports multi-turn chat (`chat --session <file>`): prior history is
  loaded from and the completed turn is appended to an append-only JSONL
  session file (`src/core/session.py`), so consecutive CLI calls continue
  one conversation — including tool-calling sessions
* Uses native provider capabilities where available (Ollama's
  schema-constrained `format` decoding, Groq's `json_object` mode) as a
  reliability optimization — gateway-side validation still always runs,
  so behavior stays identical across providers
* Counts input tokens before every request (via `tiktoken`, falling back to a heuristic);
  when a provider reports its own billed `prompt_tokens` in the response, that
  count is preferred in the log instead
* Measures request latency
* Retries transient provider failures (connection blips, HTTP 500/502/503/504)
  with exponential backoff and a stderr notice per retry
  (`src/providers/http_utils.py`); 429 and other permanent errors are never
  retried — they already classify and log correctly
* Never crashes: every failure is classified into `rate_limit | context | format | model | unknown`
  and logged, with a friendly message on stderr and a non-zero exit code
* Appends one structured JSON record per request to `logs/requests.jsonl`,
  including tool-call counts when `--tools` was used

## Repository layout

```text
src/
├── providers/        # base.py (interface) + registry.py + http_utils.py (shared retry transport) + ollama_provider.py + groq_provider.py
├── core/
│   ├── types.py          # provider-agnostic ToolSpec / ToolCall / ToolResult
│   ├── orchestrator.py    # run_turn(): the tool-call loop + schema coercion, shared by chat & structured
│   ├── session.py         # --session persistence: append-only JSONL conversation transcripts
│   ├── errors.py          # five-category error taxonomy
│   ├── logger.py          # JSONL request logging
│   └── telemetry.py       # request timing
├── structured/        # JSON Schema -> Pydantic -> validated extraction (see above)
│   ├── schema.py         # load + meta-validate + supported-subset check
│   ├── model_builder.py  # JSON Schema (subset) -> Pydantic model (internal detail)
│   └── extractor.py      # coerce_to_schema(): validate -> retry loop, via BaseProvider.chat()
├── tools/             # in-repo tool registry + executor (not a plugin system)
│   ├── registry.py        # @register-decorated tools (calculator, current_time, ...)
│   └── executor.py        # runs a ToolCall, never raises past itself
├── token_utils.py     # pre-request token counting
└── cli.py             # entry point / argument parsing + I/O only (chat + structured subcommands)
examples/
└── structured/        # sample schema + input text used in the docs above
experiments/           # sampling variance, context behavior, and failure-case logs
tests/                 # pytest suite (providers mocked, no network required)
```

## Tests

```bash
python -m pytest -q
```