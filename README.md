# LLM Gateway CLI

Provider-agnostic CLI gateway over **Ollama** (local) and **Groq** (hosted),
built as a single entry point for experimentation and benchmarking of LLMs.

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
still works for backward compatibility, but `chat` is the explicit form.

#### Non-streaming

```bash
python -m src.cli chat --provider ollama --prompt "Explain TCP handshakes"
```

#### Streaming

```bash
python -m src.cli chat --provider groq --model openai/gpt-oss-20b \
    --system "You are terse." --prompt "Explain TCP handshakes" \
    --temperature 0.2 --max-tokens 200 --stream
```

### Structured extraction

`gateway structured` extracts structured data from arbitrary text using a
JSON Schema you supply — no code, just a schema file:

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
| `description`                                             | Yes |
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

## What it does

* Switches between providers behind one `BaseProvider` interface (`src/providers/`)
* Supports both streaming and non-streaming chat responses
* Extracts structured data from text against a user-supplied JSON Schema
  (`gateway structured`), reusing the same `BaseProvider.chat()` call as
  `chat` — no separate HTTP client or provider-specific code path
* Counts input tokens before every request (via `tiktoken`, falling back to a heuristic)
* Measures request latency
* Never crashes: every failure is classified into `rate_limit | context | format | model | unknown`
  and logged, with a friendly message on stderr and a non-zero exit code
* Appends one structured JSON record per request to `logs/requests.jsonl`

## Repository layout

```text
src/
├── providers/        # base.py (interface) + ollama_provider.py + groq_provider.py
├── core/              # errors.py (taxonomy), logger.py (JSONL), telemetry.py (timing)
├── structured/        # JSON Schema -> Pydantic -> validated extraction (see above)
│   ├── schema.py         # load + meta-validate + supported-subset check
│   ├── model_builder.py  # JSON Schema (subset) -> Pydantic model (internal detail)
│   └── extractor.py      # prompt -> parse -> validate -> retry, via BaseProvider.chat()
├── token_utils.py     # pre-request token counting
└── cli.py             # entry point / orchestration (chat + structured subcommands)
examples/
└── structured/        # sample schema + input text used in the docs above
experiments/           # sampling variance, context behavior, and failure-case logs
tests/                 # pytest suite (providers mocked, no network required)
```

## Tests

```bash
python -m pytest -q
```
