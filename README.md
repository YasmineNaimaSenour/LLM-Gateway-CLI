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

### Non-streaming

```bash
python -m src.cli --provider ollama --prompt "Explain TCP handshakes"
```

### Streaming

```bash
python -m src.cli --provider groq --model llama-3.1-8b-instant \
    --system "You are terse." --prompt "Explain TCP handshakes" \
    --temperature 0.2 --max-tokens 200 --stream
```

## What it does

* Switches between providers behind one `BaseProvider` interface (`src/providers/`)
* Supports both streaming and non-streaming responses
* Counts input tokens before every request (via `tiktoken`, falling back to a heuristic)
* Measures request latency
* Never crashes: every failure is classified into `rate_limit | context | format | model | unknown`
  and logged, with a friendly message on stderr and a non-zero exit code
* Appends one structured JSON record per request to `logs/requests.jsonl`

## Repository layout

```text
src/
├── providers/       # base.py (interface) + ollama_provider.py + groq_provider.py
├── core/            # errors.py (taxonomy), logger.py (JSONL), telemetry.py (timing)
├── token_utils.py   # pre-request token counting
└── cli.py           # entry point / orchestration
experiments/          # sampling variance, context behavior, and failure-case logs
tests/                # pytest suite (providers mocked, no network required)
```

## Tests

```bash
python -m pytest -q
```
