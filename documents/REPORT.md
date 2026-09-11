# LLM Gateway — Project Report

**Status:** complete for its declared scope (M1: chat/structured; M2: tools, chat-schema, sessions) · **Test suite:** 227 tests, all passing · **Runtime target:** Python 3.10+ (developed on 3.14)

This report is the definitive technical documentation for the project. It covers what the system does, why each piece exists, how the pieces interact, and what a developer needs to know to modify or extend it safely. Every claim reflects the code as of this writing; the "future / deferred" sections distinguish what is *not* built. A short user-facing guide lives in [`README.md`](README.md); condensed, task-oriented facts for AI assistants live in [`context/`](context/). The three are complementary and do not duplicate each other.

---

## Table of contents

1. [Project goal, motivation, and scope](#1-project-goal-motivation-and-scope)
2. [Requirements](#2-requirements)
3. [Architecture overview](#3-architecture-overview)
4. [Design principles](#4-design-principles)
5. [Component reference](#5-component-reference)
6. [Data and control flows](#6-data-and-control-flows)
7. [Cross-cutting decisions](#7-cross-cutting-decisions)
8. [Configuration](#8-configuration)
9. [Testing](#9-testing)
10. [Examples](#10-examples)
11. [Dependencies](#11-dependencies)
12. [Limitations](#12-limitations)
13. [Evolution of the architecture](#13-evolution-of-the-architecture)
14. [Deferred work and future roadmap](#14-deferred-work-and-future-roadmap)
15. [Developer's guide to extending the system](#15-developers-guide-to-extending-the-system)
16. [Appendix: repository map](#appendix-repository-map)

---

## 1. Project goal, motivation, and scope

### 1.1 Goal

A **provider-agnostic CLI gateway for LLM experimentation**: one command-line tool that talks to different LLM backends (today: local **Ollama**, hosted **Groq**) behind a single interface, and layers uniform, backend-independent services on top:

- plain chat (streaming and non-streaming),
- structured output (text + JSON Schema → validated JSON),
- tool calling (model invokes registered functions in a loop),
- multi-turn sessions that survive across CLI invocations,
- structured request logging and latency telemetry for every call.

### 1.2 Motivation

Experimenting with LLMs normally means rewriting the same plumbing per provider: different HTTP shapes, different error dialects, different streaming formats, different structured-output capabilities. This project centralizes that plumbing once, so:

- **Provider comparison becomes trivial** — the same prompt can be sent to `--provider ollama` or `--provider groq` with identical downstream behavior (logging, retries, validation, tool loop).
- **Structured extraction and tool calling become provider-portable.** Providers differ wildly in native support; the gateway normalizes on top of a validate-and-retry core so behavior is identical even though underlying reliability differs.
- **Every request is observable.** One JSONL log line per call, with a uniform five-category error taxonomy, makes experimentation measurable (see `experiments/`).
- **The local path is zero-cost.** Ollama support means the whole system runs offline with no API key.

### 1.3 Scope

**In scope (implemented):** the two subcommands (`chat`, `structured`) with all the flag combinations documented in the README; the provider/tool registries; sessions; the error taxonomy; JSONL logging; token counting; HTTP retry; the structured-extraction pipeline; four guarded example schema/input pairs; a full pytest suite.

**Out of scope (deliberately not built — see §14):** a server/daemon mode, plugin or MCP-based tool discovery, provider-side cost estimation, conversation compaction/summarization, async or parallel request execution, and packaging/distribution (`pyproject.toml`).

### 1.4 Sources of truth and how to read the docs

| Document | Audience | Content |
|---|---|---|
| `README.md` | Users | What it is, install, usage examples, config |
| `REPORT.md` (this file) | Current + future developers | Full architecture, rationale, decisions, roadmap |
| `context/` | AI assistants working on the repo | Condensed facts, conventions, constraints per topic |
| `AUDIT.md` | Historical record | The 24-item code audit; every item resolved, resolution notes inline |
| `src/**` docstrings | Everyone | Module-level docstrings carry most design rationale, kept next to the code |

---

## 2. Requirements

### 2.1 Functional requirements

- **FR1 — Chat:** send a prompt (with optional system prompt) to a chosen provider/model; print the response; support streaming.
- **FR2 — Structured extraction:** given an input text file and a JSON Schema file, produce schema-validated JSON; optionally write to an output file.
- **FR3 — Schema-enforced chat:** a `chat` call may require its final answer to conform to a JSON Schema.
- **FR4 — Tool calling:** a `chat` call may expose named registered tools to the model; the gateway executes the model's tool calls and feeds results back until a final answer (bounded by `--max-tool-iterations`).
- **FR5 — Sessions:** consecutive CLI invocations sharing a `--session <file>` continue one conversation, including tool-call history and system/schema messages.
- **FR6 — Provider abstraction:** adding a backend requires implementing one interface and one decorator; zero CLI edits.
- **FR7 — Observability:** every call (success or failure) appends exactly one structured JSONL record with provider, latency, token counts, temperature, status, error classification, and tool-call counts.
- **FR8 — Never crash:** all failures are classified, logged, reported to stderr, and converted to exit code 1.
- **FR9 — Transient-failure resilience:** connection errors and 5xx are retried with backoff; permanent errors are not.

### 2.2 Non-functional requirements

- **Deterministic interface:** stdout carries only the answer (parseable by scripts); all progress, warnings, and errors go to stderr.
- **Offline-capable:** Ollama path works without network or keys; token counting falls back to a heuristic without `tiktoken`.
- **Testability without network:** the entire suite runs against mocks; no API key is ever needed in tests.
- **Crash-safety of persistence:** session files and logs are append-only JSONL; a torn trailing line is never fatal.

---

## 3. Architecture overview

### 3.1 The one-paragraph version

`src/cli.py` parses arguments and does I/O — nothing else. Both subcommands delegate to runtime primitives: `core/orchestrator.run_turn()` (the tool-call loop + schema coercion shared by both commands), `structured/extractor.py` (the validate-and-retry core), and the provider behind `providers/base.BaseProvider`. Providers translate between the runtime's provider-agnostic vocabulary (`ChatMessage`, `ToolSpec`, `ToolCall`, `ToolResult`) and their own wire formats, and share one HTTP transport with retry (`providers/http_utils.py`). Registries (`providers/registry.py`, `tools/registry.py`) make providers and tools first-class by name at import time. Failures funnel into a single error taxonomy (`core/errors.py`) and a single logging choke point (`core/logger.log_request`).

### 3.2 Package map and dependency direction

```text
              ┌──────────────────────────── CLI (src/cli.py) ────────────────────────────┐
              │  argument parsing, file I/O, stdout/stderr, logging calls, session wiring │
              └───────┬────────────────────────┬─────────────────────────┬───────────────┘
                      │                        │                         │
                      ▼                        ▼                         ▼
        ┌──────────────────────┐   ┌────────────────────────┐   ┌─────────────────────┐
        │ core/orchestrator.py │   │ structured/extractor.py│   │ core/session.py     │
        │  run_turn()          │──▶│  coerce_to_schema()    │   │ SessionStore (JSONL)│
        │  tool loop + schema  │   │  validate → retry loop │   └─────────────────────┘
        └──────┬───────────────┘   └───────┬────────────────┘
               │                           │
               │                           ▼
               │                 ┌──────────────────────────┐
               │                 │ structured/schema.py     │  load → meta-validate → subset check
               │                 │ structured/model_builder │  JSON Schema (subset) → Pydantic model
               │                 └──────────────────────────┘
               ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ providers/base.py — BaseProvider, ChatMessage, ChatResponse  │  ← the only contract
        └──────┬──────────────────────────────┬────────────────────────┘
               │                              │
      ┌────────▼─────────┐          ┌─────────▼─────────┐
      │ OllamaProvider   │          │ GroqProvider      │      each: wire-format adapter
      │ (local, :11434)  │          │ (OpenAI-compat)   │      + error mapping
      └────────┬─────────┘          └─────────┬─────────┘
               └────────────┬─────────────────┘
                            ▼
              ┌───────────────────────────────┐
              │ providers/http_utils.py       │  POST with transient-failure retry
              │ providers/registry.py         │  name → ProviderSpec → instance
              └───────────────────────────────┘

  tools/registry.py (@register, Pydantic arg models) ──▶ tools/executor.py (ToolExecutor)
  core/types.py (ToolSpec/ToolCall/ToolResult) — shared by providers, tools, orchestrator
  core/errors.py + core/logger.py + core/telemetry.py — used by everything above
```

**The dependency rule:** everything points *down* to `core` (types, errors) and `providers/base`. Nothing in `core/` imports `providers/` except `orchestrator.py`, which imports `providers.base` (the interface only — never a concrete provider). `structured/` and `tools/` import the provider interface, not implementations. Concrete providers are known only to the registry. This is what makes "provider-agnostic" true in code rather than in prose.

### 3.3 Why a runtime/CLI split

The CLI is deliberately a *thin shell*: argument parsing, file reading/writing, printing, and the error→log→exit-code choreography. All actual behavior lives in runtime primitives that take plain values and return plain results. Consequences:

- Both `chat` and `structured` share one implementation of tool looping and schema coercion instead of two drifting copies.
- A future interactive REPL, HTTP server, or library consumer calls `run_turn()` directly; nothing about it assumes a process-per-prompt CLI.
- Tests exercise runtime behavior through the same entry point the CLI uses, so CLI tests stay shallow (wiring) while runtime tests stay deep (semantics).

---

## 4. Design principles

These are the recurring, load-bearing principles; each module's docstring applies them locally.

1. **CLI is an interface, runtime is the product.** `cli.py` contains no business logic (§3.3).
2. **One shared core per capability.** The tool loop exists once (`run_turn`); the schema validate-and-retry loop exists once (`coerce_to_schema`); the HTTP retry exists once (`post_with_retry`); JSON parsing of model output exists once (`_extract_json_value`). Both subcommands compose these rather than re-implementing them.
3. **Normalize at the boundary; validate everywhere.** Providers translate their error dialects into `GatewayError`s at the HTTP boundary; the orchestrator still validates final answers gateway-side, because native provider guarantees (Ollama's grammar-constrained `format`, Groq's `json_object`) are best-effort hints, never trusted.
4. **Five-category taxonomy, subclass for detail.** All failures map to `rate_limit | context | format | model | unknown` for logging; concrete `GatewayError` subclasses (plus the `error_subtype` log field) carry the finer-grained story without multiplying log categories.
5. **Explicit contracts over implicit globals.** `__all__` declared per package (guarded by tests); `to_content_dict()` named for exactly what it does; `run_turn()` never mutates caller state and returns the full transcript.
6. **Append-only JSONL as the persistence idiom.** Request logs and session files share the same crash-safe, diffable, `jq`-able format — one JSON object per line.
7. **Registries make extension a one-file change.** Providers and tools self-register at import time; the CLI learns choices, defaults, and instantiation from the registries.
8. **stdout is the answer channel; stderr is everything else.** Deprecation warnings, retry notices, tool-loop progress, and session-save warnings all go to stderr so scripts parsing stdout are never surprised.

---

## 5. Component reference

### 5.1 `src/core/` — runtime contracts

#### `core/errors.py` — the error taxonomy

Every failure the gateway touches normalizes to exactly one of five `ErrorType` categories, chosen because they map to *different operator responses*:

| Category | Meaning | Typical operator response |
|---|---|---|
| `rate_limit` | Provider quota/throttle (429, "rate limit", "quota exceeded") | Wait; don't retry immediately |
| `context` | Prompt too long for the model's window | Shorten the prompt |
| `format` | Request or output malformed (bad schema, bad JSON, bad args, missing key, unknown provider/tool, stuck tool loop, unusable session) | Fix the request/config |
| `model` | Model doesn't exist / transport failure to reach it (404, connection refused, timeout) | Fix model name / check server |
| `unknown` | Anything else | Investigate |

Classification (`classify_exception`) uses HTTP status codes first (reliable when present), then substring markers in the message (fallback for connection-level errors). `to_gateway_error()` wraps any exception into the right `GatewayError` subclass and is **idempotent** — already-normalized errors pass through unchanged, so double-wrapping is impossible.

Concrete subclasses exist so callers can distinguish *which kind* of failure occurred even within one category; the log's additive `error_subtype` field (the class name) puts that same detail in the JSONL record:

- `FormatError` → `SchemaError` (not valid JSON Schema), `UnsupportedSchemaError` (valid but outside the supported subset), `ExtractionError` (model never produced valid schema-conforming JSON), `ToolLoopError` (model never stopped calling tools), `SessionError` (unusable session file).
- `RateLimitError`, `ContextOverflowError`, `ModelError`, `UnknownError` map 1:1 to categories.

**Why keep `ToolLoopError` in `format` rather than a sixth category?** Deliberate trade-off (AUDIT #3): the five-category taxonomy is a load-bearing interface for logs and analysis; a subclass plus `error_subtype` gives the same debugging signal with zero consumer churn. A sixth category would fragment every log query.

#### `core/types.py` — shared wire-agnostic vocabulary

`ToolSpec` (a tool's declaration: name, description, JSON-Schema parameters), `ToolCall` (a model-requested invocation with *already-parsed* `arguments: dict`), `ToolResult` (execution outcome with `is_error` flag). These live in `core/`, not `providers/base.py`, on purpose: providers *consume* them, and the tool registry/executor need them without importing the provider package at all.

#### `core/orchestrator.py` — `run_turn()`

The single runtime entry point. Signature (abridged):

```python
run_turn(provider, messages, *, tools=None, tool_executor=None, response_schema=None,
         max_tool_iterations=8, max_retries=2, temperature=0.7, max_tokens=512,
         on_tool_loop_event=None) -> OrchestrationResult
```

Behavior:

1. **Copy-on-entry.** The caller's `messages` list is never mutated (AUDIT #1); a shallow copy suffices because `ChatMessage` is a value object.
2. **Tool loop** (only when `tools` given; requires a `tool_executor`): each iteration calls `provider.chat(..., tools=tools)`; if the response carries `tool_calls`, append the assistant message (with calls), execute each call via the executor, append each result as a `tool` message; repeat until a tool-free response or `max_tool_iterations` (default 8) is exhausted → `ToolLoopError`.
3. **Schema coercion** (only when `response_schema` given): delegates to `structured.extractor.coerce_to_schema()`, passing the final tool-free response as `initial_response` so no redundant provider call is made (this is why tools+schema in one turn costs exactly one call per round plus retries).
4. **Plain path** (neither): a single `provider.chat()` call.

`OrchestrationResult` carries `text`, `data` (validated dict when schema), `tokens_out` (summed across all calls of the turn), `tokens_in` (provider-billed count from the **last** provider call — each call is billed on the full conversation so far), `tool_call_count`, `tool_iterations`, `attempts`, and `messages` — the complete transcript. Feeding `result.messages` back as the next turn's input is the documented continuation mechanism; a test pins that this never doubles history.

`on_tool_loop_event` is a deliberately minimal observer: `("thinking", round)`, `("tool", name)`, `("done", None)`. It exists so interactive UIs can show progress during tool turns — which never stream (§7.4) — without the providers needing to stream tool-call payloads (AUDIT #15). The CLI prints these as stderr progress lines; stdout stays pure.

#### `core/session.py` — `SessionStore`

A session is the conversation transcript itself, stored as append-only JSONL — one full-fidelity `ChatMessage` per line (`role`, `content`, and where applicable `tool_calls`, `tool_call_id`, `name`). Design rationale (in the module docstring and AUDIT #2):

- **Append-only = crash-safe.** A process dying mid-write loses at most the torn trailing line, which load() skips silently. No read-modify-write window.
- **The file *is* the history.** Inspectable with `cat`/`jq`; no index to corrupt; the on-disk vocabulary is the orchestrator's in-memory vocabulary 1:1.
- **Full fidelity.** System messages, assistant tool calls, and tool results are all persisted, so a tool-calling conversation resumes with complete context.

`save_session_messages()` computes the common prefix between the file's contents and the new transcript and appends only the tail — so saving the same transcript twice is a no-op, and a continuation turn writes just its new messages. A file with *no* valid records raises `SessionError` (a `FormatError`, so it logs under the existing taxonomy); a file that merely has some bad lines loads the good ones. `load_session_messages()` returns `None` when the file doesn't exist yet (fresh start).

CLI semantics (pinned by `tests/test_session_cli.py`): a failed turn is never persisted (retrying the same command resumes cleanly); on continuation turns `--system` and `--schema` are not re-injected (the session already carries them — a stderr note explains); a session-save failure after a successful answer warns but exits 0.

#### `core/logger.py` and `core/telemetry.py`

`log_request()` appends exactly one JSON object per line to `logs/requests.jsonl` (overridable via `LLM_GATEWAY_LOG_PATH`, read once at import — documented behavior). Record schema:

```json
{"timestamp": "...ISO-8601 UTC...", "provider": "ollama", "latency_ms": 123.4,
 "tokens_in": 33, "tokens_out": 12, "temperature": 0.7, "status": "success",
 "error_type": null, "error_subtype": null, "token_count_method": null,
 "tool_calls": null, "tool_iterations": null}
```

Field semantics worth knowing:

- `error_type` — one of the five category values; `null` on success. `error_subtype` — the concrete exception class name; `null` on success. Both always present as keys (analysis can rely on them).
- `token_count_method` — `"tiktoken" | "heuristic"` when *this client* produced `tokens_in`; `null` when the count came from the provider (a provider-billed count is precise by definition; labeling it "tiktoken" would be a lie) and on error paths (the pre-failure `tokens_in=0` would make any claim noise).
- `tool_calls` / `tool_iterations` — populated on tool-bearing turns; explicit `null` otherwise.

Writes are line-atomic appends; a test pins that 4 concurrent processes × 5 records produce exactly 20 parseable, loss-free lines. There is deliberately **no** log rotation — it isn't a feature; the module docstring notes where coverage would belong if it ever is.

`telemetry.py` is a small `Timer` (lazy `elapsed_ms` property, read-safe before `stop()`) plus a `measure_latency()` context manager.

### 5.2 `src/providers/` — the abstraction boundary

#### `base.py` — `BaseProvider`, `ChatMessage`, `ChatResponse`

The entire contract every backend implements:

- `chat(messages, *, temperature, max_tokens, tools, response_schema) -> ChatResponse` — non-streaming; returns the final `text`, client-side-counted `tokens_out`, parsed `tool_calls`, the raw payload, and `tokens_in` (provider-billed prompt tokens, or `None` when unreported — callers then fall back to client-side counting).
- `chat_stream(messages, *, temperature, max_tokens) -> Iterator[str]` — text-only chunks. **No `tools` parameter, by design:** the two providers stream tool-call payloads in incompatible shapes, and streaming a turn that may end in tool calls has little user-facing value (§7.4).
- `response_schema` is a **best-effort hint** on `chat()`: Ollama forwards it as grammar-constrained decoding (`format`); Groq uses `response_format: json_object` (valid JSON syntax, not schema-conforming). Callers must always validate — gateway-side validation is what actually enforces the schema, which is why behavior is provider-identical.

`ChatMessage.role` is `Literal["system","user","assistant","tool"]` (AUDIT #12 — the type system enforces what the comment used to promise). `to_content_dict()` is deliberately **content-only** (role + content; tool fields excluded) — its name says what it does (AUDIT #22); full-fidelity serialization is sessions' explicit record format.

#### `registry.py` — provider discovery

`ProviderRegistry` holds `ProviderSpec` records (name, class, `default_model`) — **not instances**. Instantiation is deferred to `get_provider(name, model, timeout=...)` so constructor side effects (e.g. Groq raising `ModelError` when `GROQ_API_KEY` is missing) happen at call time, inside the CLI's error handling, not at import time of an unrelated module. `timeout` is forwarded only to providers whose constructor accepts it (detected via `inspect.signature`) — the registry stays provider-agnostic with zero concrete-class imports (which also avoids a circular import, since providers import `register_provider` from this module). Unknown names raise `FormatError` listing registered providers — a startup-time config check, distinct from call-time provider failures.

A module-level `DEFAULT_REGISTRY` preserves import-time registration; bound-function aliases (`register_provider`, `get_provider`, `provider_names`, ...) keep every call site unchanged, while `snapshot()`/`restore()` and instance-level state enable test isolation and a future server mode (AUDIT #8).

Adding a provider: implement `BaseProvider`, decorate with `@register_provider("name", default_model=...)`, and ensure the module is imported once (list it in `providers/__init__.py`, as both built-ins are). The CLI's argparse `choices`, model defaulting, and instantiation all follow automatically.

#### `http_utils.py` — shared retry transport

`post_with_retry()` encodes a precise retry policy (AUDIT #4), each line justified in its docstring:

| Failure | Retried? | Why |
|---|---|---|
| `ConnectionError` (incl. ConnectTimeout) | ✅ | The request never completed; replay can't duplicate server-side work |
| 500/502/503/504 | ✅ | The standard replayable-fault convention |
| 429 | ❌ | Rate limiting is quota, not fault; blind immediate retry makes it worse — it already maps to `RATE_LIMIT` where the caller can choose proper backoff |
| other 4xx | ❌ | Permanent per-request failures (bad key/model/payload) |
| other 5xx | ❌ | Outside the replayable convention; classify-and-report is the honest outcome |
| read `Timeout` | ❌ | The request may still be running server-side; a retry could double-bill or double-execute a tool call |

Budget: initial attempt + 2 retries, exponential backoff 0.5s → 1.0s → 2.0s; each retry prints a one-line `[http-retry]` notice to stderr (never a silent pause). On exhaustion the last response is returned as-is — retry smooths over transients, it never hides them; the provider's classification keeps its final say. `sleep` is injectable for tests.

#### `ollama_provider.py` / `groq_provider.py`

Both are thin adapters with the same shape: build payload → `post_with_retry` → map transport errors (`ConnectionError` → `ModelError` with a helpful message; `Timeout` → `ModelError`) → on non-200 map through `to_gateway_error` with the status code → parse response shape (`FormatError` on unexpected shapes) → normalize tool calls into `ToolCall` with **already-parsed** arguments.

Provider-specific normalization worth knowing:

- **Ollama** synthesizes tool-call ids (`call_0`, ...) because its native `/api/chat` doesn't guarantee one; arguments arrive already parsed (no `json.loads`); `tokens_in` comes from `prompt_eval_count` when present. Streaming is NDJSON lines; a `done: true` chunk ends the stream. Timeout precedence: explicit argument > `OLLAMA_TIMEOUT` env > 60s default (60s chosen for interactive latency; batch users raise it — AUDIT #23). Base URL: `base_url` arg > `OLLAMA_BASE_URL` env > `http://localhost:11434`.
- **Groq** receives real tool-call ids but JSON-*encoded* argument strings — the adapter parses them (`FormatError` if malformed). `tokens_in` comes from `usage.prompt_tokens`; `tokens_out` prefers `usage.completion_tokens`. Streaming is OpenAI-style SSE (`data:` lines, `[DONE]` sentinel). API URL: `api_url` arg > `GROQ_API_URL` env > official endpoint (enables Groq-compatible proxies; AUDIT #24). API key: `api_key` arg > `GROQ_API_KEY` env; missing key raises `ModelError` at construction. Timeout precedence mirrors Ollama (`GROQ_TIMEOUT`).

### 5.3 `src/structured/` — structured outputs

The pipeline: `load_and_validate_schema()` → `build_model()` → `extract()` / `coerce_to_schema()` → plain dict.

#### `schema.py` — two failure modes, two exceptions

A `--schema` file can be wrong in two distinct ways, and callers need to tell them apart:

1. `SchemaError` — not valid JSON Schema at all (missing file, bad JSON, non-object top level, fails `jsonschema` meta-validation).
2. `UnsupportedSchemaError` — valid JSON Schema, but uses a feature outside the supported subset (`$ref`, `oneOf`, tuple `items`, ...).

The supported subset (root must be `"type": "object"`): objects/arrays/scalars (`string`, `integer`, `number`, `boolean`), nested objects/arrays at any depth, `properties`/`required` (validated to reference known properties), boolean `additionalProperties`, `enum` (any scalar values), string `description` (validated at every level — the subset checker is the single source of truth, even where meta-validation would also catch it, because it produces a clear path-qualified message), nullable types via `"type": [<type>, "null"]` (only that shape — no general unions), string `minLength`/`maxLength`/`pattern`, numeric `minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum`, array `items` (single schema)/`minItems`/`maxItems`. Everything else is rejected with a path-qualified `UnsupportedSchemaError` message (`$.properties.address.city: ...`).

#### `model_builder.py` — JSON Schema → Pydantic

Builds a Pydantic model via `create_model` from an already-validated schema (it refuses unvalidated input defensively). Mapping details that matter:

- `enum` → `Literal[...]`; constraints become Pydantic `Field` kwargs (`ge`/`le`/`gt`/`lt`, `min_length`/`max_length`, `pattern`, array bounds).
- **Property-name sanitization:** non-identifier names (hyphens, spaces, leading digits, Python keywords) become valid Python field names with a Pydantic `alias` preserving the original name; `populate_by_name=True` accepts either key, and the extractor dumps with `by_alias=True` so output round-trips to the exact original schema property names. Pinned by both-sides tests (AUDIT #16).
- `additionalProperties: false` → `extra="forbid"`; otherwise `extra="ignore"`.
- Optional (non-`required`) properties become `Optional[...]` with default `None`.

Pydantic is an **implementation detail of this package**: nothing outside `src/structured/` imports it (the tool registry goes through the package's declared API: `build_model`, `check_supported_subset`).

#### `extractor.py` — the validate-and-retry core

`coerce_to_schema(provider, messages, schema, ...)` is the one implementation both paths share:

1. Build the Pydantic model once; build a concise schema summary once (`_json_object_summary` — one line per field with required/optional, types, enum values; recurses into nested objects and notes array item types).
2. For each attempt (initial + `max_retries`, default 2): parse the response with `_extract_json_value()` — whole text as JSON, then a ` ```json ` fenced block, then the first *parseable* balanced `{...}` span found by a brace-depth walk that honors string literals (AUDIT #9: nested objects, prose braces, braces inside strings all handled; a balanced-but-non-JSON span is skipped, not fatal).
3. Validate the parsed dict against the model. Success → `ExtractionResult(data, tokens_out, attempts, raw_text, tokens_in)`.
4. Failure → append the invalid response as an assistant message plus a corrective user message carrying the **schema summary**, the previous response, and the validation error — the retry prompt stands on its own even when the original schema has scrolled out of the model's effective attention window (AUDIT #10). Then retry with a fresh call.

Note the deliberate contract difference: `run_turn()` copies its input list; `coerce_to_schema()` **mutates** the list it is given (retry appends must be visible to the next call) — both documented in their docstrings. Provider-raised `GatewayError`s (rate limits, etc.) propagate unchanged so they keep their classification.

`extract()` is the thin wrapper for the `structured` command: builds the extraction-specific system prompt ("you are a precise data-extraction engine... ONLY a single JSON object") + user message, delegates to `coerce_to_schema()`. `schema_instruction_message()` builds the distinct system message for `chat --schema` ("*once you are ready to give your final answer*...") — different wording because chat turns may use tools first; the orchestrator injects it via the CLI's `build_messages()`.

### 5.4 `src/tools/` — tool calling

#### `registry.py`

Tools are Python functions registered via `@register(name, description, parameters)` on `ToolRegistry` (a single module-level `DEFAULT_REGISTRY` preserves import-time registration; instance-level state enables isolation, AUDIT #8). Parameters are declared either as a **Pydantic model** (preferred; `model_json_schema()` is free) or a raw JSON-Schema dict (which must be `"type": "object"` and within the supported subset — reusing the structured pipeline so there is exactly one schema subset and one validation path in the codebase). Registration stores a `RegisteredTool(spec, model, func)`: the wire-facing spec, the Pydantic model used to validate incoming call arguments, and the callable.

Built-ins: `calculator` (add/subtract/multiply/divide; raises `ValueError` on divide-by-zero/unknown op) and `current_time` (IANA timezone via `zoneinfo`). `get_tools(names)` resolves a `--tools` list; unknown names raise `FormatError` **at startup, before any provider call** — a CLI misconfiguration the model couldn't recover from mid-conversation.

#### `executor.py`

`ToolExecutor.execute(call) -> ToolResult` treats tool-domain failures as **conversational, recoverable events**, never exceptions: unknown tool name, argument validation failure (`ValidationError`), or an exception inside the tool all become an error `ToolResult` fed back to the model, which can see the error and react. The dividing line: a `GatewayError` means the *runtime* failed; a tool error means the *thing the tool touched* failed — squarely inside what the model is reasoning about. Results are stringified (`str` passthrough or `json.dumps` with `default=str`).

### 5.5 `src/cli.py` — the shell

Responsibilities, exhaustively: argparse construction (choices sourced from `provider_names()`); the backward-compatibility shim (`_normalize_argv` — the flat pre-subcommand form still works but prints a one-line deprecation warning to stderr, AUDIT #7); message building (system prompt, schema-instruction injection, session-history prepend); the pre-request client-side token count (`count_message_tokens` over `to_content_dict()`s — the context-window signal); provider construction via `build_provider()` (a one-line delegation to the registry, kept as the CLI's single seam for tests to patch); dispatch to streaming/non-streaming/`run_turn`; session save; and the logging/reporting choreography.

Error handling is two nested safety nets: `except GatewayError` (already classified) then `except Exception` → `to_gateway_error` → both flow through the single `_log_and_report()` choke point, which logs the record and prints `[<category>:<ClassName>] <message>` to stderr, then returns exit code 1. The CLI structurally cannot crash (pinned by tests), and stderr's `[type:subtype]` and the log record's fields are guaranteed consistent because both come from the same line of code.

The `--timeout` flag on both subcommands flows `CLI → build_provider() → registry → constructor`; providers without the knob ignore it via the registry's signature check.

### 5.6 `src/token_utils.py` — token counting

Counts tokens **before** a request is sent (the context-window signal) and for `tokens_out` on providers that don't bill counts. `tiktoken` `cl100k_base` when available (an approximation for non-OpenAI models, fine for logging/budgeting), else a word-count heuristic (`round(words / 0.75)`). `count_method()` reports which is active — the single source of truth feeding the log's `token_count_method` field (AUDIT #13). `count_message_tokens()` adds a flat 4-token per-message overhead allowance.

---

## 6. Data and control flows

### 6.1 Plain chat (`chat --prompt "hi"`)

```text
argv → _normalize_argv (implicit 'chat' + deprecation warning) → parse args
→ build_messages: [system?, user]                      (no schema → no instruction message)
→ count_message_tokens (client-side pre-count)
→ build_provider(name, model, timeout)  ← registry
→ provider.chat(messages) → ChatResponse
→ print(text) to stdout
→ build transcript = messages + [assistant reply]      (saved to session if --session)
→ log_request(status=success, tokens_in=provider-billed or pre-count, token_count_method=... or null)
→ exit 0
```

### 6.2 Streaming chat (`--stream`)

Same, but `provider.chat_stream()` yields text chunks printed to stdout with `flush=True`; the joined text becomes the reply, `tokens_out` is the client-side count, and `tokens_in` is the pre-count (chunks carry no usage report in the current contract). Trailing newline printed once at stream end.

### 6.3 Tool calling (`--tools calculator,current_time`)

```text
get_tools(["calculator","current_time"])                ← FormatError before any network call if unknown
→ run_turn(provider, messages, tools=specs, tool_executor=ToolExecutor(...), on_tool_loop_event=stderr progress)
   loop (max 8 iterations):
     "thinking" event → provider.chat(..., tools) → response
     response.tool_calls empty? → break to final answer
     append assistant(tool_calls) → for each call:
         "tool" event → executor.execute(call) → ToolResult → append tool message
     (exhausted → ToolLoopError)
   "done" event → print(result.text)
→ log_request(tool_calls=N, tool_iterations=M)
```

Tool errors never abort the turn (§5.4); `--stream` is ignored with a stderr notice; `--max-tool-iterations` bounds the loop.

### 6.4 Chat with schema (`--schema person.json`)

```text
load_and_validate_schema(path)                          ← SchemaError / UnsupportedSchemaError before any call
→ build_messages(..., schema=schema): [system?, schema_instruction, user]
→ run_turn(..., response_schema=schema)
   (tools? → loop first, final tool-free response reused as initial_response)
   coerce_to_schema: parse → validate → (retry with corrective message up to --max-retries)
→ print(json.dumps(result.data, indent=2))
→ log_request(..., tokens_in from the accepted attempt)   (attempts is on the result object, not in the log record)
```

### 6.5 Structured extraction (`structured --input t.txt --schema s.json`)

```text
_read_text_file (FormatError on missing/empty) + load_and_validate_schema
→ tokens_in pre-count = count(input_text) + count(schema JSON)
→ extract(provider, text, schema): [extraction system prompt, user=text] → coerce_to_schema
→ print JSON (or write --output file + confirmation line)
→ log_request(..., tokens_in from the accepted attempt)
```

### 6.6 Sessions (`--session chats/demo.jsonl`)

```text
turn 1: load_session_messages → None (fresh) → messages = build_messages(...)
        → run/call → transcript → save_session_messages (creates file)
turn 2: load_session_messages → prior history
        → messages = prior + build_messages(system=None, schema=None)   ← no re-injection
        → run/call → transcript → save (prefix-match append of the new tail only)
failed turn: nothing persisted; same command retried resumes cleanly
```

### 6.7 Every failure path

```text
exception raised anywhere
→ GatewayError? keep : to_gateway_error(exc, provider=...)   (idempotent)
→ _log_and_report: log_request(status=error, error_type=category, error_subtype=ClassName)
                   print "[category:ClassName] message" → stderr
→ return exit code 1  (never an unhandled exception)
```

---

## 7. Cross-cutting decisions

This section consolidates the decisions a developer must understand before changing anything.

### 7.1 Why the CLI/runtime split (recap)

See §3.3. The practical rule: **if you're adding behavior, add it to a runtime module; `cli.py` only gains argument parsing and printing.**

### 7.2 Why the error taxonomy is five categories + subtype

Categories exist for the *log consumer* (aggregate analysis, alerting); subclasses exist for the *caller* (catch `SchemaError` specifically). Splitting them would couple every consumer to a growing enum; merging them would erase debuggability. The `error_subtype` field is the bridge. See §5.1 and AUDIT #3/#21.

### 7.3 Why provider-reported `tokens_in` is preferred, with a labeled fallback

Provider-billed prompt counts are exact and account for provider-specific tokenization; tiktoken is an approximation for non-OpenAI models. So: use the provider's count when reported, else the client-side pre-count, and record *how* a client-side count was made (`token_count_method`). Streaming always uses the pre-count (no usage in chunks). Tool loops use the last call's count (billed on the full conversation). Schema retries use the accepted attempt's count. AUDIT #11/#13.

### 7.4 Why tool-bearing and schema-coerced turns never stream

Two independent reasons, one shared consequence:

- The providers' `chat_stream()` contracts yield **text-only** chunks and cannot represent tool-call payloads (and their payload shapes are mutually incompatible). A mid-loop streamed call would have to silently drop the tool calls — the entire point of the round-trip.
- Schema enforcement applies to the final answer; streaming partial text that may later be replaced by validated JSON would mislead.

So `run_turn` handles those turns non-streaming, and `--stream` + `--tools`/`--schema` prints a stderr notice. The UX gap (silent multi-second turns) is filled by the `on_tool_loop_event` progress observer, not by streaming. A future provider that can stream tool-bearing turns would render the observer optional, not obsolete. AUDIT #15.

### 7.5 Why native provider schema features are used but never trusted

Ollama's `format` field does grammar-constrained decoding (real enforcement); Groq's `json_object` guarantees only valid JSON syntax. Using them cuts retries at zero cost. But the gateway must behave **identically across providers** — so `coerce_to_schema()` always parses and validates, and retries regardless. The hint is an optimization; the gateway is the contract. This is also why `chat --schema` output is byte-identical in shape whether the model is local or hosted.

### 7.6 Why JSONL everywhere

Append-only writes never truncate prior state (crash-safe); the format is diffable, stream-parseable, and inspectable with `cat`/`jq`; and the project treats "one JSON object per line" as a single persistence idiom shared by `logs/requests.jsonl`, session files, and (in spirit) streaming NDJSON. The alternative — one JSON array per file — needs read-modify-write and has a torn-write failure mode.

### 7.7 Why registries hold specs, not instances

Constructor side effects (Groq's missing-key `ModelError`) must surface at call time where error handling lives, not at import time of an unrelated module. Spec-holding also keeps the registry free of concrete-class imports (avoiding a circular import) and lets `timeout` be capability-detected by signature. AUDIT #5/#8.

### 7.8 Why the tool executor never raises

Model-requested tool calls are untrusted input by definition. A raised exception would abort the whole turn for something the model could recover from by adjusting arguments. Error `ToolResult`s keep the model in the loop; startup-time validation of `--tools` catches the one case the model can't fix (a misconfigured CLI). See §5.4.

### 7.9 Why `_extract_json_value` tries three strategies

Models wrap JSON in prose or fences unpredictably. Order matters: whole-text parse (fast path, no false positives) → fenced block (common wrapper) → balanced-span walk (last resort; tolerates prose and nested objects; only accepts spans that actually parse). The old first-`{`-to-last-`}` slice was silently wrong for two objects in one response (AUDIT #9).

### 7.10 Copy-on-entry vs. mutate-in-place

`run_turn()` copies and returns the transcript (callers reuse lists across turns — doubled history is a classic silent bug, AUDIT #1/#19). `coerce_to_schema()` mutates (retry appends must be visible to the next provider call, and its callers pass working copies). The asymmetry is deliberate and documented at both sites; `OrchestrationResult.messages` and the session store build on the returned-transcript contract.

### 7.11 Notable rejections (rejected alternatives)

- **Sixth error category (`TOOL_LOOP`)** — rejected: would fragment every log query; `error_subtype` gives the same signal (AUDIT #3).
- **Comprehensive `ChatMessage.to_dict()`** — rejected: token counting and full serialization are different jobs with different consumers; one method would bloat counts or invite silent data loss; rename to `to_content_dict()` instead (AUDIT #22).
- **Making `OLLAMA_TIMEOUT` mean 120s while changing the default** — rejected: would make the env var mandatory for the old behavior; env var became the general override (AUDIT #23).
- **Streaming text mid-tool-loop** — rejected: the streaming contract cannot represent tool payloads; would trade silence for wrongness (AUDIT #15).
- **`pyproject.toml` now** — deferred: no package metadata exists yet; the runtime/dev split already positions the project for `[project.optional-dependencies]` later (AUDIT #18).
- **Keeping `DEFAULT_MODELS` in the CLI** — replaced by per-provider registry defaults; the CLI no longer names any provider string or imports any concrete class (AUDIT #5).

---

## 8. Configuration

### 8.1 Environment variables

All except the last are read via `os.environ` **after** `load_dotenv()` runs at provider-module import, so setting them in `.env` works. `LLM_GATEWAY_LOG_PATH` is the exception: the logger computes its default path at its own import, which happens *before* any `load_dotenv()` call in the CLI — set it in the process environment, not `.env`.

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | Groq | *(none — required for Groq)* | API key; missing → `ModelError` at construction |
| `GROQ_API_URL` | Groq | `https://api.groq.com/openai/v1/chat/completions` | Point at a Groq-compatible proxy |
| `GROQ_TIMEOUT` | Groq | `60` | Per-request timeout (seconds) |
| `OLLAMA_BASE_URL` | Ollama | `http://localhost:11434` | Ollama server address |
| `OLLAMA_TIMEOUT` | Ollama | `60` | Per-request timeout (seconds) |
| `LLM_GATEWAY_LOG_PATH` | Logger | `logs/requests.jsonl` | Log file location (read once at import) |

Precedence everywhere: **explicit CLI flag / constructor argument > environment variable > default**. `--timeout` on both subcommands overrides the env/default for a single call.

### 8.2 Defaults worth knowing

`--temperature` 0.7 for `chat`, **0.0 for `structured`** (extraction wants determinism) · `--max-tokens` 512 · `--max-retries` 2 (schema coercion) · `--max-tool-iterations` 8 · HTTP retry: 2 retries, 0.5s→1s→2s backoff.

---

## 9. Testing

### 9.1 How to run

```bash
pip install -r requirements-dev.txt   # includes pytest
python -m pytest -q                   # 227 tests, no network, no API keys
```

### 9.2 Suite structure and philosophy

| File | Covers |
|---|---|
| `test_providers.py` | Both providers' chat/stream parsing, error mapping, timeouts, env precedence, `to_content_dict` contract, 503-recovery through the retry layer |
| `test_provider_registry.py` | Built-in registration, model defaulting, timeout forwarding, duplicate/unknown names, snapshot/restore isolation, argparse choices wired to the registry |
| `test_http_utils.py` | The full retry contract: every retryable/non-retryable status, connection errors, mid-retry recovery, backoff timing, exhaustion semantics, stderr notices |
| `test_orchestrator.py` | Tool loop semantics, transcript immutability/reuse, tools+schema composition without redundant calls, `tokens_in` semantics per path, progress-event sequences |
| `test_structured_schema.py` | Loading, meta-validation, subset enforcement at every level, both error types |
| `test_structured_model_builder.py` | Type mapping, enums, constraints, nullable, sanitization round-trips (both alias sides, `populate_by_name`) |
| `test_structured_extractor.py` | Clean/fenced/prose JSON parsing, retry behavior, corrective-message content, extraction failure |
| `test_tools.py` | Registry resolution, isolation, executor success/arg-validation/tool-exception paths |
| `test_session.py` | Store round-trips (incl. full tool fidelity), torn-line tolerance, idempotent saves, divergence handling |
| `test_session_cli.py` | CLI wiring: continuation, tool-history persistence, `--system`/`--schema` non-reinjection, failed-turn semantics, corrupt files, save-failure safety |
| `test_cli.py` | Success/error paths, both subcommands, flat-form deprecation, provider-token preference, `token_count_method`, tool-loop `error_subtype`, stderr/log consistency invariant |
| `test_errors.py` | Classification by status and message, idempotent wrapping |
| `test_logger.py` | JSONL validity, append, all optional fields, env-var override, ISO timestamps, concurrency/line-atomicity |
| `test_token_utils.py` | Method reporting, heuristic arithmetic, edge cases |
| `test_public_api.py` | Every `__all__` name resolves; top-level `src` exports nothing |
| `test_examples.py` | Examples directory matches declared pairs; every schema passes subset+build; feature matrix covered; instances validate |

Philosophy: runtime tests go deep through the same entry points the CLI uses; CLI tests stay shallow (wiring, I/O, logging) and mock `build_provider`; no test touches the network or requires a key. The `sleep` callable in `post_with_retry` and registry `snapshot()`/`restore()` exist specifically so tests can run fast and isolated. Guard tests pin non-obvious invariants: stderr/log field agreement, one-JSON-object-per-line, transcript non-mutation, alias round-trips, examples-not-rotted.

---

## 10. Examples

`examples/structured/` holds four schema/input pairs, each targeting a distinct feature mix so a user can find a minimal example for what they need (guarded by `tests/test_examples.py`, so they cannot silently rot):

| Example | Demonstrates |
|---|---|
| `person_schema.json` | nested objects, arrays, nullable fields |
| `release_notes_schema.json` | nested objects, arrays of objects, string enums, patterns |
| `ticket_schema.json` | enums, nullable strings with patterns, booleans |
| `weather_schema.json` | nullable numbers, enums, nested location |

Usage: `python -m src.cli structured --provider ollama --input examples/structured/person_input.txt --schema examples/structured/person_schema.json`. The supported-subset feature table lives in the README (user-facing) and §5.3 (developer-facing).

`experiments/` holds templates for measurement work the gateway is built to support: `sampling_variance.json` (same prompt across temperatures), `context_behavior.md` (quality/overflow degradation points), `failure_cases.json` (one deliberately induced failure per error category) — currently unfilled templates, i.e. *planned experiments*, not results.

---

## 11. Dependencies

Runtime (`requirements.txt`): `requests>=2.31` (HTTP), `pydantic>=2.0` (schema→model + validation), `jsonschema>=4.18` (meta-validation), `tiktoken>=0.7` (optional at runtime: absent → heuristic counting). **Known gap:** both providers `load_dotenv()` from `python-dotenv`, which is installed in the dev environment but **not listed in `requirements.txt`** — a fresh runtime install would fail on import. Fix is one line; it is recorded here and in `context/conventions.md` rather than silently.

Dev (`requirements-dev.txt`): `-r requirements.txt` + `pytest>=8.0`. No `pyproject.toml` (see §7.11).

Python: `>= 3.10` is required by the code's typing usage (`Literal`, `list[str]` annotations, `zoneinfo`); development happens on 3.14.

---

## 12. Limitations

Current behavior, stated plainly — not bugs, but boundaries of the implemented scope:

1. **Single-turn-per-invocation CLI.** Sessions persist history, but each invocation is still one turn; there is no interactive REPL.
2. **No streaming for tool/schema turns** — by design (§7.4); progress observer fills the gap.
3. **JSON Schema subset only** (§5.3). No `$ref`/composition/tuple items; the root must be an object.
4. **Tool args validated, not tool *outputs*.** A tool returning nonsense JSON is fed back verbatim.
5. **Token counts are approximations** for non-OpenAI models when the provider doesn't bill counts (mitigated: `token_count_method` labels them).
6. **No context-window management.** Long sessions grow until the provider errors with `context`; no compaction/summarization/truncation.
7. **No cost tracking.** Token counts are logged; prices are not modeled.
8. **No concurrency.** Sequential, synchronous requests only; no async runtime.
9. **Log/session files grow unboundedly** — no rotation or pruning.
10. **Session continuation trusts the file.** A hand-edited session is loaded as-is (only structurally invalid lines are skipped); divergence handling appends rather than rewrites history (pinned behavior).
11. **`python-dotenv` missing from `requirements.txt`** (§11) — the one dependency inconsistency found while writing this report.
12. **Two providers only.** The abstraction is proven twice; an OpenAI/Anthropic provider would exercise the interface further but exists only as documented guidance (§15.1).
13. **Heuristic error classification is substring-based** for non-status failures — a provider changing its error wording could reclassify a failure as `unknown`.

---

## 13. Evolution of the architecture

The git history shows the project's actual arc; `AUDIT.md` (24 findings, every one resolved with an inline resolution note) records the deliberate hardening pass. Stages:

1. **Scaffold (M1):** provider interface + error taxonomy; token counting + Ollama; Groq; JSONL logging + telemetry; CLI orchestration; experiment templates, tests, README.
2. **Structured outputs:** schema loading/validation, Pydantic model building, extraction with retries — first as a standalone `structured` command.
3. **Milestone 2 — tool calling:** shared tool types, registry, executor; `run_turn()` composes the tool loop with schema coercion; `chat --schema`.
4. **Audit-driven hardening (all 24 items):** copy-on-entry orchestration + transcript return (#1/#19); sessions (#2); `error_subtype` log field (#3/#21); HTTP retry transport (#4); provider registry (#5); description validation (#6); flat-form deprecation (#7); instance-based registries (#8/#20); brace-depth JSON extraction (#9); schema-summary retry prompts (#10); provider-billed token counts (#11); `Literal` roles (#12); `token_count_method` (#13); logger hardening (#14); tool-loop progress observer (#15); model-builder test completion (#16); three new example pairs (#17); requirements split (#18); `to_content_dict()` rename (#22); 60s timeouts + `--timeout` (#23); configurable Groq URL (#24).

Two structural through-lines: **composition over duplication** (each capability converged on one shared implementation: one tool loop, one retry loop, one retry transport, one JSON extractor) and **contract-first extension** (registries, declared `__all__`, typed roles — all changes that make the *next* change cheaper).

---

## 14. Deferred work and future roadmap

The audit's closing analysis named sessions, the provider registry, and transcript-returning orchestration as the unlocks for everything else — all three are now built. Remaining directions, none of which are implemented:

- **Interactive mode / REPL** — a natural consumer of `run_turn` + `SessionStore`; requires only a loop around existing primitives.
- **Server mode** — the registry instances and copy-on-entry orchestrator were shaped for it (state can be passed explicitly, not shared via globals); transport and auth are the new work.
- **More providers** — OpenAI/Anthropic/llama.cpp; the interface already separates streaming (text-only) from tool-bearing (non-streaming) calls; providers with native tool streaming would relax §7.4.
- **Fuller JSON Schema support** — `$ref`/`$defs`, composition keywords; requires extending both `schema.py` (subset check) and `model_builder.py` (mapping) in lockstep.
- **Context management** — truncation/compaction for long sessions, using `count_message_tokens` as the signal.
- **Cost estimation** — per-provider price tables over the already-logged token counts.
- **Packaging** — `pyproject.toml` with console entry point; the runtime/dev dependency split is already in place.
- **Async/parallel execution** — registries are instance-isolated; the provider interface is sync and would need an async variant or thread offloading.
- **Fill the `experiments/` templates** — the observability (per-request JSONL with taxonomy and token methods) was built for exactly this.
- **Remove the deprecated flat CLI form** — one-line change when the migration window closes (`_normalize_argv` returns argv unchanged).

---

## 15. Developer's guide to extending the system

### 15.1 Add a provider (one file + one import line)

```python
# src/providers/my_provider.py
from .base import BaseProvider, ChatResponse
from .registry import register_provider
from .http_utils import post_with_retry
from ..core.errors import to_gateway_error
from ..token_utils import count_tokens

@register_provider("myprovider", default_model="my-model-7b")
class MyProvider(BaseProvider):
    name = "myprovider"

    def __init__(self, model="my-model-7b", timeout=None):
        super().__init__(model)
        self.timeout = float(timeout or 60.0)   # accept `timeout` and the registry forwards --timeout

    def chat(self, messages, *, temperature=0.7, max_tokens=512, tools=None, response_schema=None):
        payload = self._payload(messages, temperature, max_tokens, tools, response_schema)
        resp = post_with_retry(self.url, payload=payload, timeout=self.timeout)   # shared retry transport
        if resp.status_code != 200:
            raise to_gateway_error(RuntimeError(resp.text), provider=self.name, status_code=resp.status_code)
        data = resp.json()
        return ChatResponse(text=..., tokens_out=count_tokens(text), tool_calls=..., raw=data, tokens_in=<billed or None>)

    def chat_stream(self, messages, *, temperature=0.7, max_tokens=512):
        ...  # yield text chunks only — never tool payloads
```

Then add `from . import my_provider as _my_provider  # noqa: F401` to `providers/__init__.py`. Rules: normalize tool calls to `ToolCall` with parsed dict arguments; map transport errors to `ModelError`/`FormatError` via `to_gateway_error`; treat `response_schema` as a hint only; report `tokens_in` when the API provides it, `None` when not.

### 15.2 Add a tool

```python
from pydantic import BaseModel, Field
from .registry import register

class _MyToolArgs(BaseModel):
    path: str = Field(description="File to inspect.")

@register("my_tool", "One-line description for the model.", _MyToolArgs)
def _my_tool(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")
```

Rules: raise `ValueError` for model-correctable problems (it becomes an error `ToolResult`, not a crash); return strings or JSON-serializable values; never print.

### 15.3 Extend the JSON Schema subset

Change `structured/schema.py` (subset check — add to `_UNSUPPORTED_KEYWORDS` removal or implement a new `_check_*`) **and** `structured/model_builder.py` (the corresponding mapping) in the same change, add a `tests/test_structured_schema.py` rejection/acceptance case plus a `test_structured_model_builder.py` mapping case, and — if user-visible — consider an `examples/structured/` pair (remember `test_examples.py` pins the declared pairs and feature matrix).

### 15.4 Add a log field

Add an optional keyword to `log_request()` (always present in the record, `None` when not applicable — the record schema is load-bearing), set it from the CLI's logging calls, and extend `tests/test_logger.py`. Follow the `error_subtype`/`token_count_method` precedent: optional parameter, explicit semantics in the docstring, `null` when not applicable.

### 15.5 Safety rules for changing core behavior

- `run_turn()` must keep copy-on-entry and return the full transcript — sessions and multi-turn continuation depend on it.
- Keep the five-category taxonomy stable; add subclasses + use `error_subtype`, not new categories.
- Anything on stdout stays parseable; user-facing chatter goes to stderr.
- New persistence uses append-only JSONL; torn trailing lines must never be fatal.
- Every `GatewayError` flows through `_log_and_report()` — don't log/report errors ad hoc elsewhere.
- `chat_stream()` yields text only; don't smuggle tool payloads through it.

---

## Appendix: repository map

```text
README.md               user-facing guide (install, usage, config, subset table)
REPORT.md               this report — architecture, rationale, decisions, roadmap
AUDIT.md                historical 24-item audit, all resolved (gitignored)
context/                condensed AI-assistant context files (per topic)
requirements.txt        runtime deps (requests, pydantic, jsonschema, tiktoken)
requirements-dev.txt    + pytest
.env.example            template for .env (keys, endpoints, timeouts)
src/
  cli.py                argparse + I/O shell; both subcommands; logging choreography
  token_utils.py        tiktoken/heuristic counting; count_method()
  core/
    errors.py           five-category taxonomy + subclasses + classify/wrap
    logger.py           log_request() — one JSONL record per call
    telemetry.py        Timer / measure_latency
    types.py            ToolSpec / ToolCall / ToolResult (wire-agnostic)
    orchestrator.py     run_turn() — tool loop + schema coercion; OrchestrationResult
    session.py          SessionStore — append-only JSONL transcripts
  providers/
    base.py             BaseProvider / ChatMessage / ChatResponse
    registry.py         ProviderRegistry / ProviderSpec / DEFAULT_REGISTRY
    http_utils.py       post_with_retry — transient-failure policy
    ollama_provider.py  local backend adapter (:11434, NDJSON streaming)
    groq_provider.py    hosted backend adapter (OpenAI-compatible, SSE streaming)
  structured/
    schema.py           load + meta-validate + supported-subset check
    model_builder.py    JSON Schema (subset) → Pydantic model
    extractor.py        coerce_to_schema / extract — validate-and-retry core
  tools/
    registry.py         @register decorator, built-ins (calculator, current_time)
    executor.py         ToolExecutor — never raises; error ToolResults
examples/structured/    four schema/input pairs (test-guarded)
experiments/            measurement templates (sampling variance, context, failures)
tests/                  16 test modules, 227 tests — offline, mocked
```
