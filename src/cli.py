"""LLM Gateway CLI — provider-agnostic entry point.

Usage:
    # chat (default subcommand — "chat" may be omitted, kept for backward compatibility)
    python -m src.cli --provider ollama --model llama3.2 --prompt "Hello"
    python -m src.cli chat --provider groq --model llama-3.1-8b-instant --prompt "Hello" --stream
    python -m src.cli chat --provider ollama --system "You are terse." --prompt "Explain TCP" \
        --temperature 0.2 --max-tokens 200 --stream

    # structured extraction: input text + JSON Schema -> validated JSON
    python -m src.cli structured --provider ollama --input notes.txt --schema person.json
    python -m src.cli structured --provider groq --input notes.txt --schema person.json \
        --output result.json --max-retries 3

Every call — success or failure — writes one structured JSONL record to
logs/requests.jsonl and never raises past main(): errors are caught,
classified, logged, and reported to stderr with a non-zero exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .core.errors import FormatError, GatewayError, to_gateway_error
from .core.logger import log_request
from .core.telemetry import Timer
from .providers.base import BaseProvider, ChatMessage
from .providers.groq_provider import GroqProvider
from .providers.ollama_provider import OllamaProvider
from .structured.extractor import extract as run_extraction
from .structured.schema import load_and_validate_schema
from .token_utils import count_message_tokens, count_tokens

DEFAULT_MODELS = {
    "ollama": "llama3.2",
    "groq": "openai/gpt-oss-20b",
}

_COMMANDS = {"chat", "structured"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-gateway", description="Provider-agnostic LLM gateway CLI.")
    subparsers = parser.add_subparsers(dest="command")

    chat_parser = subparsers.add_parser("chat", help="Send a single chat prompt to a provider.")
    _add_chat_arguments(chat_parser)

    structured_parser = subparsers.add_parser(
        "structured", help="Extract structured data from text using a JSON Schema."
    )
    _add_structured_arguments(structured_parser)

    return parser


def _add_chat_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["ollama", "groq"], required=True, help="Which backend to use.")
    parser.add_argument("--model", default=None, help="Model name (defaults per-provider).")
    parser.add_argument("--prompt", required=True, help="User prompt.")
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512, dest="max_tokens")
    parser.add_argument("--stream", action="store_true", help="Stream the response token-by-token.")


def _add_structured_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["ollama", "groq"], required=True, help="Which backend to use.")
    parser.add_argument("--model", default=None, help="Model name (defaults per-provider).")
    parser.add_argument(
        "--input", required=True, dest="input_path", help="Path to a text file with the input to extract from."
    )
    parser.add_argument(
        "--schema",
        required=True,
        dest="schema_path",
        help="Path to a JSON Schema file describing the target structure.",
    )
    parser.add_argument(
        "--output", default=None, dest="output_path", help="Write the extracted JSON here instead of stdout."
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512, dest="max_tokens")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        dest="max_retries",
        help="Retries on unparsable/invalid model output before giving up.",
    )


def _normalize_argv(argv: List[str]) -> List[str]:
    """Backward compatibility: allow omitting the 'chat' subcommand entirely.

    `--provider ollama --prompt hi` (the pre-subcommand CLI shape) is treated
    as `chat --provider ollama --prompt hi`.
    """
    if not argv:
        return ["chat"]
    if argv[0] in _COMMANDS or argv[0] in ("-h", "--help"):
        return argv
    return ["chat", *argv]


def build_provider(provider_name: str, model: Optional[str]) -> BaseProvider:
    model = model or DEFAULT_MODELS[provider_name]
    if provider_name == "ollama":
        return OllamaProvider(model=model)
    if provider_name == "groq":
        return GroqProvider(model=model)
    raise ValueError(f"Unknown provider: {provider_name}")  # unreachable: argparse restricts choices


def build_messages(system: Optional[str], prompt: str) -> List[ChatMessage]:
    messages: List[ChatMessage] = []
    if system:
        messages.append(ChatMessage(role="system", content=system))
    messages.append(ChatMessage(role="user", content=prompt))
    return messages


def _run_sync(provider: BaseProvider, messages: List[ChatMessage], args: argparse.Namespace) -> int:
    response = provider.chat(messages, temperature=args.temperature, max_tokens=args.max_tokens)
    print(response.text)
    return response.tokens_out


def _run_stream(provider: BaseProvider, messages: List[ChatMessage], args: argparse.Namespace) -> int:
    chunks: List[str] = []
    for chunk in provider.chat_stream(messages, temperature=args.temperature, max_tokens=args.max_tokens):
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print()  # trailing newline once the stream ends
    return count_tokens("".join(chunks))


def _read_text_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FormatError(f"Input file not found: {path}")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise FormatError(f"Could not read input file {path}: {exc}", cause=exc) from exc
    if not text.strip():
        raise FormatError(f"Input file {path} is empty.")
    return text


def _log_and_report(exc: GatewayError, args: argparse.Namespace, tokens_in: int, latency_ms: float) -> None:
    log_request(
        provider=args.provider,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=0,
        temperature=args.temperature,
        status="error",
        error_type=exc.error_type.value,
    )
    # error_type keeps the 5-category log taxonomy; the class name gives the
    # sharper distinction (e.g. SchemaError vs UnsupportedSchemaError vs
    # ExtractionError) without changing what gets logged.
    print(f"[{exc.error_type.value}:{type(exc).__name__}] {exc}", file=sys.stderr)


def _main_chat(args: argparse.Namespace) -> int:
    messages = build_messages(args.system, args.prompt)
    tokens_in = count_message_tokens([m.to_dict() for m in messages])

    timer = Timer().start()
    try:
        provider = build_provider(args.provider, args.model)
        tokens_out = _run_stream(provider, messages, args) if args.stream else _run_sync(provider, messages, args)
    except GatewayError as exc:
        timer.stop()
        _log_and_report(exc, args, tokens_in, timer.elapsed_ms)
        return 1
    except Exception as exc:  # last-resort safety net: the CLI must never crash
        timer.stop()
        gw_exc = to_gateway_error(exc, provider=args.provider)
        _log_and_report(gw_exc, args, tokens_in, timer.elapsed_ms)
        return 1

    timer.stop()
    log_request(
        provider=args.provider,
        latency_ms=timer.elapsed_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        temperature=args.temperature,
        status="success",
        error_type=None,
    )
    return 0


def _main_structured(args: argparse.Namespace) -> int:
    timer = Timer().start()
    tokens_in = 0
    try:
        input_text = _read_text_file(args.input_path)
        schema = load_and_validate_schema(args.schema_path)
        tokens_in = count_tokens(input_text) + count_tokens(json.dumps(schema))

        provider = build_provider(args.provider, args.model)
        result = run_extraction(
            provider,
            input_text,
            schema,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
        )
    except GatewayError as exc:
        timer.stop()
        _log_and_report(exc, args, tokens_in, timer.elapsed_ms)
        return 1
    except Exception as exc:  # last-resort safety net: the CLI must never crash
        timer.stop()
        gw_exc = to_gateway_error(exc, provider=args.provider)
        _log_and_report(gw_exc, args, tokens_in, timer.elapsed_ms)
        return 1

    timer.stop()
    output_json = json.dumps(result.data, indent=2, ensure_ascii=False)
    if args.output_path:
        Path(args.output_path).write_text(output_json + "\n", encoding="utf-8")
        print(f"Wrote extracted data to {args.output_path}")
    else:
        print(output_json)

    log_request(
        provider=args.provider,
        latency_ms=timer.elapsed_ms,
        tokens_in=tokens_in,
        tokens_out=result.tokens_out,
        temperature=args.temperature,
        status="success",
        error_type=None,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(_normalize_argv(raw_argv))

    if args.command == "structured":
        return _main_structured(args)
    return _main_chat(args)


if __name__ == "__main__":
    sys.exit(main())
