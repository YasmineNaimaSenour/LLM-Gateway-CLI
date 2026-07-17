"""LLM Gateway CLI — provider-agnostic entry point.

Usage:
    python -m src.cli --provider ollama --model llama3.2 --prompt "Hello"
    python -m src.cli --provider groq --model llama-3.1-8b-instant --prompt "Hello" --stream
    python -m src.cli --provider ollama --system "You are terse." --prompt "Explain TCP" \
        --temperature 0.2 --max-tokens 200 --stream

Every call — success or failure — writes one structured JSONL record to
logs/requests.jsonl and never raises past main(): errors are caught,
classified, logged, and reported to stderr with a non-zero exit code.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .core.errors import GatewayError, to_gateway_error
from .core.logger import log_request
from .core.telemetry import Timer
from .providers.base import BaseProvider, ChatMessage
from .providers.groq_provider import GroqProvider
from .providers.ollama_provider import OllamaProvider
from .token_utils import count_message_tokens, count_tokens

DEFAULT_MODELS = {
    "ollama": "llama3.2",
    "groq": "llama-3.1-8b-instant",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-gateway", description="Provider-agnostic LLM gateway CLI.")
    parser.add_argument("--provider", choices=["ollama", "groq"], required=True, help="Which backend to use.")
    parser.add_argument("--model", default=None, help="Model name (defaults per-provider).")
    parser.add_argument("--prompt", required=True, help="User prompt.")
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512, dest="max_tokens")
    parser.add_argument("--stream", action="store_true", help="Stream the response token-by-token.")
    return parser


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
    print(f"[{exc.error_type.value}] {exc}", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
