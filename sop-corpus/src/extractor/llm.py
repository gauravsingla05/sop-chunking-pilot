"""Extractor LLM wrapper — four backends, same interface.

Backends (selected automatically; override with EXTRACTOR_BACKEND env var):

  claude-cli      (default when available, subscription-billed)
      Spawns the locally-installed `claude` Code CLI in `--print
      --output-format json` mode. Counts against your Claude Code
      subscription quota, not the Anthropic API.

  gemini-api      (free tier covers the entire pilot)
      Calls Google's Gemini 2.0 Flash via the google-genai SDK.
      Requires GEMINI_API_KEY (free key from aistudio.google.com).

  openai-api      (cheap pay-per-call: ~$0.20 for the full pilot)
      Calls OpenAI's chat completions API in JSON-mode.
      Requires OPENAI_API_KEY.

  anthropic-api   (opt-in)
      Calls the Anthropic SDK with prompt caching. Used only when
      EXTRACTOR_BACKEND=anthropic-api OR when no other backend is
      available but ANTHROPIC_API_KEY is set.

All four backends accept the same prompt and return the same
`ExtractResult`. Running the pilot twice with different backends
gives the paper a cross-judge robustness check.

Why claude-cli is default:
  - You're already paying for the subscription; double-billing through
    the API on top of that is wasteful.
  - One process spawn per chunk is slower (~2s overhead) but for a
    1000-call pilot the wall-clock is fine and the cost is zero
    incremental.

Why dry-run still exists:
  - The whole pipeline (chunkers, merger, metrics) can be exercised
    offline against `EMPTY_GRAPH` outputs to catch shape bugs cheaply.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .schema import EMPTY_GRAPH
from .prompt import SYSTEM_PROMPT, chunk_user_prompt


# --- Auto-load API keys from a .env file (lightweight, no python-dotenv dep).
# Searched in order: $SOP_DOTENV (explicit override), repo-root/.env,
# the current working directory's .env. Only sets keys not already in env.
_DOTENV_CANDIDATES = [
    Path(os.environ["SOP_DOTENV"]) if os.environ.get("SOP_DOTENV") else None,
    Path(__file__).resolve().parents[3] / ".env",   # repo-root/.env
    Path.cwd() / ".env",
]
_DOTENV_CANDIDATES = [p for p in _DOTENV_CANDIDATES if p is not None]

def _load_dotenv_once():
    for p in _DOTENV_CANDIDATES:
        if not p.is_file():
            continue
        try:
            for raw in p.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
            break   # first hit wins
        except Exception:
            continue

_load_dotenv_once()

log = logging.getLogger("extractor.llm")

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "sonnet")
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4000"))
CLAUDE_CLI_TIMEOUT = int(os.environ.get("CLAUDE_CLI_TIMEOUT", "120"))   # per-call seconds

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

try:
    from google import genai as google_genai
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False

_HAS_CLAUDE_CLI = shutil.which("claude") is not None

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")


def _has_gemini_key() -> bool:
    return bool(
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENAI_API_KEY")
    )


def _select_backend() -> str:
    """Decide which backend to use at import time. The choice is sticky
    for the process so we don't accidentally interleave billing.

    Priority (when no override):
      1. claude-cli  — zero incremental cost via subscription
      2. gemini-api  — free tier
      3. openai-api  — cheap pay-per-call
      4. anthropic-api — pay-per-call (most expensive option)
    """
    forced = os.environ.get("EXTRACTOR_BACKEND", "").strip().lower()
    if forced in ("claude-cli", "gemini-api", "openai-api", "anthropic-api"):
        return forced
    if _HAS_CLAUDE_CLI:
        return "claude-cli"
    if _HAS_GEMINI and _has_gemini_key():
        return "gemini-api"
    if _HAS_OPENAI and os.environ.get("OPENAI_API_KEY"):
        return "openai-api"
    if _HAS_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic-api"
    return "claude-cli"   # let _claude_cli raise a clear error if it's missing


BACKEND = _select_backend()
log.info("extractor backend: %s", BACKEND)


# ----------------------------------------------------------------- types

@dataclass
class ExtractResult:
    graph: dict
    raw_response: Optional[str] = None
    error: Optional[str] = None
    cached_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


# ----------------------------------------------------------------- client

_client = None

def _get_client():
    global _client
    if _client is None:
        if not _HAS_ANTHROPIC:
            raise RuntimeError(
                "anthropic SDK not installed. Run `pip install anthropic`."
            )
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Either export it or use --dry-run."
            )
        _client = anthropic.Anthropic()
    return _client


# ----------------------------------------------------------------- parse

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _parse_json(s: str) -> dict | None:
    """Best-effort JSON parse. The model is asked for naked JSON but we
    handle accidental ```json fences."""
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE.search(s)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Last resort: try to find the first '{' ... matching '}'.
    start = s.find("{")
    if start >= 0:
        depth = 0
        for i, c in enumerate(s[start:], start=start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def _validate_shape(g: dict) -> dict:
    """Defensive normalisation — ensure the four top-level keys exist as
    lists. Extractor downstream code can assume this shape unconditionally."""
    out = dict(EMPTY_GRAPH)
    if isinstance(g, dict):
        for k in EMPTY_GRAPH:
            v = g.get(k)
            out[k] = v if isinstance(v, list) else []
    return out


# ----------------------------------------------------------------- call

def _build_full_prompt(chunk_text: str, chunk_idx: int, doc_title: str) -> str:
    """Both backends use a single combined prompt — the anthropic-api path
    splits it into system+user, but for the CLI we pipe one stream."""
    return SYSTEM_PROMPT + "\n\n" + chunk_user_prompt(
        chunk_text, chunk_idx=chunk_idx, doc_title=doc_title,
    )


def _via_claude_cli(prompt: str, *, model: str) -> ExtractResult:
    """Subprocess-backed call routed through the Claude Code CLI."""
    if not _HAS_CLAUDE_CLI:
        return ExtractResult(
            graph=dict(EMPTY_GRAPH),
            error="claude CLI not found on PATH; install Claude Code or set ANTHROPIC_API_KEY",
        )
    cmd = ["claude", "-p", "--output-format", "json", "--model", model]
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=CLAUDE_CLI_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return ExtractResult(graph=dict(EMPTY_GRAPH), error="claude CLI timeout")
    except Exception as e:
        return ExtractResult(graph=dict(EMPTY_GRAPH), error=f"claude CLI error: {e}")

    if proc.returncode != 0:
        return ExtractResult(
            graph=dict(EMPTY_GRAPH),
            error=f"claude CLI exit={proc.returncode}: {proc.stderr.strip()[:300]}",
        )

    # Outer envelope from `claude --output-format json`.
    try:
        env = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return ExtractResult(
            graph=dict(EMPTY_GRAPH),
            error=f"claude CLI returned non-JSON envelope: {e}; head={proc.stdout[:200]!r}",
            raw_response=proc.stdout,
        )

    if env.get("is_error"):
        return ExtractResult(
            graph=dict(EMPTY_GRAPH),
            error=f"claude reported error: {env.get('subtype')}",
            raw_response=proc.stdout,
        )

    body = env.get("result") or ""
    parsed = _parse_json(body)
    graph = _validate_shape(parsed) if parsed else dict(EMPTY_GRAPH)

    usage = env.get("usage") or {}
    return ExtractResult(
        graph=graph,
        raw_response=body,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cached_tokens=usage.get("cache_read_input_tokens"),
    )


def _via_anthropic_api(prompt_user: str, *, model: str, max_tokens: int) -> ExtractResult:
    """Direct Anthropic SDK call. Uses prompt caching on the system slot."""
    client = _get_client()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt_user}],
        )
    except Exception as e:
        msg = str(e).lower()
        if any(t in msg for t in ("overloaded", "rate_limit", "timeout", "connection")):
            time.sleep(2.5)
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=[{
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    messages=[{"role": "user", "content": prompt_user}],
                )
            except Exception as e2:
                return ExtractResult(graph=dict(EMPTY_GRAPH), error=str(e2))
        else:
            return ExtractResult(graph=dict(EMPTY_GRAPH), error=str(e))

    text = ""
    for block in (resp.content or []):
        if getattr(block, "type", None) == "text":
            text += block.text

    parsed = _parse_json(text)
    graph = _validate_shape(parsed) if parsed else dict(EMPTY_GRAPH)

    usage = getattr(resp, "usage", None)
    return ExtractResult(
        graph=graph,
        raw_response=text,
        cached_tokens=getattr(usage, "cache_read_input_tokens", None) if usage else None,
        input_tokens=getattr(usage, "input_tokens", None) if usage else None,
        output_tokens=getattr(usage, "output_tokens", None) if usage else None,
    )


_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not _HAS_OPENAI:
            raise RuntimeError("openai SDK not installed. Run `pip install openai`.")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set.")
        _openai_client = openai.OpenAI()
    return _openai_client


def _via_openai_api(prompt_user: str, *, model: str, max_tokens: int) -> ExtractResult:
    """OpenAI chat completions in JSON-object mode. Cheap and fast; no
    server-side caching but the system prompt is small enough that it
    doesn't matter for ~1k calls."""
    client = _get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt_user},
            ],
        )
    except Exception as e:
        msg = str(e).lower()
        if any(t in msg for t in ("overload", "rate_limit", "timeout", "connection", "429")):
            time.sleep(2.5)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt_user},
                    ],
                )
            except Exception as e2:
                return ExtractResult(graph=dict(EMPTY_GRAPH), error=str(e2))
        else:
            return ExtractResult(graph=dict(EMPTY_GRAPH), error=str(e))

    text = ""
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        pass

    parsed = _parse_json(text)
    graph = _validate_shape(parsed) if parsed else dict(EMPTY_GRAPH)

    usage = getattr(resp, "usage", None)
    return ExtractResult(
        graph=graph,
        raw_response=text,
        input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        cached_tokens=getattr(usage, "prompt_cached_tokens", None) if usage else None,
    )


_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if not _HAS_GEMINI:
            raise RuntimeError(
                "google-genai SDK not installed. Run `pip install google-genai`."
            )
        if not _has_gemini_key():
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY / GOOGLE_GENAI_API_KEY) not set. "
                "Get a free key at https://aistudio.google.com/apikey."
            )
        key = (os.environ.get("GEMINI_API_KEY")
               or os.environ.get("GOOGLE_API_KEY")
               or os.environ.get("GOOGLE_GENAI_API_KEY"))
        # Hard timeout — pilot-001-gemini got stuck for 18h on a single
        # call with no timeout configured. 90s is generous for Flash but
        # bounded enough that one bad call doesn't tank the whole run.
        _gemini_client = google_genai.Client(
            api_key=key,
            http_options=google_genai.types.HttpOptions(timeout=90_000),  # ms
        )
    return _gemini_client


def _via_gemini_api(prompt_user: str, *, model: str, max_tokens: int) -> ExtractResult:
    """Gemini 2.0 Flash with response_mime_type=application/json so the
    model returns valid JSON without prose. System prompt is included as
    a system_instruction (small enough we don't bother caching)."""
    client = _get_gemini_client()
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt_user,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "max_output_tokens": max_tokens,
                "temperature": 0,
            },
        )
    except Exception as e:
        msg = str(e).lower()
        if any(t in msg for t in ("overload", "rate", "timeout", "deadline", "429", "503")):
            time.sleep(2.5)
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt_user,
                    config={
                        "system_instruction": SYSTEM_PROMPT,
                        "response_mime_type": "application/json",
                        "max_output_tokens": max_tokens,
                        "temperature": 0,
                    },
                )
            except Exception as e2:
                return ExtractResult(graph=dict(EMPTY_GRAPH), error=str(e2))
        else:
            return ExtractResult(graph=dict(EMPTY_GRAPH), error=str(e))

    text = ""
    try:
        text = (resp.text or "").strip()
    except Exception:
        # Defensive: some response shapes need a different accessor
        try:
            text = resp.candidates[0].content.parts[0].text
        except Exception:
            text = ""

    parsed = _parse_json(text)
    graph = _validate_shape(parsed) if parsed else dict(EMPTY_GRAPH)

    usage = getattr(resp, "usage_metadata", None)
    return ExtractResult(
        graph=graph,
        raw_response=text,
        input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
        output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
        cached_tokens=getattr(usage, "cached_content_token_count", None) if usage else None,
    )


def extract_from_chunk(
    chunk_text: str,
    *,
    chunk_idx: int,
    doc_title: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    dry_run: bool = False,
) -> ExtractResult:
    """Unified entry point. Routes to the configured backend."""
    if dry_run:
        return ExtractResult(graph=dict(EMPTY_GRAPH), raw_response="<dry-run>")

    user = chunk_user_prompt(chunk_text, chunk_idx=chunk_idx, doc_title=doc_title)

    if BACKEND == "claude-cli":
        prompt = _build_full_prompt(chunk_text, chunk_idx, doc_title)
        return _via_claude_cli(prompt, model=model)
    if BACKEND == "gemini-api":
        return _via_gemini_api(user, model=GEMINI_MODEL, max_tokens=max_tokens)
    if BACKEND == "openai-api":
        return _via_openai_api(user, model=OPENAI_MODEL, max_tokens=max_tokens)
    if BACKEND == "anthropic-api":
        return _via_anthropic_api(user, model=model, max_tokens=max_tokens)
    return ExtractResult(graph=dict(EMPTY_GRAPH), error=f"unknown backend {BACKEND!r}")
