"""Multi-backend web search for Prime Agent's IPython kernel.

The module defines `run()`, so the kernel exposes it as an async callable:

    await websearch("prime agent latest release")
    await websearch.backends()
    await websearch.search("query")      # raw SearchResult objects
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Sequence

import httpx

from ._backends import BACKENDS, BackendError, ResultItem, SearchResult
from .config import (
    AUTO_ORDER,
    ENABLE_HINTS,
    MAX_QUERY_CHARS,
    Settings,
    load_settings,
    wants_every_backend,
)

__all__ = ["run", "backends", "search", "SearchResult", "ResultItem", "Outcome"]
__version__ = "0.1.0"

_USER_AGENT = "prime-agent-websearch/0.1 (+https://github.com/sehoon123/prime-agent-websearch)"


@dataclass
class Outcome:
    """Everything one search call produced, including what went wrong."""

    query: str
    settings: Settings
    results: list[SearchResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def skipped(self) -> list[str]:
        return [name for name in self.settings.order if not self.settings.available(name)]

    @property
    def used(self) -> list[str]:
        return [result.backend for result in self.results]


def _redact(text: str, secrets: Sequence[str]) -> str:
    """Replace any credential that leaked into text (e.g. an echoed API error)."""
    for secret in secrets:
        if secret and len(secret) >= 8 and secret in text:
            text = text.replace(secret, "***")
    return text


async def _execute(
    query: str,
    num_results: Optional[int],
    provider: Optional[str],
    model: Optional[str],
    timeout: Optional[float],
) -> Outcome:
    text = (query or "").strip()[:MAX_QUERY_CHARS]
    if not text:
        raise ValueError("query must not be empty")

    settings = load_settings(num_results=num_results, timeout=timeout, provider=provider, model=model)
    secrets = settings.secrets
    every = wants_every_backend(provider)
    outcome = Outcome(query=text, settings=settings)

    eligible = [name for name in settings.order if settings.available(name)]
    if not eligible:
        hints = "; ".join(f"{name}: {ENABLE_HINTS[name]}" for name in settings.order if name in ENABLE_HINTS)
        raise RuntimeError(f"no search backend is configured or reachable. {hints}")

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    async with httpx.AsyncClient(
        timeout=settings.timeout,
        headers={"user-agent": _USER_AGENT},
        limits=limits,
    ) as client:
        for name in eligible:
            try:
                result = await BACKENDS[name](client, text, settings)
            except BackendError as error:
                outcome.failures.append(f"{name}: {_redact(str(error), secrets)}")
                continue
            except Exception as error:  # a backend must never break the kernel
                outcome.failures.append(f"{name}: unexpected {type(error).__name__}")
                continue
            if result.empty:
                outcome.failures.append(f"{name}: no results")
                continue
            result.detail = _redact(result.detail, secrets)
            outcome.results.append(result)
            if not every:
                break

    if not outcome.results:
        raise RuntimeError("all search backends failed: " + " | ".join(outcome.failures))
    return outcome


async def search(
    query: str,
    *,
    num_results: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> list[SearchResult]:
    """Run the query and return raw `SearchResult` objects.

    Use this from Python when the rendered text of `run()` is not what you want.
    Each result carries `backend`, `detail`, `answer`, `items[].{title,url,snippet}`
    and `queries`. Raises `RuntimeError` when every eligible backend fails.
    """
    outcome = await _execute(query, num_results, provider, model, timeout)
    return outcome.results


def _render_result(result: SearchResult, limit: int) -> str:
    head = f"## {result.backend}"
    if result.detail:
        head += f" ({result.detail})"
    lines = [head]

    if result.answer:
        lines += ["", result.answer.strip()]

    if result.items:
        lines += ["", "### Sources"]
        for index, item in enumerate(result.items[:limit], start=1):
            lines.append(f"{index}. {item.title}")
            lines.append(f"   {item.url}")
            if item.snippet:
                lines.append(f"   {item.snippet}")

    if result.queries:
        lines += ["", "### Searches run", "; ".join(result.queries)]
    return "\n".join(lines)


def _render(outcome: Outcome) -> str:
    body = "\n\n".join(_render_result(result, outcome.settings.num_results) for result in outcome.results)
    trailer = [f"used: {', '.join(outcome.used)}"]
    if outcome.skipped:
        trailer.append(f"not configured: {', '.join(outcome.skipped)}")
    if outcome.failures:
        trailer.append("failed: " + " | ".join(outcome.failures))
    return f"# websearch: {outcome.query}\n\n{body}\n---\nbackends " + " · ".join(trailer)


async def run(
    query: str,
    num_results: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    """Search the web and return a grounded answer with source URLs.

    Args:
        query: What to search for.
        num_results: Results to return, 1-20 (default 5, or PRIME_AGENT_WEBSEARCH_NUM_RESULTS).
        provider: "auto" (default) tries backends in order and returns the first
            hit; "all" queries every configured backend; a name or comma-separated
            list restricts it to those. Available: gemini, tavily, brave, serper,
            exa, searxng, ddg.
        model: Pin the Gemini grounding model, e.g. "gemini-2.5-flash".
        timeout: HTTP timeout in seconds (default 45).

    Returns:
        Markdown text: the answer when a backend produced one, numbered sources
        with real URLs, the searches that were run, and which backends were used,
        unconfigured, or failed. Never raises; failures come back as text.
    """
    try:
        outcome = await _execute(query, num_results, provider, model, timeout)
    except (ValueError, RuntimeError) as error:
        return f"websearch failed: {error}"
    return _render(outcome)


async def backends() -> str:
    """List every search backend, whether it is usable here, and how to enable it."""
    settings = load_settings()
    lines = ["# websearch backends", ""]
    for name in AUTO_ORDER:
        ready = settings.available(name)
        detail = ""
        if name == "gemini" and ready:
            detail = ", ".join(
                f"{endpoint.label} ({len(endpoint.keys)} key{'' if len(endpoint.keys) == 1 else 's'})"
                for endpoint in settings.gemini_endpoints
            )
        elif name == "searxng" and ready:
            detail = settings.searxng_url or ""
        lines.append(f"- {'ready ' if ready else 'off   '} {name}" + (f" - {detail}" if detail else ""))
        if not ready:
            lines.append(f"           enable: {ENABLE_HINTS[name]}")
    lines += ["", f"auto order: {', '.join(AUTO_ORDER)}"]
    return "\n".join(lines)


def cli() -> None:  # pragma: no cover - for `python -m websearch` outside the kernel
    import sys

    query = " ".join(sys.argv[1:]).strip()
    print(asyncio.run(run(query) if query else backends()))
