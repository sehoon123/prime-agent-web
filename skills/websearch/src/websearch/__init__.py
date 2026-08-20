"""Multi-backend web search for Prime Agent's IPython kernel.

The module defines `run()`, so the kernel exposes it as an async callable:

    await websearch("prime agent latest release")
    await websearch("kernel panic", recency="week", domains=["lwn.net"])
    await websearch.backends()
    await websearch.search("query")      # raw SearchResult objects

Batching is plain Python - the kernel is the composition layer:

    import asyncio
    answers = await asyncio.gather(*(websearch(q) for q in queries))
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import httpx

from ._backends import BACKENDS, BackendError, ResultItem, SearchResult
from .config import (
    AUTO_ORDER,
    ENABLE_HINTS,
    MAX_QUERY_CHARS,
    RECENCY_VALUES,
    SearchQuery,
    Settings,
    load_settings,
    parse_domains,
    parse_recency,
    wants_every_backend,
)

__all__ = ["run", "backends", "search", "SearchResult", "ResultItem", "Outcome", "clear_cache"]
__version__ = "0.2.0"

_USER_AGENT = "prime-agent-websearch/0.2 (+https://github.com/sehoon123/prime-agent-websearch)"

# Repeated identical searches inside one kernel session are common in agent loops
# (retries, reformulations, subagents). Cache them briefly to protect quota.
_CACHE: dict[tuple[Any, ...], tuple[float, "Outcome"]] = {}
_CACHE_MAX_ENTRIES = 64


@dataclass
class Outcome:
    """Everything one search call produced, including what went wrong."""

    query: SearchQuery
    settings: Settings
    results: list[SearchResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    cached: bool = False

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


def clear_cache() -> None:
    """Drop the in-process result cache."""
    _CACHE.clear()


def _cache_get(key: tuple[Any, ...], ttl: float) -> Optional[Outcome]:
    if ttl <= 0:
        return None
    entry = _CACHE.get(key)
    if not entry:
        return None
    stored_at, outcome = entry
    if time.monotonic() - stored_at > ttl:
        _CACHE.pop(key, None)
        return None
    return outcome


def _cache_put(key: tuple[Any, ...], outcome: Outcome, ttl: float) -> None:
    if ttl <= 0:
        return
    if len(_CACHE) >= _CACHE_MAX_ENTRIES:
        oldest = min(_CACHE, key=lambda existing: _CACHE[existing][0])
        _CACHE.pop(oldest, None)
    _CACHE[key] = (time.monotonic(), outcome)


async def _attempt(
    client: httpx.AsyncClient,
    name: str,
    query: SearchQuery,
    settings: Settings,
    secrets: Sequence[str],
) -> tuple[Optional[SearchResult], Optional[str]]:
    """Run one backend. Returns (result, failure) with exactly one set."""
    try:
        result = await BACKENDS[name](client, query, settings)
    except BackendError as error:
        return None, f"{name}: {_redact(str(error), secrets)}"
    except Exception as error:  # a backend must never break the kernel
        return None, f"{name}: unexpected {type(error).__name__}"
    if result.empty:
        return None, f"{name}: no results"
    result.detail = _redact(result.detail, secrets)
    return result, None


async def _execute(
    query: str,
    num_results: Optional[int],
    provider: Optional[str],
    model: Optional[str],
    timeout: Optional[float],
    recency: Optional[str],
    domains: Union[str, Sequence[str], None],
) -> Outcome:
    text = (query or "").strip()[:MAX_QUERY_CHARS]
    if not text:
        raise ValueError("query must not be empty")

    settings = load_settings(num_results=num_results, timeout=timeout, provider=provider, model=model)
    include, exclude = parse_domains(domains)
    request = SearchQuery(
        text=text,
        num_results=settings.num_results,
        recency=parse_recency(recency),
        include_domains=include,
        exclude_domains=exclude,
    )
    every = wants_every_backend(provider)

    eligible = [name for name in settings.order if settings.available(name)]
    if not eligible:
        hints = "; ".join(f"{name}: {ENABLE_HINTS[name]}" for name in settings.order if name in ENABLE_HINTS)
        raise RuntimeError(f"no search backend is configured or reachable. {hints}")

    cache_key = (*request.cache_key, tuple(eligible), every)
    cached = _cache_get(cache_key, settings.cache_ttl)
    if cached is not None:
        return Outcome(
            query=cached.query,
            settings=settings,
            results=cached.results,
            failures=cached.failures,
            cached=True,
        )

    secrets = settings.secrets
    outcome = Outcome(query=request, settings=settings)
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    async with httpx.AsyncClient(
        timeout=settings.timeout,
        headers={"user-agent": _USER_AGENT},
        limits=limits,
    ) as client:
        if every:
            # Fan out concurrently: latency is one backend, not the sum.
            attempts = await asyncio.gather(
                *(_attempt(client, name, request, settings, secrets) for name in eligible)
            )
            for result, failure in attempts:
                if result is not None:
                    outcome.results.append(result)
                elif failure:
                    outcome.failures.append(failure)
        else:
            for name in eligible:
                result, failure = await _attempt(client, name, request, settings, secrets)
                if result is not None:
                    outcome.results.append(result)
                    break
                if failure:
                    outcome.failures.append(failure)

    if not outcome.results:
        raise RuntimeError("all search backends failed: " + " | ".join(outcome.failures))
    _cache_put(cache_key, outcome, settings.cache_ttl)
    return outcome


async def search(
    query: str,
    *,
    num_results: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    recency: Optional[str] = None,
    domains: Union[str, Sequence[str], None] = None,
) -> list[SearchResult]:
    """Run the query and return raw `SearchResult` objects.

    Use this from Python when the rendered text of `run()` is not what you want.
    Each result carries `backend`, `detail`, `answer`, `items[].{title,url,snippet}`,
    `queries` and `dropped`. Raises `RuntimeError` when every eligible backend fails.
    """
    outcome = await _execute(query, num_results, provider, model, timeout, recency, domains)
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
    if result.dropped:
        lines.append(f"\n({result.dropped} result(s) removed by the domain filter)")
    return "\n".join(lines)


def _render(outcome: Outcome) -> str:
    query = outcome.query
    header = f"# websearch: {query.text}"
    constraints = []
    if query.recency:
        constraints.append(f"last {query.recency}")
    if query.include_domains:
        constraints.append("only " + ", ".join(query.include_domains))
    if query.exclude_domains:
        constraints.append("excluding " + ", ".join(query.exclude_domains))
    if constraints:
        header += f"  ({'; '.join(constraints)})"

    body = "\n\n".join(_render_result(result, query.num_results) for result in outcome.results)

    trailer = [f"used: {', '.join(outcome.used)}"]
    if outcome.cached:
        trailer.append("from cache")
    if outcome.skipped:
        trailer.append(f"not configured: {', '.join(outcome.skipped)}")
    if outcome.failures:
        trailer.append("failed: " + " | ".join(outcome.failures))
    return f"{header}\n\n{body}\n---\nbackends " + " · ".join(trailer)


async def run(
    query: str,
    num_results: Optional[int] = None,
    provider: Optional[str] = None,
    recency: Optional[str] = None,
    domains: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    """Search the web and return a grounded answer with source URLs.

    Args:
        query: What to search for.
        num_results: Results to return, 1-20 (default 5, or PRIME_AGENT_WEBSEARCH_NUM_RESULTS).
        provider: "auto" (default) tries backends in order and returns the first
            hit; "all" queries every configured backend concurrently; a name or
            comma-separated list restricts it to those. Available: gemini, tavily,
            brave, serper, exa, searxng, ddg.
        recency: Only results from the last "day", "week", "month" or "year".
        domains: Comma-separated domains to restrict to; prefix one with "-" to
            exclude it, e.g. "github.com,-reddit.com".
        model: Pin the Gemini grounding model, e.g. "gemini-2.5-flash".
        timeout: HTTP timeout in seconds (default 45).

    Returns:
        Markdown text: the answer when a backend produced one (with [n] citation
        markers where the provider reported them), numbered sources with real URLs,
        the searches that were run, and which backends were used, unconfigured, or
        failed. Never raises; failures come back as text.
    """
    try:
        outcome = await _execute(query, num_results, provider, model, timeout, recency, domains)
    except (ValueError, RuntimeError) as error:
        return f"websearch failed: {error}"
    return _render(outcome)


async def backends() -> str:
    """List every search backend, whether it is usable here, and how to enable it."""
    settings = load_settings()
    lines = ["# websearch backends", ""]
    for name in AUTO_ORDER:
        ready = settings.available(name)
        detail = settings.describe(name) if ready else ""
        lines.append(f"- {'ready ' if ready else 'off   '} {name}" + (f" - {detail}" if detail else ""))
        if not ready:
            lines.append(f"           enable: {ENABLE_HINTS[name]}")
    lines += [
        "",
        f"auto order: {', '.join(AUTO_ORDER)}",
        f"recency values: {', '.join(RECENCY_VALUES)}",
        f"cache: {'off' if settings.cache_ttl <= 0 else f'{settings.cache_ttl:g}s in-process'}",
    ]
    return "\n".join(lines)


def cli() -> None:  # pragma: no cover - for `python -m websearch` outside the kernel
    import sys

    query = " ".join(sys.argv[1:]).strip()
    print(asyncio.run(run(query) if query else backends()))
