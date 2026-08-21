"""Multi-backend web search for Prime Agent's IPython kernel.

The module defines `run()`, so the kernel exposes it as an async callable:

    await websearch("prime agent latest release")
    await websearch("kernel panic", recency="week", domains=["lwn.net"])
    await websearch.backends()
    await websearch.search("query")      # raw SearchResult objects

Batching is plain Python - the kernel is the composition layer:

    import asyncio
    answers = await asyncio.gather(*(websearch(q) for q in queries))

Ordering learns from this session's own trajectory: a backend that keeps
failing sits out a doubling cooldown, and one success restores it.

    await websearch.run("first query")   # gemini fails -> cools down
    await websearch.run("another query") # skips straight to the next backend
    print(await websearch.health())     # the evidence behind that order
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field, replace
from html import unescape as html_unescape
from typing import Any, Optional, Sequence, Union

import httpx

from . import _health
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

__all__ = [
    "run",
    "backends",
    "health",
    "reset_health",
    "search",
    "SearchResult",
    "ResultItem",
    "Outcome",
    "clear_cache",
]
__version__ = "0.6.3"

_USER_AGENT = "prime-agent-websearch/0.6.3 (+https://github.com/sehoon123/prime-agent-web)"

# Session-scoped backend health, refined from what actually happened here.
_HEALTH = _health.HealthTracker()
_HEALTH_GENERATION = 0

# Repeated identical searches inside one kernel session are common in agent loops
# (retries, reformulations, subagents). Cache them briefly to protect quota.
_CACHE: dict[tuple[Any, ...], tuple[float, "Outcome"]] = {}
_CACHE_MAX_ENTRIES = 64
_CACHE_GENERATION = 0
_CACHE_DIGEST_KEY = os.urandom(32)


@dataclass
class Outcome:
    """Everything one search call produced, including what went wrong."""

    query: SearchQuery
    settings: Settings
    results: list[SearchResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    """Backends skipped as cooling down on this call. A cache hit never sets it:
    nothing was attempted, so claiming skips would be stale information."""
    cached: bool = False

    @property
    def skipped(self) -> list[str]:
        return [name for name in self.settings.order if not self.settings.available(name)]

    @property
    def used(self) -> list[str]:
        return [result.backend for result in self.results]


def _copy_search_result(result: SearchResult) -> SearchResult:
    """Copy one result and every mutable object reachable from it."""
    return replace(
        result,
        items=[replace(item) for item in result.items],
        queries=list(result.queries),
    )


def _copy_outcome(outcome: Outcome) -> Outcome:
    """Keep callers from mutating the in-process cache through raw results."""
    return replace(
        outcome,
        results=[_copy_search_result(result) for result in outcome.results],
        failures=list(outcome.failures),
        deferred=list(outcome.deferred),
    )


_ENTITY = re.compile(r"&(?:#[xX][0-9A-Fa-f]+|#[0-9]+|[A-Za-z][A-Za-z0-9]+);")


def _decode_control_entities(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        decoded = html_unescape(match.group(0))
        if decoded != match.group(0) and any(
            character.isspace()
            or unicodedata.category(character) in {"Cc", "Zl", "Zp"}
            or (
                unicodedata.category(character) == "Cf"
                and character not in {"\u200c", "\u200d"}
            )
            for character in decoded
        ):
            return decoded
        return match.group(0)

    return _ENTITY.sub(replace, text)


def _redact(text: str, secrets: Sequence[str]) -> str:
    """Replace credentials longest-first and remove terminal control bytes."""
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            text = text.replace(secret, "***")
    text = _decode_control_entities(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n").replace("\t", " ")
    text = "".join(
        character
        for character in text
        if unicodedata.category(character) != "Cf"
        or character in {"\u200c", "\u200d"}
    )
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

def clear_cache() -> None:
    """Drop the cache and prevent already-running searches from repopulating it."""
    global _CACHE_GENERATION
    _CACHE_GENERATION += 1
    _CACHE.clear()


def _now() -> float:
    """Monotonic clock, indirected so tests can control cache expiry exactly.

    `time.monotonic()` counts from an arbitrary origin (uptime on Linux), so a
    test must never assume a specific absolute value.
    """
    return time.monotonic()


def _configuration_fingerprint(settings: Settings, eligible: Sequence[str]) -> str:
    """Hash content-affecting endpoint configuration without storing credentials."""
    material: list[Any] = []
    if "gemini" in eligible:
        material.append(
            (
                "gemini",
                tuple(
                    (endpoint.label, endpoint.base_url, endpoint.models)
                    for endpoint in settings.gemini_endpoints
                ),
            )
        )
    if "searxng" in eligible:
        material.append(("searxng", settings.searxng_url))
    material.append(("credentials", settings.secrets))
    return hashlib.blake2s(
        repr(material).encode("utf-8"), key=_CACHE_DIGEST_KEY, digest_size=8
    ).hexdigest()


def _cache_get(key: tuple[Any, ...], ttl: float) -> Optional[Outcome]:
    if ttl <= 0:
        return None
    entry = _CACHE.get(key)
    if not entry:
        return None
    stored_at, outcome = entry
    if _now() - stored_at > ttl:
        _CACHE.pop(key, None)
        return None
    return _copy_outcome(outcome)


def _cache_put(key: tuple[Any, ...], outcome: Outcome, ttl: float) -> None:
    if ttl <= 0:
        return
    if key not in _CACHE and len(_CACHE) >= _CACHE_MAX_ENTRIES:
        oldest = min(_CACHE, key=lambda existing: _CACHE[existing][0])
        _CACHE.pop(oldest, None)
    cached = _copy_outcome(outcome)
    # Results need no credentials after the request. Do not retain auth material
    # merely because a successful answer remains cached for a few minutes.
    cached.settings = replace(cached.settings, auth={}, searxng_url=None)
    cached.failures = []
    cached.deferred = []
    _CACHE[key] = (_now(), cached)


def _failure_text(name: str, message: str, secrets: Sequence[str]) -> str:
    """Prefix a backend exactly once and redact any echoed credential."""
    text = _redact(message.strip(), secrets)
    lowered = text.casefold()
    prefix = name.casefold()
    if lowered == prefix or lowered.startswith((prefix + ":", prefix + " ")):
        return text
    return f"{name}: {text}"


def _redact_result(result: SearchResult, secrets: Sequence[str]) -> SearchResult:
    """Redact provider-controlled text before it is rendered or cached."""
    result.detail = _redact(result.detail, secrets)
    if result.answer is not None:
        result.answer = _redact(result.answer, secrets)
    kept_items: list[ResultItem] = []
    removed_credential_url = False
    for item in result.items:
        if any(secret and secret in item.url for secret in secrets):
            result.dropped += 1
            removed_credential_url = True
            continue
        item.title = _redact(item.title, secrets)
        item.snippet = _redact(item.snippet, secrets)
        kept_items.append(item)
    result.items = kept_items
    # Once a supporting URL is removed, provider citation numbers are no longer
    # trustworthy. Keep the safe sources but discard the generated answer.
    if removed_credential_url:
        result.answer = None
    result.queries = [_redact(query, secrets) for query in result.queries]
    return result


async def _attempt(
    client: httpx.AsyncClient,
    name: str,
    query: SearchQuery,
    settings: Settings,
    secrets: Sequence[str],
    health_generation: int,
) -> tuple[Optional[SearchResult], Optional[str]]:
    """Run one backend and update health at the instant it completes.

    Recording here preserves real completion order across concurrent searches.
    An empty result set is neutral: it neither cools nor restores a backend.
    """
    try:
        result = await BACKENDS[name](client, query, settings)
    except BackendError as error:
        failure = _failure_text(name, str(error), secrets)
        if health_generation == _HEALTH_GENERATION:
            _HEALTH.record_failure(name, failure, settings.cooldown_base)
        return None, failure
    except Exception as error:  # a backend must never break the kernel
        failure = f"{name}: unexpected {type(error).__name__}"
        if health_generation == _HEALTH_GENERATION:
            _HEALTH.record_failure(name, failure, settings.cooldown_base)
        return None, failure
    if result.empty:
        return None, f"{name}: no results"
    _redact_result(result, secrets)
    if result.empty:
        return None, f"{name}: no safe results after credential redaction"
    if health_generation == _HEALTH_GENERATION:
        _HEALTH.record_success(name)
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
    health_generation = _HEALTH_GENERATION
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

    # Adaptive order from this session's trajectory: backends that keep failing
    # sit their cooldown out. Explicit requests are never refused - naming
    # providers or asking for "all" overrides health and attempts everything.
    health_enabled = settings.cooldown_base > 0
    if not health_enabled:
        # Clear old deadlines as soon as the setting is disabled. Otherwise a
        # later re-enable could resurrect a cooldown that the user turned off.
        _HEALTH.partition(eligible, enabled=False)
    if every:
        deferred: list[str] = []
        attempt_order = eligible
    else:
        ready, cooling = _HEALTH.partition(eligible, enabled=health_enabled)
        deferred = cooling if ready else []
        attempt_order = ready or eligible

    cache_generation = _CACHE_GENERATION
    effective_model = settings.gemini_model if "gemini" in eligible else None
    configuration = _configuration_fingerprint(settings, eligible)
    cache_key = (*request.cache_key, tuple(eligible), every, effective_model, configuration)
    cached = _cache_get(cache_key, settings.cache_ttl)
    if cached is not None:
        return Outcome(
            query=cached.query,
            settings=settings,
            results=cached.results,
            cached=True,
        )

    secrets = settings.secrets
    outcome = Outcome(query=request, settings=settings, deferred=deferred)
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    async with httpx.AsyncClient(
        timeout=settings.timeout,
        headers={"user-agent": _USER_AGENT},
        limits=limits,
    ) as client:
        if every:
            # Fan out concurrently: latency is one backend, not the sum.
            attempts = await asyncio.gather(
                *(
                    _attempt(
                        client, name, request, settings, secrets, health_generation
                    )
                    for name in attempt_order
                )
            )
            for result, failure in attempts:
                if result is not None:
                    outcome.results.append(result)
                elif failure:
                    outcome.failures.append(failure)
        else:
            for name in attempt_order:
                result, failure = await _attempt(
                    client, name, request, settings, secrets, health_generation
                )
                if result is not None:
                    outcome.results.append(result)
                    break
                if failure:
                    outcome.failures.append(failure)

        # Last resort before giving up: the ready chain produced nothing, so
        # every cooled backend gets one try. This keeps the pre-0.6 guarantee
        # that an automatic search only fails after every configured backend
        # has seen the query.
        if deferred and not outcome.results:
            for name in list(deferred):
                outcome.deferred.remove(name)
                result, failure = await _attempt(
                    client, name, request, settings, secrets, health_generation
                )
                if result is not None:
                    outcome.results.append(result)
                    break
                if failure:
                    outcome.failures.append(failure)

    if not outcome.results:
        raise RuntimeError("all search backends failed: " + " | ".join(outcome.failures))
    if cache_generation == _CACHE_GENERATION:
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
        lines.append(
            f"\n({result.dropped} result(s) removed by URL safety, "
            "credential redaction, or the domain filter)"
        )
    return "\n".join(lines)


def _single_line(value: str) -> str:
    value = _decode_control_entities(value)
    normalized = "".join(
        " "
        if character.isspace()
        or unicodedata.category(character) in {"Cc", "Zl", "Zp"}
        or (
            unicodedata.category(character) == "Cf"
            and character not in {"\u200c", "\u200d"}
        )
        else character
        for character in value
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _render(outcome: Outcome) -> str:
    query = outcome.query
    header = f"# websearch: {_single_line(query.text)}"
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
    if outcome.deferred:
        trailer.append("cooled: " + ", ".join(outcome.deferred) + " (see websearch.health())")
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
            comma- or whitespace-separated list restricts it to those. Available:
            gemini, tavily, brave, serper, exa, searxng, ddg.
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
    except (TypeError, ValueError, RuntimeError) as error:
        return f"websearch failed: {error}"
    except Exception as error:
        return f"websearch failed: unexpected {type(error).__name__}"
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
        (
            "cooldown: off"
            if settings.cooldown_base <= 0
            else f"cooldown: {settings.cooldown_base:g}s base, doubling per consecutive failure"
        ),
    ]
    return "\n".join(lines)


async def health() -> str:
    """Show per-backend health for this session and the evidence behind it."""
    settings = load_settings()
    if settings.cooldown_base <= 0:
        _HEALTH.partition(list(AUTO_ORDER), enabled=False)
    return _health.render(_HEALTH, list(AUTO_ORDER), base=settings.cooldown_base)


def reset_health() -> None:
    """Clear health and ignore completions from searches already in progress."""
    global _HEALTH_GENERATION
    _HEALTH_GENERATION += 1
    _HEALTH.reset()


def cli() -> None:  # pragma: no cover - for `python -m websearch` outside the kernel
    import sys

    query = " ".join(sys.argv[1:]).strip()
    print(asyncio.run(run(query) if query else backends()))
