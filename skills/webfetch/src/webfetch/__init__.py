"""Fetch a URL as readable markdown for Prime Agent's IPython kernel.

The module defines `run()`, so the kernel exposes it as an async callable:

    print(await webfetch("https://docs.example.com/guide"))
    doc = await webfetch.fetch("https://arxiv.org/pdf/2605.09998")
    docs = await webfetch.fetch([url_a, url_b])           # concurrent

`run()` renders a bounded summary for reading; `fetch()` returns `Document`
objects with all text extracted under the configured body cap, because slicing and
storing belong in the kernel.

Successful fetches are kept in a small session cache (default TTL 300s,
PRIME_AGENT_WEBFETCH_CACHE_TTL): later retries and reformulations can reuse the
same extracted Document instead of fetching it again. Concurrent duplicate calls
are independent. `webfetch.clear_cache()` drops completed entries.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence, Union
from urllib.parse import urlsplit

import httpx

from . import _gemini
from ._extract import (
    Extracted,
    html_to_markdown,
    html_to_text,
    limit_pdf_pages,
    looks_like_html,
    looks_like_pdf,
    looks_like_text,
    pdf_to_text,
    rewrite_url,
    save_binary,
    tidy,
)
from ._robots import RobotsCache, RobotsDeniedError
from ._safety import (
    DEFAULT_MAX_BYTES,
    USER_AGENT_AUTONOMOUS,
    USER_AGENT_MANUAL,
    FetchError,
    Resolver,
    TooLargeError,
    UnsafeUrlError,
    check_host_resolves_public,
    check_url_syntax,
    default_resolver,
    guarded_get,
)

__all__ = [
    "run",
    "fetch",
    "Document",
    "clear_cache",
    "FetchError",
    "UnsafeUrlError",
    "TooLargeError",
    "gemini_available",
]
__version__ = "0.6.2"

MODES = ("markdown", "text", "raw")
DEFAULT_MAX_CHARS = 20_000
DEFAULT_TIMEOUT = 45.0
MAX_CONCURRENCY = 6

# Successful fetches are reused within this kernel session. Errors are never
# cached - they are exactly what a retry should retry.
_DOC_CACHE: dict[tuple[Any, ...], tuple[float, Document]] = {}
_DOC_CACHE_TTL_DEFAULT = 300.0
_DOC_CACHE_MAX_ENTRIES = 32
_DOC_CACHE_MAX_CHARS = 2_000_000
_DOC_CACHE_GENERATION = 0


def _doc_cache_clock() -> float:
    """Monotonic clock, indirected so tests can control expiry exactly."""
    return time.monotonic()


def clear_cache() -> None:
    """Drop the cache and prevent already-running fetches from repopulating it."""
    global _DOC_CACHE_GENERATION
    _DOC_CACHE_GENERATION += 1
    _DOC_CACHE.clear()


def _doc_cache_ttl() -> float:
    return max(0.0, _env_float("PRIME_AGENT_WEBFETCH_CACHE_TTL", _DOC_CACHE_TTL_DEFAULT))


def _cache_key(
    target: str,
    mode: str,
    prompt: Optional[str],
    gemini: Optional[bool],
    model: Optional[str],
    cap: int,
    max_pages: Optional[int],
    autonomous: bool,
    gemini_fingerprint: str,
) -> tuple[Any, ...]:
    """Only arguments that change content or fetch policy belong in the key."""
    return (
        target.strip(),
        mode,
        prompt or "",
        gemini,
        model or "",
        cap,
        max_pages,
        autonomous,
        gemini_fingerprint,
    )


def _copy_document(document: Document) -> Document:
    """Shallow copy whose mutable list fields are duplicated.

    Agents annotate Documents freely - notes especially. Without this, one
    caller's annotations would leak into the cache and then into every later
    hit, and hits would mutate what earlier callers still hold.
    """
    return replace(
        document,
        notes=list(document.notes),
        retrieved_urls=list(document.retrieved_urls),
    )


def _doc_cache_get(key: tuple[Any, ...], ttl: float) -> Optional[Document]:
    if ttl <= 0:
        return None
    entry = _DOC_CACHE.get(key)
    if entry is None:
        return None
    stored_at, document = entry
    if _doc_cache_clock() - stored_at > ttl:
        _DOC_CACHE.pop(key, None)
        return None
    # A copy per hit, annotated for provenance.
    hit = _copy_document(document)
    hit.notes.append("from session cache")
    return hit


def _doc_cache_put(key: tuple[Any, ...], document: Document) -> None:
    # A saved binary is an external mutable file, not an isolated value. If the
    # caller deletes or overwrites it, a cached Document would point at damage.
    if (
        document.saved_path is not None
        or len(document.text) > _DOC_CACHE_MAX_CHARS
        or any(note.startswith("Gemini scan failed (") for note in document.notes)
    ):
        return
    if key not in _DOC_CACHE and len(_DOC_CACHE) >= _DOC_CACHE_MAX_ENTRIES:
        oldest = min(_DOC_CACHE, key=lambda existing: _DOC_CACHE[existing][0])
        _DOC_CACHE.pop(oldest, None)
    _DOC_CACHE[key] = (_doc_cache_clock(), _copy_document(document))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ[name])
    except (KeyError, ValueError):
        return default
    return value if math.isfinite(value) else default


@dataclass
class Document:
    """One document; rendering may truncate `text`, and the body cap may limit it."""

    url: str
    """The URL as requested."""
    final_url: str
    """Where the request actually landed after redirects and rewrites."""
    kind: str
    """html, pdf, text, binary, answer, or error."""
    text: str = ""
    title: Optional[str] = None
    content_type: str = ""
    status: int = 0
    bytes_len: int = 0
    pages: Optional[int] = None
    saved_path: Optional[str] = None
    error: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    answer: Optional[str] = None
    """Model answer, when a prompt was given or a page needed model extraction."""
    source: str = "local"
    """local, gemini-url-context, gemini-video, or gemini-pdf."""
    retrieved_urls: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    def __len__(self) -> int:
        return len(self.text)


def _extract_body(body_bytes: bytes, content_type: str, text: str, mode: str, max_pages: Optional[int]) -> Extracted:
    if mode == "raw":
        if looks_like_html(body_bytes, content_type):
            return Extracted(kind="html", text=text)
        if looks_like_text(body_bytes, content_type):
            return Extracted(kind="text", text=text)
        # A byte body cannot be represented losslessly in Document.text; use the
        # same secure file path as other complete binaries.
        return Extracted(kind="binary", text="")
    if looks_like_pdf(body_bytes, content_type):
        return pdf_to_text(body_bytes, max_pages=max_pages)
    if looks_like_html(body_bytes, content_type):
        if mode == "text":
            return html_to_text(text)
        return html_to_markdown(text)
    if looks_like_text(body_bytes, content_type):
        return Extracted(kind="text", text=text if mode == "raw" else tidy(text))
    return Extracted(kind="binary", text="")


async def _gemini_document(
    client: httpx.AsyncClient,
    url: str,
    source: str,
    coroutine: Any,
    notes: list[str],
) -> Document:
    """Wrap a Gemini call as a Document, keeping failures inside the Document."""
    try:
        answer = await coroutine
    except _gemini.GeminiUnavailable as error:
        return Document(url=url, final_url=url, kind="error", error=str(error), notes=notes)
    except RuntimeError as error:
        return Document(
            url=url, final_url=url, kind="error", error=f"{source} failed: {error}", notes=notes
        )
    except Exception as error:
        return Document(
            url=url,
            final_url=url,
            kind="error",
            error=f"{source} failed unexpectedly: {type(error).__name__}",
            notes=notes,
        )
    return Document(
        url=url,
        final_url=url,
        kind="answer",
        text=answer.text,
        content_type="",
        source=source,
        answer=answer.text,
        retrieved_urls=answer.retrieved_urls,
        notes=notes + [f"{source} via {answer.detail}"],
    )


async def _fetch_one(
    client: httpx.AsyncClient,
    url: str,
    mode: str,
    max_bytes: int,
    max_pages: Optional[int],
    resolver: Optional[Resolver],
    robots: Optional[RobotsCache],
    prompt: Optional[str],
    gemini: Optional[bool],
    model: Optional[str],
    timeout: Optional[float],
) -> Document:
    try:
        original = check_url_syntax(url)
    except UnsafeUrlError as error:
        return Document(url=url, final_url=url, kind="error", error=str(error))
    target, rewrite_note = rewrite_url(original)
    notes = [rewrite_note] if rewrite_note else []
    allow_gemini = gemini is not False

    # Validate before *any* retrieval path. Prompted URL-context calls used to
    # hand private targets to Gemini before the local guard had a chance to run.
    try:
        checked = check_url_syntax(target)
        await check_host_resolves_public(
            urlsplit(checked).hostname or "", resolver, timeout
        )
    except UnsafeUrlError as error:
        return Document(url=url, final_url=target, kind="error", error=str(error), notes=notes)
    except FetchError:
        # A local DNS failure can still be recovered by Gemini's server-side URL
        # context. A private DNS answer is UnsafeUrlError and was refused above.
        pass

    # Autonomous policy applies to video and Gemini URL-context retrieval too,
    # not only to the later local GET.
    if robots is not None:
        try:
            verdict = await robots.check(client, target)
        except UnsafeUrlError as error:
            return Document(url=url, final_url=target, kind="error", error=str(error), notes=notes)
        except Exception as error:  # an unavailable policy file must not break a fetch
            verdict = None
            notes.append(f"robots.txt check skipped ({type(error).__name__})")
        if verdict is not None and not verdict.allowed:
            return Document(url=url, final_url=target, kind="error", error=verdict.reason, notes=notes)

    # Tier 1: a video can only be read by the model.
    if _gemini.is_video_url(target):
        if not allow_gemini:
            return Document(
                url=url,
                final_url=target,
                kind="error",
                error="reading a video needs Gemini, but gemini=False was passed",
                notes=notes,
            )
        return await _gemini_document(
            client,
            target,
            "gemini-video",
            _gemini.describe_video(client, target, prompt, model=model, timeout=timeout),
            notes,
        )

    # Tier 2: an explicit question, or gemini=True, goes through url_context, which
    # also reaches pages that need JavaScript or block scripted clients.
    model_requested = allow_gemini and (bool(prompt) or gemini is True)
    if model_requested and not _gemini.available():
        return Document(
            url=url,
            final_url=target,
            kind="error",
            error=_gemini.UNAVAILABLE,
            notes=notes,
        )
    if model_requested:
        document = await _gemini_document(
            client,
            target,
            "gemini-url-context",
            _gemini.answer_about_url(client, target, prompt or "Summarise this page.", model=model, timeout=timeout),
            notes,
        )
        if document.ok:
            return document
        # A local page dump does not satisfy an explicit question or forced-model
        # request, and caching it would suppress a later recovery after 429/5xx.
        return document

    async def guard_redirect(candidate: str) -> None:
        if robots is None:
            return
        try:
            verdict = await robots.check(client, candidate)
        except UnsafeUrlError:
            raise
        except Exception as error:
            notes.append(f"redirect robots.txt check skipped ({type(error).__name__})")
            return
        if not verdict.allowed:
            raise RobotsDeniedError(verdict.reason)

    try:
        body = await guarded_get(
            client,
            target,
            max_bytes=max_bytes,
            resolver=resolver,
            timeout=timeout,
            redirect_guard=guard_redirect,
        )
    except RobotsDeniedError as error:
        return Document(url=url, final_url=target, kind="error", error=str(error), notes=notes)
    except UnsafeUrlError as error:
        # Never hand a refused target to a model either.
        return Document(url=url, final_url=target, kind="error", error=str(error), notes=notes)
    except FetchError as error:
        # Tier 3: blocked or JavaScript-only pages often still work server-side.
        if allow_gemini and _gemini.available():
            fallback = await _gemini_document(
                client,
                target,
                "gemini-url-context",
                _gemini.answer_about_url(
                    client,
                    target,
                    prompt or "Extract the readable content of this page as markdown.",
                    model=model,
                    timeout=timeout,
                ),
                notes + [f"local fetch failed ({error})"],
            )
            if fallback.ok:
                return fallback
            notes = fallback.notes + [f"gemini fallback failed ({fallback.error})"]
        return Document(url=url, final_url=target, kind="error", error=str(error), notes=notes)

    if body.truncated:
        # Partial archives, PDFs, images and other binaries are corrupt. Never
        # report or save them as successful downloads.
        if not (
            looks_like_html(body.content, body.content_type)
            or looks_like_text(body.content, body.content_type)
        ):
            return Document(
                url=url,
                final_url=body.final_url,
                kind="error",
                error=str(TooLargeError(body.final_url, max_bytes)),
                content_type=body.content_type,
                status=body.status,
                bytes_len=len(body.content),
                notes=notes,
            )
        notes.append(f"body capped at {max_bytes:,} bytes; text is incomplete")
    if body.final_url != target:
        notes.append(f"redirected to {body.final_url}")

    try:
        extracted = await asyncio.to_thread(
            _extract_body,
            body.content,
            body.content_type,
            body.text,
            mode,
            max_pages,
        )
        if extracted.kind == "binary":
            extracted = await asyncio.to_thread(
                save_binary, body.content, body.final_url, body.content_type
            )
    except Exception as error:
        message = str(error) if isinstance(error, RuntimeError) else f"extraction failed: {type(error).__name__}"
        return Document(
            url=url,
            final_url=body.final_url,
            kind="error",
            error=message,
            content_type=body.content_type,
            status=body.status,
            bytes_len=len(body.content),
            notes=notes,
        )

    # Tier 3b: a PDF with no text layer is a scan; only vision can read it.
    if (
        extracted.kind == "pdf"
        and allow_gemini
        and any("scanned" in note for note in extracted.notes)
        and _gemini.available()
    ):
        try:
            model_content = await asyncio.to_thread(
                limit_pdf_pages, body.content, max_pages
            )
        except RuntimeError as error:
            extracted.notes.append(f"Gemini scan skipped ({error})")
        else:
            escalated = await _gemini_document(
                client,
                body.final_url,
                "gemini-pdf",
                _gemini.read_pdf(client, model_content, prompt, model=model, timeout=timeout),
                notes + extracted.notes,
            )
            if escalated.ok:
                escalated.kind = "pdf"
                escalated.content_type = body.content_type
                escalated.status = body.status
                escalated.bytes_len = len(body.content)
                escalated.pages = extracted.pages
                return escalated
            extracted.notes.append(f"Gemini scan failed ({escalated.error})")

    return Document(
        url=url,
        final_url=body.final_url,
        kind=extracted.kind,
        text=extracted.text,
        title=extracted.title,
        content_type=body.content_type,
        status=body.status,
        bytes_len=len(body.content),
        pages=extracted.pages,
        saved_path=extracted.saved_path,
        notes=notes + extracted.notes,
    )


async def fetch(
    url: Union[str, Sequence[str]],
    *,
    mode: str = "markdown",
    prompt: Optional[str] = None,
    gemini: Optional[bool] = None,
    model: Optional[str] = None,
    max_bytes: Optional[int] = None,
    max_pages: Optional[int] = None,
    timeout: Optional[float] = None,
    respect_robots: Optional[bool] = None,
    resolver: Optional[Resolver] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> Union[Document, list[Document]]:
    """Fetch one URL or many, returning `Document` objects under the body cap.

    A list of URLs is fetched concurrently (bounded), and a failure becomes a
    Document with `kind="error"` instead of raising, so one bad URL cannot lose the
    others. `respect_robots` defaults to True (see PRIME_AGENT_WEBFETCH_RESPECT_ROBOTS);
    set it False when the user explicitly asked for a page. `prompt` asks a question
    about the page instead of dumping it; `gemini=False` keeps everything local and
    `gemini=True` forces the model path. `resolver` and `transport` are injection
    points for tests and custom networking.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)} (got {mode!r})")

    if isinstance(url, str):
        urls = [url.strip()]
    else:
        try:
            raw_urls = list(url)
        except TypeError as error:
            raise ValueError("url must be an http(s) URL or a sequence of URL strings") from error
        if any(not isinstance(target, str) for target in raw_urls):
            raise ValueError("every URL must be a string")
        urls = [target.strip() for target in raw_urls]
    if not urls:
        raise ValueError("no URL was given")
    if max_pages is not None and (type(max_pages) is not int or max_pages < 1):
        raise ValueError("max_pages must be a positive integer")
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 1):
        raise ValueError("max_bytes must be a positive integer")
    if timeout is not None and (
        not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive finite number")

    cap = max_bytes if max_bytes is not None else _env_int("PRIME_AGENT_WEBFETCH_MAX_BYTES", DEFAULT_MAX_BYTES)
    seconds = timeout if timeout is not None else _env_float("PRIME_AGENT_WEBFETCH_TIMEOUT", DEFAULT_TIMEOUT)
    if seconds <= 0:  # invalid environment values fall back; explicit values were rejected above
        seconds = DEFAULT_TIMEOUT
    autonomous = (
        respect_robots
        if respect_robots is not None
        else _env_bool("PRIME_AGENT_WEBFETCH_RESPECT_ROBOTS", True)
    )
    user_agent = USER_AGENT_AUTONOMOUS if autonomous else USER_AGENT_MANUAL
    resolution_cache: dict[str, tuple[str, ...]] = {}
    base_resolver = resolver or default_resolver

    async def cached_resolver(hostname: str) -> Sequence[str]:
        key = hostname.lower().rstrip(".")
        cached = resolution_cache.get(key)
        if cached is not None:
            return cached
        resolved = tuple(await base_resolver(hostname))
        resolution_cache[key] = resolved
        return resolved

    robots = (
        RobotsCache(
            user_agent=user_agent,
            resolver=cached_resolver,
            timeout=min(10.0, seconds),
        )
        if autonomous
        else None
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # An injected transport or resolver means a non-default environment (tests,
    # custom networking); such results must never be served from or added to
    # the cache. TTL=0 must disable storage too, not only reads.
    ttl = _doc_cache_ttl()
    reuse_cache = transport is None and resolver is None and ttl > 0
    cache_generation = _DOC_CACHE_GENERATION
    gemini_fingerprint = _gemini.cache_fingerprint()
    cap_effective = max(1, cap)

    async with httpx.AsyncClient(
        timeout=seconds,
        headers={"user-agent": user_agent, "accept-language": "en,*;q=0.5"},
        limits=httpx.Limits(max_connections=MAX_CONCURRENCY, max_keepalive_connections=4),
        transport=transport,
    ) as client:

        async def one(target: str) -> Document:
            key = _cache_key(
                target,
                mode,
                prompt,
                gemini,
                model,
                cap_effective,
                max_pages,
                autonomous,
                gemini_fingerprint,
            )
            if reuse_cache:
                hit = _doc_cache_get(key, ttl)
                if hit is not None:
                    return hit
            try:
                async with semaphore:
                    document = await _fetch_one(
                        client,
                        target,
                        mode,
                        cap_effective,
                        max_pages,
                        cached_resolver,
                        robots,
                        prompt,
                        gemini,
                        model,
                        seconds,
                    )
            except Exception as error:
                document = Document(
                    url=target,
                    final_url=target,
                    kind="error",
                    error=f"unexpected {type(error).__name__}",
                )
            if reuse_cache and document.ok and cache_generation == _DOC_CACHE_GENERATION:
                _doc_cache_put(key, document)
            return document

        documents = await asyncio.gather(*(one(target) for target in urls))

    return documents[0] if isinstance(url, str) else list(documents)


def _safe_line(value: object) -> str:
    normalized = "".join(
        " "
        if character.isspace()
        or unicodedata.category(character) in {"Cc", "Zl", "Zp"}
        or (
            unicodedata.category(character) == "Cf"
            and character not in {"\u200c", "\u200d"}
        )
        else character
        for character in str(value)
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _render(document: Document, max_chars: int) -> str:
    if not document.ok:
        return f"# webfetch failed: {_safe_line(document.url)}\n\n{_safe_line(document.error)}" + (
            "\n\n" + "\n".join(f"note: {_safe_line(note)}" for note in document.notes)
            if document.notes
            else ""
        )

    facts = [document.kind]
    if document.source != "local":
        facts.append(document.source)
    if document.bytes_len:
        facts.append(f"{document.bytes_len:,} bytes")
    if document.content_type:
        facts.append(document.content_type)
    if document.pages:
        facts.append(f"{document.pages} pages")
    if document.text:
        facts.append(f"{len(document.text):,} chars extracted")

    header = [f"# webfetch: {_safe_line(document.final_url)}"]
    if document.title:
        header.append(f"**{_safe_line(document.title)}**")
    header.append(" · ".join(_safe_line(fact) for fact in facts))
    for note in document.notes:
        header.append(f"note: {_safe_line(note)}")
    if document.retrieved_urls:
        header.append(
            "retrieved: "
            + ", ".join(_safe_line(url) for url in document.retrieved_urls[:5])
        )

    body = document.text
    if max_chars > 0 and len(body) > max_chars:
        body = (
            body[:max_chars]
            + f"\n\n[truncated at {max_chars:,} of {len(document.text):,} chars — "
            "use `doc = await webfetch.fetch(url)` and slice `doc.text` for the rest]"
        )
    if not body:
        body = document.saved_path or "(no text content)"
    return "\n".join(header) + "\n\n" + body


async def run(
    url: str,
    prompt: Optional[str] = None,
    mode: str = "markdown",
    max_chars: Optional[int] = None,
    max_pages: Optional[int] = None,
    max_bytes: Optional[int] = None,
    respect_robots: Optional[bool] = None,
    gemini: Optional[bool] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    """Fetch a URL and return its content as readable text.

    Args:
        url: An http(s) URL. GitHub blob links are rewritten to raw file contents.
            A YouTube link is read as a video when Gemini is configured.
        prompt: Ask a question about the page instead of returning all of it. Uses
            Gemini's url_context tool, which also reaches pages that need JavaScript
            or block scripted clients.
        mode: "markdown" (default) converts HTML while keeping headings, code blocks
            and link targets; "text" returns plain text; "raw" returns the body
            exactly as served, for JSON and other machine formats.
        max_chars: Truncate the rendered output (default 20000, 0 disables). The full
            text is always available through `webfetch.fetch(url)`.
        max_pages: For PDFs, extract only the first N pages.
        max_bytes: Body size cap (default 10 MB). Large PDFs need a higher value.
        respect_robots: Check robots.txt first (default True). Set False when the
            user explicitly asked for this page.
        gemini: None (default) uses Gemini only where local extraction cannot work -
            videos, scanned PDFs, a given prompt, or a blocked page. False keeps
            everything local. True forces the model path.
        model: Pin the Gemini model, e.g. "gemini-2.5-flash".
        timeout: HTTP timeout in seconds (default 45).

    Returns:
        Markdown text: the final URL, title, content facts, any notes, then the
        content. PDFs come back with `--- page N ---` markers. Non-text bodies are
        saved to a temp file and the path is reported. Never raises; failures are
        returned as text.
    """
    limit = max_chars if max_chars is not None else _env_int("PRIME_AGENT_WEBFETCH_MAX_CHARS", DEFAULT_MAX_CHARS)
    try:
        document = await fetch(
            url,
            mode=mode,
            prompt=prompt,
            gemini=gemini,
            model=model,
            max_pages=max_pages,
            max_bytes=max_bytes,
            respect_robots=respect_robots,
            timeout=timeout,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        return f"webfetch failed: {error}"
    except Exception as error:
        return f"webfetch failed: unexpected {type(error).__name__}"
    assert isinstance(document, Document)
    return _render(document, max(0, limit))


def gemini_available() -> bool:
    """True when a Gemini endpoint is discoverable, enabling video, prompt and scan support."""
    return _gemini.available()


def cli() -> None:  # pragma: no cover - for `python -m webfetch` outside the kernel
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m webfetch <url> [markdown|text|raw]")
        raise SystemExit(2)
    mode = sys.argv[2] if len(sys.argv) > 2 else "markdown"
    print(asyncio.run(run(sys.argv[1], mode=mode)))
