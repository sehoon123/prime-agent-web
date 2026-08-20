"""Fetch a URL as readable markdown for Prime Agent's IPython kernel.

The module defines `run()`, so the kernel exposes it as an async callable:

    print(await webfetch("https://docs.example.com/guide"))
    doc = await webfetch.fetch("https://arxiv.org/pdf/2605.09998")
    docs = await webfetch.fetch([url_a, url_b])           # concurrent

`run()` renders a bounded summary for reading; `fetch()` returns `Document`
objects with the full text, because slicing and storing belong in the kernel.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import httpx

from ._extract import (
    Extracted,
    html_to_markdown,
    html_to_text,
    looks_like_html,
    looks_like_pdf,
    looks_like_text,
    pdf_to_text,
    rewrite_url,
    save_binary,
    tidy,
)
from ._robots import RobotsCache
from ._safety import (
    DEFAULT_MAX_BYTES,
    USER_AGENT_AUTONOMOUS,
    USER_AGENT_MANUAL,
    FetchError,
    Resolver,
    TooLargeError,
    UnsafeUrlError,
    guarded_get,
)

__all__ = ["run", "fetch", "Document", "FetchError", "UnsafeUrlError", "TooLargeError"]
__version__ = "0.3.0"

MODES = ("markdown", "text", "raw")
DEFAULT_MAX_CHARS = 20_000
DEFAULT_TIMEOUT = 45.0
MAX_CONCURRENCY = 6

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
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Document:
    """One fetched document. `text` is the full extraction, never truncated."""

    url: str
    """The URL as requested."""
    final_url: str
    """Where the request actually landed after redirects and rewrites."""
    kind: str
    """html, pdf, text, binary, or error."""
    text: str = ""
    title: Optional[str] = None
    content_type: str = ""
    status: int = 0
    bytes_len: int = 0
    pages: Optional[int] = None
    saved_path: Optional[str] = None
    error: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    def __len__(self) -> int:
        return len(self.text)


def _extract_body(body_bytes: bytes, content_type: str, text: str, mode: str, max_pages: Optional[int]) -> Extracted:
    if looks_like_pdf(body_bytes, content_type):
        if mode == "raw":
            return Extracted(kind="pdf", text="", notes=["raw mode does not decode PDF bytes"])
        return pdf_to_text(body_bytes, max_pages=max_pages)
    if looks_like_html(body_bytes, content_type):
        if mode == "raw":
            return Extracted(kind="html", text=text)
        if mode == "text":
            return html_to_text(text)
        return html_to_markdown(text)
    if looks_like_text(body_bytes, content_type):
        return Extracted(kind="text", text=text if mode == "raw" else tidy(text))
    return Extracted(kind="binary", text="")


async def _fetch_one(
    client: httpx.AsyncClient,
    url: str,
    mode: str,
    max_bytes: int,
    max_pages: Optional[int],
    resolver: Optional[Resolver],
    robots: Optional[RobotsCache],
) -> Document:
    target, rewrite_note = rewrite_url(url.strip())
    notes = [rewrite_note] if rewrite_note else []

    if robots is not None:
        try:
            verdict = await robots.check(client, target)
        except Exception as error:  # robots must never break a fetch
            verdict = None
            notes.append(f"robots.txt check skipped ({type(error).__name__})")
        if verdict is not None and not verdict.allowed:
            return Document(url=url, final_url=target, kind="error", error=verdict.reason, notes=notes)

    try:
        body = await guarded_get(client, target, max_bytes=max_bytes, resolver=resolver)
    except FetchError as error:
        return Document(url=url, final_url=target, kind="error", error=str(error), notes=notes)

    if body.truncated:
        # A partial PDF or archive is unparseable, so say so instead of failing
        # later with a cryptic library error.
        if looks_like_pdf(body.content, body.content_type):
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
        extracted = _extract_body(body.content, body.content_type, body.text, mode, max_pages)
        if extracted.kind == "binary":
            extracted = save_binary(body.content, body.final_url, body.content_type)
    except RuntimeError as error:
        return Document(
            url=url,
            final_url=body.final_url,
            kind="error",
            error=str(error),
            content_type=body.content_type,
            status=body.status,
            bytes_len=len(body.content),
            notes=notes,
        )

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
    max_bytes: Optional[int] = None,
    max_pages: Optional[int] = None,
    timeout: Optional[float] = None,
    respect_robots: Optional[bool] = None,
    resolver: Optional[Resolver] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> Union[Document, list[Document]]:
    """Fetch one URL or many, returning `Document` objects with full text.

    A list of URLs is fetched concurrently (bounded), and a failure becomes a
    Document with `kind="error"` instead of raising, so one bad URL cannot lose the
    others. `respect_robots` defaults to True (see PRIME_AGENT_WEBFETCH_RESPECT_ROBOTS);
    set it False when the user explicitly asked for a page. `resolver` and
    `transport` are injection points for tests and custom networking.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)} (got {mode!r})")

    urls = [url] if isinstance(url, str) else list(url)
    if not urls:
        raise ValueError("no URL was given")

    cap = max_bytes if max_bytes is not None else _env_int("PRIME_AGENT_WEBFETCH_MAX_BYTES", DEFAULT_MAX_BYTES)
    seconds = timeout if timeout is not None else _env_float("PRIME_AGENT_WEBFETCH_TIMEOUT", DEFAULT_TIMEOUT)
    autonomous = (
        respect_robots
        if respect_robots is not None
        else _env_bool("PRIME_AGENT_WEBFETCH_RESPECT_ROBOTS", True)
    )
    user_agent = USER_AGENT_AUTONOMOUS if autonomous else USER_AGENT_MANUAL
    robots = RobotsCache(user_agent=user_agent) if autonomous else None
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async with httpx.AsyncClient(
        timeout=max(1.0, seconds),
        headers={"user-agent": user_agent, "accept-language": "en,*;q=0.5"},
        limits=httpx.Limits(max_connections=MAX_CONCURRENCY, max_keepalive_connections=4),
        transport=transport,
    ) as client:

        async def one(target: str) -> Document:
            async with semaphore:
                return await _fetch_one(client, target, mode, max(1024, cap), max_pages, resolver, robots)

        documents = await asyncio.gather(*(one(target) for target in urls))

    return documents[0] if isinstance(url, str) else list(documents)


def _render(document: Document, max_chars: int) -> str:
    if not document.ok:
        return f"# webfetch failed: {document.url}\n\n{document.error}" + (
            "\n\n" + "\n".join(f"note: {note}" for note in document.notes) if document.notes else ""
        )

    facts = [document.kind, f"{document.bytes_len:,} bytes"]
    if document.content_type:
        facts.append(document.content_type)
    if document.pages:
        facts.append(f"{document.pages} pages")
    if document.text:
        facts.append(f"{len(document.text):,} chars extracted")

    header = [f"# webfetch: {document.final_url}"]
    if document.title:
        header.append(f"**{document.title}**")
    header.append(" · ".join(facts))
    for note in document.notes:
        header.append(f"note: {note}")

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
    mode: str = "markdown",
    max_chars: Optional[int] = None,
    max_pages: Optional[int] = None,
    max_bytes: Optional[int] = None,
    respect_robots: Optional[bool] = None,
    timeout: Optional[float] = None,
) -> str:
    """Fetch a URL and return its content as readable text.

    Args:
        url: An http(s) URL. GitHub blob links are rewritten to raw file contents.
        mode: "markdown" (default) converts HTML while keeping headings, code blocks
            and link targets; "text" returns plain text; "raw" returns the body
            exactly as served, for JSON and other machine formats.
        max_chars: Truncate the rendered output (default 20000, 0 disables). The full
            text is always available through `webfetch.fetch(url)`.
        max_pages: For PDFs, extract only the first N pages.
        max_bytes: Body size cap (default 10 MB). Large PDFs need a higher value.
        respect_robots: Check robots.txt first (default True). Set False when the
            user explicitly asked for this page.
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
            max_pages=max_pages,
            max_bytes=max_bytes,
            respect_robots=respect_robots,
            timeout=timeout,
        )
    except ValueError as error:
        return f"webfetch failed: {error}"
    assert isinstance(document, Document)
    return _render(document, max(0, limit))


def cli() -> None:  # pragma: no cover - for `python -m webfetch` outside the kernel
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m webfetch <url> [markdown|text|raw]")
        raise SystemExit(2)
    mode = sys.argv[2] if len(sys.argv) > 2 else "markdown"
    print(asyncio.run(run(sys.argv[1], mode=mode)))
