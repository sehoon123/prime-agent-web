"""Content extraction: HTML to markdown, PDF to text, binaries to a file.

The goal is structure preservation, not prettiness: headings, code blocks and link
targets are what make a fetched page useful to a coding agent. A readability-style
main-content extractor was measured and rejected for this workload - on API
reference pages it dropped every heading and link.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import gettempdir
from typing import Optional
from urllib.parse import urlsplit

# Elements that are never content. Removed before conversion.
BOILERPLATE_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "iframe",
    "form",
    "nav",
    "header",
    "footer",
    "aside",
)
# Containers that usually hold the real content, in preference order.
CONTENT_SELECTORS = ("main", "article", "[role=main]", "#content", ".markdown-body", "body")

PDF_MAGIC = b"%PDF-"
TEXT_CONTENT_TYPES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/csv",
        "text/xml",
        "application/json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/javascript",
        "application/x-ndjson",
    }
)
HTML_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml"})

_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.M)


@dataclass
class Extracted:
    """Extraction result for one body."""

    kind: str
    """One of html, pdf, text, binary."""
    text: str = ""
    title: Optional[str] = None
    saved_path: Optional[str] = None
    pages: Optional[int] = None
    notes: list[str] = field(default_factory=list)


def tidy(text: str) -> str:
    return _BLANK_LINES.sub("\n\n", _TRAILING_SPACE.sub("", text)).strip()


def rewrite_url(url: str) -> tuple[str, Optional[str]]:
    """Rewrite URLs whose HTML is useless to a machine-readable equivalent.

    Returns (url, note). GitHub `blob` links become raw file contents; a bare
    repository URL is left alone with a hint, because cloning belongs to the agent
    (`git clone`), not to a fetch helper.
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    segments = [segment for segment in parts.path.split("/") if segment]

    if host in ("github.com", "www.github.com"):
        if len(segments) >= 5 and segments[2] in ("blob", "raw"):
            owner, repo, _, ref, *rest = segments
            raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{'/'.join(rest)}"
            return raw, "rewritten to raw.githubusercontent.com for exact file contents"
        if len(segments) == 2:
            return url, (
                f"this is a repository root; `git clone https://github.com/{segments[0]}/{segments[1]}` "
                "gives real file contents instead of rendered HTML"
            )
    return url, None


def looks_like_pdf(content: bytes, content_type: str) -> bool:
    return content_type == "application/pdf" or content[:5] == PDF_MAGIC


def looks_like_html(content: bytes, content_type: str) -> bool:
    if content_type in HTML_CONTENT_TYPES:
        return True
    if content_type:
        return False
    head = content[:512].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<head" in head


def looks_like_text(content: bytes, content_type: str) -> bool:
    if content_type in TEXT_CONTENT_TYPES or content_type.startswith("text/"):
        return True
    if content_type:
        return False
    sample = content[:2048]
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def html_title(soup: object) -> Optional[str]:
    title = getattr(soup, "title", None)
    text = getattr(title, "string", None) if title is not None else None
    if isinstance(text, str) and text.strip():
        return re.sub(r"\s+", " ", text).strip()[:300]
    heading = soup.find("h1") if hasattr(soup, "find") else None  # type: ignore[union-attr]
    if heading is not None:
        value = heading.get_text(" ", strip=True)
        if value:
            return value[:300]
    return None


def html_to_markdown(html: str, *, strip_boilerplate: bool = True) -> Extracted:
    """Convert HTML to markdown, keeping headings, code blocks and link targets."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as error:  # pragma: no cover - declared dependency
        raise RuntimeError("beautifulsoup4 is required to extract HTML") from error
    try:
        from markdownify import markdownify
    except ImportError as error:  # pragma: no cover - declared dependency
        raise RuntimeError("markdownify is required to extract HTML") from error

    soup = BeautifulSoup(html, "html.parser")
    title = html_title(soup)
    notes: list[str] = []

    if strip_boilerplate:
        for element in soup(list(BOILERPLATE_TAGS)):
            element.decompose()
        for comment_parent in soup.find_all(string=lambda value: isinstance(value, str) and False):
            comment_parent.extract()  # pragma: no cover - placeholder for clarity

        root = None
        for selector in CONTENT_SELECTORS:
            try:
                root = soup.select_one(selector)
            except Exception:  # a malformed selector must never break a fetch
                root = None
            if root is not None and root.get_text(strip=True):
                if selector != "body":
                    notes.append(f"extracted from <{selector}>")
                break
        target = root if root is not None else soup
    else:
        target = soup

    markdown = markdownify(str(target), heading_style="ATX", bullets="-", escape_underscores=False)
    return Extracted(kind="html", text=tidy(markdown), title=title, notes=notes)


def html_to_text(html: str) -> Extracted:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = html_title(soup)
    for element in soup(list(BOILERPLATE_TAGS)):
        element.decompose()
    return Extracted(kind="html", text=tidy(soup.get_text("\n", strip=True)), title=title)


def pdf_to_text(content: bytes, *, max_pages: Optional[int] = None) -> Extracted:
    """Extract text per page, with page markers so the agent can cite locations."""
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - declared dependency
        raise RuntimeError("pypdf is required to extract PDFs") from error
    from io import BytesIO

    try:
        # strict=False keeps slightly malformed PDFs readable, as most real ones are.
        reader = PdfReader(BytesIO(content), strict=False)
    except Exception as error:
        raise RuntimeError(f"could not parse PDF: {type(error).__name__}") from error

    notes: list[str] = []
    total = len(reader.pages)
    limit = total if max_pages is None else min(total, max_pages)
    if limit < total:
        notes.append(f"first {limit} of {total} pages")

    chunks: list[str] = []
    for index in range(limit):
        try:
            page_text = reader.pages[index].extract_text() or ""
        except Exception:
            page_text = ""
            notes.append(f"page {index + 1} could not be extracted")
        chunks.append(f"--- page {index + 1} ---\n{page_text.strip()}")

    body = tidy("\n\n".join(chunks))
    title = None
    try:
        metadata = reader.metadata
        if metadata and metadata.title:
            title = str(metadata.title)[:300]
    except Exception:
        title = None
    if not body.replace("--- page", "").strip("- \n0123456789"):
        notes.append("no extractable text layer; this may be a scanned PDF")
    return Extracted(kind="pdf", text=body, title=title, pages=total, notes=notes)


def save_binary(content: bytes, url: str, content_type: str) -> Extracted:
    """Write a non-text body to a temp file and report where it landed."""
    suffix = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
        "application/zip": ".zip",
        "application/gzip": ".gz",
        "application/octet-stream": ".bin",
    }.get(content_type, "")
    if not suffix:
        tail = Path(urlsplit(url).path).suffix
        suffix = tail if 1 < len(tail) <= 6 else ".bin"

    digest = hashlib.sha256(content).hexdigest()[:16]
    path = Path(gettempdir()) / f"webfetch-{digest}{suffix}"
    path.write_bytes(content)
    note = f"binary body ({content_type or 'unknown type'}, {len(content):,} bytes) saved to {path}"
    if content_type.startswith("image/"):
        note += "; use the attach-image skill or PIL to inspect it"
    return Extracted(kind="binary", text="", saved_path=str(path), notes=[note])
