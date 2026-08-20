"""Optional Gemini-backed capabilities.

Three things a local fetcher cannot do, in the tiered style used by pi-web-access
(deterministic extraction first, model extraction as a fallback):

- answer a question about a page, including pages that need JavaScript or block
  scripted clients, via the `url_context` tool (Gemini fetches server-side);
- read a video, via a YouTube `fileData` part;
- read a scanned PDF that has no text layer, via an inline PDF part.

Endpoint and key discovery is reused from the `websearch` skill shipped in the same
package, so there is one source of truth for "which Gemini endpoints does this host
have". When only `webfetch` is installed, these features report themselves as
unavailable and everything else keeps working.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import httpx

# Inline request payload ceiling for generateContent; larger PDFs need the Files API.
MAX_INLINE_BYTES = 18 * 1024 * 1024
FAILOVER_STATUSES = frozenset({401, 402, 403, 408, 409, 425, 429, 500, 502, 503, 504})

_YOUTUBE_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be")
_YOUTUBE_PATHS = re.compile(r"^/(watch|shorts/|live/|embed/|v/)")

DEFAULT_VIDEO_PROMPT = (
    "Describe this video: what it covers, what is shown on screen, and any code, "
    "commands or file names that appear. Note timestamps for key moments."
)
DEFAULT_PDF_PROMPT = (
    "Transcribe this document to markdown. Preserve headings, tables, code and "
    "formulas. Do not summarise or omit content."
)

UNAVAILABLE = (
    "no Gemini endpoint is available. Install the `websearch` skill from the same "
    "package (it provides endpoint discovery) and configure a google-generative-ai "
    "provider in models.json, or set GEMINI_API_KEY."
)


class GeminiUnavailable(RuntimeError):
    """No usable Gemini endpoint on this host."""


@dataclass
class GeminiAnswer:
    text: str
    detail: str
    """endpoint/model that produced it."""
    retrieved_urls: list[str] = field(default_factory=list)


def is_video_url(url: str) -> bool:
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host in ("youtu.be", "www.youtu.be"):
        return bool(parts.path.strip("/"))
    if host in _YOUTUBE_HOSTS:
        return bool(_YOUTUBE_PATHS.match(parts.path))
    return False


def _endpoints() -> Sequence[Any]:
    """Gemini endpoints discovered by the websearch skill, if it is installed."""
    try:
        from websearch.config import gemini_endpoints  # type: ignore import-not-found
    except ImportError:
        return ()
    try:
        return gemini_endpoints()
    except Exception:
        return ()


def available() -> bool:
    return bool(_endpoints())


def _extract_text(payload: dict[str, Any]) -> tuple[str, list[str]]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return "", []
    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    text = ""
    if isinstance(parts, list):
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()

    urls: list[str] = []
    metadata = candidate.get("urlContextMetadata")
    if isinstance(metadata, dict):
        for entry in metadata.get("urlMetadata") or []:
            if isinstance(entry, dict):
                url = entry.get("retrievedUrl") or entry.get("retrieved_url")
                if isinstance(url, str) and url not in urls:
                    urls.append(url)
    return text, urls


async def generate(
    client: httpx.AsyncClient,
    parts: list[dict[str, Any]],
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> GeminiAnswer:
    """Call generateContent, failing over across endpoints and keys.

    Raises GeminiUnavailable when nothing is configured, or RuntimeError with the
    collected failures when every endpoint refused.
    """
    endpoints = _endpoints()
    if not endpoints:
        raise GeminiUnavailable(UNAVAILABLE)

    body: dict[str, Any] = {"contents": [{"role": "user", "parts": parts}]}
    if tools:
        body["tools"] = tools

    failures: list[str] = []
    for endpoint in endpoints:
        chosen = endpoint.pick_model(model)
        if not chosen:
            failures.append(f"{endpoint.label}: no usable model")
            continue
        url = f"{endpoint.base_url}/models/{chosen}:generateContent"
        for key in endpoint.keys:
            try:
                response = await client.post(
                    url,
                    headers={"x-goog-api-key": key, "content-type": "application/json"},
                    json=body,
                    timeout=timeout,
                )
            except httpx.HTTPError as error:
                failures.append(f"{endpoint.label}/{chosen}: {type(error).__name__}")
                continue

            if response.status_code >= 400:
                detail = ""
                try:
                    error_payload = response.json()
                    if isinstance(error_payload, dict) and isinstance(error_payload.get("error"), dict):
                        detail = str(error_payload["error"].get("message") or "")[:160]
                except ValueError:
                    detail = response.text[:160]
                failures.append(f"{endpoint.label}/{chosen}: HTTP {response.status_code} {detail}".strip())
                if response.status_code in FAILOVER_STATUSES:
                    continue
                break  # a 400 will repeat with the same request shape

            try:
                payload = response.json()
            except ValueError:
                failures.append(f"{endpoint.label}/{chosen}: non-JSON response")
                continue

            text, urls = _extract_text(payload)
            if not text:
                failures.append(f"{endpoint.label}/{chosen}: empty response")
                continue
            return GeminiAnswer(text=text, detail=f"{endpoint.label}/{chosen}", retrieved_urls=urls)

    raise RuntimeError("; ".join(failures) or "every Gemini endpoint failed")


async def answer_about_url(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> GeminiAnswer:
    """Answer `prompt` about `url` with the url_context tool (server-side fetch)."""
    parts = [{"text": f"{prompt.strip()}\n\nUse this page: {url}"}]
    return await generate(
        client, parts, tools=[{"url_context": {}}], model=model, timeout=timeout
    )


async def describe_video(
    client: httpx.AsyncClient,
    url: str,
    prompt: Optional[str] = None,
    *,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> GeminiAnswer:
    """Read a YouTube video: transcript-level content plus what is on screen."""
    parts = [
        {"fileData": {"fileUri": url}},
        {"text": (prompt or DEFAULT_VIDEO_PROMPT).strip()},
    ]
    return await generate(client, parts, model=model, timeout=timeout)


async def read_pdf(
    client: httpx.AsyncClient,
    content: bytes,
    prompt: Optional[str] = None,
    *,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> GeminiAnswer:
    """Read a PDF whose text layer is missing or unusable (scans, figures)."""
    if len(content) > MAX_INLINE_BYTES:
        raise RuntimeError(
            f"PDF is {len(content):,} bytes, over the {MAX_INLINE_BYTES:,}-byte inline limit "
            "for model extraction; pass max_pages or extract locally"
        )
    parts = [
        {"inlineData": {"mimeType": "application/pdf", "data": base64.b64encode(content).decode("ascii")}},
        {"text": (prompt or DEFAULT_PDF_PROMPT).strip()},
    ]
    return await generate(client, parts, model=model, timeout=timeout)
