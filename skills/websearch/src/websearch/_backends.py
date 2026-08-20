"""Search backend implementations (private module).

Named with a leading underscore because `websearch.backends` is the public
helper that lists which backends are usable on this host.

Every backend is an async function taking an `httpx.AsyncClient` and returning a
`SearchResult`. Failures raise `BackendError`, which the caller turns into a
failover attempt. No backend ever puts a credential into an exception message.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from html import unescape
from typing import Any, Sequence
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from .config import AI_STUDIO_FALLBACK_MODELS, GeminiEndpoint, Settings

# Statuses worth retrying on another key or backend.
FAILOVER_STATUSES = frozenset({401, 402, 403, 408, 409, 425, 429, 500, 502, 503, 504})

_GROUNDING_REDIRECT_HOST = "vertexaisearch.cloud.google.com"


class BackendError(RuntimeError):
    """A backend attempt failed; the caller may try another key or backend."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = True) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class ResultItem:
    title: str
    url: str
    snippet: str = ""


@dataclass
class SearchResult:
    backend: str
    detail: str = ""
    answer: str | None = None
    items: list[ResultItem] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.answer and not self.items


def _clean(value: Any, limit: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    text = unescape(re.sub(r"\s+", " ", value)).strip()
    return text[:limit]


def _raise_for_status(response: httpx.Response, backend: str) -> None:
    if response.status_code < 400:
        return
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or error.get("status") or "")
            elif isinstance(error, str):
                detail = error
            detail = detail or str(payload.get("message") or "")
    except ValueError:
        detail = response.text[:200]
    message = f"{backend} returned HTTP {response.status_code}"
    if detail:
        message = f"{message}: {_clean(detail, 200)}"
    raise BackendError(message, status=response.status_code, retryable=response.status_code in FAILOVER_STATUSES)


async def _request(client: httpx.AsyncClient, backend: str, method: str, url: str, **kwargs: Any) -> httpx.Response:
    try:
        response = await client.request(method, url, **kwargs)
    except httpx.HTTPError as error:
        raise BackendError(f"{backend} failed before an HTTP response: {type(error).__name__}") from error
    _raise_for_status(response, backend)
    return response


def _json_body(response: httpx.Response, backend: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as error:
        raise BackendError(f"{backend} returned a non-JSON response", retryable=False) from error
    if not isinstance(payload, dict):
        raise BackendError(f"{backend} returned an unexpected payload", retryable=False)
    return payload


# --------------------------------------------------------------------------- #
# Gemini (Google Search grounding)
# --------------------------------------------------------------------------- #


async def _resolve_redirect(client: httpx.AsyncClient, url: str) -> str:
    """Turn a grounding redirect link into the publisher URL, best effort."""
    if _GROUNDING_REDIRECT_HOST not in url:
        return url
    for method in ("HEAD", "GET"):
        try:
            response = await client.request(method, url, follow_redirects=True, timeout=15.0)
        except httpx.HTTPError:
            continue
        final = str(response.url)
        if _GROUNDING_REDIRECT_HOST not in final:
            return final
    return url


async def _resolve_redirects(client: httpx.AsyncClient, items: Sequence[ResultItem]) -> None:
    targets = [item for item in items if _GROUNDING_REDIRECT_HOST in item.url]
    if not targets:
        return
    resolved = await asyncio.gather(
        *(_resolve_redirect(client, item.url) for item in targets),
        return_exceptions=True,
    )
    for item, value in zip(targets, resolved):
        if isinstance(value, str) and value:
            item.url = value


async def _gemini_studio_models(client: httpx.AsyncClient, endpoint: GeminiEndpoint, key: str) -> tuple[str, ...]:
    """List models for the public endpoint, falling back to known ids."""
    try:
        response = await client.get(f"{endpoint.base_url}/models", headers={"x-goog-api-key": key})
        if response.status_code >= 400:
            return AI_STUDIO_FALLBACK_MODELS
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return AI_STUDIO_FALLBACK_MODELS
    names: list[str] = []
    models = payload.get("models") if isinstance(payload, dict) else None
    if isinstance(models, list):
        for model in models:
            if not isinstance(model, dict):
                continue
            name = model.get("name")
            methods = model.get("supportedGenerationMethods")
            if not isinstance(name, str):
                continue
            if isinstance(methods, list) and "generateContent" not in methods:
                continue
            names.append(name.split("/")[-1])
    return tuple(names) or AI_STUDIO_FALLBACK_MODELS


def _parse_gemini(payload: dict[str, Any], num_results: int) -> tuple[str | None, list[ResultItem], list[str]]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise BackendError("gemini returned no candidates", retryable=False)
    candidate = candidates[0] if isinstance(candidates[0], dict) else {}

    parts = ((candidate.get("content") or {}).get("parts")) if isinstance(candidate.get("content"), dict) else None
    answer = ""
    if isinstance(parts, list):
        answer = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()

    metadata = candidate.get("groundingMetadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    items: list[ResultItem] = []
    seen: set[str] = set()
    for chunk in metadata.get("groundingChunks") or []:
        if not isinstance(chunk, dict):
            continue
        web = chunk.get("web")
        if not isinstance(web, dict):
            continue
        url = web.get("uri")
        if not isinstance(url, str) or url in seen:
            continue
        seen.add(url)
        items.append(ResultItem(title=_clean(web.get("title")) or url, url=url))
        if len(items) >= num_results:
            break

    queries = [q for q in (metadata.get("webSearchQueries") or []) if isinstance(q, str)]
    if not answer and not items:
        raise BackendError("gemini returned no grounded content", retryable=False)
    return (answer or None), items, queries


async def search_gemini(client: httpx.AsyncClient, query: str, settings: Settings) -> SearchResult:
    endpoints = settings.gemini_endpoints
    if not endpoints:
        raise BackendError("no gemini endpoint is configured", retryable=False)

    errors: list[str] = []
    for endpoint in endpoints:
        for key in endpoint.keys:
            models = endpoint.models
            if not models:
                models = await _gemini_studio_models(client, endpoint, key)
            model = GeminiEndpoint(endpoint.label, endpoint.base_url, models, endpoint.keys).pick_model(
                settings.gemini_model
            )
            if not model:
                errors.append(f"{endpoint.label}: no usable model")
                continue

            url = f"{endpoint.base_url}/models/{model}:generateContent"
            headers = {"x-goog-api-key": key, "content-type": "application/json"}
            # google_search is the Gemini 2+ tool; google_search_retrieval is the
            # older name still required by some gateways.
            for tool in ({"google_search": {}}, {"google_search_retrieval": {}}):
                body = {
                    "contents": [{"role": "user", "parts": [{"text": query}]}],
                    "tools": [tool],
                }
                try:
                    response = await _request(client, "gemini", "POST", url, headers=headers, json=body)
                    answer, items, queries = _parse_gemini(_json_body(response, "gemini"), settings.num_results)
                except BackendError as error:
                    errors.append(f"{endpoint.label}/{model}: {error}")
                    # A 400 usually means this gateway rejected the tool shape;
                    # anything else is a key/endpoint problem, so stop retrying.
                    if error.status == 400:
                        continue
                    break

                await _resolve_redirects(client, items)
                return SearchResult(
                    backend="gemini",
                    detail=f"{endpoint.label}/{model}",
                    answer=answer,
                    items=items,
                    queries=queries,
                )
    raise BackendError("; ".join(errors) or "every gemini endpoint failed")


# --------------------------------------------------------------------------- #
# Credential-based JSON APIs
# --------------------------------------------------------------------------- #


async def search_serper(client: httpx.AsyncClient, query: str, settings: Settings) -> SearchResult:
    key = settings.simple_key("serper")
    if not key:
        raise BackendError("no serper credential is configured", retryable=False)
    response = await _request(
        client,
        "serper",
        "POST",
        "https://google.serper.dev/search",
        headers={"X-API-KEY": key, "content-type": "application/json"},
        json={"q": query, "num": settings.num_results},
    )
    payload = _json_body(response, "serper")

    answer_parts: list[str] = []
    box = payload.get("answerBox")
    if isinstance(box, dict):
        answer_parts.append(_clean(box.get("answer") or box.get("snippet")))
    graph = payload.get("knowledgeGraph")
    if isinstance(graph, dict):
        title = _clean(graph.get("title"))
        description = _clean(graph.get("description"))
        if title or description:
            answer_parts.append(" - ".join(part for part in (title, description) if part))

    items: list[ResultItem] = []
    for entry in payload.get("organic") or []:
        if not isinstance(entry, dict):
            continue
        url = entry.get("link")
        if not isinstance(url, str):
            continue
        items.append(ResultItem(_clean(entry.get("title")) or url, url, _clean(entry.get("snippet"))))
        if len(items) >= settings.num_results:
            break

    answer = "\n".join(part for part in answer_parts if part) or None
    return SearchResult(backend="serper", detail="google.serper.dev", answer=answer, items=items)


async def search_tavily(client: httpx.AsyncClient, query: str, settings: Settings) -> SearchResult:
    key = settings.simple_key("tavily")
    if not key:
        raise BackendError("no tavily credential is configured", retryable=False)
    response = await _request(
        client,
        "tavily",
        "POST",
        "https://api.tavily.com/search",
        headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
        json={
            "query": query,
            "max_results": settings.num_results,
            "include_answer": True,
            "search_depth": "basic",
        },
    )
    payload = _json_body(response, "tavily")
    items: list[ResultItem] = []
    for entry in payload.get("results") or []:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str):
            continue
        items.append(ResultItem(_clean(entry.get("title")) or url, url, _clean(entry.get("content"))))
        if len(items) >= settings.num_results:
            break
    return SearchResult(
        backend="tavily",
        detail="api.tavily.com",
        answer=_clean(payload.get("answer"), 1500) or None,
        items=items,
    )


async def search_brave(client: httpx.AsyncClient, query: str, settings: Settings) -> SearchResult:
    key = settings.simple_key("brave")
    if not key:
        raise BackendError("no brave credential is configured", retryable=False)
    response = await _request(
        client,
        "brave",
        "GET",
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
        params={"q": query, "count": settings.num_results},
    )
    payload = _json_body(response, "brave")
    web = payload.get("web")
    entries = web.get("results") if isinstance(web, dict) else None
    items: list[ResultItem] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str):
            continue
        items.append(ResultItem(_clean(entry.get("title")) or url, url, _clean(entry.get("description"))))
        if len(items) >= settings.num_results:
            break
    return SearchResult(backend="brave", detail="api.search.brave.com", items=items)


async def search_exa(client: httpx.AsyncClient, query: str, settings: Settings) -> SearchResult:
    key = settings.simple_key("exa")
    if not key:
        raise BackendError("no exa credential is configured", retryable=False)
    response = await _request(
        client,
        "exa",
        "POST",
        "https://api.exa.ai/search",
        headers={"x-api-key": key, "content-type": "application/json"},
        json={
            "query": query,
            "numResults": settings.num_results,
            "contents": {"text": {"maxCharacters": 500}},
        },
    )
    payload = _json_body(response, "exa")
    items: list[ResultItem] = []
    for entry in payload.get("results") or []:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str):
            continue
        snippet = entry.get("text") or entry.get("summary") or entry.get("highlights")
        if isinstance(snippet, list):
            snippet = " ".join(str(part) for part in snippet)
        items.append(ResultItem(_clean(entry.get("title")) or url, url, _clean(snippet)))
        if len(items) >= settings.num_results:
            break
    return SearchResult(backend="exa", detail="api.exa.ai", items=items)


async def search_searxng(client: httpx.AsyncClient, query: str, settings: Settings) -> SearchResult:
    base = settings.searxng_url
    if not base:
        raise BackendError("SEARXNG_URL is not set", retryable=False)
    response = await _request(
        client,
        "searxng",
        "GET",
        f"{base}/search",
        params={"q": query, "format": "json"},
        headers={"Accept": "application/json"},
    )
    payload = _json_body(response, "searxng")
    items: list[ResultItem] = []
    for entry in payload.get("results") or []:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str):
            continue
        items.append(ResultItem(_clean(entry.get("title")) or url, url, _clean(entry.get("content"))))
        if len(items) >= settings.num_results:
            break
    if not items:
        raise BackendError("searxng returned no results (instance may block format=json)", retryable=False)
    return SearchResult(backend="searxng", detail=urlparse(base).netloc or base, items=items)


# --------------------------------------------------------------------------- #
# DuckDuckGo (no credential)
# --------------------------------------------------------------------------- #

_DDG_ENDPOINTS = ("https://html.duckduckgo.com/html/", "https://lite.duckduckgo.com/lite/")
_DDG_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_ANCHOR_RE = re.compile(r'<a[^>]+class="[^"]*result(?:__a|-link)[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def _unwrap_ddg_url(href: str) -> str:
    url = href.strip()
    if url.startswith("//"):
        url = f"https:{url}"
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return url


def _parse_ddg_with_bs4(html: str, limit: int) -> list[ResultItem]:
    try:
        from bs4 import BeautifulSoup  # type: ignore import-not-found
    except ImportError:
        return []
    soup = BeautifulSoup(html, "html.parser")
    items: list[ResultItem] = []
    for anchor in soup.select("a.result__a, a.result-link"):
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        container = anchor.find_parent(["div", "tr", "table", "body"])
        snippet = ""
        if container is not None:
            node = container.select_one(".result__snippet, .result-snippet, td.result-snippet")
            if node is None and container.name == "tr":
                sibling = container.find_next_sibling("tr")
                node = sibling.select_one(".result-snippet") if sibling else None
            if node is not None:
                snippet = node.get_text(" ", strip=True)
        items.append(ResultItem(anchor.get_text(" ", strip=True), _unwrap_ddg_url(href), _clean(snippet)))
        if len(items) >= limit:
            break
    return items


def _parse_ddg_with_regex(html: str, limit: int) -> list[ResultItem]:
    items: list[ResultItem] = []
    for href, raw_title in _ANCHOR_RE.findall(html):
        title = _clean(_TAG_RE.sub(" ", raw_title))
        url = _unwrap_ddg_url(href)
        items.append(ResultItem(title or url, url))
        if len(items) >= limit:
            break
    return items


async def search_ddg(client: httpx.AsyncClient, query: str, settings: Settings) -> SearchResult:
    errors: list[str] = []
    for url in _DDG_ENDPOINTS:
        try:
            response = await _request(
                client,
                "ddg",
                "POST",
                url,
                headers={"user-agent": _DDG_USER_AGENT},
                data={"q": query},
                follow_redirects=True,
            )
        except BackendError as error:
            errors.append(str(error))
            continue
        html = response.text
        items = _parse_ddg_with_bs4(html, settings.num_results) or _parse_ddg_with_regex(html, settings.num_results)
        if items:
            return SearchResult(backend="ddg", detail=urlparse(url).netloc, items=items)
        errors.append(f"no results parsed from {urlparse(url).netloc}")
    raise BackendError("; ".join(errors) or "duckduckgo returned no results")


BACKENDS = {
    "gemini": search_gemini,
    "serper": search_serper,
    "tavily": search_tavily,
    "brave": search_brave,
    "exa": search_exa,
    "searxng": search_searxng,
    "ddg": search_ddg,
}
