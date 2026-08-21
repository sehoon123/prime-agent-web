"""Search backend implementations (private module).

Named with a leading underscore because `websearch.backends` is the public helper
that lists which backends are usable on this host.

Every backend is an async function `(client, query, settings) -> SearchResult`.
Failures raise `BackendError`, which the caller turns into a failover attempt. No
backend ever puts a credential into an exception message, and no URL taken from a
provider response is followed before it passes `is_public_http_url`.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import dataclass, field
from html import unescape
from typing import Any, Optional, Sequence
from urllib.parse import parse_qs, urljoin, urlparse

import httpx

from .config import (
    AI_STUDIO_FALLBACK_MODELS,
    GeminiEndpoint,
    SearchQuery,
    Settings,
    is_public_http_url,
    recency_start_date,
    safe_endpoint_label,
)

# Statuses worth retrying on another key or backend.
FAILOVER_STATUSES = frozenset({401, 402, 403, 408, 409, 425, 429, 500, 502, 503, 504})

GROUNDING_REDIRECT_HOST = "vertexaisearch.cloud.google.com"
REDIRECT_TIMEOUT = 10.0
MAX_REDIRECT_HOPS = 5
MAX_BACKEND_BYTES = 5 * 1024 * 1024
MAX_BACKEND_ITEMS = 1000
MAX_GEMINI_ANSWER_BYTES = 200_000
MAX_GROUNDING_SUPPORTS = 2000
MAX_SUPPORT_INDICES = 100
MAX_REDIRECT_CONCURRENCY = 8
REDIRECT_RESOLUTION_BUDGET = 5.0


class BackendError(RuntimeError):
    """A backend attempt failed; the caller may try another key or backend."""

    def __init__(self, message: str, *, status: Optional[int] = None, retryable: bool = True) -> None:
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
    answer: Optional[str] = None
    items: list[ResultItem] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    dropped: int = 0
    """Results removed by the client-side domain filter."""

    @property
    def empty(self) -> bool:
        return not self.answer and not self.items


def _clean(value: Any, limit: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    decoded = unescape(value)
    text = "".join(
        " "
        if character.isspace()
        or unicodedata.category(character) in {"Cc", "Zl", "Zp"}
        or (
            unicodedata.category(character) == "Cf"
            and character not in {"\u200c", "\u200d"}
        )
        else character
        for character in decoded
    )
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _finish(result: SearchResult, query: SearchQuery) -> SearchResult:
    """Apply URL safety, the client-side domain filter, and the result cap."""
    kept: list[ResultItem] = []
    for item in result.items:
        item.url = item.url.strip()
        if is_public_http_url(item.url) and query.allows(unescape(item.url)):
            kept.append(item)
        else:
            result.dropped += 1
    result.items = kept[: query.num_results]
    if (query.include_domains or query.exclude_domains) and not result.items:
        # A provider-generated answer cannot be proven in-scope without at least
        # one surviving supporting URL.
        result.answer = None
    return result


def _redact_known(text: str, secrets: Sequence[str]) -> str:
    for secret in sorted(set(secrets), key=len, reverse=True):
        if not secret:
            continue
        text = text.replace(secret, "***")
    return text


def _raise_for_status(
    response: httpx.Response,
    backend: str,
    secrets: Sequence[str] = (),
) -> None:
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
        detail = response.text
    detail = _redact_known(detail, secrets)
    message = f"{backend} returned HTTP {response.status_code}"
    if detail:
        message = f"{message}: {_clean(detail, 200)}"
    raise BackendError(
        message,
        status=response.status_code,
        retryable=response.status_code in FAILOVER_STATUSES,
    )


async def _request(
    client: httpx.AsyncClient,
    backend: str,
    method: str,
    url: str,
    *,
    secrets: Sequence[str] = (),
    **kwargs: Any,
) -> httpx.Response:
    """Issue one provider request without eagerly buffering an unbounded body."""
    try:
        follow_redirects = bool(kwargs.pop("follow_redirects", False))
        request = client.build_request(method, url, **kwargs)
        request.headers["accept-encoding"] = "identity"
        response = await client.send(
            request, stream=True, follow_redirects=follow_redirects
        )
        try:
            encoding = response.headers.get("content-encoding", "").strip().lower()
            if encoding not in ("", "identity"):
                raise BackendError(
                    f"{backend} ignored identity encoding ({encoding})",
                    retryable=False,
                )
            content = bytearray()
            if response.is_stream_consumed:
                content.extend(response.content[: MAX_BACKEND_BYTES + 1])
            else:
                async for chunk in response.aiter_raw(chunk_size=65536):
                    remaining = MAX_BACKEND_BYTES + 1 - len(content)
                    if remaining <= 0:
                        break
                    content.extend(chunk[:remaining])
                    if len(content) > MAX_BACKEND_BYTES:
                        break
        finally:
            await response.aclose()
    except BackendError:
        raise
    except httpx.HTTPError as error:
        raise BackendError(
            f"{backend} failed before a complete HTTP response: {type(error).__name__}"
        ) from error

    if len(content) > MAX_BACKEND_BYTES:
        raise BackendError(
            f"{backend} response exceeded {MAX_BACKEND_BYTES:,} bytes",
            retryable=False,
        )
    bounded = httpx.Response(
        response.status_code,
        headers=response.headers,
        content=bytes(content),
        request=request,
    )
    _raise_for_status(bounded, backend, secrets)
    return bounded


def _entries(value: Any) -> Sequence[Any]:
    return value[:MAX_BACKEND_ITEMS] if isinstance(value, list) else ()

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


def _is_grounding_redirect(url: str) -> bool:
    """Match the Google redirector by hostname, never by attacker-controlled text."""
    try:
        parts = urlparse(url)
        return (
            parts.scheme == "https"
            and (parts.hostname or "").lower().rstrip(".") == GROUNDING_REDIRECT_HOST
            and parts.port in (None, 443)
            and "@" not in parts.netloc
        )
    except ValueError:
        return False


async def _resolve_redirect(client: httpx.AsyncClient, url: str) -> str:
    """Resolve a grounding redirect to its publisher URL without fetching it.

    Uses `follow_redirects=False` and reads `Location`, so the publisher host is
    never contacted. Only the exact Google redirector hostname may receive HEAD.
    """
    if not is_public_http_url(url) or not _is_grounding_redirect(url):
        return url
    current = url
    for _ in range(MAX_REDIRECT_HOPS):
        try:
            request = client.build_request(
                "HEAD",
                current,
                headers={"accept-encoding": "identity"},
                timeout=REDIRECT_TIMEOUT,
            )
            response = await client.send(
                request, stream=True, follow_redirects=False
            )
            try:
                location = response.headers.get("location")
            finally:
                await response.aclose()
        except httpx.HTTPError:
            return url
        if not location:
            return url if current == url else current
        candidate = urljoin(current, location)
        if not is_public_http_url(candidate):
            return url
        current = candidate
        if not _is_grounding_redirect(current):
            return current
    return url


async def _resolve_redirects(
    client: httpx.AsyncClient,
    items: Sequence[ResultItem],
    *,
    max_candidates: int,
) -> None:
    targets = [
        item for item in items if _is_grounding_redirect(item.url)
    ][:max_candidates]
    if not targets:
        return
    semaphore = asyncio.Semaphore(MAX_REDIRECT_CONCURRENCY)

    async def resolve(item: ResultItem) -> None:
        async with semaphore:
            value = await _resolve_redirect(client, item.url)
            if value:
                item.url = value

    tasks = [asyncio.create_task(resolve(item)) for item in targets]
    try:
        done, pending = await asyncio.wait(
            tasks, timeout=REDIRECT_RESOLUTION_BUDGET
        )
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        # A failed redirect is left unresolved and dropped by _finish_grounded.
        try:
            task.result()
        except (Exception, asyncio.CancelledError):
            pass


async def _gemini_studio_models(client: httpx.AsyncClient, endpoint: GeminiEndpoint, key: str) -> tuple[str, ...]:
    """List models for the public endpoint, falling back to known ids."""
    try:
        response = await _request(
            client,
            "gemini models",
            "GET",
            f"{endpoint.base_url}/models",
            secrets=(key,),
            headers={"x-goog-api-key": key},
        )
        payload = response.json()
    except (BackendError, ValueError):
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


def _annotate_citations(
    answer: str,
    metadata: dict[str, Any],
    source_numbers: dict[int, int],
    part_ranges: dict[int, tuple[int, int]],
) -> str:
    """Append markers whose chunk indices still map to displayed sources."""
    supports = metadata.get("groundingSupports")
    if not answer or not isinstance(supports, list) or not source_numbers:
        return answer.strip()

    encoded = answer.encode("utf-8")
    # Merge supports at the same byte offset. Inserting them separately reverses
    # provider order and can duplicate the same source marker.
    insertions: dict[int, set[int]] = {}
    for support in supports[:MAX_GROUNDING_SUPPORTS]:
        if not isinstance(support, dict):
            continue
        segment = support.get("segment")
        indices = support.get("groundingChunkIndices")
        if not isinstance(segment, dict) or not isinstance(indices, list):
            continue
        relative_end = segment.get("endIndex")
        if "partIndex" in segment:
            part_index = segment.get("partIndex")
        else:
            part_index = next(iter(part_ranges)) if len(part_ranges) == 1 else 0
        if type(relative_end) is not int or type(part_index) is not int:
            continue
        part_range = part_ranges.get(part_index)
        if part_range is None:
            continue
        part_start, part_end = part_range
        end = part_start + relative_end
        if not part_start < end <= part_end:
            continue
        if end > len(encoded) or (
            end < len(encoded) and encoded[end] & 0xC0 == 0x80
        ):
            continue
        while end > part_start and encoded[end - 1 : end].isspace():
            end -= 1
        numbers = {
            source_numbers[index]
            for index in indices[:MAX_SUPPORT_INDICES]
            if type(index) is int and index in source_numbers
        }
        if numbers and end > 0:
            insertions.setdefault(end, set()).update(numbers)

    pieces: list[bytes] = []
    cursor = 0
    for end in sorted(insertions):
        pieces.append(encoded[cursor:end])
        pieces.append(
            "".join(f"[{number}]" for number in sorted(insertions[end])).encode("utf-8")
        )
        cursor = end
    pieces.append(encoded[cursor:])
    return b"".join(pieces).decode("utf-8").strip()


def _parse_gemini(
    payload: dict[str, Any],
) -> tuple[
    Optional[str],
    list[ResultItem],
    list[str],
    dict[str, Any],
    dict[int, int],
    dict[int, tuple[int, int]],
]:
    """Parse grounding data while retaining each provider chunk's item index."""
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise BackendError("gemini returned no candidates", retryable=False)
    candidate = candidates[0] if isinstance(candidates[0], dict) else {}

    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    answer_parts: list[str] = []
    part_ranges: dict[int, tuple[int, int]] = {}
    byte_offset = 0
    if isinstance(parts, list):
        for part_index, part in enumerate(parts):
            text = part.get("text") if isinstance(part, dict) else None
            if isinstance(text, str):
                encoded_length = len(text.encode("utf-8"))
                part_ranges[part_index] = (byte_offset, byte_offset + encoded_length)
                answer_parts.append(text)
                byte_offset += encoded_length
    answer = "".join(answer_parts)
    answer_bytes = answer.encode("utf-8")
    if len(answer_bytes) > MAX_GEMINI_ANSWER_BYTES:
        cut = MAX_GEMINI_ANSWER_BYTES
        while cut > 0 and answer_bytes[cut] & 0xC0 == 0x80:
            cut -= 1
        answer = (
            answer_bytes[:cut].decode("utf-8")
            + f"\n\n[provider answer truncated at {MAX_GEMINI_ANSWER_BYTES:,} UTF-8 bytes]"
        )
        part_ranges = {
            index: (start, min(end, cut))
            for index, (start, end) in part_ranges.items()
            if start < cut
        }

    metadata = candidate.get("groundingMetadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    items: list[ResultItem] = []
    item_by_url: dict[str, int] = {}
    chunk_to_item: dict[int, int] = {}
    chunks = metadata.get("groundingChunks")
    if isinstance(chunks, list):
        for chunk_index, chunk in enumerate(chunks[:MAX_BACKEND_ITEMS]):
            if not isinstance(chunk, dict):
                continue
            web = chunk.get("web")
            if not isinstance(web, dict):
                continue
            url = web.get("uri")
            if not isinstance(url, str):
                continue
            url = url.strip()
            if not url:
                continue
            item_index = item_by_url.get(url)
            if item_index is None:
                if len(items) >= MAX_BACKEND_ITEMS:
                    continue
                item_index = len(items)
                item_by_url[url] = item_index
                items.append(ResultItem(title=_clean(web.get("title")) or url, url=url))
            chunk_to_item[chunk_index] = item_index

    queries = [
        cleaned
        for query in _entries(metadata.get("webSearchQueries"))
        if (cleaned := _clean(query, 500))
    ]
    if not answer.strip() and not items:
        raise BackendError("gemini returned no grounded content", retryable=False)
    return (
        answer if answer.strip() else None,
        items,
        queries,
        metadata,
        chunk_to_item,
        part_ranges,
    )


def _finish_grounded(
    items: list[ResultItem],
    chunk_to_item: dict[int, int],
    query: SearchQuery,
) -> tuple[list[ResultItem], dict[int, int], int]:
    """Filter resolved sources and map original chunks to final source numbers."""
    kept: list[ResultItem] = []
    item_to_source: dict[int, int] = {}
    source_by_url: dict[str, int] = {}
    dropped = 0
    for item_index, item in enumerate(items):
        item.url = item.url.strip()
        if (
            _is_grounding_redirect(item.url)
            or _is_grounding_redirect(unescape(item.url))
            or not is_public_http_url(item.url)
            or not query.allows(unescape(item.url))
        ):
            dropped += 1
            continue
        source_number = source_by_url.get(item.url)
        if source_number is None:
            if len(kept) >= query.num_results:
                continue
            kept.append(item)
            source_number = len(kept)
            source_by_url[item.url] = source_number
        item_to_source[item_index] = source_number

    chunk_to_source = {
        chunk_index: item_to_source[item_index]
        for chunk_index, item_index in chunk_to_item.items()
        if item_index in item_to_source
    }
    return kept, chunk_to_source, dropped


async def search_gemini(client: httpx.AsyncClient, query: SearchQuery, settings: Settings) -> SearchResult:
    endpoints = settings.gemini_endpoints
    if not endpoints:
        raise BackendError("no gemini endpoint is configured", retryable=False)

    # Grounding has no filter fields, so constraints go into the prompt itself.
    prompt = query.operator_text(with_recency_hint=True)
    errors: list[str] = []
    for endpoint in endpoints:
        for key in endpoint.keys:
            # A pinned model is already usable evidence; do not delay it behind a
            # model-list request that a gateway may not expose.
            usable = endpoint
            if not settings.gemini_model and not endpoint.models:
                usable = endpoint.with_models(await _gemini_studio_models(client, endpoint, key))
            model = usable.pick_model(settings.gemini_model)
            if not model:
                errors.append(f"{endpoint.label}: no usable model")
                continue

            url = f"{usable.base_url}/models/{model}:generateContent"
            headers = {"x-goog-api-key": key, "content-type": "application/json"}
            # google_search is the Gemini 2+ tool; google_search_retrieval is the
            # older name still required by some gateways.
            retry_next_key = True
            for tool in ({"google_search": {}}, {"google_search_retrieval": {}}):
                body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "tools": [tool]}
                try:
                    response = await _request(
                        client,
                        "gemini",
                        "POST",
                        url,
                        secrets=settings.secrets,
                        headers=headers,
                        json=body,
                    )
                    (
                        answer,
                        items,
                        queries,
                        metadata,
                        chunk_to_item,
                        part_ranges,
                    ) = _parse_gemini(_json_body(response, "gemini"))
                except BackendError as error:
                    errors.append(f"{endpoint.label}/{model}: {error}")
                    # A 400 may mean this gateway needs the legacy tool shape.
                    # If both shapes fail, another key would repeat the same error.
                    if error.status == 400:
                        retry_next_key = False
                        continue
                    retry_next_key = error.retryable
                    break

                await _resolve_redirects(
                    client,
                    items,
                    max_candidates=max(10, query.num_results * 2),
                )
                items, source_numbers, dropped = _finish_grounded(items, chunk_to_item, query)
                if (query.include_domains or query.exclude_domains) and not items:
                    answer = None
                if answer:
                    answer = _annotate_citations(
                        answer, metadata, source_numbers, part_ranges
                    )
                return SearchResult(
                    backend="gemini",
                    detail=f"{usable.label}/{model}",
                    answer=answer,
                    items=items,
                    queries=queries,
                    dropped=dropped,
                )
            if not retry_next_key:
                break

    raise BackendError("; ".join(errors) or "every gemini endpoint failed")


# --------------------------------------------------------------------------- #
# Credential-based JSON APIs
# --------------------------------------------------------------------------- #

SERPER_TBS = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}
BRAVE_FRESHNESS = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}
DDG_DF = {"day": "d", "week": "w", "month": "m", "year": "y"}


async def search_serper(client: httpx.AsyncClient, query: SearchQuery, settings: Settings) -> SearchResult:
    key = settings.simple_key("serper")
    if not key:
        raise BackendError("no serper credential is configured", retryable=False)
    body: dict[str, Any] = {"q": query.operator_text(), "num": query.num_results}
    if query.recency:
        body["tbs"] = SERPER_TBS[query.recency]
    response = await _request(
        client,
        "serper",
        "POST",
        "https://google.serper.dev/search",
        secrets=settings.secrets,
        headers={"X-API-KEY": key, "content-type": "application/json"},
        json=body,
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
    for entry in _entries(payload.get("organic")):
        if not isinstance(entry, dict):
            continue
        url = entry.get("link")
        if not isinstance(url, str):
            continue
        items.append(ResultItem(_clean(entry.get("title")) or url, url, _clean(entry.get("snippet"))))

    answer = "\n".join(part for part in answer_parts if part) or None
    return _finish(
        SearchResult(backend="serper", detail="google.serper.dev", answer=answer, items=items), query
    )


async def search_tavily(client: httpx.AsyncClient, query: SearchQuery, settings: Settings) -> SearchResult:
    key = settings.simple_key("tavily")
    if not key:
        raise BackendError("no tavily credential is configured", retryable=False)
    body: dict[str, Any] = {
        "query": query.text,
        "max_results": query.num_results,
        "include_answer": True,
        "search_depth": "basic",
    }
    if query.recency:
        body["time_range"] = query.recency
    if query.include_domains:
        body["include_domains"] = list(query.include_domains)
    if query.exclude_domains:
        body["exclude_domains"] = list(query.exclude_domains)

    response = await _request(
        client,
        "tavily",
        "POST",
        "https://api.tavily.com/search",
        secrets=settings.secrets,
        headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
        json=body,
    )
    payload = _json_body(response, "tavily")
    items: list[ResultItem] = []
    for entry in _entries(payload.get("results")):
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str):
            continue
        items.append(ResultItem(_clean(entry.get("title")) or url, url, _clean(entry.get("content"))))
    return _finish(
        SearchResult(
            backend="tavily",
            detail="api.tavily.com",
            answer=_clean(payload.get("answer"), 1500) or None,
            items=items,
        ),
        query,
    )


async def search_brave(client: httpx.AsyncClient, query: SearchQuery, settings: Settings) -> SearchResult:
    key = settings.simple_key("brave")
    if not key:
        raise BackendError("no brave credential is configured", retryable=False)
    params: dict[str, Any] = {
        "q": query.operator_text(),
        # Domain filters are query operators here, so ask for more and filter.
        "count": 20 if (query.include_domains or query.exclude_domains) else query.num_results,
    }
    if query.recency:
        params["freshness"] = BRAVE_FRESHNESS[query.recency]
    response = await _request(
        client,
        "brave",
        "GET",
        "https://api.search.brave.com/res/v1/web/search",
        secrets=settings.secrets,
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
        params=params,
    )
    payload = _json_body(response, "brave")
    web = payload.get("web")
    entries = web.get("results") if isinstance(web, dict) else None
    items: list[ResultItem] = []
    for entry in _entries(entries):
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str):
            continue
        items.append(ResultItem(_clean(entry.get("title")) or url, url, _clean(entry.get("description"))))
    return _finish(SearchResult(backend="brave", detail="api.search.brave.com", items=items), query)


async def search_exa(client: httpx.AsyncClient, query: SearchQuery, settings: Settings) -> SearchResult:
    key = settings.simple_key("exa")
    if not key:
        raise BackendError("no exa credential is configured", retryable=False)
    body: dict[str, Any] = {
        "query": query.text,
        "numResults": query.num_results,
        "contents": {"text": {"maxCharacters": 500}},
    }
    if query.recency:
        body["startPublishedDate"] = recency_start_date(query.recency)
    if query.include_domains:
        body["includeDomains"] = list(query.include_domains)
    if query.exclude_domains:
        body["excludeDomains"] = list(query.exclude_domains)

    response = await _request(
        client,
        "exa",
        "POST",
        "https://api.exa.ai/search",
        secrets=settings.secrets,
        headers={"x-api-key": key, "content-type": "application/json"},
        json=body,
    )
    payload = _json_body(response, "exa")
    items: list[ResultItem] = []
    for entry in _entries(payload.get("results")):
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str):
            continue
        snippet = entry.get("text") or entry.get("summary") or entry.get("highlights")
        if isinstance(snippet, list):
            snippet = " ".join(str(part) for part in snippet)
        items.append(ResultItem(_clean(entry.get("title")) or url, url, _clean(snippet)))
    return _finish(SearchResult(backend="exa", detail="api.exa.ai", items=items), query)


async def search_searxng(client: httpx.AsyncClient, query: SearchQuery, settings: Settings) -> SearchResult:
    base = settings.searxng_url
    if not base:
        raise BackendError("SEARXNG_URL is not set", retryable=False)
    params: dict[str, Any] = {"q": query.operator_text(), "format": "json"}
    if query.recency:
        params["time_range"] = query.recency
    response = await _request(
        client,
        "searxng",
        "GET",
        f"{base}/search",
        secrets=settings.secrets,
        params=params,
        headers={"Accept": "application/json"},
    )
    payload = _json_body(response, "searxng")
    items: list[ResultItem] = []
    for entry in _entries(payload.get("results")):
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str):
            continue
        items.append(ResultItem(_clean(entry.get("title")) or url, url, _clean(entry.get("content"))))
    if not items:
        raise BackendError("searxng returned no results (instance may block format=json)", retryable=False)
    return _finish(
        SearchResult(
            backend="searxng",
            detail=urlparse(safe_endpoint_label(base)).netloc or "configured endpoint",
            items=items,
        ),
        query,
    )


# --------------------------------------------------------------------------- #
# DuckDuckGo (no credential)
# --------------------------------------------------------------------------- #

_DDG_ENDPOINTS = ("https://html.duckduckgo.com/html/", "https://lite.duckduckgo.com/lite/")
_DDG_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_ANCHOR_RE = re.compile(
    r'<a[^>]+class="[^"]*result(?:__a|-link)[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I
)
_TAG_RE = re.compile(r"<[^>]+>")


def _unwrap_ddg_url(href: str) -> str:
    url = href.strip()
    if url.startswith("//"):
        url = f"https:{url}"
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    host = (parsed.hostname or "").lower().rstrip(".")
    if (host == "duckduckgo.com" or host.endswith(".duckduckgo.com")) and parsed.path.startswith("/l"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            # parse_qs has already decoded the query component exactly once.
            return target[0]
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


async def search_ddg(client: httpx.AsyncClient, query: SearchQuery, settings: Settings) -> SearchResult:
    data: dict[str, str] = {"q": query.operator_text()}
    if query.recency:
        data["df"] = DDG_DF[query.recency]
    # Domain operators are unreliable on the HTML endpoints, so over-fetch and
    # let the client-side filter in _finish() do the work.
    want = 20 if (query.include_domains or query.exclude_domains) else query.num_results

    errors: list[str] = []
    for url in _DDG_ENDPOINTS:
        try:
            response = await _request(
                client,
                "ddg",
                "POST",
                url,
                secrets=settings.secrets,
                headers={"user-agent": _DDG_USER_AGENT},
                data=data,
                follow_redirects=False,
            )
        except BackendError as error:
            errors.append(str(error))
            continue

        if response.is_redirect:
            errors.append(
                f"{urlparse(url).netloc} redirected the request (HTTP {response.status_code})"
            )
            continue
        html = response.text
        items = _parse_ddg_with_bs4(html, want) or _parse_ddg_with_regex(html, want)
        if items:
            return _finish(
                SearchResult(backend="ddg", detail=urlparse(url).netloc, items=items), query
            )
        # DuckDuckGo answers 202 with an empty body when it rate-limits a client.
        if response.status_code == 202:
            errors.append(f"{urlparse(url).netloc} rate-limited the request (HTTP 202)")
        else:
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
