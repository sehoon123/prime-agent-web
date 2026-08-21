"""Bounded HTTP responses from configured Gemini-compatible providers."""

from __future__ import annotations

from typing import Any

import httpx

MAX_PROVIDER_BYTES = 5 * 1024 * 1024


class ProviderResponseTooLarge(RuntimeError):
    """A provider returned more data than this client will buffer."""


async def request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_bytes: int = MAX_PROVIDER_BYTES,
    **kwargs: Any,
) -> httpx.Response:
    """Send one request and return a bounded, fully buffered response."""
    request = client.build_request(method, url, **kwargs)
    request.headers["accept-encoding"] = "identity"
    response = await client.send(request, stream=True, follow_redirects=False)
    try:
        if response.is_stream_consumed:
            content = response.content[: max_bytes + 1]
        else:
            encoding = response.headers.get("content-encoding", "").strip().lower()
            if encoding not in ("", "identity"):
                raise RuntimeError(
                    f"provider ignored identity encoding ({encoding})"
                )
            buffered = bytearray()
            async for chunk in response.aiter_raw(chunk_size=65536):
                remaining = max_bytes + 1 - len(buffered)
                if remaining <= 0:
                    break
                buffered.extend(chunk[:remaining])
                if len(buffered) > max_bytes:
                    break
            content = bytes(buffered)
    finally:
        await response.aclose()

    if len(content) > max_bytes:
        raise ProviderResponseTooLarge(
            f"provider response exceeded {max_bytes:,} bytes"
        )
    return httpx.Response(
        response.status_code,
        headers=response.headers,
        content=content,
        request=request,
    )
