"""URL validation and guarded HTTP fetching.

This skill fetches URLs the model found in untrusted places - search results, page
content, user text. A naive fetch is an SSRF primitive: a prompt-injected link can
point an authenticated client at cloud metadata or an internal service. Every
request therefore goes through `guarded_get`, which validates the URL *and every
redirect hop* both syntactically and by DNS resolution, and caps body size.
"""

from __future__ import annotations

import asyncio
import ipaddress
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Sequence
from urllib.parse import urljoin, urlsplit

import httpx

MAX_REDIRECTS = 5
DEFAULT_MAX_BYTES = 10 * 1024 * 1024

# Identify the client and its autonomy, following the convention used by the
# official MCP fetch server so operators can recognise agent traffic in logs.
USER_AGENT_AUTONOMOUS = (
    "prime-agent-webfetch/0.3 (Autonomous; +https://github.com/sehoon123/prime-agent-web)"
)
USER_AGENT_MANUAL = (
    "prime-agent-webfetch/0.3 (User-Specified; +https://github.com/sehoon123/prime-agent-web)"
)

_BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata",
        "metadata.google.internal",
        "instance-data",
    }
)
_BLOCKED_SUFFIXES = (".local", ".internal", ".localdomain", ".home.arpa", ".onion")

Resolver = Callable[[str], Awaitable[Sequence[str]]]


class FetchError(RuntimeError):
    """A fetch could not be completed. Message is safe to show the model."""


class TooLargeError(FetchError):
    """The body exceeds the byte cap. Message says how to raise it."""

    def __init__(self, url: str, max_bytes: int, actual: Optional[int] = None) -> None:
        size = f"{actual:,} bytes" if actual else "more than the cap"
        super().__init__(
            f"{url} is {size}, over the {max_bytes:,}-byte limit. "
            f"Retry with max_bytes={max(max_bytes * 4, (actual or 0) + 1_048_576):,} "
            "(or set PRIME_AGENT_WEBFETCH_MAX_BYTES) if the whole body is needed."
        )
        self.max_bytes = max_bytes
        self.actual = actual


class UnsafeUrlError(FetchError):
    """The URL, or a redirect it produced, targets a non-public address."""


def is_public_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def check_url_syntax(url: str) -> str:
    """Validate scheme/authority and return the normalized URL.

    Rejects non-http(s) schemes, embedded credentials, and hostnames that are
    literal private addresses or well-known internal names. Raises UnsafeUrlError.
    """
    raw = (url or "").strip()
    if not raw:
        raise UnsafeUrlError("no URL was given")
    try:
        parts = urlsplit(raw)
    except ValueError as error:
        raise UnsafeUrlError(f"malformed URL: {error}") from error

    if parts.scheme not in ("http", "https"):
        raise UnsafeUrlError(
            f"only http(s) URLs can be fetched, got {parts.scheme or 'no'} scheme. "
            "Read local files with open() instead."
        )
    if "@" in (parts.netloc or ""):
        raise UnsafeUrlError("URLs carrying credentials are refused")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise UnsafeUrlError("URL has no host")
    if host in _BLOCKED_HOSTS or host.endswith(_BLOCKED_SUFFIXES):
        raise UnsafeUrlError(f"refusing to fetch internal host {host!r}")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if "." not in host:
            raise UnsafeUrlError(f"refusing to fetch bare internal name {host!r}") from None
    else:
        if not is_public_ip(host):
            raise UnsafeUrlError(f"refusing to fetch non-public address {host!r}")
    return raw


async def default_resolver(hostname: str) -> Sequence[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(hostname, None)
    return [info[4][0] for info in infos]


async def check_host_resolves_public(hostname: str, resolver: Optional[Resolver] = None) -> None:
    """DNS preflight: every resolved address must be public.

    Catches names that legitimately resolve to private space (`foo.example.com ->
    10.0.0.5`), which syntax checks cannot see.
    """
    try:
        ipaddress.ip_address(hostname)
        return  # literal addresses were already checked syntactically
    except ValueError:
        pass

    resolve = resolver or default_resolver
    try:
        addresses = await resolve(hostname)
    except Exception as error:  # DNS failure is a fetch failure, not a safety verdict
        raise FetchError(f"could not resolve {hostname}: {type(error).__name__}") from error
    if not addresses:
        raise FetchError(f"could not resolve {hostname}")
    for address in addresses:
        if not is_public_ip(address):
            raise UnsafeUrlError(f"{hostname} resolves to non-public address {address}")


@dataclass
class FetchedBody:
    """A completed, size-capped HTTP response."""

    requested_url: str
    final_url: str
    status: int
    content_type: str
    content: bytes
    encoding: Optional[str]
    truncated: bool = False
    redirects: int = 0

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding or "utf-8", errors="replace")


async def guarded_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    resolver: Optional[Resolver] = None,
    headers: Optional[dict[str, str]] = None,
) -> FetchedBody:
    """GET `url`, validating every hop and streaming at most `max_bytes`.

    Redirects are followed manually so each hop is validated before it is
    requested; `httpx`'s own follow_redirects would jump to an unvalidated host.
    """
    current = check_url_syntax(url)
    await check_host_resolves_public(urlsplit(current).hostname or "", resolver)

    for hop in range(MAX_REDIRECTS + 1):
        try:
            request = client.build_request("GET", current, headers=headers)
            response = await client.send(request, stream=True, follow_redirects=False)
        except httpx.HTTPError as error:
            raise FetchError(f"request failed: {type(error).__name__}: {error}") from error

        # Reject oversized bodies from the header before spending bandwidth on them.
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            await response.aclose()
            raise TooLargeError(current, max_bytes, int(declared))

        location = response.headers.get("location")
        if response.is_redirect and location:
            await response.aclose()
            if hop == MAX_REDIRECTS:
                raise FetchError(f"too many redirects (>{MAX_REDIRECTS}) starting at {url}")
            candidate = check_url_syntax(urljoin(current, location))
            await check_host_resolves_public(urlsplit(candidate).hostname or "", resolver)
            current = candidate
            continue

        try:
            chunks: list[bytes] = []
            size = 0
            truncated = False
            async for chunk in response.aiter_bytes():
                chunks.append(chunk)
                size += len(chunk)
                if size >= max_bytes:
                    truncated = True
                    break
        finally:
            await response.aclose()

        body = b"".join(chunks)[:max_bytes]
        if response.status_code >= 400:
            detail = body[:200].decode("utf-8", errors="replace").strip()
            raise FetchError(
                f"HTTP {response.status_code} from {current}" + (f": {detail}" if detail else "")
            )
        return FetchedBody(
            requested_url=url,
            final_url=current,
            status=response.status_code,
            content_type=(response.headers.get("content-type") or "").split(";")[0].strip().lower(),
            content=body,
            encoding=response.encoding,
            truncated=truncated,
            redirects=hop,
        )

    raise FetchError(f"too many redirects (>{MAX_REDIRECTS}) starting at {url}")
