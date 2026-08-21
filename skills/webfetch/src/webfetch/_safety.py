"""URL validation and guarded HTTP fetching.

This skill fetches URLs the model found in untrusted places - search results, page
content, user text. A naive fetch is an SSRF primitive: a prompt-injected link can
point an authenticated client at cloud metadata or an internal service. Every
request therefore goes through `guarded_get`, which validates the URL *and every
redirect hop* both syntactically and by DNS resolution, and caps body size.
"""

from __future__ import annotations

import asyncio
import codecs
import html
import ipaddress
import re
import unicodedata
import zlib
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Sequence
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

MAX_REDIRECTS = 5
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
MAX_URL_CHARS = 8192

# Identify the client and its autonomy, following the convention used by the
# official MCP fetch server so operators can recognise agent traffic in logs.
USER_AGENT_AUTONOMOUS = (
    "prime-agent-webfetch/0.6.2 (Autonomous; +https://github.com/sehoon123/prime-agent-web)"
)
USER_AGENT_MANUAL = (
    "prime-agent-webfetch/0.6.2 (User-Specified; +https://github.com/sehoon123/prime-agent-web)"
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
    except (TypeError, ValueError):
        return False
    if getattr(address, "scope_id", None) is not None:
        return False
    return bool(
        address.is_global
        and not address.is_reserved
        and not address.is_multicast
        and not getattr(address, "is_site_local", False)
    )


def _looks_like_noncanonical_ipv4(host: str) -> bool:
    labels = host.split(".")
    if not 1 <= len(labels) <= 4 or any(not label for label in labels):
        return False
    return all(
        label.isdigit()
        or (
            label.lower().startswith("0x")
            and len(label) > 2
            and all(character in "0123456789abcdef" for character in label[2:].lower())
        )
        for label in labels
    )


def _validate_url_candidate(raw: str) -> None:
    try:
        parts = urlsplit(raw)
        # urlsplit defers malformed-port validation until .port is accessed.
        _ = parts.port
    except ValueError as error:
        raise UnsafeUrlError(f"malformed URL: {error}") from error

    if parts.scheme not in ("http", "https"):
        raise UnsafeUrlError(
            f"only http(s) URLs can be fetched, got {parts.scheme or 'no'} scheme. "
            "Read local files with open() instead."
        )
    if "%" in (parts.netloc or ""):
        raise UnsafeUrlError("URL authority cannot contain percent escapes")
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
        if _looks_like_noncanonical_ipv4(host):
            raise UnsafeUrlError(f"refusing noncanonical numeric hostname {host!r}") from None
        if "." not in host:
            raise UnsafeUrlError(f"refusing to fetch bare internal name {host!r}") from None
    else:
        if not is_public_ip(host):
            raise UnsafeUrlError(f"refusing to fetch non-public address {host!r}")


def check_url_syntax(url: str) -> str:
    """Validate the literal and renderer-decoded URL, returning the literal URL."""
    supplied = url or ""
    if len(supplied) > MAX_URL_CHARS:
        raise UnsafeUrlError(f"URL exceeds {MAX_URL_CHARS:,} characters")
    decoded = html.unescape(supplied)
    if any(
        character.isspace()
        or character == "\\"
        or unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
        for character in decoded
    ):
        raise UnsafeUrlError("URL contains whitespace, controls, or backslashes")
    raw = supplied.strip()
    if not raw:
        raise UnsafeUrlError("no URL was given")
    _validate_url_candidate(raw)
    decoded = html.unescape(raw)
    if decoded != raw:
        _validate_url_candidate(decoded)
    return raw


async def default_resolver(hostname: str) -> Sequence[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(hostname, None)
    return [info[4][0] for info in infos]


async def check_host_resolves_public(
    hostname: str,
    resolver: Optional[Resolver] = None,
    timeout: Optional[float] = None,
) -> Sequence[str]:
    """DNS preflight: return resolved addresses only when every one is public.

    Catches names that legitimately resolve to private space (`foo.example.com ->
    10.0.0.5`), which syntax checks cannot see.
    """
    try:
        ipaddress.ip_address(hostname)
        return (hostname,)  # literal addresses were already checked syntactically
    except ValueError:
        pass

    resolve = resolver or default_resolver
    try:
        if timeout is None:
            addresses = await resolve(hostname)
        else:
            addresses = await asyncio.wait_for(resolve(hostname), timeout=max(0.001, timeout))
    except Exception as error:  # DNS failure is a fetch failure, not a safety verdict
        raise FetchError(f"could not resolve {hostname}: {type(error).__name__}") from error
    if not addresses:
        raise FetchError(f"could not resolve {hostname}")
    for address in addresses:
        if not is_public_ip(address):
            raise UnsafeUrlError(f"{hostname} resolves to non-public address {address}")
    return tuple(addresses)


def _pin_request_target(
    client: httpx.AsyncClient,
    url: str,
    address: Optional[str],
    headers: Optional[dict[str, str]],
) -> tuple[str, dict[str, str], dict[str, object]]:
    """Connect the native transport to a vetted IP while preserving Host and TLS SNI.

    A separate DNS check is vulnerable to rebinding because the HTTP transport
    resolves the hostname again. Injected transports are caller-controlled and
    do not open native sockets, so they keep the original URL for testability.
    """
    transport = getattr(client, "_transport", None)
    if not isinstance(transport, httpx.AsyncHTTPTransport) or address is None:
        return url, dict(headers or {}), {}

    parts = urlsplit(url)
    original = httpx.URL(url)
    host = f"[{address}]" if ":" in address else address
    if parts.port is not None:
        host += f":{parts.port}"
    target = urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    request_headers = dict(headers or {})
    logical_host = original.raw_host.decode("ascii").rstrip(".")
    logical_authority = f"[{logical_host}]" if ":" in logical_host else logical_host
    if original.port is not None:
        logical_authority += f":{original.port}"
    request_headers["host"] = logical_authority
    # The pool is keyed by the pinned IP. Closing each response prevents a TLS
    # connection for one hostname being reused for another hostname on that IP.
    request_headers["connection"] = "close"
    extensions: dict[str, object] = {}
    if parts.scheme == "https":
        extensions["sni_hostname"] = logical_host
    return target, request_headers, extensions


def _origin_key(url: str) -> tuple[str, str, Optional[int]]:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    port = parts.port or (443 if scheme == "https" else 80 if scheme == "http" else None)
    return scheme, (parts.hostname or "").lower().rstrip("."), port


async def _send_pinned_request(
    client: httpx.AsyncClient,
    request: httpx.Request,
    logical_url: str,
    *,
    native_transport: bool,
) -> httpx.Response:
    """Use logical proxy routing while the actual request URL remains IP-pinned."""
    selector = getattr(client, "_transport_for_url", None)
    if native_transport and callable(selector):
        transport = selector(httpx.URL(logical_url))
        response = await transport.handle_async_request(request)
        response.request = request
        return response
    return await client.send(request, stream=True, follow_redirects=False)


_HTML_META_CHARSET = re.compile(
    br"<meta[^>]+charset\s*=\s*[\"']?([A-Za-z0-9._-]+)", re.IGNORECASE
)


def _html_meta_encoding(content: bytes) -> Optional[str]:
    match = _HTML_META_CHARSET.search(content[:4096])
    if match is None:
        return None
    try:
        name = match.group(1).decode("ascii")
        return codecs.lookup(name).name
    except (LookupError, UnicodeDecodeError):
        return None


async def _read_bounded_body(
    response: httpx.Response, max_bytes: int
) -> tuple[bytes, bool]:
    """Read one decoded byte beyond the cap without decoder-sized allocations."""
    if response.is_stream_consumed:
        # httpx has already transfer-decoded in-memory transport responses.
        # Such injected transports are caller-controlled; cap the available body
        # directly and never feed decoded bytes through the wire decoder again.
        content = response.content
        return content[:max_bytes], len(content) > max_bytes

    encoding = response.headers.get("content-encoding", "").strip().lower()
    gzip_encoded = encoding in ("gzip", "x-gzip")
    if encoding in ("", "identity"):
        decoder: Optional[zlib.Decompress] = None
    elif gzip_encoded:
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif encoding == "deflate":
        decoder = zlib.decompressobj()
    else:
        raise FetchError(f"unsupported content encoding for bounded fetch: {encoding}")

    limit = max_bytes + 1
    content = bytearray()
    raw_seen = 0

    try:
        async for raw_chunk in response.aiter_raw(chunk_size=65536):
            raw_seen += len(raw_chunk)
            if raw_seen > max_bytes + 65536:
                raise TooLargeError(str(response.url), max_bytes, raw_seen)
            if decoder is None:
                content.extend(raw_chunk[: limit - len(content)])
            else:
                pending = raw_chunk
                while pending and len(content) < limit:
                    if decoder.eof:
                        if not gzip_encoded:
                            raise FetchError("compressed response has trailing data")
                        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
                    before_pending = len(pending)
                    before_content = len(content)
                    content.extend(decoder.decompress(pending, limit - len(content)))
                    if decoder.eof:
                        pending = decoder.unused_data or decoder.unconsumed_tail
                    else:
                        pending = decoder.unconsumed_tail
                    if (
                        len(pending) == before_pending
                        and len(content) == before_content
                        and len(content) < limit
                    ):
                        raise FetchError("compressed response decoder made no progress")
            if len(content) >= limit:
                break
        if decoder is not None and len(content) < limit and not decoder.eof:
            content.extend(decoder.flush(limit - len(content)))
            if not decoder.eof:
                raise FetchError("compressed response ended before its checksum")
    except zlib.error as error:
        raise FetchError(f"invalid compressed response: {error}") from error

    return bytes(content[:max_bytes]), len(content) > max_bytes


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
    timeout: Optional[float] = None,
    raise_for_status: bool = True,
    redirect_guard: Optional[Callable[[str], Awaitable[None]]] = None,
    reject_declared_oversize: bool = True,
) -> FetchedBody:
    """GET `url`, validating every hop and streaming at most `max_bytes`.

    Redirects are followed manually so each hop is validated before it is
    requested; `httpx`'s own follow_redirects would jump to an unvalidated host.
    `raise_for_status=False` is for policy files that must inspect 401/403.
    """
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    current = check_url_syntax(url)
    addresses = await check_host_resolves_public(
        urlsplit(current).hostname or "", resolver, timeout
    )
    bounded_headers = {
        key: value for key, value in dict(headers or {}).items() if key.lower() != "cookie"
    }
    bounded_headers.setdefault("accept-encoding", "identity")
    origin_cookies: dict[tuple[str, str, Optional[int]], httpx.Cookies] = {}

    for hop in range(MAX_REDIRECTS + 1):
        native_transport = isinstance(
            getattr(client, "_transport", None), httpx.AsyncHTTPTransport
        )
        logical_request = httpx.Request("GET", current, headers=bounded_headers)
        cookie_jar = origin_cookies.setdefault(_origin_key(current), httpx.Cookies())
        cookie_jar.set_cookie_header(logical_request)
        hop_headers = dict(logical_request.headers)
        connect_addresses: list[Optional[str]] = (
            list(addresses) if native_transport else [None]
        )
        last_connect_error: Optional[httpx.HTTPError] = None
        response: Optional[httpx.Response] = None
        for address in connect_addresses:
            try:
                request_url, request_headers, extensions = _pin_request_target(
                    client, current, address, hop_headers
                )
                request_timeout = (
                    None
                    if timeout is None
                    else httpx.Timeout(
                        timeout,
                        connect=max(0.001, timeout / len(connect_addresses)),
                    )
                )
                if request_timeout is None:
                    request = client.build_request(
                        "GET", request_url, headers=request_headers, extensions=extensions
                    )
                else:
                    request = client.build_request(
                        "GET",
                        request_url,
                        headers=request_headers,
                        timeout=request_timeout,
                        extensions=extensions,
                    )
                # The logical hostname is carried in Host/SNI, but httpx's cookie
                # jar sees the pinned IP. Never let that implementation detail
                # leak a cookie between two hostnames sharing an address.
                logical_cookie = hop_headers.get("cookie")
                request.headers.pop("cookie", None)
                if logical_cookie:
                    request.headers["cookie"] = logical_cookie
                response = await _send_pinned_request(
                    client,
                    request,
                    current,
                    native_transport=native_transport,
                )
                logical_response = httpx.Response(
                    response.status_code,
                    headers=response.headers,
                    request=logical_request,
                )
                cookie_jar.extract_cookies(logical_response)
                break
            except (httpx.ConnectError, httpx.ConnectTimeout) as error:
                last_connect_error = error
                continue
            except (httpx.HTTPError, httpx.InvalidURL) as error:
                raise FetchError(
                    f"request failed: {type(error).__name__}: {error}"
                ) from error
        if response is None:
            assert last_connect_error is not None
            raise FetchError(
                f"request failed: {type(last_connect_error).__name__}: {last_connect_error}"
            ) from last_connect_error

        # Reject oversized bodies from the header before spending bandwidth on them.
        declared = response.headers.get("content-length")
        content_encoding = response.headers.get("content-encoding", "").strip().lower()
        if (
            reject_declared_oversize
            and content_encoding in ("", "identity")
            and declared
            and declared.isdigit()
            and int(declared) > max_bytes
        ):
            await response.aclose()
            raise TooLargeError(current, max_bytes, int(declared))

        location = response.headers.get("location")
        if response.is_redirect and location:
            await response.aclose()
            if hop == MAX_REDIRECTS:
                raise FetchError(f"too many redirects (>{MAX_REDIRECTS}) starting at {url}")
            candidate = check_url_syntax(urljoin(current, location))
            addresses = await check_host_resolves_public(
                urlsplit(candidate).hostname or "", resolver, timeout
            )
            if redirect_guard is not None:
                await redirect_guard(candidate)
            current = candidate
            continue

        try:
            try:
                body, truncated = await _read_bounded_body(response, max_bytes)
            except httpx.HTTPError as error:
                raise FetchError(
                    f"response body failed: {type(error).__name__}: {error}"
                ) from error
        finally:
            await response.aclose()

        if raise_for_status and response.status_code >= 400:
            detail = body[:200].decode("utf-8", errors="replace").strip()
            raise FetchError(
                f"HTTP {response.status_code} from {current}"
                + (f": {detail}" if detail else "")
            )
        content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        encoding = response.charset_encoding
        if encoding is not None:
            try:
                encoding = codecs.lookup(encoding).name
            except LookupError:
                encoding = None
        if encoding is None and content_type in ("text/html", "application/xhtml+xml"):
            encoding = _html_meta_encoding(body)

        return FetchedBody(
            requested_url=url,
            final_url=current,
            status=response.status_code,
            content_type=content_type,
            content=body,
            encoding=encoding or "utf-8",
            truncated=truncated,
            redirects=hop,
        )

    raise FetchError(f"too many redirects (>{MAX_REDIRECTS}) starting at {url}")
