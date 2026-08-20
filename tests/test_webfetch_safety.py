"""Offline tests for URL validation, redirect handling, size caps, and robots.txt."""

from __future__ import annotations

import unittest
from typing import Any, Callable, Optional, Sequence
from unittest import mock

import httpx

from webfetch import _safety
from webfetch._robots import RobotsCache
from webfetch._safety import FetchError, TooLargeError, UnsafeUrlError, guarded_get


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)


async def public_resolver(hostname: str) -> Sequence[str]:
    return ["93.184.216.34"]


async def private_resolver(hostname: str) -> Sequence[str]:
    return ["10.0.0.5"]


async def mixed_resolver(hostname: str) -> Sequence[str]:
    # A public and a private record: the private one must still block the fetch.
    return ["93.184.216.34", "127.0.0.1"]


class UrlSyntaxTest(unittest.TestCase):
    def test_accepts_public_http_urls(self) -> None:
        for url in ("https://example.com/a", "http://sub.example.co.uk/b?c=1", "https://8.8.8.8/x"):
            self.assertEqual(_safety.check_url_syntax(url), url)

    def test_rejects_private_and_internal_targets(self) -> None:
        for url in (
            "http://127.0.0.1:8080/admin",
            "http://localhost/admin",
            "http://10.0.0.5/",
            "http://192.168.1.1/",
            "http://172.16.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://[fd00::1]/",
            "http://metadata.google.internal/x",
            "http://printer.local/",
            "http://vault.internal/",
            "http://intranet/",
            "http://facebookcorewwwi.onion/",
        ):
            with self.assertRaises(UnsafeUrlError, msg=url):
                _safety.check_url_syntax(url)

    def test_rejects_other_schemes_and_credentials(self) -> None:
        for url in (
            "file:///etc/passwd",
            "ftp://example.com/x",
            "javascript:alert(1)",
            "data:text/html,<script>",
            "https://user:pass@example.com/",
            "",
            "   ",
        ):
            with self.assertRaises(UnsafeUrlError, msg=url):
                _safety.check_url_syntax(url)

    def test_file_scheme_error_suggests_open(self) -> None:
        with self.assertRaises(UnsafeUrlError) as ctx:
            _safety.check_url_syntax("file:///etc/passwd")
        self.assertIn("open()", str(ctx.exception))


class DnsPreflightTest(unittest.IsolatedAsyncioTestCase):
    async def test_public_resolution_passes(self) -> None:
        await _safety.check_host_resolves_public("example.com", public_resolver)

    async def test_private_resolution_is_blocked(self) -> None:
        with self.assertRaises(UnsafeUrlError) as ctx:
            await _safety.check_host_resolves_public("evil.example.com", private_resolver)
        self.assertIn("10.0.0.5", str(ctx.exception))

    async def test_any_private_record_blocks(self) -> None:
        with self.assertRaises(UnsafeUrlError):
            await _safety.check_host_resolves_public("split.example.com", mixed_resolver)

    async def test_resolution_failure_is_a_fetch_error(self) -> None:
        async def failing(hostname: str) -> Sequence[str]:
            raise OSError("nope")

        with self.assertRaises(FetchError):
            await _safety.check_host_resolves_public("gone.example.com", failing)

    async def test_literal_ip_skips_dns(self) -> None:
        async def must_not_run(hostname: str) -> Sequence[str]:
            raise AssertionError("DNS must not be consulted for a literal IP")

        await _safety.check_host_resolves_public("8.8.8.8", must_not_run)


class GuardedGetTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_body_and_metadata(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="hello", headers={"content-type": "text/plain; charset=utf-8"})

        async with client_for(handler) as client:
            body = await guarded_get(client, "https://example.com/x", resolver=public_resolver)
        self.assertEqual(body.status, 200)
        self.assertEqual(body.content_type, "text/plain")
        self.assertEqual(body.text, "hello")
        self.assertEqual(body.redirects, 0)

    async def test_redirects_are_validated_per_hop(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            if request.url.path == "/start":
                return httpx.Response(302, headers={"location": "https://example.com/end"})
            return httpx.Response(200, text="done")

        async with client_for(handler) as client:
            body = await guarded_get(client, "https://example.com/start", resolver=public_resolver)
        self.assertEqual(body.final_url, "https://example.com/end")
        self.assertEqual(body.redirects, 1)
        self.assertEqual(len(seen), 2)

    async def test_redirect_into_private_space_is_blocked_before_request(self) -> None:
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

        async with client_for(handler) as client:
            with self.assertRaises(UnsafeUrlError):
                await guarded_get(client, "https://example.com/start", resolver=public_resolver)
        self.assertEqual(requested, ["https://example.com/start"])

    async def test_redirect_to_host_resolving_private_is_blocked(self) -> None:
        async def resolver(hostname: str) -> Sequence[str]:
            return ["10.1.2.3"] if hostname == "inside.example.com" else ["93.184.216.34"]

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "example.com":
                return httpx.Response(301, headers={"location": "https://inside.example.com/x"})
            raise AssertionError("must never reach the private host")

        async with client_for(handler) as client:
            with self.assertRaises(UnsafeUrlError):
                await guarded_get(client, "https://example.com/start", resolver=resolver)

    async def test_redirect_limit(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            index = int(request.url.params.get("n", "0"))
            return httpx.Response(302, headers={"location": f"https://example.com/?n={index + 1}"})

        async with client_for(handler) as client:
            with self.assertRaises(FetchError) as ctx:
                await guarded_get(client, "https://example.com/?n=0", resolver=public_resolver)
        self.assertIn("too many redirects", str(ctx.exception))

    async def test_content_length_header_rejects_before_download(self) -> None:
        streamed = {"bytes": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            streamed["bytes"] += 1
            return httpx.Response(200, content=b"x" * 100, headers={"content-length": "100000000"})

        async with client_for(handler) as client:
            with self.assertRaises(TooLargeError) as ctx:
                await guarded_get(client, "https://example.com/big", max_bytes=1024, resolver=public_resolver)
        message = str(ctx.exception)
        self.assertIn("100,000,000 bytes", message)
        self.assertIn("max_bytes=", message)

    async def test_unannounced_body_is_capped_by_streaming(self) -> None:
        async def chunks() -> Any:
            for _ in range(40):
                yield b"y" * 1000

        def handler(request: httpx.Request) -> httpx.Response:
            # A chunked response has no content-length, so only the stream guard
            # can stop it - this is the case a header check cannot catch.
            return httpx.Response(200, content=chunks(), headers={"content-type": "text/plain"})

        async with client_for(handler) as client:
            body = await guarded_get(client, "https://example.com/big", max_bytes=4096, resolver=public_resolver)
        self.assertTrue(body.truncated)
        self.assertEqual(len(body.content), 4096)

    async def test_http_error_includes_status_and_detail(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Not Found here")

        async with client_for(handler) as client:
            with self.assertRaises(FetchError) as ctx:
                await guarded_get(client, "https://example.com/missing", resolver=public_resolver)
        self.assertIn("HTTP 404", str(ctx.exception))
        self.assertIn("Not Found here", str(ctx.exception))

    async def test_transport_error_is_wrapped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        async with client_for(handler) as client:
            with self.assertRaises(FetchError) as ctx:
                await guarded_get(client, "https://example.com/x", resolver=public_resolver)
        self.assertIn("ConnectError", str(ctx.exception))


ROBOTS_BODY = """
# a comment
User-agent: *
Disallow: /private/
Allow: /
"""


class RobotsTest(unittest.IsolatedAsyncioTestCase):
    def cache(self) -> RobotsCache:
        return RobotsCache(user_agent="prime-agent-webfetch/0.3 (Autonomous; +https://example.invalid)")

    async def test_disallowed_path_is_refused(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertTrue(request.url.path.endswith("/robots.txt"))
            return httpx.Response(200, text=ROBOTS_BODY)

        async with client_for(handler) as client:
            verdict = await self.cache().check(client, "https://example.com/private/x")
        self.assertFalse(verdict.allowed)
        self.assertIn("respect_robots=False", verdict.reason)

    async def test_allowed_path_passes(self) -> None:
        async with client_for(lambda request: httpx.Response(200, text=ROBOTS_BODY)) as client:
            verdict = await self.cache().check(client, "https://example.com/public/x")
        self.assertTrue(verdict.allowed)

    async def test_missing_robots_allows(self) -> None:
        async with client_for(lambda request: httpx.Response(404, text="nope")) as client:
            verdict = await self.cache().check(client, "https://example.com/x")
        self.assertTrue(verdict.allowed)

    async def test_forbidden_robots_disallows(self) -> None:
        # MCP fetch convention: 401/403 on robots.txt means "do not fetch".
        async with client_for(lambda request: httpx.Response(403, text="")) as client:
            verdict = await self.cache().check(client, "https://example.com/x")
        self.assertFalse(verdict.allowed)
        self.assertIn("403", verdict.reason)

    async def test_unreachable_robots_allows(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        async with client_for(handler) as client:
            verdict = await self.cache().check(client, "https://example.com/x")
        self.assertTrue(verdict.allowed)

    async def test_robots_is_fetched_once_per_origin(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, text=ROBOTS_BODY)

        cache = self.cache()
        async with client_for(handler) as client:
            await cache.check(client, "https://example.com/a")
            await cache.check(client, "https://example.com/b")
            await cache.check(client, "https://other.example.com/c")
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
