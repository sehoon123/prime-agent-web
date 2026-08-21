"""Offline tests for URL validation, redirect handling, size caps, and robots.txt."""

from __future__ import annotations

import asyncio
import gzip
import random
import unittest
from typing import Any, Callable, Sequence

import httpx

from webfetch import _robots, _safety
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
            "http://100.64.0.1/",
            "http://[64:ff9b::a9fe:a9fe]/",
            "http://[::1]/",
            "http://[fd00::1]/",
            "http://[fec0::1]/",
            "http://metadata.google.internal/x",
            "http://printer.local/",
            "http://vault.internal/",
            "http://intranet/",
            "http://facebookcorewwwi.onion/",
            "https://safe.example/x\nFAKE",
            "https://safe.example/x\x1b[31m",
            "https://safe.example/x\u202eTXT",
            "https://safe.example/x&#x202e;TXT",
            "https://127&period;0.0&period;1/x",
            "https://user&commat;evil.example/x",
            "https://127%2e0.0%2e1/x",
            "https://169%2e254.169%2e254/x",
            "https://127.1/x",
            "https://0177.0.0.1/x",
            "https://0x7f.1/x",
            "https://169.254.43518/x",
        ):
            with self.assertRaises(UnsafeUrlError, msg=url):
                _safety.check_url_syntax(url)

    def test_html_entity_validation_does_not_rewrite_wire_url(self) -> None:
        url = "https://example.com/search?q=a&amp;b=c"
        self.assertEqual(_safety.check_url_syntax(url), url)

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

    async def test_native_transport_connects_to_the_vetted_ip(self) -> None:
        captured: list[httpx.Request] = []

        class RecordingTransport(httpx.AsyncHTTPTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                captured.append(request)
                return httpx.Response(200, content=b"ok", request=request)

        async with httpx.AsyncClient(transport=RecordingTransport()) as client:
            body = await guarded_get(
                client, "https://example.com/path", resolver=public_resolver
            )
        self.assertEqual(body.final_url, "https://example.com/path")
        self.assertEqual(captured[0].url.host, "93.184.216.34")
        self.assertEqual(captured[0].headers["host"], "example.com")
        self.assertEqual(captured[0].extensions["sni_hostname"], "example.com")
        self.assertEqual(captured[0].headers["connection"], "close")

    async def test_pinned_transport_tries_all_vetted_addresses(self) -> None:
        hosts: list[str | None] = []

        class FailoverTransport(httpx.AsyncHTTPTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                hosts.append(request.url.host)
                if request.url.host == "8.8.8.8":
                    raise httpx.ConnectError("first address down", request=request)
                return httpx.Response(200, content=b"ok", request=request)

        async def resolver(hostname: str) -> Sequence[str]:
            return ["8.8.8.8", "93.184.216.34"]

        async with httpx.AsyncClient(transport=FailoverTransport()) as client:
            body = await guarded_get(client, "https://example.com/", resolver=resolver)
        self.assertEqual(body.text, "ok")
        self.assertEqual(hosts, ["8.8.8.8", "93.184.216.34"])

    async def test_pinned_ip_never_sends_cookie_across_logical_hosts(self) -> None:
        requests: list[httpx.Request] = []

        class CookieTransport(httpx.AsyncHTTPTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                requests.append(request)
                headers = {"set-cookie": "sid=SECRET; Path=/"} if len(requests) == 1 else {}
                return httpx.Response(200, content=b"ok", headers=headers, request=request)

        async with httpx.AsyncClient(transport=CookieTransport()) as client:
            await guarded_get(client, "https://a.example/", resolver=public_resolver)
            await guarded_get(client, "https://b.example/", resolver=public_resolver)
        self.assertNotIn("cookie", requests[1].headers)
        self.assertEqual(requests[0].url.host, requests[1].url.host)
        self.assertNotEqual(requests[0].headers["host"], requests[1].headers["host"])

    async def test_same_origin_redirect_preserves_response_cookie(self) -> None:
        requests: list[httpx.Request] = []

        class CookieRedirectTransport(httpx.AsyncHTTPTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                requests.append(request)
                if request.url.path == "/start":
                    return httpx.Response(
                        302,
                        headers={"location": "/end", "set-cookie": "sid=ok; Path=/"},
                        request=request,
                    )
                return httpx.Response(200, content=b"ok", request=request)

        async with httpx.AsyncClient(transport=CookieRedirectTransport()) as client:
            await guarded_get(client, "https://a.example/start", resolver=public_resolver)
        self.assertNotIn("cookie", requests[0].headers)
        self.assertEqual(requests[1].headers.get("cookie"), "sid=ok")

    async def test_cross_origin_redirect_does_not_forward_cookie(self) -> None:
        requests: list[httpx.Request] = []

        class CookieRedirectTransport(httpx.AsyncHTTPTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                requests.append(request)
                if len(requests) == 1:
                    return httpx.Response(
                        302,
                        headers={
                            "location": "https://b.example/end",
                            "set-cookie": "sid=secret; Domain=.example; Path=/",
                        },
                        request=request,
                    )
                return httpx.Response(200, content=b"ok", request=request)

        async with httpx.AsyncClient(transport=CookieRedirectTransport()) as client:
            await guarded_get(client, "https://a.example/start", resolver=public_resolver)
        self.assertNotIn("cookie", requests[1].headers)

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

    async def test_exact_size_stream_is_not_marked_truncated(self) -> None:
        async def chunks() -> Any:
            yield b"x" * 1024

        async with client_for(lambda request: httpx.Response(200, content=chunks())) as client:
            body = await guarded_get(
                client,
                "https://example.com/exact",
                max_bytes=1024,
                resolver=public_resolver,
            )
        self.assertFalse(body.truncated)
        self.assertEqual(len(body.content), 1024)

    async def test_gzip_bomb_is_decoded_under_the_output_cap(self) -> None:
        compressed = gzip.compress(b"x" * (20 * 1024 * 1024))

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield compressed

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["accept-encoding"], "identity")
            return httpx.Response(
                200,
                headers={"content-encoding": "gzip", "content-type": "text/plain"},
                stream=Stream(),
            )

        async with client_for(handler) as client:
            body = await guarded_get(
                client,
                "https://example.com/bomb",
                max_bytes=1024,
                resolver=public_resolver,
            )
        self.assertEqual(body.content, b"x" * 1024)
        self.assertTrue(body.truncated)

    async def test_concatenated_gzip_members_are_all_decoded(self) -> None:
        compressed = gzip.compress(b"first") + gzip.compress(b"second")

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield compressed

        async with client_for(
            lambda request: httpx.Response(
                200,
                headers={"content-encoding": "gzip"},
                stream=Stream(),
            )
        ) as client:
            body = await guarded_get(
                client, "https://example.com/concat", max_bytes=100, resolver=public_resolver
            )
        self.assertEqual(body.content, b"firstsecond")
        self.assertFalse(body.truncated)

    async def test_compressed_wire_length_does_not_reject_exact_decoded_body(self) -> None:
        decoded = random.Random(0).randbytes(1000)
        compressed = gzip.compress(decoded)
        self.assertGreater(len(compressed), len(decoded))

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield compressed

        async with client_for(
            lambda request: httpx.Response(
                200,
                headers={
                    "content-encoding": "gzip",
                    "content-length": str(len(compressed)),
                },
                stream=Stream(),
            )
        ) as client:
            body = await guarded_get(
                client, "https://example.com/compressed", max_bytes=1000, resolver=public_resolver
            )
        self.assertEqual(body.content, decoded)
        self.assertFalse(body.truncated)

    async def test_one_large_transport_chunk_is_bounded(self) -> None:
        async def chunks() -> Any:
            yield b"x" * 1_000_000

        async with client_for(lambda request: httpx.Response(200, content=chunks())) as client:
            body = await guarded_get(
                client,
                "https://example.com/large-chunk",
                max_bytes=1024,
                resolver=public_resolver,
            )
        self.assertTrue(body.truncated)
        self.assertEqual(len(body.content), 1024)

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

    async def test_stream_read_error_is_wrapped(self) -> None:
        async def broken_stream() -> Any:
            yield b"prefix"
            raise httpx.ReadError("broken stream")

        async with client_for(
            lambda request: httpx.Response(200, content=broken_stream())
        ) as client:
            with self.assertRaises(FetchError) as ctx:
                await guarded_get(
                    client, "https://example.com/x", resolver=public_resolver
                )
        self.assertIn("ReadError", str(ctx.exception))

    async def test_malformed_port_is_a_safe_url_error(self) -> None:
        async with client_for(lambda request: httpx.Response(200)) as client:
            with self.assertRaises(UnsafeUrlError):
                await guarded_get(
                    client,
                    "https://example.com:not-a-port/x",
                    resolver=public_resolver,
                )

    async def test_dns_resolution_obeys_timeout(self) -> None:
        async def slow(hostname: str) -> Sequence[str]:
            await asyncio.sleep(60)
            return ["93.184.216.34"]

        async with client_for(lambda request: httpx.Response(200)) as client:
            with self.assertRaises(FetchError) as ctx:
                await guarded_get(
                    client,
                    "https://example.com/x",
                    resolver=slow,
                    timeout=0.01,
                )
        self.assertIn("TimeoutError", str(ctx.exception))

    async def test_html_meta_charset_is_used_without_header_charset(self) -> None:
        html = '<meta charset="shift_jis"><body>日本語</body>'.encode("shift_jis")
        async with client_for(
            lambda request: httpx.Response(
                200, content=html, headers={"content-type": "text/html"}
            )
        ) as client:
            body = await guarded_get(
                client, "https://example.com/x", resolver=public_resolver
            )
        self.assertIn("日本語", body.text)

    async def test_invalid_header_charset_falls_back_to_utf8(self) -> None:
        async with client_for(
            lambda request: httpx.Response(
                200,
                content="café".encode(),
                headers={"content-type": "text/plain; charset=no-such-codec"},
            )
        ) as client:
            body = await guarded_get(
                client, "https://example.com/x", resolver=public_resolver
            )
        self.assertEqual(body.text, "café")

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
        return RobotsCache(
            user_agent="prime-agent-webfetch/0.6.2 (Autonomous; +https://example.invalid)",
            resolver=public_resolver,
        )

    async def test_disallowed_path_is_refused(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertTrue(request.url.path.endswith("/robots.txt"))
            return httpx.Response(200, text=ROBOTS_BODY)

        async with client_for(handler) as client:
            verdict = await self.cache().check(client, "https://example.com/private/x")
        self.assertFalse(verdict.allowed)
        self.assertIn("respect_robots=False", verdict.reason)

    async def test_longest_rule_wins_regardless_of_file_order(self) -> None:
        bodies = (
            "User-agent: *\nAllow: /\nDisallow: /private\n",
            "User-agent: *\nDisallow: /private\nAllow: /private/public\n",
        )
        expected = (False, True)
        for body, allowed in zip(bodies, expected):
            with self.subTest(body=body):
                async with client_for(
                    lambda request, body=body: httpx.Response(200, text=body)
                ) as client:
                    verdict = await self.cache().check(
                        client, "https://example.com/private/public"
                    )
                self.assertEqual(verdict.allowed, allowed)

    async def test_wildcard_terminal_anchor_and_allow_tie(self) -> None:
        body = """User-agent: *
Disallow: /files/*.pdf$
Allow: /files/public.pdf$
Allow: /same$
Disallow: /same$
"""
        async with client_for(lambda request: httpx.Response(200, text=body)) as client:
            cache = self.cache()
            public = await cache.check(client, "https://example.com/files/public.pdf")
            private = await cache.check(client, "https://example.com/files/private.pdf")
            suffix = await cache.check(client, "https://example.com/files/private.pdf?download=1")
            tie = await cache.check(client, "https://example.com/same")
        self.assertTrue(public.allowed)
        self.assertFalse(private.allowed)
        self.assertTrue(suffix.allowed)
        self.assertTrue(tie.allowed)

    async def test_unreserved_percent_escapes_cannot_bypass_rule(self) -> None:
        body = "User-agent: *\nDisallow: /private\n"
        async with client_for(lambda request: httpx.Response(200, text=body)) as client:
            cache = self.cache()
            encoded_first = await cache.check(
                client, "https://example.com/%70rivate"
            )
            encoded_middle = await cache.check(
                client, "https://example.com/priv%61te"
            )
            reserved_slash = await cache.check(
                client, "https://example.com/%2Fprivate"
            )
        self.assertFalse(encoded_first.allowed)
        self.assertFalse(encoded_middle.allowed)
        self.assertTrue(reserved_slash.allowed)

    async def test_many_wildcards_do_not_backtrack_exponentially(self) -> None:
        pattern = "/" + "*a" * 200 + "b"
        body = f"User-agent: *\nDisallow: {pattern}\n"
        async with client_for(lambda request: httpx.Response(200, text=body)) as client:
            verdict = await self.cache().check(
                client, "https://example.com/" + "a" * 200 + "c"
            )
        self.assertTrue(verdict.allowed)

    def test_match_work_budget_fails_closed(self) -> None:
        body = "User-agent: *\n" + "Disallow: /*Z\n" * 32 + "Disallow: /a\n"
        policy = _robots._parse_policy(body)
        target = "https://example.com/" + "a" * 8000
        self.assertIsNone(policy.can_fetch(self.cache().user_agent, target))

    def test_long_url_with_many_plain_rules_does_not_exhaust_budget(self) -> None:
        body = (
            "User-agent: *\n"
            + "".join(f"Disallow: /admin/section-{index}/\n" for index in range(600))
            + "Allow: /\n"
        )
        policy = _robots._parse_policy(body)
        target = "https://example.com/wanted?x=" + "a" * 2000
        self.assertTrue(policy.can_fetch(self.cache().user_agent, target))

    async def test_oversized_robots_file_is_ignored(self) -> None:
        async with client_for(
            lambda request: httpx.Response(
                200, content=b"#" * (_robots.MAX_ROBOTS_BYTES + 1)
            )
        ) as client:
            verdict = await self.cache().check(client, "https://example.com/ok")
        self.assertTrue(verdict.allowed)

    async def test_unicode_and_utf8_escaped_paths_are_equivalent(self) -> None:
        for rule, target in (
            ("/café", "/caf%C3%A9"),
            ("/caf%C3%A9", "/café"),
        ):
            with self.subTest(rule=rule, target=target):
                policy = _robots._parse_policy(f"User-agent: *\nDisallow: {rule}\n")
                self.assertFalse(
                    policy.can_fetch(
                        self.cache().user_agent, "https://example.com" + target
                    )
                )

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

    async def test_robots_redirect_to_private_space_is_refused_before_request(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(
                302,
                headers={"location": "http://169.254.169.254/latest/meta-data/"},
            )

        async with client_for(handler) as client:
            with self.assertRaises(UnsafeUrlError):
                await self.cache().check(client, "https://example.com/x")
        self.assertEqual(calls, ["https://example.com/robots.txt"])

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
