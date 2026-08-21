"""Offline tests for the optional Gemini tiers. No network, no credentials."""

from __future__ import annotations

import base64
import json
import unittest
from io import BytesIO
from typing import Any, Callable, Sequence
from unittest import mock

import httpx

import webfetch
from webfetch import _gemini
from websearch import config as search_config


def answer_payload(text: str, retrieved: Sequence[str] = ()) -> dict[str, Any]:
    candidate: dict[str, Any] = {"content": {"parts": [{"text": text}]}}
    if retrieved:
        candidate["urlContextMetadata"] = {
            "urlMetadata": [{"retrievedUrl": url, "urlRetrievalStatus": "SUCCESS"} for url in retrieved]
        }
    return {"candidates": [candidate]}


class FakeEndpoint:
    """Mirrors websearch.config.GeminiEndpoint's surface, which is all _gemini uses."""

    def __init__(self, label: str, base_url: str, models: Sequence[str], keys: Sequence[str]) -> None:
        self.label = label
        self.base_url = base_url
        self.models = tuple(models)
        self.keys = tuple(keys)

    def pick_model(self, pinned: str | None) -> str | None:
        return pinned or (self.models[0] if self.models else None)


CORP = FakeEndpoint("corp", "https://gw.example.com/v1beta", ("gemini-3.6-flash",), ("sk-corp",))


def patch_endpoints(case: unittest.TestCase, *endpoints: FakeEndpoint) -> None:
    patcher = mock.patch.object(_gemini, "_endpoints", lambda: tuple(endpoints))
    case.addCleanup(patcher.stop)
    patcher.start()


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)


async def public_resolver(hostname: str) -> Sequence[str]:
    return ["93.184.216.34"]


def make_pdf(page_count: int) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class VideoUrlDetectionTest(unittest.TestCase):
    def test_youtube_forms_are_detected(self) -> None:
        for url in (
            "https://www.youtube.com/watch?v=abc123",
            "https://youtube.com/shorts/xyz",
            "https://youtu.be/abc123",
            "https://m.youtube.com/watch?v=abc",
            "https://www.youtube.com/live/abc",
        ):
            self.assertTrue(_gemini.is_video_url(url), url)

    def test_other_urls_are_not_videos(self) -> None:
        for url in (
            "https://youtube.com/",
            "https://youtube.com/@channel",
            "https://example.com/watch?v=abc",
            "https://vimeo.com/12345",
            "https://notyoutube.com/watch?v=a",
            "https://youtube.com/watchlist",
        ):
            self.assertFalse(_gemini.is_video_url(url), url)


class GenerateTest(unittest.IsolatedAsyncioTestCase):
    def test_cache_fingerprint_changes_when_only_key_rotates(self) -> None:
        endpoint_a = search_config.GeminiEndpoint(
            "gateway", "https://gateway.example/v1", ("model",), ("KEY-A",), "test"
        )
        endpoint_b = search_config.GeminiEndpoint(
            "gateway", "https://gateway.example/v1", ("model",), ("KEY-B",), "test"
        )
        with mock.patch.object(_gemini, "_endpoints", return_value=(endpoint_a,)):
            first = _gemini.cache_fingerprint()
        with mock.patch.object(_gemini, "_endpoints", return_value=(endpoint_b,)):
            second = _gemini.cache_fingerprint()
        self.assertNotEqual(first, second)
        self.assertNotIn("KEY", first + second)

    async def test_unavailable_without_endpoints(self) -> None:
        patch_endpoints(self)
        async with client_for(lambda request: httpx.Response(200)) as client:
            with self.assertRaises(_gemini.GeminiUnavailable) as ctx:
                await _gemini.generate(client, [{"text": "hi"}])
        self.assertIn("websearch", str(ctx.exception))

    async def test_empty_ai_studio_model_list_uses_documented_fallback(self) -> None:
        endpoint = search_config.GeminiEndpoint(
            "google-ai-studio",
            search_config.AI_STUDIO_BASE_URL,
            (),
            ("AIza-test-key-123456789",),
        )
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json=answer_payload("works"))

        with mock.patch.object(
            search_config, "gemini_endpoints", return_value=(endpoint,)
        ):
            async with client_for(handler) as client:
                answer = await _gemini.generate(client, [{"text": "hi"}])
        self.assertEqual(answer.text, "works")
        self.assertTrue(any("/models/gemini-flash-latest:" in url for url in seen))

    async def test_provider_echoes_are_redacted_and_unsafe_urls_are_dropped(self) -> None:
        secret = "AIza-leaked-secret-123456789"
        endpoint = FakeEndpoint(
            "corp", "https://gw.example.com/v1beta", ("m",), (secret,)
        )
        patch_endpoints(self, endpoint)

        def failure(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                json={"error": {"message": f"denied key {secret}"}},
            )

        async with client_for(failure) as client:
            with self.assertRaises(RuntimeError) as ctx:
                await _gemini.generate(client, [{"text": "hi"}])
        self.assertNotIn(secret, str(ctx.exception))
        self.assertIn("***", str(ctx.exception))

        payload = answer_payload(
            f"answer {secret}",
            [
                f"https://example.com/?key={secret}",
                "http://169.254.169.254/latest/meta-data/",
            ],
        )
        async with client_for(lambda request: httpx.Response(200, json=payload)) as client:
            answer = await _gemini.generate(client, [{"text": "hi"}])
        self.assertNotIn(secret, answer.text)
        self.assertEqual(answer.retrieved_urls, [])

    async def test_short_and_prior_failover_keys_are_redacted(self) -> None:
        first = "short"
        second = "SECONDSECRET-123"
        endpoint = FakeEndpoint(
            "corp", "https://gw.example.com/v1beta", ("m",), (first, second)
        )
        patch_endpoints(self, endpoint)

        def handler(request: httpx.Request) -> httpx.Response:
            if request.headers["x-goog-api-key"] == first:
                return httpx.Response(
                    429, json={"error": {"message": f"echo {first}"}}
                )
            return httpx.Response(200, json=answer_payload(f"repeats {first}"))

        async with client_for(handler) as client:
            answer = await _gemini.generate(client, [{"text": "hi"}])
        self.assertEqual(answer.text, "repeats ***")

        only_short = FakeEndpoint(
            "corp", "https://gw.example.com/v1beta", ("m",), (first,)
        )
        patch_endpoints(self, only_short)
        async with client_for(
            lambda request: httpx.Response(
                400, json={"error": {"message": f"echo {first}"}}
            )
        ) as client:
            with self.assertRaises(RuntimeError) as ctx:
                await _gemini.generate(client, [{"text": "hi"}])
        self.assertNotIn(first, str(ctx.exception))
        self.assertIn("***", str(ctx.exception))

    async def test_oversized_provider_response_is_rejected(self) -> None:
        patch_endpoints(self, CORP)

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"x" * (5 * 1024 * 1024 + 1)

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["accept-encoding"], "identity")
            return httpx.Response(200, stream=Stream())

        async with client_for(handler) as client:
            with self.assertRaises(RuntimeError) as ctx:
                await _gemini.generate(client, [{"text": "hi"}])
        self.assertIn("provider response exceeded", str(ctx.exception))

    async def test_url_context_request_shape_and_parse(self) -> None:
        patch_endpoints(self, CORP)
        bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["x-goog-api-key"], "sk-corp")
            self.assertTrue(str(request.url).endswith("/models/gemini-3.6-flash:generateContent"))
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=answer_payload("Two abstractions.", ["https://example.com/a"]))

        async with client_for(handler) as client:
            result = await _gemini.answer_about_url(client, "https://example.com/a", "What are they?")

        self.assertEqual(bodies[0]["tools"], [{"url_context": {}}])
        self.assertIn("https://example.com/a", bodies[0]["contents"][0]["parts"][0]["text"])
        self.assertIn("What are they?", bodies[0]["contents"][0]["parts"][0]["text"])
        self.assertEqual(result.text, "Two abstractions.")
        self.assertEqual(result.retrieved_urls, ["https://example.com/a"])
        self.assertEqual(result.detail, "corp/gemini-3.6-flash")

    async def test_video_request_uses_file_data(self) -> None:
        patch_endpoints(self, CORP)
        bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=answer_payload("A video about networks."))

        async with client_for(handler) as client:
            result = await _gemini.describe_video(client, "https://youtu.be/abc")

        parts = bodies[0]["contents"][0]["parts"]
        self.assertEqual(parts[0], {"fileData": {"fileUri": "https://youtu.be/abc"}})
        self.assertIn("what is shown on screen", parts[1]["text"])
        self.assertNotIn("tools", bodies[0])
        self.assertEqual(result.text, "A video about networks.")

    async def test_pdf_request_is_inlined_as_base64(self) -> None:
        patch_endpoints(self, CORP)
        pdf = make_pdf(1)
        bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=answer_payload("# Transcribed"))

        async with client_for(handler) as client:
            await _gemini.read_pdf(client, pdf)

        inline = bodies[0]["contents"][0]["parts"][0]["inlineData"]
        self.assertEqual(inline["mimeType"], "application/pdf")
        self.assertEqual(base64.b64decode(inline["data"]), pdf)

    async def test_oversized_pdf_is_never_inlined(self) -> None:
        # Over the inline ceiling the Files API path is used instead; see
        # test_webfetch_files.py for the upload protocol itself.
        patch_endpoints(self, CORP)
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(404, text="no files api here")

        async with client_for(handler) as client:
            with self.assertRaises(RuntimeError) as ctx:
                await _gemini.read_pdf(client, b"%PDF-" + b"x" * _gemini.MAX_INLINE_BYTES)
        self.assertIn("inline limit", str(ctx.exception))
        self.assertTrue(all("generateContent" not in url for url in seen))
        self.assertTrue(any("/upload/" in url for url in seen))

    async def test_key_and_endpoint_failover(self) -> None:
        dead = FakeEndpoint("dead", "https://dead.example.com/v1beta", ("m",), ("k-bad",))
        live = FakeEndpoint("live", "https://live.example.com/v1beta", ("m",), ("k-bad2", "k-good"))
        patch_endpoints(self, dead, live)
        seen: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.url.host or "", request.headers["x-goog-api-key"]))
            if request.url.host == "dead.example.com":
                return httpx.Response(429, json={"error": {"message": "slow down"}})
            if request.headers["x-goog-api-key"] == "k-bad2":
                return httpx.Response(401, json={"error": {"message": "bad key"}})
            return httpx.Response(200, json=answer_payload("ok"))

        async with client_for(handler) as client:
            result = await _gemini.generate(client, [{"text": "hi"}])
        self.assertEqual(result.detail, "live/m")
        self.assertEqual(len(seen), 3)

    async def test_400_does_not_retry_the_same_shape(self) -> None:
        patch_endpoints(self, FakeEndpoint("corp", "https://gw.example.com/v1beta", ("m",), ("k1", "k2")))
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json={"error": {"message": "bad request"}})

        async with client_for(handler) as client:
            with self.assertRaises(RuntimeError):
                await _gemini.generate(client, [{"text": "hi"}])
        self.assertEqual(calls["n"], 1)

    async def test_empty_response_is_a_failure(self) -> None:
        patch_endpoints(self, CORP)
        async with client_for(lambda request: httpx.Response(200, json={"candidates": []})) as client:
            with self.assertRaises(RuntimeError) as ctx:
                await _gemini.generate(client, [{"text": "hi"}])
        self.assertIn("empty response", str(ctx.exception))


class TierRoutingTest(unittest.IsolatedAsyncioTestCase):
    """Which tier handles which input, and what happens when Gemini is off."""

    def setUp(self) -> None:
        patcher = mock.patch.dict(
            "os.environ", {"PRIME_AGENT_WEBFETCH_RESPECT_ROBOTS": "0"}, clear=False
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    async def test_video_url_routes_to_video_tier(self) -> None:
        patch_endpoints(self, CORP)

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertIn("generateContent", str(request.url))
            return httpx.Response(200, json=answer_payload("Video summary."))

        document = await webfetch.fetch(
            "https://youtu.be/abc", transport=httpx.MockTransport(handler), resolver=public_resolver
        )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(document.source, "gemini-video")
        self.assertEqual(document.kind, "answer")
        self.assertEqual(document.answer, "Video summary.")

    async def test_prompt_routes_to_url_context(self) -> None:
        patch_endpoints(self, CORP)
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return httpx.Response(200, json=answer_payload("Answered.", ["https://example.com/p"]))

        document = await webfetch.fetch(
            "https://example.com/p",
            prompt="What is this?",
            transport=httpx.MockTransport(handler),
            resolver=public_resolver,
        )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(document.source, "gemini-url-context")
        self.assertEqual(document.retrieved_urls, ["https://example.com/p"])
        # The page itself was never fetched locally.
        self.assertTrue(all("generateContent" in path for path in paths))

    async def test_requested_model_without_endpoint_is_an_error(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, text="local page")

        with mock.patch.object(_gemini, "available", return_value=False):
            document = await webfetch.fetch(
                "https://example.com/p",
                prompt="Q?",
                respect_robots=False,
                transport=httpx.MockTransport(handler),
                resolver=public_resolver,
            )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(document.kind, "error")
        self.assertIn("no Gemini endpoint is available", document.error or "")
        self.assertEqual(calls, [])

    async def test_no_prompt_stays_local(self) -> None:
        patch_endpoints(self, CORP)

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertNotIn("generateContent", str(request.url))
            return httpx.Response(200, text="<html><body><main><h1>Local</h1></main></body></html>",
                                  headers={"content-type": "text/html"})

        document = await webfetch.fetch(
            "https://example.com/p", transport=httpx.MockTransport(handler), resolver=public_resolver
        )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(document.source, "local")
        self.assertIn("# Local", document.text)

    async def test_gemini_false_refuses_video_instead_of_fetching_it(self) -> None:
        patch_endpoints(self, CORP)

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("nothing should be requested")

        document = await webfetch.fetch(
            "https://youtu.be/abc", gemini=False, transport=httpx.MockTransport(handler)
        )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(document.kind, "error")
        self.assertIn("gemini=False", document.error or "")

    async def test_gemini_false_keeps_prompt_local(self) -> None:
        patch_endpoints(self, CORP)

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertNotIn("generateContent", str(request.url))
            return httpx.Response(200, text="<html><body><p>plain</p></body></html>",
                                  headers={"content-type": "text/html"})

        document = await webfetch.fetch(
            "https://example.com/p",
            prompt="ignored",
            gemini=False,
            transport=httpx.MockTransport(handler),
            resolver=public_resolver,
        )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(document.source, "local")

    async def test_blocked_page_falls_back_to_url_context(self) -> None:
        patch_endpoints(self, CORP)

        def handler(request: httpx.Request) -> httpx.Response:
            if "generateContent" in str(request.url):
                return httpx.Response(200, json=answer_payload("Recovered content."))
            return httpx.Response(403, text="forbidden")

        document = await webfetch.fetch(
            "https://example.com/blocked", transport=httpx.MockTransport(handler), resolver=public_resolver
        )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(document.source, "gemini-url-context")
        self.assertTrue(any("local fetch failed" in note for note in document.notes))

    async def test_unsafe_url_is_never_handed_to_the_model(self) -> None:
        patch_endpoints(self, CORP)

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no request may be made for a refused target")

        for options in ({}, {"prompt": "read it"}, {"gemini": True}):
            with self.subTest(options=options):
                document = await webfetch.fetch(
                    "http://169.254.169.254/latest/meta-data/",
                    respect_robots=False,
                    transport=httpx.MockTransport(handler),
                    **options,
                )
                assert isinstance(document, webfetch.Document)
                self.assertEqual(document.kind, "error")
                self.assertIn("non-public", document.error or "")

    async def test_robots_policy_precedes_prompted_url_context(self) -> None:
        patch_endpoints(self, CORP)
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            if "generateContent" in str(request.url):
                raise AssertionError("robots refusal must precede Gemini retrieval")
            return httpx.Response(200, text="User-agent: *\nDisallow: /\n")

        document = await webfetch.fetch(
            "https://example.com/private",
            prompt="read it",
            respect_robots=True,
            resolver=public_resolver,
            transport=httpx.MockTransport(handler),
        )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(document.kind, "error")
        self.assertIn("robots.txt", document.error or "")
        self.assertEqual(requested, ["https://example.com/robots.txt"])

    async def test_scanned_pdf_escalates_to_vision(self) -> None:
        patch_endpoints(self, CORP)
        pdf = make_pdf(2)

        def handler(request: httpx.Request) -> httpx.Response:
            if "generateContent" in str(request.url):
                return httpx.Response(200, json=answer_payload("# Page one\ntranscribed"))
            return httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"})

        document = await webfetch.fetch(
            "https://example.com/scan.pdf", transport=httpx.MockTransport(handler), resolver=public_resolver
        )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(document.source, "gemini-pdf")
        self.assertEqual(document.kind, "pdf")
        self.assertEqual(document.pages, 2)
        self.assertEqual(document.bytes_len, len(pdf))
        self.assertIn("transcribed", document.text)

    async def test_scanned_pdf_honors_max_pages_before_gemini(self) -> None:
        patch_endpoints(self, CORP)
        pdf = make_pdf(2)
        pages_sent: list[int] = []

        async def read_limited(
            client: httpx.AsyncClient,
            content: bytes,
            prompt: str | None = None,
            **kwargs: Any,
        ) -> _gemini.GeminiAnswer:
            from pypdf import PdfReader

            pages_sent.append(len(PdfReader(BytesIO(content)).pages))
            return _gemini.GeminiAnswer("limited", "fake")

        with mock.patch.object(_gemini, "read_pdf", new=read_limited):
            document = await webfetch.fetch(
                "https://example.com/scan.pdf",
                max_pages=1,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200, content=pdf, headers={"content-type": "application/pdf"}
                    )
                ),
                resolver=public_resolver,
            )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(pages_sent, [1])
        self.assertEqual(document.text, "limited")

    async def test_scanned_pdf_without_gemini_keeps_the_note(self) -> None:
        patch_endpoints(self)  # nothing configured
        pdf = make_pdf(1)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"})

        document = await webfetch.fetch(
            "https://example.com/scan.pdf", transport=httpx.MockTransport(handler), resolver=public_resolver
        )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(document.source, "local")
        self.assertEqual(document.kind, "pdf")
        self.assertTrue(any("scanned" in note for note in document.notes))

    async def test_url_context_failure_does_not_cache_an_ignored_prompt(self) -> None:
        patch_endpoints(self, CORP)
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if "generateContent" in str(request.url):
                return httpx.Response(500, json={"error": {"message": "model down"}})
            return httpx.Response(
                200,
                text="<html><body><main><p>local text</p></main></body></html>",
                headers={"content-type": "text/html"},
            )

        document = await webfetch.fetch(
            "https://example.com/p",
            prompt="What is this?",
            transport=httpx.MockTransport(handler),
            resolver=public_resolver,
        )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(document.kind, "error")
        self.assertIn("gemini-url-context failed", document.error or "")
        self.assertFalse(any(url == "https://example.com/p" for url in calls))


class GeminiRenderTest(unittest.IsolatedAsyncioTestCase):
    async def test_answer_document_shows_source_and_retrieved(self) -> None:
        document = webfetch.Document(
            url="https://example.com/p",
            final_url="https://example.com/p",
            kind="answer",
            text="The answer.",
            source="gemini-url-context",
            answer="The answer.",
            retrieved_urls=["https://example.com/p"],
            notes=["gemini-url-context via corp/gemini-3.6-flash"],
        )
        text = webfetch._render(document, 20_000)
        self.assertIn("gemini-url-context", text)
        self.assertIn("retrieved: https://example.com/p", text)
        self.assertIn("The answer.", text)

    async def test_gemini_available_helper(self) -> None:
        patch_endpoints(self, CORP)
        self.assertTrue(webfetch.gemini_available())
        patch_endpoints(self)
        self.assertFalse(webfetch.gemini_available())


if __name__ == "__main__":
    unittest.main()
