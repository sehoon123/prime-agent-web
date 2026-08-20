"""Offline tests for every backend, using httpx.MockTransport. No network."""

from __future__ import annotations

import json
import os
import unittest
from typing import Any, Callable
from unittest import mock

import httpx

import websearch
from websearch import _backends as backends
from websearch import config

GEMINI_PAYLOAD: dict[str, Any] = {
    "candidates": [
        {
            "content": {"parts": [{"text": "Spain won the 2026 World Cup."}]},
            "groundingMetadata": {
                "groundingChunks": [
                    {"web": {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AAA", "title": "foxsports.com"}},
                    {"web": {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/BBB", "title": "britannica.com"}},
                    {"web": {"uri": "https://example.org/plain", "title": "example.org"}},
                ],
                "webSearchQueries": ["who won the 2026 world cup"],
            },
        }
    ]
}

DDG_HTML = """
<div class="result results_links">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Freal.example.com%2Fpage&amp;rut=x">Real <b>Page</b></a>
  <a class="result__snippet">A snippet about the page.</a>
</div>
<div class="result results_links">
  <a class="result__a" href="https://direct.example.com/two">Second result</a>
  <a class="result__snippet">Second snippet.</a>
</div>
"""


def settings_for(**overrides: Any) -> config.Settings:
    base: dict[str, Any] = {
        "num_results": 5,
        "timeout": 5.0,
        "order": config.AUTO_ORDER,
        "gemini_model": None,
        "searxng_url": None,
        "auth": {},
    }
    base.update(overrides)
    return config.Settings(**base)


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)


class GeminiBackendTest(unittest.IsolatedAsyncioTestCase):
    def endpoint_settings(self) -> config.Settings:
        settings = settings_for(auth={"corp": {"type": "api_key", "key": "sk-corp-secret-value"}})
        endpoint = config.GeminiEndpoint(
            label="corp",
            base_url="https://gw.example.com/v1beta",
            models=("gemini-3.6-flash",),
            keys=("sk-corp-secret-value",),
        )
        patcher = mock.patch.object(
            config.Settings, "gemini_endpoints", property(lambda self: (endpoint,))
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        return settings

    async def test_parses_answer_sources_and_queries(self) -> None:
        settings = self.endpoint_settings()
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(f"{request.method} {request.url}")
            if "grounding-api-redirect" in str(request.url):
                target = "https://foxsports.com/story" if request.url.path.endswith("AAA") else "https://britannica.com/x"
                return httpx.Response(301, headers={"location": target})
            if str(request.url).startswith("https://foxsports.com") or str(request.url).startswith("https://britannica.com"):
                return httpx.Response(200, text="ok")
            body = json.loads(request.content)
            self.assertEqual(body["tools"], [{"google_search": {}}])
            self.assertEqual(request.headers["x-goog-api-key"], "sk-corp-secret-value")
            return httpx.Response(200, json=GEMINI_PAYLOAD)

        async with client_for(handler) as client:
            result = await backends.search_gemini(client, "who won the 2026 world cup", settings)

        self.assertEqual(result.backend, "gemini")
        self.assertEqual(result.detail, "corp/gemini-3.6-flash")
        self.assertEqual(result.answer, "Spain won the 2026 World Cup.")
        self.assertEqual(result.queries, ["who won the 2026 world cup"])
        self.assertEqual(
            [item.url for item in result.items],
            ["https://foxsports.com/story", "https://britannica.com/x", "https://example.org/plain"],
        )

    async def test_falls_back_to_legacy_tool_name_on_400(self) -> None:
        settings = self.endpoint_settings()
        tools_seen: list[Any] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "grounding-api-redirect" in str(request.url):
                return httpx.Response(200, text="ok")
            body = json.loads(request.content)
            tools_seen.append(body["tools"][0])
            if "google_search" in body["tools"][0]:
                return httpx.Response(400, json={"error": {"message": "Unknown name \"google_search\""}})
            return httpx.Response(200, json=GEMINI_PAYLOAD)

        async with client_for(handler) as client:
            result = await backends.search_gemini(client, "q", settings)

        self.assertEqual(tools_seen, [{"google_search": {}}, {"google_search_retrieval": {}}])
        self.assertEqual(result.answer, "Spain won the 2026 World Cup.")

    async def test_key_failover_on_401(self) -> None:
        endpoint = config.GeminiEndpoint(
            label="corp",
            base_url="https://gw.example.com/v1beta",
            models=("gemini-3.6-flash",),
            keys=("sk-dead-key-value", "sk-live-key-value"),
        )
        patcher = mock.patch.object(config.Settings, "gemini_endpoints", property(lambda self: (endpoint,)))
        self.addCleanup(patcher.stop)
        patcher.start()

        def handler(request: httpx.Request) -> httpx.Response:
            if "grounding-api-redirect" in str(request.url):
                return httpx.Response(200, text="ok")
            if request.headers["x-goog-api-key"] == "sk-dead-key-value":
                return httpx.Response(401, json={"error": {"message": "invalid key"}})
            return httpx.Response(200, json=GEMINI_PAYLOAD)

        async with client_for(handler) as client:
            result = await backends.search_gemini(client, "q", settings_for())
        self.assertEqual(result.answer, "Spain won the 2026 World Cup.")

    async def test_no_endpoint_is_not_retryable(self) -> None:
        patcher = mock.patch.object(config.Settings, "gemini_endpoints", property(lambda self: ()))
        self.addCleanup(patcher.stop)
        patcher.start()
        async with client_for(lambda request: httpx.Response(200)) as client:
            with self.assertRaises(backends.BackendError) as ctx:
                await backends.search_gemini(client, "q", settings_for())
        self.assertFalse(ctx.exception.retryable)


class CredentialBackendTest(unittest.IsolatedAsyncioTestCase):
    async def test_serper(self) -> None:
        settings = settings_for(auth={"serper": {"type": "api_key", "key": "sk-serper"}})
        payload = {
            "answerBox": {"answer": "42"},
            "knowledgeGraph": {"title": "Thing", "description": "A thing."},
            "organic": [
                {"title": "First", "link": "https://a.example.com", "snippet": "one"},
                {"title": "Second", "link": "https://b.example.com", "snippet": "two"},
            ],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["X-API-KEY"], "sk-serper")
            self.assertEqual(json.loads(request.content)["num"], 5)
            return httpx.Response(200, json=payload)

        with mock.patch.dict(os.environ, {}, clear=True):
            async with client_for(handler) as client:
                result = await backends.search_serper(client, "q", settings)
        self.assertIn("42", result.answer or "")
        self.assertIn("Thing - A thing.", result.answer or "")
        self.assertEqual([item.url for item in result.items], ["https://a.example.com", "https://b.example.com"])

    async def test_tavily(self) -> None:
        settings = settings_for(auth={"tavily": {"type": "api_key", "key": "tvly-key"}})
        payload = {
            "answer": "Short answer.",
            "results": [{"title": "T", "url": "https://t.example.com", "content": "body"}],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["authorization"], "Bearer tvly-key")
            self.assertTrue(json.loads(request.content)["include_answer"])
            return httpx.Response(200, json=payload)

        with mock.patch.dict(os.environ, {}, clear=True):
            async with client_for(handler) as client:
                result = await backends.search_tavily(client, "q", settings)
        self.assertEqual(result.answer, "Short answer.")
        self.assertEqual(result.items[0].snippet, "body")

    async def test_brave(self) -> None:
        settings = settings_for(auth={"brave": {"type": "api_key", "key": "brave-key"}})
        payload = {"web": {"results": [{"title": "B", "url": "https://b.example.com", "description": "desc"}]}}

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["X-Subscription-Token"], "brave-key")
            self.assertEqual(request.url.params["count"], "5")
            return httpx.Response(200, json=payload)

        with mock.patch.dict(os.environ, {}, clear=True):
            async with client_for(handler) as client:
                result = await backends.search_brave(client, "q", settings)
        self.assertEqual(result.items[0].snippet, "desc")

    async def test_exa_joins_highlight_lists(self) -> None:
        settings = settings_for(auth={"exa": {"type": "api_key", "key": "exa-key"}})
        payload = {"results": [{"title": "E", "url": "https://e.example.com", "highlights": ["a", "b"]}]}

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["x-api-key"], "exa-key")
            return httpx.Response(200, json=payload)

        with mock.patch.dict(os.environ, {}, clear=True):
            async with client_for(handler) as client:
                result = await backends.search_exa(client, "q", settings)
        self.assertEqual(result.items[0].snippet, "a b")

    async def test_searxng_requires_results(self) -> None:
        settings = settings_for(searxng_url="https://searx.example.com")

        async with client_for(lambda request: httpx.Response(200, json={"results": []})) as client:
            with self.assertRaises(backends.BackendError):
                await backends.search_searxng(client, "q", settings)

        payload = {"results": [{"title": "S", "url": "https://s.example.com", "content": "c"}]}
        async with client_for(lambda request: httpx.Response(200, json=payload)) as client:
            result = await backends.search_searxng(client, "q", settings)
        self.assertEqual(result.detail, "searx.example.com")

    async def test_missing_credential_is_not_retryable(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            async with client_for(lambda request: httpx.Response(200)) as client:
                for backend in (backends.search_serper, backends.search_tavily, backends.search_brave, backends.search_exa):
                    with self.assertRaises(backends.BackendError) as ctx:
                        await backend(client, "q", settings_for())
                    self.assertFalse(ctx.exception.retryable)


class DuckDuckGoTest(unittest.IsolatedAsyncioTestCase):
    async def test_parses_and_unwraps_redirects(self) -> None:
        async with client_for(lambda request: httpx.Response(200, text=DDG_HTML)) as client:
            result = await backends.search_ddg(client, "q", settings_for())
        self.assertEqual(result.backend, "ddg")
        self.assertEqual(result.items[0].url, "https://real.example.com/page")
        self.assertEqual(result.items[0].title, "Real Page")
        self.assertEqual(result.items[1].url, "https://direct.example.com/two")

    async def test_regex_parser_matches_bs4_urls(self) -> None:
        items = backends._parse_ddg_with_regex(DDG_HTML, 5)
        self.assertEqual(
            [item.url for item in items],
            ["https://real.example.com/page", "https://direct.example.com/two"],
        )

    async def test_second_endpoint_is_tried(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.host)
            if request.url.host.startswith("html."):
                return httpx.Response(503, text="nope")
            return httpx.Response(200, text=DDG_HTML)

        async with client_for(handler) as client:
            result = await backends.search_ddg(client, "q", settings_for())
        self.assertEqual(calls, ["html.duckduckgo.com", "lite.duckduckgo.com"])
        self.assertTrue(result.items)


class RenderingTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_renders_answer_sources_and_trailer(self) -> None:
        result = backends.SearchResult(
            backend="gemini",
            detail="corp/gemini-3.6-flash",
            answer="An answer.",
            items=[backends.ResultItem("Title", "https://example.com", "snippet")],
            queries=["q1", "q2"],
        )

        async def fake_execute(*args: Any, **kwargs: Any) -> websearch.Outcome:
            with mock.patch.object(config, "read_first_json", return_value={}):
                settings = config.load_settings()
            return websearch.Outcome(
                query="my query",
                settings=settings,
                results=[result],
                failures=["serper: no results"],
            )

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(config, "read_first_json", return_value={}):
                with mock.patch.object(websearch, "_execute", fake_execute):
                    text = await websearch.run("my query")

        self.assertIn("# websearch: my query", text)
        self.assertIn("## gemini (corp/gemini-3.6-flash)", text)
        self.assertIn("1. Title", text)
        self.assertIn("https://example.com", text)
        self.assertIn("q1; q2", text)
        self.assertIn("used: gemini", text)
        self.assertIn("failed: serper: no results", text)
        self.assertIn("not configured:", text)

    async def test_run_reports_failure_without_raising(self) -> None:
        async def failing(*args: Any, **kwargs: Any) -> websearch.Outcome:
            raise RuntimeError("all search backends failed: ddg: boom")

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(config, "read_first_json", return_value={}):
                with mock.patch.object(websearch, "_execute", failing):
                    text = await websearch.run("q")
        self.assertTrue(text.startswith("websearch failed:"))

    async def test_empty_query_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            await websearch.search("   ")

    async def test_secrets_never_reach_output(self) -> None:
        secret = "sk-super-secret-value-1234"
        endpoint = config.GeminiEndpoint("corp", "https://gw.example.com/v1beta", ("gemini-3.6-flash",), (secret,))
        patcher = mock.patch.object(config.Settings, "gemini_endpoints", property(lambda self: (endpoint,)))
        self.addCleanup(patcher.stop)
        patcher.start()

        def handler(request: httpx.Request) -> httpx.Response:
            # Echo the key back inside the error, the worst realistic case.
            return httpx.Response(403, json={"error": {"message": f"key {secret} is banned"}})

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(config, "read_first_json", return_value={}):
                async with client_for(handler) as client:
                    with self.assertRaises(backends.BackendError) as ctx:
                        await backends.search_gemini(client, "q", settings_for())
                    raw = str(ctx.exception)
                    self.assertIn(secret, raw)  # backends may include it...
                    self.assertNotIn(secret, websearch._redact(raw, (secret,)))  # ...run() must not


class BackendsListingTest(unittest.IsolatedAsyncioTestCase):
    async def test_lists_every_backend_with_hints(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(config, "read_first_json", return_value={}):
                text = await websearch.backends()
        for name in config.AUTO_ORDER:
            self.assertIn(name, text)
        self.assertIn("ready  ddg", text)
        self.assertIn("enable:", text)


if __name__ == "__main__":
    unittest.main()
