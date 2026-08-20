"""Offline tests for orchestration: failover, fan-out, cache, rendering."""

from __future__ import annotations

import os
import unittest
from typing import Any
from unittest import mock

import httpx

import websearch
from websearch import _backends as backends
from websearch import config


def result(backend: str, url: str = "https://example.com", **overrides: Any) -> backends.SearchResult:
    base: dict[str, Any] = {
        "backend": backend,
        "detail": f"{backend}.example",
        "answer": f"{backend} answer",
        "items": [backends.ResultItem(f"{backend} title", url, "snippet")],
    }
    base.update(overrides)
    return backends.SearchResult(**base)


class OrchestrationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        websearch.clear_cache()
        patcher = mock.patch.object(config, "read_first_json", return_value={})
        self.addCleanup(patcher.stop)
        patcher.start()
        env = mock.patch.dict(os.environ, {"PRIME_AGENT_WEBSEARCH_CACHE_TTL": "0"}, clear=True)
        self.addCleanup(env.stop)
        env.start()

    def fake_backends(self, **implementations: Any) -> None:
        registry = dict.fromkeys(config.AUTO_ORDER)
        for name, impl in implementations.items():
            registry[name] = impl
        patcher = mock.patch.dict(backends.BACKENDS, registry, clear=True)
        self.addCleanup(patcher.stop)
        patcher.start()
        available = mock.patch.object(
            config.Settings, "available", lambda self, backend: backend in implementations
        )
        self.addCleanup(available.stop)
        available.start()

    async def test_auto_stops_at_first_success(self) -> None:
        calls: list[str] = []

        async def failing(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("gemini")
            raise backends.BackendError("gemini: boom")

        async def working(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("ddg")
            return result("ddg")

        never = mock.AsyncMock(side_effect=AssertionError("must not run"))
        self.fake_backends(gemini=failing, ddg=working, tavily=never)

        text = await websearch.run("q", provider="gemini,ddg")
        self.assertEqual(calls, ["gemini", "ddg"])
        self.assertIn("## ddg", text)
        self.assertIn("failed: gemini", text)

    async def test_all_runs_backends_concurrently(self) -> None:
        started: list[str] = []

        def make(name: str) -> Any:
            async def impl(client: Any, query: Any, settings: Any) -> Any:
                started.append(name)
                # Yield control so a sequential implementation would order calls.
                import asyncio

                await asyncio.sleep(0.01 if name == "gemini" else 0)
                return result(name, url=f"https://{name}.example.com")

            return impl

        self.fake_backends(gemini=make("gemini"), ddg=make("ddg"))
        results = await websearch.search("q", provider="all")
        self.assertEqual({item.backend for item in results}, {"gemini", "ddg"})
        # Both started before either finished.
        self.assertEqual(started, ["gemini", "ddg"])

    async def test_unexpected_exception_does_not_break_the_call(self) -> None:
        async def exploding(client: Any, query: Any, settings: Any) -> Any:
            raise ZeroDivisionError("bug in a backend")

        async def working(client: Any, query: Any, settings: Any) -> Any:
            return result("ddg")

        self.fake_backends(gemini=exploding, ddg=working)
        text = await websearch.run("q")
        self.assertIn("## ddg", text)
        self.assertIn("unexpected ZeroDivisionError", text)

    async def test_all_backends_failing_returns_text_not_exception(self) -> None:
        async def failing(client: Any, query: Any, settings: Any) -> Any:
            raise backends.BackendError("nope")

        self.fake_backends(ddg=failing)
        text = await websearch.run("q")
        self.assertTrue(text.startswith("websearch failed:"))
        self.assertIn("all search backends failed", text)

    async def test_no_backend_configured_explains_how_to_enable(self) -> None:
        self.fake_backends()  # nothing available
        text = await websearch.run("q")
        self.assertIn("no search backend is configured", text)
        self.assertIn("GEMINI_API_KEY", text)

    async def test_empty_query_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            await websearch.search("   ")

    async def test_invalid_recency_is_reported(self) -> None:
        async def working(client: Any, query: Any, settings: Any) -> Any:
            return result("ddg")

        self.fake_backends(ddg=working)
        with self.assertRaises(ValueError):
            await websearch.search("q", recency="fortnight")

    async def test_query_is_truncated(self) -> None:
        seen: list[config.SearchQuery] = []

        async def capture(client: Any, query: config.SearchQuery, settings: Any) -> Any:
            seen.append(query)
            return result("ddg")

        self.fake_backends(ddg=capture)
        await websearch.search("x" * (config.MAX_QUERY_CHARS + 500))
        self.assertEqual(len(seen[0].text), config.MAX_QUERY_CHARS)


class CacheTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        websearch.clear_cache()
        patcher = mock.patch.object(config, "read_first_json", return_value={})
        self.addCleanup(patcher.stop)
        patcher.start()

    def install(self, ttl: str) -> list[int]:
        calls: list[int] = []

        async def impl(client: Any, query: Any, settings: Any) -> Any:
            calls.append(1)
            return result("ddg")

        patcher = mock.patch.dict(backends.BACKENDS, {"ddg": impl}, clear=True)
        self.addCleanup(patcher.stop)
        patcher.start()
        available = mock.patch.object(config.Settings, "available", lambda self, backend: backend == "ddg")
        self.addCleanup(available.stop)
        available.start()
        env = mock.patch.dict(os.environ, {"PRIME_AGENT_WEBSEARCH_CACHE_TTL": ttl}, clear=True)
        self.addCleanup(env.stop)
        env.start()
        return calls

    async def test_repeat_query_is_served_from_cache(self) -> None:
        calls = self.install("300")
        first = await websearch.run("same query")
        second = await websearch.run("same query")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("from cache", first)
        self.assertIn("from cache", second)

    async def test_different_parameters_are_different_entries(self) -> None:
        calls = self.install("300")
        await websearch.run("q")
        await websearch.run("q", recency="week")
        await websearch.run("q", domains="github.com")
        await websearch.run("q", num_results=9)
        self.assertEqual(len(calls), 4)

    async def test_cache_can_be_disabled(self) -> None:
        calls = self.install("0")
        await websearch.run("q")
        await websearch.run("q")
        self.assertEqual(len(calls), 2)

    async def test_expired_entry_is_refetched(self) -> None:
        calls = self.install("300")
        await websearch.run("q")
        # Age every entry past the TTL.
        for key, (_, outcome) in list(websearch._CACHE.items()):
            websearch._CACHE[key] = (0.0, outcome)
        await websearch.run("q")
        self.assertEqual(len(calls), 2)


class RenderingTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        websearch.clear_cache()
        patcher = mock.patch.object(config, "read_first_json", return_value={})
        self.addCleanup(patcher.stop)
        patcher.start()
        env = mock.patch.dict(os.environ, {"PRIME_AGENT_WEBSEARCH_CACHE_TTL": "0"}, clear=True)
        self.addCleanup(env.stop)
        env.start()

    async def test_full_render(self) -> None:
        rich = result(
            "gemini",
            detail="corp/gemini-3.6-flash",
            answer="An answer.[1]",
            items=[backends.ResultItem("Title", "https://example.com", "snippet")],
            queries=["q1", "q2"],
            dropped=2,
        )

        async def impl(client: Any, query: Any, settings: Any) -> Any:
            return rich

        with mock.patch.dict(backends.BACKENDS, {"gemini": impl}, clear=True):
            with mock.patch.object(config.Settings, "available", lambda self, backend: backend == "gemini"):
                text = await websearch.run("my query", recency="week", domains="example.com,-spam.com")

        self.assertIn("# websearch: my query  (last week; only example.com; excluding spam.com)", text)
        self.assertIn("## gemini (corp/gemini-3.6-flash)", text)
        self.assertIn("An answer.[1]", text)
        self.assertIn("1. Title", text)
        self.assertIn("   https://example.com", text)
        self.assertIn("q1; q2", text)
        self.assertIn("2 result(s) removed by the domain filter", text)
        self.assertIn("used: gemini", text)


class BackendsListingTest(unittest.IsolatedAsyncioTestCase):
    async def test_lists_backends_hints_and_cache_state(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(config, "read_first_json", return_value={}):
                text = await websearch.backends()
        for name in config.AUTO_ORDER:
            self.assertIn(name, text)
        self.assertIn("ready  ddg", text)
        self.assertIn("enable:", text)
        self.assertIn("recency values:", text)
        self.assertIn("cache:", text)

    async def test_shows_credential_source_not_value(self) -> None:
        auth = {"tavily": {"type": "api_key", "key": "tvly-secret-value"}}
        with mock.patch.dict(os.environ, {"BRAVE_API_KEY": "brave-secret-value"}, clear=True):
            with mock.patch.object(config, "read_first_json", return_value=auth):
                text = await websearch.backends()
        self.assertIn("auth.json:tavily", text)
        self.assertIn("$BRAVE_API_KEY", text)
        self.assertNotIn("tvly-secret-value", text)
        self.assertNotIn("brave-secret-value", text)


if __name__ == "__main__":
    unittest.main()
