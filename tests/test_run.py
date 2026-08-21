"""Offline tests for orchestration: failover, fan-out, cache, rendering."""

from __future__ import annotations

import asyncio
import os
import unittest
from dataclasses import replace
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
        websearch.reset_health()
        patcher = mock.patch.object(config, "read_first_json", return_value={})
        self.addCleanup(patcher.stop)
        patcher.start()
        env = mock.patch.dict(
            os.environ,
            {"PRIME_AGENT_WEBSEARCH_CACHE_TTL": "0", "PRIME_AGENT_WEBSEARCH_COOLDOWN": "0"},
            clear=True,
        )
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

    async def test_restricted_fanout_keeps_success_and_failure(self) -> None:
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

    async def test_auto_stops_at_first_success(self) -> None:
        calls: list[str] = []

        async def working(client: Any, query: Any, settings: Any) -> Any:
            calls.append("gemini")
            return result("gemini")

        async def must_not_run(client: Any, query: Any, settings: Any) -> Any:
            calls.append("ddg")
            raise AssertionError("auto must stop after the first result")

        self.fake_backends(gemini=working, ddg=must_not_run)
        text = await websearch.run("q", provider="auto")
        self.assertEqual(calls, ["gemini"])
        self.assertIn("## gemini", text)

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

    async def test_backend_name_is_not_duplicated_in_failures(self) -> None:
        async def failing(client: Any, query: Any, settings: Any) -> Any:
            raise backends.BackendError("gemini returned HTTP 429")

        self.fake_backends(gemini=failing)
        text = await websearch.run("q", provider="gemini")
        self.assertIn("gemini returned HTTP 429", text)
        self.assertNotIn("gemini: gemini", text)

    def test_query_header_is_single_line_and_bidi_safe(self) -> None:
        self.assertEqual(websearch._single_line("q\nFAKE\u202e"), "q FAKE")
        self.assertEqual(websearch._single_line("q&#x202e;FAKE"), "q FAKE")
        self.assertEqual(
            websearch._single_line("👨\u200d👩\u200d👧"), "👨\u200d👩\u200d👧"
        )

    def test_configuration_fingerprint_changes_when_key_rotates(self) -> None:
        from types import SimpleNamespace

        common = {"gemini_endpoints": (), "searxng_url": None}
        first = SimpleNamespace(**common, secrets=("KEY-A",))
        second = SimpleNamespace(**common, secrets=("KEY-B",))
        self.assertNotEqual(
            websearch._configuration_fingerprint(first, ("tavily",)),
            websearch._configuration_fingerprint(second, ("tavily",)),
        )

    def test_basic_username_is_not_a_bearer_redaction_value(self) -> None:
        settings = config.Settings(
            num_results=5,
            timeout=10.0,
            order=("searxng",),
            gemini_model=None,
            searxng_url="https://user:p%40ss@searx.example",
            cache_ttl=0.0,
            auth={},
        )
        self.assertNotIn("user", settings.secrets)
        result = backends.SearchResult(
            "searxng",
            items=[backends.ResultItem("users", "https://example.com/users/alice")],
        )
        websearch._redact_result(result, settings.secrets)
        self.assertEqual(len(result.items), 1)

    async def test_every_provider_controlled_field_is_redacted(self) -> None:
        self.assertEqual(websearch._redact("echo short", ("short",)), "echo ***")
        self.assertEqual(websearch._redact("Linux X11", ("X",)), "Linux ***11")
        self.assertEqual(websearch._redact("echo X", ("X",)), "echo ***")
        self.assertEqual(websearch._redact("prefixshortsuffix", ("short",)), "prefix***suffix")
        self.assertEqual(websearch._redact("ok\rOVERWRITE", ()), "ok\nOVERWRITE")
        self.assertEqual(websearch._redact("ABCDEF", ("ABC", "ABCDEF")), "***")
        secret = "sk-secret-provider-value-12345"

        async def leaking(client: Any, query: Any, settings: Any) -> Any:
            return backends.SearchResult(
                backend="ddg",
                detail=f"detail {secret}",
                answer=f"answer {secret}",
                items=[
                    backends.ResultItem(
                        f"title {secret}",
                        "https://example.com/safe",
                        f"snippet {secret}",
                    ),
                    backends.ResultItem(
                        "credential URL",
                        f"https://example.com/?token={secret}",
                    ),
                ],
                queries=[f"query {secret}"],
            )

        self.fake_backends(ddg=leaking)
        with mock.patch.object(config.Settings, "secrets", property(lambda self: (secret,))):
            text = await websearch.run("q")
        self.assertNotIn(secret, text)
        self.assertGreaterEqual(text.count("***"), 4)
        self.assertNotIn("token=", text)
        self.assertIn("result(s) removed", text)

    async def test_run_never_raises_for_wrong_argument_types(self) -> None:
        cases = (
            ((123,), {}),
            (("q",), {"num_results": "5"}),
            (("q",), {"domains": 5}),
        )
        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs):
                text = await websearch.run(*args, **kwargs)  # type: ignore[arg-type]
                self.assertTrue(text.startswith("websearch failed:"))

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
        self.addCleanup(websearch.clear_cache)
        websearch.reset_health()
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

    async def test_entry_within_ttl_is_reused(self) -> None:
        calls = self.install("300")
        clock = [1_000.0]
        with mock.patch.object(websearch, "_now", lambda: clock[0]):
            await websearch.run("q")
            clock[0] += 299.0  # still inside the 300s window
            text = await websearch.run("q")
        self.assertEqual(len(calls), 1)
        self.assertIn("from cache", text)

    async def test_expired_entry_is_refetched(self) -> None:
        calls = self.install("300")
        # A fake clock: time.monotonic() counts from an arbitrary origin (uptime),
        # so absolute values must never be assumed - only deltas.
        clock = [1_000.0]
        with mock.patch.object(websearch, "_now", lambda: clock[0]):
            await websearch.run("q")
            clock[0] += 301.0  # past the 300s TTL
            text = await websearch.run("q")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("from cache", text)

    async def test_gemini_model_is_part_of_the_cache_key(self) -> None:
        calls: list[str | None] = []

        async def gemini(client: Any, query: Any, settings: config.Settings) -> Any:
            calls.append(settings.gemini_model)
            return result("gemini", answer=settings.gemini_model)

        backend_patch = mock.patch.dict(backends.BACKENDS, {"gemini": gemini}, clear=True)
        available_patch = mock.patch.object(config.Settings, "available", lambda self, name: name == "gemini")
        env_patch = mock.patch.dict(os.environ, {"PRIME_AGENT_WEBSEARCH_CACHE_TTL": "300"}, clear=True)
        with backend_patch, available_patch, env_patch:
            first = await websearch.search("q", model="model-a")
            second = await websearch.search("q", model="model-b")
            third = await websearch.search("q", model="model-b")

        self.assertEqual(calls, ["model-a", "model-b"])
        self.assertEqual([first[0].answer, second[0].answer, third[0].answer], ["model-a", "model-b", "model-b"])

    async def test_searxng_endpoint_is_part_of_the_cache_configuration(self) -> None:
        calls: list[str | None] = []

        async def searxng(client: Any, query: Any, settings: config.Settings) -> Any:
            calls.append(settings.searxng_url)
            return result("searxng", answer=settings.searxng_url)

        with mock.patch.dict(backends.BACKENDS, {"searxng": searxng}, clear=True), mock.patch.dict(
            os.environ,
            {
                "PRIME_AGENT_WEBSEARCH_CACHE_TTL": "300",
                "SEARXNG_URL": "https://one.example",
            },
            clear=True,
        ):
            first = await websearch.search("q", provider="searxng")
            os.environ["SEARXNG_URL"] = "https://two.example"
            second = await websearch.search("q", provider="searxng")
        self.assertEqual(calls, ["https://one.example", "https://two.example"])
        self.assertEqual(first[0].answer, "https://one.example")
        self.assertEqual(second[0].answer, "https://two.example")

    async def test_raw_results_cannot_mutate_cached_results(self) -> None:
        calls = self.install("300")
        first = await websearch.search("q")
        first[0].items[0].title = "poisoned"
        first[0].queries.append("poisoned")
        first.clear()

        second = await websearch.search("q")
        self.assertEqual(len(calls), 1)
        self.assertEqual(second[0].items[0].title, "ddg title")
        self.assertEqual(second[0].queries, [])

        second[0].items.clear()
        third = await websearch.search("q")
        self.assertEqual(len(third[0].items), 1)

    async def test_clear_during_search_prevents_late_repopulation(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        calls: list[int] = []

        async def blocked(client: Any, query: Any, settings: Any) -> Any:
            calls.append(1)
            started.set()
            await release.wait()
            return result("ddg")

        with mock.patch.dict(backends.BACKENDS, {"ddg": blocked}, clear=True), mock.patch.object(
            config.Settings, "available", lambda self, name: name == "ddg"
        ), mock.patch.dict(os.environ, {"PRIME_AGENT_WEBSEARCH_CACHE_TTL": "300"}, clear=True):
            pending = asyncio.create_task(websearch.run("q"))
            await started.wait()
            websearch.clear_cache()
            release.set()
            await pending
            self.assertEqual(websearch._CACHE, {})
            await websearch.run("q")

        self.assertEqual(len(calls), 2)

    def test_replacing_an_existing_full_cache_entry_does_not_evict_another(self) -> None:
        settings = config.load_settings()
        for index in range(websearch._CACHE_MAX_ENTRIES):
            outcome = websearch.Outcome(config.SearchQuery(f"q{index}"), settings, results=[result("ddg")])
            websearch._cache_put((index,), outcome, 300.0)
        keys_before = set(websearch._CACHE)

        replacement = websearch.Outcome(config.SearchQuery("replacement"), settings, results=[result("ddg")])
        websearch._cache_put((websearch._CACHE_MAX_ENTRIES - 1,), replacement, 300.0)
        self.assertEqual(set(websearch._CACHE), keys_before)

    def test_cache_does_not_retain_settings_credentials(self) -> None:
        settings = replace(
            config.load_settings(),
            auth={"credential": {"key": "secret"}},
            searxng_url="https://user:password@searx.example",
        )
        outcome = websearch.Outcome(config.SearchQuery("q"), settings, results=[result("ddg")])
        websearch._cache_put(("q",), outcome, 300.0)
        stored = websearch._CACHE[("q",)][1]
        self.assertEqual(stored.settings.auth, {})
        self.assertIsNone(stored.settings.searxng_url)
        self.assertEqual(settings.auth, {"credential": {"key": "secret"}})


class RenderingTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        websearch.clear_cache()
        websearch.reset_health()
        patcher = mock.patch.object(config, "read_first_json", return_value={})
        self.addCleanup(patcher.stop)
        patcher.start()
        env = mock.patch.dict(
            os.environ,
            {"PRIME_AGENT_WEBSEARCH_CACHE_TTL": "0", "PRIME_AGENT_WEBSEARCH_COOLDOWN": "0"},
            clear=True,
        )
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
        self.assertIn(
            "2 result(s) removed by URL safety, credential redaction, or the domain filter",
            text,
        )
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
