"""Offline tests for trajectory-driven backend health.

A failing backend earns a session cooldown that doubles per consecutive miss;
one success resets it. Explicit provider lists and provider="all" are never
refused - health only reorders the automatic fallback chain.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from typing import Any
from unittest import mock

import httpx

import websearch
from websearch import _backends as backends
from websearch import _health
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


class HealthTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        websearch.clear_cache()
        websearch.reset_health()
        patcher = mock.patch.object(config, "read_first_json", return_value={})
        self.addCleanup(patcher.stop)
        patcher.start()
        env = mock.patch.dict(os.environ, {"PRIME_AGENT_WEBSEARCH_CACHE_TTL": "0"}, clear=True)
        self.addCleanup(env.stop)
        env.start()

    def tearDown(self) -> None:
        websearch.reset_health()

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

    async def test_failure_earns_cooldown_and_next_call_skips_it(self) -> None:
        calls: list[str] = []

        async def failing(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("gemini")
            raise backends.BackendError("quota exhausted")

        async def working(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("ddg")
            return result("ddg")

        self.fake_backends(gemini=failing, ddg=working)

        # Default auto chain: health reorders it after a failure.
        first = await websearch.run("q")
        self.assertEqual(calls, ["gemini", "ddg"])
        self.assertIn("failed: gemini", first)

        second = await websearch.run("q")
        self.assertEqual(calls, ["gemini", "ddg", "ddg"])
        self.assertIn("cooled: gemini", second)

    async def test_named_provider_is_never_refused_while_cooling(self) -> None:
        calls: list[str] = []

        async def failing(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("gemini")
            raise backends.BackendError("gemini: down")

        self.fake_backends(gemini=failing)
        await websearch.run("q", provider="gemini")
        await websearch.run("q", provider="gemini")
        self.assertEqual(calls, ["gemini", "gemini"])

    async def test_all_mode_ignores_health(self) -> None:
        calls: list[str] = []

        async def failing(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("gemini")
            raise backends.BackendError("gemini: down")

        async def working(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("ddg")
            return result("ddg")

        self.fake_backends(gemini=failing, ddg=working)
        await websearch.run("q", provider="all")
        await websearch.run("q", provider="all")
        # The fan-out attempts every configured backend on both calls.
        self.assertEqual(calls.count("gemini"), 2)
        self.assertEqual(calls.count("ddg"), 2)

    async def test_empty_results_cool_nothing(self) -> None:
        calls: list[str] = []

        async def empty(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("tavily")
            return result("tavily", answer=None, items=[])

        self.fake_backends(tavily=empty, ddg=empty)
        await websearch.run("q", provider="tavily")
        await websearch.run("q", provider="tavily")
        self.assertEqual(calls, ["tavily", "tavily"])
        text = await websearch.health()
        self.assertIn("- tavily: ok", text)

    def test_multiplier_doubles_and_caps(self) -> None:
        tracker = _health.HealthTracker()
        with mock.patch.object(_health, "_now", lambda: 1000.0):
            for expected_multiplier in (1, 2, 4, 8, 8):
                tracker.record_failure("gemini", "boom", 120.0)
                state = tracker.get("gemini")
                self.assertEqual(state.multiplier, expected_multiplier)
                self.assertEqual(state.until, 1000.0 + 120.0 * expected_multiplier)
            tracker.record_success("gemini")
            state = tracker.get("gemini")
            self.assertEqual((state.failures, state.until), (0, 0.0))

    def test_cooldown_expires(self) -> None:
        tracker = _health.HealthTracker()
        clock = {"now": 100.0}
        with mock.patch.object(_health, "_now", lambda: clock["now"]):
            tracker.record_failure("gemini", "boom", 60.0)
            self.assertEqual(tracker.partition(["gemini", "ddg"]), (["ddg"], ["gemini"]))
            clock["now"] = 160.0
            self.assertEqual(tracker.partition(["gemini", "ddg"]), (["gemini", "ddg"], []))
            rendered = _health.render(tracker, ["gemini"], base=60.0)
        self.assertIn("ready; cooldown expired after 1 consecutive failure(s)", rendered)
        self.assertNotIn("recovered", rendered)

    async def test_disabled_by_env(self) -> None:
        calls: list[str] = []

        async def failing(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("gemini")
            raise backends.BackendError("gemini: down")

        async def working(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("ddg")
            return result("ddg")

        self.fake_backends(gemini=failing, ddg=working)
        with mock.patch.dict(os.environ, {"PRIME_AGENT_WEBSEARCH_COOLDOWN": "0"}):
            await websearch.run("q")
            second = await websearch.run("q")
        self.assertEqual(calls, ["gemini", "ddg", "gemini", "ddg"])
        self.assertNotIn("cooled:", second)

    async def test_disabling_cooldown_clears_an_active_deadline(self) -> None:
        calls: list[str] = []

        async def failing(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("gemini")
            raise backends.BackendError("down")

        async def working(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("ddg")
            return result("ddg")

        self.fake_backends(gemini=failing, ddg=working)
        with mock.patch.dict(os.environ, {"PRIME_AGENT_WEBSEARCH_COOLDOWN": "120"}):
            await websearch.run("q1")
        with mock.patch.dict(os.environ, {"PRIME_AGENT_WEBSEARCH_COOLDOWN": "0"}):
            second = await websearch.run("q2")
            health = await websearch.health()
        self.assertEqual(calls, ["gemini", "ddg", "gemini", "ddg"])
        self.assertNotIn("cooled:", second)
        self.assertIn("cooldown off", health)
        self.assertNotIn("gemini: cooling", health)

    async def test_health_updates_follow_attempt_completion_order(self) -> None:
        gemini_success = asyncio.Event()
        release_ddg = asyncio.Event()

        async def gemini(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            if query.text == "older-success":
                gemini_success.set()
                return result("gemini")
            await gemini_success.wait()
            raise backends.BackendError("newer failure")

        async def ddg(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            await release_ddg.wait()
            return result("ddg")

        self.fake_backends(gemini=gemini, ddg=ddg)
        older = asyncio.create_task(websearch.run("older-success", provider="all"))
        await gemini_success.wait()
        await websearch.run("newer-failure", provider="gemini")
        self.assertIn("- gemini: cooling", await websearch.health())

        # Finishing an unrelated slow sibling must not replay the older Gemini
        # success and erase the newer failure.
        release_ddg.set()
        await older
        self.assertIn("- gemini: cooling", await websearch.health())

    def test_multiplier_bounds_work_before_exponentiation(self) -> None:
        state = _health.BackendHealth(failures=1_000_000)
        self.assertEqual(state.multiplier, _health.MAX_MULTIPLIER)

    async def test_reset_ignores_late_completion_from_an_old_search(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def delayed_failure(
            client: httpx.AsyncClient,
            query: config.SearchQuery,
            settings: config.Settings,
        ) -> Any:
            started.set()
            await release.wait()
            raise backends.BackendError("late failure")

        self.fake_backends(gemini=delayed_failure)
        pending = asyncio.create_task(websearch.run("q", provider="gemini"))
        await started.wait()
        websearch.reset_health()
        release.set()
        await pending
        self.assertIn("- gemini: ok", await websearch.health())

    async def test_health_call_with_cooldown_off_clears_old_deadlines(self) -> None:
        websearch._HEALTH.record_failure("gemini", "down", 120.0)
        with mock.patch.dict(os.environ, {"PRIME_AGENT_WEBSEARCH_COOLDOWN": "0"}):
            self.assertIn("cooldown off", await websearch.health())
        with mock.patch.dict(os.environ, {"PRIME_AGENT_WEBSEARCH_COOLDOWN": "120"}):
            health = await websearch.health()
        self.assertIn("- gemini: ready; cooldown expired", health)
        self.assertNotIn("- gemini: cooling", health)

    async def test_reset_health_restores_the_static_order(self) -> None:
        calls: list[str] = []

        async def failing(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("gemini")
            raise backends.BackendError("gemini: down")

        async def working(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("ddg")
            return result("ddg")

        self.fake_backends(gemini=failing, ddg=working)
        await websearch.run("q")
        websearch.reset_health()
        await websearch.run("q")
        self.assertEqual(calls, ["gemini", "ddg", "gemini", "ddg"])

    async def test_cached_hits_do_not_claim_cooled_skips(self) -> None:
        calls: list[str] = []

        async def failing(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("gemini")
            raise backends.BackendError("quota exhausted")

        async def working(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("ddg")
            return result("ddg")

        self.fake_backends(gemini=failing, ddg=working)
        with mock.patch.dict(os.environ, {"PRIME_AGENT_WEBSEARCH_CACHE_TTL": "300"}):
            await websearch.run("q")   # gemini fails -> cools; ddg answers; cached
            second = await websearch.run("q")   # pure cache hit: nothing attempted
        self.assertEqual(calls, ["gemini", "ddg"])
        self.assertIn("from cache", second)
        # Gemini is still cooling right now, yet a cached answer must not claim
        # backends were consulted, skipped, or failed on this call.
        self.assertNotIn("cooled:", second)
        self.assertNotIn("failed:", second)
        self.assertIn("- gemini: cooling", await websearch.health())

    async def test_health_labels_failures_when_cooldown_is_disabled(self) -> None:
        calls: list[str] = []

        async def failing(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("gemini")
            raise backends.BackendError("quota exhausted")

        self.fake_backends(gemini=failing)
        with mock.patch.dict(os.environ, {"PRIME_AGENT_WEBSEARCH_COOLDOWN": "0"}):
            await websearch.run("q", provider="gemini")
            await websearch.run("q", provider="gemini")
            text = await websearch.health()
        self.assertEqual(calls, ["gemini", "gemini"])
        self.assertIn("- gemini: 2 recent failure(s), cooldown off", text)
        self.assertNotIn("(recovered)", text)

    async def test_last_resort_attempts_are_not_reported_as_deferred(self) -> None:
        calls: list[str] = []

        async def gemini(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("gemini")
            raise backends.BackendError("gemini down")

        async def tavily(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("tavily")
            if query.text == "q1":
                raise backends.BackendError("tavily down")
            return result("tavily")

        async def ddg(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("ddg")
            if query.text == "q2":
                return result("ddg", answer=None, items=[])
            return result("ddg")

        self.fake_backends(gemini=gemini, tavily=tavily, ddg=ddg)
        await websearch.run("q1")  # gemini/tavily cool; ddg answers
        second = await websearch.run("q2")  # ddg empty; both cooled backends are attempted

        self.assertEqual(calls, ["gemini", "tavily", "ddg", "ddg", "gemini", "tavily"])
        self.assertIn("## tavily", second)
        self.assertIn("failed: ddg: no results | gemini down", second)
        self.assertNotIn("cooled:", second)

    async def test_health_renders_the_evidence(self) -> None:
        calls: list[str] = []

        async def failing(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("gemini")
            raise backends.BackendError("quota exhausted")

        async def working(client: httpx.AsyncClient, query: config.SearchQuery, settings: config.Settings) -> Any:
            calls.append("ddg")
            return result("ddg")

        self.fake_backends(gemini=failing, ddg=working)
        await websearch.run("q")

        text = await websearch.health()
        self.assertIn("# websearch health", text)
        self.assertIn("- gemini: cooling", text)
        self.assertIn("consecutive failure(s): gemini: quota exhausted", text)
        self.assertIn("- ddg: ok", text)
        self.assertIn("PRIME_AGENT_WEBSEARCH_COOLDOWN=0", text)


if __name__ == "__main__":
    unittest.main()
