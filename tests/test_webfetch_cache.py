"""Offline tests for the webfetch session document cache.

The cache exists because agent loops retry and subagents refetch the same
pages; it must serve copies (never the stored instance), skip errors, expire,
and stay out of the way when a transport or resolver was injected.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import unittest
from typing import Any
from unittest import mock

import webfetch


def make_document(url: str, text: str = "# Hello\n\nBody.", kind: str = "html") -> webfetch.Document:
    return webfetch.Document(url=url, final_url=url, kind=kind, text=text)


class DocumentCacheTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        webfetch.clear_cache()
        env = mock.patch.dict(os.environ, {"PRIME_AGENT_WEBFETCH_CACHE_TTL": "300"}, clear=False)
        self.addCleanup(env.stop)
        env.start()

    def tearDown(self) -> None:
        webfetch.clear_cache()

    def install_fetch_one(self) -> dict[str, int]:
        calls: dict[str, int] = {"n": 0}

        async def fake_fetch_one(*args: Any, **kwargs: Any) -> webfetch.Document:
            calls["n"] += 1
            url = args[1] if len(args) > 1 else kwargs.get("url", "?")
            return make_document(str(url))

        patcher = mock.patch.object(webfetch, "_fetch_one", new=fake_fetch_one)
        self.addCleanup(patcher.stop)
        patcher.start()
        return calls

    def install_clock(self, start: float = 1000.0) -> dict[str, float]:
        clock = {"now": start}

        patcher = mock.patch.object(webfetch, "_doc_cache_clock", lambda: clock["now"])
        self.addCleanup(patcher.stop)
        patcher.start()
        return clock

    async def test_second_fetch_is_served_from_the_session_cache(self) -> None:
        calls = self.install_fetch_one()
        first = await webfetch.fetch("https://x.test/guide")
        second = await webfetch.fetch("https://x.test/guide")
        self.assertEqual(calls["n"], 1)
        self.assertEqual(second.text, first.text)
        self.assertIn("from session cache", second.notes)
        self.assertNotIn("from session cache", first.notes)

    async def test_content_changing_arguments_bypass_the_hit(self) -> None:
        calls = self.install_fetch_one()
        await webfetch.fetch("https://x.test/guide")
        await webfetch.fetch("https://x.test/guide", mode="text")
        await webfetch.fetch("https://x.test/guide", prompt="summarise")
        await webfetch.fetch("https://x.test/guide", max_bytes=4096)
        self.assertEqual(calls["n"], 4)

    async def test_errors_are_never_cached(self) -> None:
        calls: dict[str, int] = {"n": 0}

        async def failing_fetch_one(*args: Any, **kwargs: Any) -> webfetch.Document:
            calls["n"] += 1
            doc = make_document("https://x.test/broken", kind="error", text="")
            doc.error = "connection refused"
            return doc

        with mock.patch.object(webfetch, "_fetch_one", new=failing_fetch_one):
            first = await webfetch.fetch("https://x.test/broken")
            second = await webfetch.fetch("https://x.test/broken")
        self.assertEqual(calls["n"], 2)
        self.assertFalse(first.ok)
        self.assertFalse(second.ok)

    async def test_entries_expire_after_the_ttl(self) -> None:
        calls = self.install_fetch_one()
        clock = self.install_clock(start=1000.0)
        await webfetch.fetch("https://x.test/guide")
        clock["now"] = 1000.0 + 300.0  # exactly the default TTL: still valid
        await webfetch.fetch("https://x.test/guide")
        self.assertEqual(calls["n"], 1)
        clock["now"] = 1300.5  # past expiry: fetched again
        await webfetch.fetch("https://x.test/guide")
        self.assertEqual(calls["n"], 2)

    async def test_clear_cache_forces_a_refetch(self) -> None:
        calls = self.install_fetch_one()
        await webfetch.fetch("https://x.test/guide")
        webfetch.clear_cache()
        await webfetch.fetch("https://x.test/guide")
        self.assertEqual(calls["n"], 2)

    async def test_oversized_documents_are_not_stored(self) -> None:
        calls: dict[str, int] = {"n": 0}
        big = make_document("https://x.test/big", text="#" * (webfetch._DOC_CACHE_MAX_CHARS + 1))

        async def big_fetch_one(*args: Any, **kwargs: Any) -> webfetch.Document:
            calls["n"] += 1
            return big

        with mock.patch.object(webfetch, "_fetch_one", new=big_fetch_one):
            await webfetch.fetch("https://x.test/big")
            again = await webfetch.fetch("https://x.test/big")
        self.assertEqual(calls["n"], 2)
        self.assertNotIn("from session cache", again.notes)

    async def test_cache_stays_bounded(self) -> None:
        calls = self.install_fetch_one()
        urls = [f"https://x.test/page-{i}" for i in range(webfetch._DOC_CACHE_MAX_ENTRIES + 4)]
        for url in urls:
            await webfetch.fetch(url)
        self.assertLessEqual(len(webfetch._DOC_CACHE), webfetch._DOC_CACHE_MAX_ENTRIES)
        # The very first page fell out of the cache and is fetched again.
        await webfetch.fetch(urls[0])
        self.assertEqual(calls["n"], len(urls) + 1)

    async def test_cache_entries_are_isolated_from_caller_mutation(self) -> None:
        calls = self.install_fetch_one()
        first = await webfetch.fetch("https://x.test/guide")
        first.notes.append("caller annotation")
        first.retrieved_urls.append("https://x.test/leaked")

        second = await webfetch.fetch("https://x.test/guide")
        self.assertEqual(calls["n"], 1)
        self.assertEqual(second.notes, ["from session cache"])
        self.assertEqual(second.retrieved_urls, [])

        # And mutating a hit must not poison later hits either.
        second.notes.append("hit annotation")
        third = await webfetch.fetch("https://x.test/guide")
        self.assertEqual(third.notes, ["from session cache"])

    async def test_gemini_configuration_change_invalidates_content_cache(self) -> None:
        calls = self.install_fetch_one()
        fingerprints = iter(("without-gemini", "with-gemini"))
        with mock.patch.object(
            webfetch._gemini,
            "cache_fingerprint",
            side_effect=lambda: next(fingerprints),
        ):
            await webfetch.fetch("https://x.test/guide")
            await webfetch.fetch("https://x.test/guide")
        self.assertEqual(calls["n"], 2)

    async def test_ttl_zero_disables_storage_as_well_as_reads(self) -> None:
        calls = self.install_fetch_one()
        with mock.patch.dict(os.environ, {"PRIME_AGENT_WEBFETCH_CACHE_TTL": "0"}):
            await webfetch.fetch("https://x.test/guide")
            self.assertEqual(webfetch._DOC_CACHE, {})

        # Enabling caching later must fetch again; an off-period result cannot
        # appear as a hit after the setting changes.
        await webfetch.fetch("https://x.test/guide")
        await webfetch.fetch("https://x.test/guide")
        self.assertEqual(calls["n"], 2)

    async def test_failed_scan_escalation_is_not_cached(self) -> None:
        degraded = webfetch.Document(
            url="https://example.com/scan.pdf",
            final_url="https://example.com/scan.pdf",
            kind="pdf",
            notes=["Gemini scan failed (HTTP 429)"],
        )
        recovered = webfetch.Document(
            url=degraded.url,
            final_url=degraded.final_url,
            kind="pdf",
            text="recovered scan text",
            source="gemini-pdf",
        )
        with mock.patch.object(
            webfetch, "_fetch_one", mock.AsyncMock(side_effect=[degraded, recovered])
        ) as fetch_one:
            first = await webfetch.fetch(degraded.url)
            second = await webfetch.fetch(degraded.url)
        self.assertEqual(first.text, "")
        self.assertEqual(second.text, "recovered scan text")
        self.assertEqual(fetch_one.await_count, 2)

    async def test_clear_during_fetch_prevents_late_repopulation(self) -> None:
        calls: dict[str, int] = {"n": 0}
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked(*args: Any, **kwargs: Any) -> webfetch.Document:
            calls["n"] += 1
            started.set()
            await release.wait()
            return make_document(str(args[1]))

        with mock.patch.object(webfetch, "_fetch_one", new=blocked):
            pending = asyncio.create_task(webfetch.fetch("https://x.test/guide"))
            await started.wait()
            webfetch.clear_cache()
            release.set()
            await pending
            self.assertEqual(webfetch._DOC_CACHE, {})
            await webfetch.fetch("https://x.test/guide")

        self.assertEqual(calls["n"], 2)

    async def test_manual_and_autonomous_policies_use_separate_entries(self) -> None:
        calls: list[bool] = []
        fetch_one_signature = inspect.signature(webfetch._fetch_one)

        async def fetch_one(*args: Any, **kwargs: Any) -> webfetch.Document:
            bound = fetch_one_signature.bind(*args, **kwargs)
            calls.append(bound.arguments["robots"] is not None)
            return make_document(str(bound.arguments["url"]))

        with mock.patch.object(webfetch, "_fetch_one", new=fetch_one):
            await webfetch.fetch("https://x.test/guide", respect_robots=False)
            await webfetch.fetch("https://x.test/guide", respect_robots=True)
            await webfetch.fetch("https://x.test/guide", respect_robots=True)
        self.assertEqual(calls, [False, True])

    async def test_documents_backed_by_mutable_files_are_not_cached(self) -> None:
        calls: dict[str, int] = {"n": 0}

        async def binary(*args: Any, **kwargs: Any) -> webfetch.Document:
            calls["n"] += 1
            return webfetch.Document(
                url=str(args[1]),
                final_url=str(args[1]),
                kind="binary",
                saved_path="/tmp/caller-may-delete-this.bin",
            )

        with mock.patch.object(webfetch, "_fetch_one", new=binary):
            await webfetch.fetch("https://x.test/file.bin")
            await webfetch.fetch("https://x.test/file.bin")
        self.assertEqual(calls["n"], 2)
        self.assertEqual(webfetch._DOC_CACHE, {})

    def test_replacing_an_existing_full_cache_entry_does_not_evict_another(self) -> None:
        for index in range(webfetch._DOC_CACHE_MAX_ENTRIES):
            webfetch._doc_cache_put((index,), make_document(f"https://x.test/{index}"))
        keys_before = set(webfetch._DOC_CACHE)

        last_key = (webfetch._DOC_CACHE_MAX_ENTRIES - 1,)
        webfetch._doc_cache_put(last_key, make_document("https://x.test/replacement"))
        self.assertEqual(set(webfetch._DOC_CACHE), keys_before)

    async def test_run_reports_cache_hits_in_rendered_notes(self) -> None:
        self.install_fetch_one()
        await webfetch.run("https://x.test/guide")
        rendered = await webfetch.run("https://x.test/guide")
        self.assertIn("note: from session cache", rendered)


if __name__ == "__main__":
    unittest.main()
