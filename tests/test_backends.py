"""Offline tests for every backend, using httpx.MockTransport. No network."""

from __future__ import annotations

import json
import os
import unittest
from typing import Any, Callable, Optional
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
                    {
                        "web": {
                            "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AAA",
                            "title": "foxsports.com",
                        }
                    },
                    {
                        "web": {
                            "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/BBB",
                            "title": "britannica.com",
                        }
                    },
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
        "cache_ttl": 0.0,
        "auth": {},
    }
    base.update(overrides)
    return config.Settings(**base)


def query_for(text: str = "q", **overrides: Any) -> config.SearchQuery:
    base: dict[str, Any] = {"text": text, "num_results": 5}
    base.update(overrides)
    return config.SearchQuery(**base)


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)


def patch_endpoints(case: unittest.TestCase, *endpoints: config.GeminiEndpoint) -> None:
    patcher = mock.patch.object(
        config.Settings, "gemini_endpoints", property(lambda self: tuple(endpoints))
    )
    case.addCleanup(patcher.stop)
    patcher.start()


CORP = config.GeminiEndpoint(
    label="corp",
    base_url="https://gw.example.com/v1beta",
    models=("gemini-3.6-flash",),
    keys=("sk-corp-secret-value",),
    source="models.json:corp",
)


class RequestBoundsTest(unittest.IsolatedAsyncioTestCase):
    async def test_provider_requests_force_identity_and_cap_streams(self) -> None:
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"x" * (backends.MAX_BACKEND_BYTES + 1)

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["accept-encoding"], "identity")
            return httpx.Response(200, stream=Stream())

        async with client_for(handler) as client:
            with self.assertRaises(backends.BackendError) as ctx:
                await backends._request(client, "test", "GET", "https://api.example/x")
        self.assertIn("response exceeded", str(ctx.exception))


class DomainAnswerTest(unittest.TestCase):
    def test_safe_html_entities_in_url_are_not_rewritten(self) -> None:
        url = "https://example.com/search?q=a&amp;b=c"
        result = backends._finish(
            backends.SearchResult("test", items=[backends.ResultItem("x", url)]),
            config.SearchQuery("q"),
        )
        self.assertEqual(result.items[0].url, url)

    def test_answer_is_dropped_without_in_scope_support(self) -> None:
        result = backends.SearchResult(
            backend="tavily",
            answer="SECRET FROM BLOCKED DOMAIN",
            items=[backends.ResultItem("blocked", "https://blocked.example/x")],
        )
        finished = backends._finish(
            result,
            config.SearchQuery("q", include_domains=("allowed.example",)),
        )
        self.assertIsNone(finished.answer)
        self.assertTrue(finished.empty)
        self.assertEqual(finished.dropped, 1)


class GeminiBackendTest(unittest.IsolatedAsyncioTestCase):
    async def test_parses_answer_sources_and_queries(self) -> None:
        patch_endpoints(self, CORP)
        seen_bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "grounding-api-redirect" in str(request.url):
                target = (
                    "https://foxsports.com/story"
                    if request.url.path.endswith("AAA")
                    else "https://britannica.com/x"
                )
                return httpx.Response(301, headers={"location": target})
            seen_bodies.append(json.loads(request.content))
            self.assertEqual(request.headers["x-goog-api-key"], "sk-corp-secret-value")
            return httpx.Response(200, json=GEMINI_PAYLOAD)

        async with client_for(handler) as client:
            result = await backends.search_gemini(client, query_for("who won"), settings_for())

        self.assertEqual(seen_bodies[0]["tools"], [{"google_search": {}}])
        self.assertEqual(result.detail, "corp/gemini-3.6-flash")
        self.assertEqual(result.answer, "Spain won the 2026 World Cup.")
        self.assertEqual(result.queries, ["who won the 2026 world cup"])
        self.assertEqual(
            [item.url for item in result.items],
            ["https://foxsports.com/story", "https://britannica.com/x", "https://example.org/plain"],
        )

    async def test_redirects_are_resolved_without_fetching_the_target(self) -> None:
        patch_endpoints(self, CORP)
        methods: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            methods.append((request.method, str(request.url)))
            if "grounding-api-redirect" in str(request.url):
                return httpx.Response(302, headers={"location": "https://publisher.example.com/story"})
            return httpx.Response(200, json=GEMINI_PAYLOAD)

        async with client_for(handler) as client:
            result = await backends.search_gemini(client, query_for(), settings_for())

        self.assertIn("https://publisher.example.com/story", [item.url for item in result.items])
        # Only HEAD on the redirector; the publisher itself is never requested.
        self.assertTrue(all("publisher.example.com" not in url for _, url in methods))
        self.assertTrue(any(method == "HEAD" for method, _ in methods))

    async def test_redirect_to_private_address_is_rejected(self) -> None:
        patch_endpoints(self, CORP)
        redirector = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AAA"

        def handler(request: httpx.Request) -> httpx.Response:
            if "grounding-api-redirect" in str(request.url):
                # SSRF attempt: cloud metadata service.
                return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})
            return httpx.Response(200, json=GEMINI_PAYLOAD)

        async with client_for(handler) as client:
            result = await backends.search_gemini(client, query_for(), settings_for())

        urls = [item.url for item in result.items]
        self.assertNotIn("http://169.254.169.254/latest/meta-data/", urls)
        self.assertNotIn(redirector, urls)  # unresolved redirectors are not useful sources
        self.assertGreaterEqual(result.dropped, 1)

    async def test_inline_citation_markers_from_grounding_supports(self) -> None:
        patch_endpoints(self, CORP)
        answer = "Spain won. Argentina lost."
        payload = {
            "candidates": [
                {
                    "content": {"parts": [{"text": answer}]},
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://a.example.com", "title": "a"}},
                            {"web": {"uri": "https://b.example.com", "title": "b"}},
                        ],
                        "groundingSupports": [
                            {"segment": {"endIndex": 11}, "groundingChunkIndices": [0]},
                            {"segment": {"endIndex": len(answer)}, "groundingChunkIndices": [0, 1]},
                        ],
                    },
                }
            ]
        }
        async with client_for(lambda request: httpx.Response(200, json=payload)) as client:
            result = await backends.search_gemini(client, query_for(), settings_for())
        self.assertEqual(result.answer, "Spain won.[1] Argentina lost.[1][2]")

    async def test_redirect_detection_never_matches_attacker_controlled_text(self) -> None:
        requested: list[str] = []
        item = backends.ResultItem(
            "trap",
            "http://169.254.169.254/latest?vertexaisearch.cloud.google.com",
        )

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            return httpx.Response(200)

        async with client_for(handler) as client:
            await backends._resolve_redirects(client, [item], max_candidates=10)
        self.assertEqual(requested, [])
        result = backends._finish(backends.SearchResult("gemini", items=[item]), query_for())
        self.assertEqual(result.items, [])
        self.assertEqual(result.dropped, 1)

    def test_grounding_filter_runs_before_result_cap(self) -> None:
        answer = "Claim"
        chunks = [
            {"web": {"uri": f"http://10.0.0.{index + 1}/x", "title": "blocked"}}
            for index in range(10)
        ]
        chunks.append(
            {"web": {"uri": "https://allowed.example/x", "title": "allowed"}}
        )
        payload = {
            "candidates": [
                {
                    "content": {"parts": [{"text": answer}]},
                    "groundingMetadata": {
                        "groundingChunks": chunks,
                        "groundingSupports": [
                            {
                                "segment": {"endIndex": len(answer)},
                                "groundingChunkIndices": [10],
                            }
                        ],
                    },
                }
            ]
        }
        parsed, items, _, metadata, chunk_map, ranges = backends._parse_gemini(payload)
        kept, source_numbers, _ = backends._finish_grounded(
            items,
            chunk_map,
            config.SearchQuery("q", include_domains=("allowed.example",)),
        )
        self.assertEqual([item.url for item in kept], ["https://allowed.example/x"])
        self.assertEqual(
            backends._annotate_citations(parsed or "", metadata, source_numbers, ranges),
            "Claim[1]",
        )

    async def test_citations_are_renumbered_after_safety_and_domain_filtering(self) -> None:
        patch_endpoints(self, CORP)
        answer = "Grounded claim."
        payload = {
            "candidates": [
                {
                    "content": {"parts": [{"text": None}, {"text": answer}]},
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "http://169.254.169.254/latest", "title": "private"}},
                            {"web": {"uri": "https://blocked.example/x", "title": "blocked"}},
                            {"web": {"uri": "https://allowed.example/a", "title": "allowed"}},
                            {"web": {"uri": "https://allowed.example/a", "title": "duplicate"}},
                        ],
                        "groundingSupports": [
                            {
                                "segment": {"partIndex": 1, "endIndex": len(answer)},
                                "groundingChunkIndices": [0, 1, 2, 3],
                            }
                        ],
                    },
                }
            ]
        }
        async with client_for(lambda request: httpx.Response(200, json=payload)) as client:
            result = await backends.search_gemini(
                client,
                query_for(include_domains=("allowed.example",)),
                settings_for(),
            )

        self.assertEqual([item.url for item in result.items], ["https://allowed.example/a"])
        self.assertEqual(result.answer, "Grounded claim.[1]")
        self.assertEqual(result.dropped, 2)

    async def test_pinned_model_skips_unneeded_model_listing(self) -> None:
        endpoint = config.GeminiEndpoint(
            "model-less", "https://gw.example.com/v1beta", (), ("key",)
        )
        patch_endpoints(self, endpoint)
        requested: list[tuple[str, str]] = []
        payload = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "answer"}]},
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://example.com/x", "title": "x"}}
                        ]
                    },
                }
            ]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append((request.method, str(request.url)))
            if request.method == "GET":
                raise AssertionError("a pinned model must not trigger model discovery")
            return httpx.Response(200, json=payload)

        async with client_for(handler) as client:
            result = await backends.search_gemini(
                client, query_for(), settings_for(gemini_model="pinned-model")
            )
        self.assertEqual(result.detail, "model-less/pinned-model")
        self.assertTrue(all(method != "GET" for method, _ in requested))

    async def test_citation_offsets_respect_raw_whitespace_and_part_index(self) -> None:
        patch_endpoints(self, CORP)
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "  First. "}, {"text": "Second."}]
                    },
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://example.com/x", "title": "x"}}
                        ],
                        "groundingSupports": [
                            {
                                "segment": {"partIndex": 1, "endIndex": 7},
                                "groundingChunkIndices": [0],
                            }
                        ],
                    },
                }
            ]
        }
        async with client_for(lambda request: httpx.Response(200, json=payload)) as client:
            result = await backends.search_gemini(client, query_for(), settings_for())
        self.assertEqual(result.answer, "First. Second.[1]")

        payload["candidates"][0]["content"] = {"parts": [{"text": "  Claim."}]}
        payload["candidates"][0]["groundingMetadata"]["groundingSupports"][0]["segment"] = {
            "endIndex": 8
        }
        async with client_for(lambda request: httpx.Response(200, json=payload)) as client:
            result = await backends.search_gemini(client, query_for(), settings_for())
        self.assertEqual(result.answer, "Claim.[1]")

    async def test_recency_and_domains_go_into_the_prompt(self) -> None:
        patch_endpoints(self, CORP)
        prompts: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "grounding-api-redirect" in str(request.url):
                return httpx.Response(200, text="ok")
            prompts.append(json.loads(request.content)["contents"][0]["parts"][0]["text"])
            return httpx.Response(200, json=GEMINI_PAYLOAD)

        query = query_for(
            "kernel panic",
            recency="week",
            include_domains=("lwn.net", "kernel.org"),
            exclude_domains=("reddit.com",),
        )
        async with client_for(handler) as client:
            await backends.search_gemini(client, query, settings_for())

        prompt = prompts[0]
        self.assertIn("kernel panic", prompt)
        self.assertIn("(site:lwn.net OR site:kernel.org)", prompt)
        self.assertIn("-site:reddit.com", prompt)
        self.assertIn("last week", prompt)

    async def test_falls_back_to_legacy_tool_name_on_400(self) -> None:
        patch_endpoints(self, CORP)
        tools_seen: list[Any] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "grounding-api-redirect" in str(request.url):
                return httpx.Response(200, text="ok")
            body = json.loads(request.content)
            tools_seen.append(body["tools"][0])
            if "google_search" in body["tools"][0]:
                return httpx.Response(400, json={"error": {"message": 'Unknown name "google_search"'}})
            return httpx.Response(200, json=GEMINI_PAYLOAD)

        async with client_for(handler) as client:
            result = await backends.search_gemini(client, query_for(), settings_for())

        self.assertEqual(tools_seen, [{"google_search": {}}, {"google_search_retrieval": {}}])
        self.assertEqual(result.answer, "Spain won the 2026 World Cup.")

    async def test_key_failover_on_401(self) -> None:
        patch_endpoints(
            self,
            config.GeminiEndpoint(
                label="corp",
                base_url="https://gw.example.com/v1beta",
                models=("gemini-3.6-flash",),
                keys=("sk-dead-key-value", "sk-live-key-value"),
            ),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if "grounding-api-redirect" in str(request.url):
                return httpx.Response(200, text="ok")
            if request.headers["x-goog-api-key"] == "sk-dead-key-value":
                return httpx.Response(401, json={"error": {"message": "invalid key"}})
            return httpx.Response(200, json=GEMINI_PAYLOAD)

        async with client_for(handler) as client:
            result = await backends.search_gemini(client, query_for(), settings_for())
        self.assertEqual(result.answer, "Spain won the 2026 World Cup.")

    async def test_endpoint_failover(self) -> None:
        dead = config.GeminiEndpoint("dead", "https://dead.example.com/v1beta", ("gemini-3.6-flash",), ("k1",))
        live = config.GeminiEndpoint("live", "https://live.example.com/v1beta", ("gemini-3.6-flash",), ("k2",))
        patch_endpoints(self, dead, live)

        def handler(request: httpx.Request) -> httpx.Response:
            if "grounding-api-redirect" in str(request.url):
                return httpx.Response(200, text="ok")
            if request.url.host == "dead.example.com":
                return httpx.Response(429, json={"error": {"message": "slow down"}})
            return httpx.Response(200, json=GEMINI_PAYLOAD)

        async with client_for(handler) as client:
            result = await backends.search_gemini(client, query_for(), settings_for())
        self.assertEqual(result.detail, "live/gemini-3.6-flash")

    async def test_ai_studio_lists_models_when_none_are_declared(self) -> None:
        patch_endpoints(
            self,
            config.GeminiEndpoint("google-ai-studio", config.AI_STUDIO_BASE_URL, (), ("sk-studio",)),
        )
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            if request.url.path.endswith("/models"):
                return httpx.Response(
                    200,
                    json={
                        "models": [
                            {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]},
                            {"name": "models/gemini-9-flash", "supportedGenerationMethods": ["generateContent"]},
                        ]
                    },
                )
            if "grounding-api-redirect" in str(request.url):
                return httpx.Response(200, text="ok")
            return httpx.Response(200, json=GEMINI_PAYLOAD)

        async with client_for(handler) as client:
            result = await backends.search_gemini(client, query_for(), settings_for())
        self.assertEqual(result.detail, "google-ai-studio/gemini-9-flash")

    async def test_no_endpoint_is_not_retryable(self) -> None:
        patch_endpoints(self)
        async with client_for(lambda request: httpx.Response(200)) as client:
            with self.assertRaises(backends.BackendError) as ctx:
                await backends.search_gemini(client, query_for(), settings_for())
        self.assertFalse(ctx.exception.retryable)


class CredentialBackendTest(unittest.IsolatedAsyncioTestCase):
    async def test_serper_maps_recency_to_tbs(self) -> None:
        settings = settings_for(auth={"serper": {"type": "api_key", "key": "sk-serper"}})
        payload = {
            "answerBox": {"answer": "42"},
            "knowledgeGraph": {"title": "Thing", "description": "A thing."},
            "organic": [
                {"title": "First", "link": "https://a.example.com", "snippet": "one"},
                {"title": "Second", "link": "https://b.example.com", "snippet": "two"},
            ],
        }
        bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["X-API-KEY"], "sk-serper")
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=payload)

        with mock.patch.dict(os.environ, {}, clear=True):
            async with client_for(handler) as client:
                result = await backends.search_serper(
                    client, query_for("q", recency="month", include_domains=("a.example.com",)), settings
                )

        self.assertEqual(bodies[0]["tbs"], "qdr:m")
        self.assertEqual(bodies[0]["q"], "q site:a.example.com")
        self.assertIn("42", result.answer or "")
        self.assertIn("Thing - A thing.", result.answer or "")
        # b.example.com is outside the include filter.
        self.assertEqual([item.url for item in result.items], ["https://a.example.com"])
        self.assertEqual(result.dropped, 1)

    async def test_tavily_uses_native_filter_fields(self) -> None:
        settings = settings_for(auth={"tavily": {"type": "api_key", "key": "tvly-key"}})
        payload = {
            "answer": "Short answer.",
            "results": [{"title": "T", "url": "https://t.example.com", "content": "body"}],
        }
        bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["authorization"], "Bearer tvly-key")
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=payload)

        with mock.patch.dict(os.environ, {}, clear=True):
            async with client_for(handler) as client:
                result = await backends.search_tavily(
                    client,
                    query_for(
                        "q",
                        recency="day",
                        include_domains=("t.example.com",),
                        exclude_domains=("spam.example.com",),
                    ),
                    settings,
                )

        body = bodies[0]
        self.assertEqual(body["time_range"], "day")
        self.assertEqual(body["include_domains"], ["t.example.com"])
        self.assertEqual(body["exclude_domains"], ["spam.example.com"])
        self.assertEqual(body["query"], "q")  # no site: operators when native fields exist
        self.assertEqual(result.answer, "Short answer.")

    async def test_brave_maps_recency_to_freshness(self) -> None:
        settings = settings_for(auth={"brave": {"type": "api_key", "key": "brave-key"}})
        payload = {"web": {"results": [{"title": "B", "url": "https://b.example.com", "description": "desc"}]}}
        seen: list[httpx.URL] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["X-Subscription-Token"], "brave-key")
            seen.append(request.url)
            return httpx.Response(200, json=payload)

        with mock.patch.dict(os.environ, {}, clear=True):
            async with client_for(handler) as client:
                await backends.search_brave(client, query_for("q", recency="year"), settings)
                await backends.search_brave(client, query_for("q", include_domains=("x.com",)), settings)

        self.assertEqual(seen[0].params["freshness"], "py")
        self.assertEqual(seen[0].params["count"], "5")
        # With domain operators, over-fetch and filter client-side.
        self.assertEqual(seen[1].params["count"], "20")
        self.assertIn("site:x.com", seen[1].params["q"])

    async def test_exa_maps_recency_to_start_date(self) -> None:
        settings = settings_for(auth={"exa": {"type": "api_key", "key": "exa-key"}})
        payload = {"results": [{"title": "E", "url": "https://e.example.com", "highlights": ["a", "b"]}]}
        bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["x-api-key"], "exa-key")
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=payload)

        with mock.patch.dict(os.environ, {}, clear=True):
            async with client_for(handler) as client:
                result = await backends.search_exa(
                    client, query_for("q", recency="week", include_domains=("e.example.com",)), settings
                )

        self.assertRegex(bodies[0]["startPublishedDate"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(bodies[0]["includeDomains"], ["e.example.com"])
        self.assertEqual(result.items[0].snippet, "a b")

    async def test_searxng_time_range_and_empty_results(self) -> None:
        settings = settings_for(searxng_url="https://searx.example.com")

        async with client_for(lambda request: httpx.Response(200, json={"results": []})) as client:
            with self.assertRaises(backends.BackendError):
                await backends.search_searxng(client, query_for(), settings)

        payload = {"results": [{"title": "S", "url": "https://s.example.com", "content": "c"}]}
        seen: list[httpx.URL] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url)
            return httpx.Response(200, json=payload)

        async with client_for(handler) as client:
            result = await backends.search_searxng(client, query_for("q", recency="month"), settings)
        self.assertEqual(seen[0].params["time_range"], "month")
        self.assertEqual(result.detail, "searx.example.com")

    async def test_missing_credential_is_not_retryable(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            async with client_for(lambda request: httpx.Response(200)) as client:
                for backend in (
                    backends.search_serper,
                    backends.search_tavily,
                    backends.search_brave,
                    backends.search_exa,
                    backends.search_searxng,
                ):
                    with self.assertRaises(backends.BackendError) as ctx:
                        await backend(client, query_for(), settings_for())
                    self.assertFalse(ctx.exception.retryable)


class DuckDuckGoTest(unittest.IsolatedAsyncioTestCase):
    async def test_parses_and_unwraps_redirects(self) -> None:
        async with client_for(lambda request: httpx.Response(200, text=DDG_HTML)) as client:
            result = await backends.search_ddg(client, query_for(), settings_for())
        self.assertEqual(result.items[0].url, "https://real.example.com/page")
        self.assertEqual(result.items[0].title, "Real Page")
        self.assertEqual(result.items[1].url, "https://direct.example.com/two")

    def test_redirect_target_is_decoded_exactly_once(self) -> None:
        wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%252Fb"
        self.assertEqual(backends._unwrap_ddg_url(wrapped), "https://example.com/a%2Fb")

    async def test_recency_uses_df_parameter(self) -> None:
        seen: list[bytes] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.content)
            return httpx.Response(200, text=DDG_HTML)

        async with client_for(handler) as client:
            await backends.search_ddg(client, query_for("q", recency="week"), settings_for())
        self.assertIn(b"df=w", seen[0])

    async def test_domain_filter_is_applied_client_side(self) -> None:
        async with client_for(lambda request: httpx.Response(200, text=DDG_HTML)) as client:
            result = await backends.search_ddg(
                client, query_for("q", exclude_domains=("direct.example.com",)), settings_for()
            )
        self.assertEqual([item.url for item in result.items], ["https://real.example.com/page"])
        self.assertEqual(result.dropped, 1)

    async def test_regex_parser_matches_bs4_urls(self) -> None:
        items = backends._parse_ddg_with_regex(DDG_HTML, 5)
        self.assertEqual(
            [item.url for item in items],
            ["https://real.example.com/page", "https://direct.example.com/two"],
        )

    async def test_endpoint_redirect_is_not_followed_blindly(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if request.url.host == "html.duckduckgo.com":
                return httpx.Response(
                    302,
                    headers={"location": "http://169.254.169.254/latest/meta-data/"},
                )
            return httpx.Response(200, text=DDG_HTML)

        async with client_for(handler) as client:
            result = await backends.search_ddg(client, query_for(), settings_for())
        self.assertTrue(result.items)
        self.assertEqual(
            calls,
            ["https://html.duckduckgo.com/html/", "https://lite.duckduckgo.com/lite/"],
        )

    async def test_rate_limit_is_reported_clearly(self) -> None:
        async with client_for(lambda request: httpx.Response(202, text="")) as client:
            with self.assertRaises(backends.BackendError) as ctx:
                await backends.search_ddg(client, query_for(), settings_for())
        self.assertIn("rate-limited", str(ctx.exception))
        self.assertIn("HTTP 202", str(ctx.exception))

    async def test_second_endpoint_is_tried(self) -> None:
        calls: list[Optional[str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.host)
            if (request.url.host or "").startswith("html."):
                return httpx.Response(503, text="nope")
            return httpx.Response(200, text=DDG_HTML)

        async with client_for(handler) as client:
            result = await backends.search_ddg(client, query_for(), settings_for())
        self.assertEqual(calls, ["html.duckduckgo.com", "lite.duckduckgo.com"])
        self.assertTrue(result.items)


class RedactionTest(unittest.IsolatedAsyncioTestCase):
    def test_clean_removes_html_encoded_and_terminal_controls(self) -> None:
        self.assertEqual(backends._clean("Title&#10;Injected\x1b[31m"), "Title Injected [31m")
        self.assertEqual(backends._clean("A&#x2028;INJECT\u202e"), "A INJECT")

    def test_error_secret_is_redacted_before_detail_cutoff(self) -> None:
        secret = "CUTOFF-SECRET-123"
        response = httpx.Response(
            400,
            json={"error": {"message": "x" * 195 + secret}},
            request=httpx.Request("GET", "https://api.example"),
        )
        with self.assertRaises(backends.BackendError) as ctx:
            backends._raise_for_status(response, "provider", (secret,))
        self.assertNotIn(secret, str(ctx.exception))
        self.assertNotIn("CUTOFF-", str(ctx.exception))

    async def test_secret_echoed_by_provider_is_redacted(self) -> None:
        secret = "sk-super-secret-value-1234"
        patch_endpoints(
            self, config.GeminiEndpoint("corp", "https://gw.example.com/v1beta", ("gemini-3.6-flash",), (secret,))
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"error": {"message": f"key {secret} is banned"}})

        with mock.patch.dict(os.environ, {}, clear=True):
            async with client_for(handler) as client:
                with self.assertRaises(backends.BackendError) as ctx:
                    await backends.search_gemini(client, query_for(), settings_for())
        raw = str(ctx.exception)
        self.assertNotIn(secret, raw)
        self.assertIn("***", raw)
        self.assertNotIn(secret, websearch._redact(raw, (secret,)))


if __name__ == "__main__":
    unittest.main()
