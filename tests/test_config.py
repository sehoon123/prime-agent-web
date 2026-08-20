"""Offline tests for backend discovery. No network, no host files."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from websearch import config


class SecretResolutionTest(unittest.TestCase):
    def test_literal_value(self) -> None:
        self.assertEqual(config.resolve_secret("sk-literal"), "sk-literal")

    def test_env_var_name_wins(self) -> None:
        with mock.patch.dict(os.environ, {"MY_KEY": "sk-from-env"}, clear=False):
            self.assertEqual(config.resolve_secret("MY_KEY"), "sk-from-env")

    def test_command_reference_is_skipped(self) -> None:
        self.assertIsNone(config.resolve_secret("!security find-generic-password -ws x"))

    def test_blank_and_non_string(self) -> None:
        self.assertIsNone(config.resolve_secret("   "))
        self.assertIsNone(config.resolve_secret(None))
        self.assertIsNone(config.resolve_secret(42))


class CredentialLookupTest(unittest.TestCase):
    auth = {
        "serper": {"type": "api_key", "key": "sk-serper"},
        "oauthy": {"type": "oauth", "access": "nope"},
    }

    def test_api_key_entry(self) -> None:
        self.assertEqual(config.auth_credential(self.auth, "serper"), "sk-serper")

    def test_oauth_entry_is_ignored(self) -> None:
        self.assertIsNone(config.auth_credential(self.auth, "oauthy"))

    def test_env_beats_auth_file(self) -> None:
        with mock.patch.dict(os.environ, {"SERPER_API_KEY": "sk-env"}, clear=False):
            self.assertEqual(config.credential(self.auth, ("serper",), ("SERPER_API_KEY",)), "sk-env")

    def test_falls_back_to_auth_file(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.credential(self.auth, ("serper",), ("SERPER_API_KEY",)), "sk-serper")


class GeminiEndpointDiscoveryTest(unittest.TestCase):
    models_json = {
        "providers": {
            "corp-gemini": {
                "baseUrl": "https://gw.example.com/v1beta",
                "api": "google-generative-ai",
                "models": [{"id": "gemini-3.1-pro"}, {"id": "gemini-3.6-flash"}],
            },
            "corp-claude": {
                "baseUrl": "https://gw.example.com",
                "api": "anthropic-messages",
                "models": [{"id": "claude-sonnet-5"}],
            },
            "mixed": {
                "baseUrl": "https://mixed.example.com/v1beta/",
                "models": [
                    {"id": "gpt-x", "api": "openai-completions"},
                    {"id": "gemini-mixed-flash", "api": "google-generative-ai"},
                ],
            },
            "keyless-gemini": {
                "baseUrl": "https://nokey.example.com",
                "api": "google-generative-ai",
                "models": [{"id": "gemini-3.6-flash"}],
            },
        }
    }
    auth = {
        "corp-gemini": {"type": "api_key", "key": "sk-corp"},
        "mixed": {"type": "api_key", "key": "sk-mixed"},
    }

    def endpoints(self, rotator: config.RotatorKeys | None = None) -> tuple[config.GeminiEndpoint, ...]:
        with mock.patch.dict(os.environ, {}, clear=True):
            return config.gemini_endpoints(
                models_json=self.models_json,
                auth=self.auth,
                rotator=rotator or config.RotatorKeys(),
            )

    def test_only_google_api_providers_with_keys(self) -> None:
        labels = [endpoint.label for endpoint in self.endpoints()]
        self.assertEqual(labels, ["corp-gemini", "mixed"])

    def test_model_level_api_is_detected(self) -> None:
        mixed = next(e for e in self.endpoints() if e.label == "mixed")
        self.assertEqual(mixed.models, ("gemini-mixed-flash",))

    def test_base_url_trailing_slash_removed(self) -> None:
        mixed = next(e for e in self.endpoints() if e.label == "mixed")
        self.assertEqual(mixed.base_url, "https://mixed.example.com/v1beta")

    def test_prefers_flash_model(self) -> None:
        corp = next(e for e in self.endpoints() if e.label == "corp-gemini")
        self.assertEqual(corp.pick_model(None), "gemini-3.6-flash")

    def test_pinned_model_wins(self) -> None:
        corp = next(e for e in self.endpoints() if e.label == "corp-gemini")
        self.assertEqual(corp.pick_model("gemini-3.1-pro"), "gemini-3.1-pro")

    def test_rotator_keys_are_appended_without_duplicates(self) -> None:
        rotator = config.RotatorKeys(by_provider={"corp-gemini": ["sk-corp", "sk-pool-2", "sk-pool-3"]})
        corp = next(e for e in self.endpoints(rotator) if e.label == "corp-gemini")
        self.assertEqual(corp.keys, ("sk-corp", "sk-pool-2", "sk-pool-3"))

    def test_ai_studio_endpoint_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "sk-studio"}, clear=True):
            endpoints = config.gemini_endpoints(
                models_json={}, auth={}, rotator=config.RotatorKeys()
            )
        self.assertEqual([e.label for e in endpoints], ["google-ai-studio"])
        self.assertEqual(endpoints[0].base_url, config.AI_STUDIO_BASE_URL)


class RotatorParsingTest(unittest.TestCase):
    def load(self, document: dict, env: dict[str, str] | None = None) -> config.RotatorKeys:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "key-rotator.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            environment = {"PRIME_AGENT_WEBSEARCH_KEY_ROTATOR": str(path), **(env or {})}
            with mock.patch.dict(os.environ, environment, clear=True):
                return config.load_rotator_keys()

    def test_multi_pool_shape(self) -> None:
        keys = self.load(
            {
                "pools": [
                    {
                        "poolId": "a",
                        "targets": [{"provider": "p-gemini"}, {"provider": "p-claude"}],
                        "keys": [{"id": "k1", "value": "sk-1"}, {"id": "k2", "value": "sk-2"}],
                    },
                    {
                        "poolId": "b",
                        "targets": [{"provider": "q-gemini"}],
                        "keys": [{"id": "k3", "value": "sk-3"}],
                    },
                ]
            }
        )
        self.assertEqual(keys.get("p-gemini"), ("sk-1", "sk-2"))
        self.assertEqual(keys.get("p-claude"), ("sk-1", "sk-2"))
        self.assertEqual(keys.get("q-gemini"), ("sk-3",))
        self.assertEqual(keys.get("unknown"), ())

    def test_legacy_single_pool_shape(self) -> None:
        keys = self.load({"provider": "solo", "api": "google-generative-ai", "keys": [{"value": "sk-solo"}]})
        self.assertEqual(keys.get("solo"), ("sk-solo",))

    def test_env_backed_keys(self) -> None:
        keys = self.load(
            {"provider": "solo", "keys": [{"env": "POOL_KEY_1"}, {"env": "MISSING_VAR"}]},
            env={"POOL_KEY_1": "sk-env-pool"},
        )
        self.assertEqual(keys.get("solo"), ("sk-env-pool",))

    def test_missing_file_is_not_fatal(self) -> None:
        with mock.patch.dict(
            os.environ, {"PRIME_AGENT_WEBSEARCH_KEY_ROTATOR": "/nonexistent/key-rotator.json"}, clear=True
        ):
            with mock.patch.object(config, "read_first_json", return_value={}):
                self.assertEqual(config.load_rotator_keys().by_provider, {})


class OrderParsingTest(unittest.TestCase):
    def test_auto_and_all_use_full_order(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.parse_order("auto"), config.AUTO_ORDER)
            self.assertEqual(config.parse_order("all"), config.AUTO_ORDER)
            self.assertEqual(config.parse_order(None), config.AUTO_ORDER)

    def test_single_and_chain(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.parse_order("ddg"), ("ddg",))
            self.assertEqual(config.parse_order("gemini,ddg"), ("gemini", "ddg"))
            self.assertEqual(config.parse_order("GEMINI, DDG"), ("gemini", "ddg"))

    def test_unknown_backend_raises(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                config.parse_order("bing")
        self.assertIn("bing", str(ctx.exception))

    def test_env_default(self) -> None:
        with mock.patch.dict(os.environ, {"PRIME_AGENT_WEBSEARCH_PROVIDER": "ddg"}, clear=True):
            self.assertEqual(config.parse_order(None), ("ddg",))

    def test_wants_every_backend(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(config.wants_every_backend("all"))
            self.assertTrue(config.wants_every_backend("gemini,ddg"))
            self.assertFalse(config.wants_every_backend("auto"))
            self.assertFalse(config.wants_every_backend("ddg"))


class SettingsTest(unittest.TestCase):
    def test_clamps_and_defaults(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(config, "read_first_json", return_value={}):
                self.assertEqual(config.load_settings(num_results=999).num_results, config.MAX_NUM_RESULTS)
                self.assertEqual(config.load_settings(num_results=0).num_results, 1)
                self.assertEqual(config.load_settings().num_results, config.DEFAULT_NUM_RESULTS)
                self.assertEqual(config.load_settings(timeout=0.1).timeout, 1.0)

    def test_env_defaults(self) -> None:
        env = {"PRIME_AGENT_WEBSEARCH_NUM_RESULTS": "7", "PRIME_AGENT_WEBSEARCH_TIMEOUT": "12.5"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(config, "read_first_json", return_value={}):
                settings = config.load_settings()
        self.assertEqual(settings.num_results, 7)
        self.assertEqual(settings.timeout, 12.5)

    def test_ddg_always_available_others_gated(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(config, "read_first_json", return_value={}):
                settings = config.load_settings()
                self.assertTrue(settings.available("ddg"))
                self.assertFalse(settings.available("serper"))
                self.assertFalse(settings.available("gemini"))
                self.assertFalse(settings.available("searxng"))

    def test_agent_dirs_honour_overrides(self) -> None:
        with mock.patch.dict(os.environ, {"PRIME_AGENT_CODING_AGENT_DIR": "/tmp/custom-agent"}, clear=True):
            dirs = config.agent_dirs()
        self.assertEqual(dirs[0], Path("/tmp/custom-agent"))
        self.assertIn(Path.home() / ".prime" / "agent", dirs)


if __name__ == "__main__":
    unittest.main()


class DomainFilterTest(unittest.TestCase):
    def test_normalize_domain_forms(self) -> None:
        for raw, expected in [
            ("github.com", "github.com"),
            ("  GitHub.COM ", "github.com"),
            ("https://github.com/owner/repo", "github.com"),
            ("-reddit.com", "reddit.com"),
            ("http://user@site.example.com/x?y=1", "site.example.com"),
            (".github.com.", "github.com"),
        ]:
            self.assertEqual(config.normalize_domain(raw), expected, raw)

    def test_normalize_domain_rejects_junk(self) -> None:
        for raw in ("", "  ", "-", "two words", "under_score.com", "http://"):
            self.assertIsNone(config.normalize_domain(raw), raw)

    def test_parse_domains_from_list_and_string(self) -> None:
        self.assertEqual(
            config.parse_domains(["github.com", "-reddit.com", "https://lwn.net/x"]),
            (("github.com", "lwn.net"), ("reddit.com",)),
        )
        self.assertEqual(
            config.parse_domains("github.com, -reddit.com"),
            (("github.com",), ("reddit.com",)),
        )
        self.assertEqual(config.parse_domains(None), ((), ()))
        self.assertEqual(config.parse_domains([]), ((), ()))

    def test_parse_domains_dedupes(self) -> None:
        self.assertEqual(config.parse_domains(["a.com", "A.com", "a.com"]), (("a.com",), ()))


class RecencyTest(unittest.TestCase):
    def test_accepts_names_and_shorthands(self) -> None:
        self.assertEqual(config.parse_recency("week"), "week")
        self.assertEqual(config.parse_recency(" DAY "), "day")
        self.assertEqual(config.parse_recency("m"), "month")
        self.assertIsNone(config.parse_recency(None))
        self.assertIsNone(config.parse_recency("  "))

    def test_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            config.parse_recency("fortnight")

    def test_start_date_is_in_the_past(self) -> None:
        from datetime import datetime, timezone

        now = datetime(2026, 3, 15, tzinfo=timezone.utc)
        self.assertEqual(config.recency_start_date("day", now), "2026-03-14")
        self.assertEqual(config.recency_start_date("week", now), "2026-03-08")
        self.assertEqual(config.recency_start_date("year", now), "2025-03-15")


class SearchQueryTest(unittest.TestCase):
    def test_operator_text_builds_site_filters(self) -> None:
        single = config.SearchQuery("q", include_domains=("a.com",))
        self.assertEqual(single.operator_text(), "q site:a.com")

        multi = config.SearchQuery("q", include_domains=("a.com", "b.com"), exclude_domains=("c.com",))
        self.assertEqual(multi.operator_text(), "q (site:a.com OR site:b.com) -site:c.com")

        plain = config.SearchQuery("q")
        self.assertEqual(plain.operator_text(), "q")

    def test_recency_hint_is_opt_in(self) -> None:
        query = config.SearchQuery("q", recency="week")
        self.assertEqual(query.operator_text(), "q")
        self.assertEqual(query.operator_text(with_recency_hint=True), "q (published within the last week)")

    def test_allows_matches_subdomains(self) -> None:
        query = config.SearchQuery("q", include_domains=("example.com",))
        self.assertTrue(query.allows("https://example.com/a"))
        self.assertTrue(query.allows("https://docs.example.com/a"))
        self.assertFalse(query.allows("https://notexample.com/a"))
        self.assertFalse(query.allows("https://other.org/a"))

    def test_exclude_wins_over_include(self) -> None:
        query = config.SearchQuery("q", include_domains=("example.com",), exclude_domains=("bad.example.com",))
        self.assertTrue(query.allows("https://good.example.com/a"))
        self.assertFalse(query.allows("https://bad.example.com/a"))

    def test_no_filter_allows_everything(self) -> None:
        self.assertTrue(config.SearchQuery("q").allows("https://anything.example"))

    def test_cache_key_distinguishes_parameters(self) -> None:
        base = config.SearchQuery("q", num_results=5)
        self.assertNotEqual(base.cache_key, config.SearchQuery("q", num_results=6).cache_key)
        self.assertNotEqual(base.cache_key, config.SearchQuery("q", recency="week").cache_key)
        self.assertNotEqual(base.cache_key, config.SearchQuery("q", include_domains=("a.com",)).cache_key)
        self.assertEqual(base.cache_key, config.SearchQuery("q", num_results=5).cache_key)


class UrlSafetyTest(unittest.TestCase):
    def test_public_urls_pass(self) -> None:
        for url in (
            "https://example.com/a",
            "http://sub.example.co.uk/b?c=1",
            "https://8.8.8.8/dns",
        ):
            self.assertTrue(config.is_public_http_url(url), url)

    def test_private_and_local_targets_are_blocked(self) -> None:
        for url in (
            "http://169.254.169.254/latest/meta-data/",   # cloud metadata
            "http://metadata.google.internal/x",
            "http://127.0.0.1:8080/admin",
            "http://localhost/admin",
            "http://10.0.0.5/",
            "http://192.168.1.1/",
            "http://172.16.0.1/",
            "http://[::1]/",
            "http://[fd00::1]/",
            "http://printer.local/",
            "http://vault.internal/",
            "http://intranet/",                            # bare internal label
        ):
            self.assertFalse(config.is_public_http_url(url), url)

    def test_non_http_schemes_and_credentials_are_blocked(self) -> None:
        for url in (
            "file:///etc/passwd",
            "ftp://example.com/x",
            "gopher://example.com/",
            "javascript:alert(1)",
            "https://user:pass@example.com/",
            "",
            "   ",
        ):
            self.assertFalse(config.is_public_http_url(url), url)


class CredentialSourceTest(unittest.TestCase):
    def test_source_reports_env_var_name(self) -> None:
        with mock.patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-x"}, clear=True):
            found = config.find_credential({}, ("tavily",), ("TAVILY_API_KEY",))
        assert found
        self.assertEqual(found.source, "$TAVILY_API_KEY")

    def test_source_reports_auth_entry(self) -> None:
        auth = {"tavily": {"type": "api_key", "key": "tvly-x"}}
        with mock.patch.dict(os.environ, {}, clear=True):
            found = config.find_credential(auth, ("tavily",), ("TAVILY_API_KEY",))
        assert found
        self.assertEqual(found.source, "auth.json:tavily")

    def test_gemini_endpoint_source_mentions_rotator(self) -> None:
        models_json = {
            "providers": {
                "corp": {
                    "baseUrl": "https://gw.example.com/v1beta",
                    "api": "google-generative-ai",
                    "models": [{"id": "gemini-3.6-flash"}],
                }
            }
        }
        auth = {"corp": {"type": "api_key", "key": "sk-corp"}}
        rotator = config.RotatorKeys(by_provider={"corp": ["sk-pool-1", "sk-pool-2"]})
        with mock.patch.dict(os.environ, {}, clear=True):
            endpoints = config.gemini_endpoints(models_json=models_json, auth=auth, rotator=rotator)
        self.assertIn("models.json:corp", endpoints[0].source)
        self.assertIn("key-rotator (2 pool keys)", endpoints[0].source)
