# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-20

### Added

- `recency` argument (`day`, `week`, `month`, `year`, or `d`/`w`/`m`/`y`), mapped
  to each backend's native parameter: Tavily `time_range`, Brave `freshness`,
  Serper `tbs=qdr:*`, Exa `startPublishedDate`, SearXNG `time_range`, DuckDuckGo
  `df`, and a prompt hint for Gemini grounding.
- `domains` argument accepting a list or comma-separated string, with `-` to
  exclude. Uses native fields on Tavily and Exa, `site:` / `-site:` operators
  elsewhere, and re-checks every result client-side. Removed results are reported
  as a count in the output.
- Inline `[n]` citation markers in Gemini answers, built from
  `groundingMetadata.groundingSupports`, so individual claims map to sources.
- Concurrent fan-out: `provider="all"` and explicit chains now query backends with
  `asyncio.gather` instead of sequentially.
- In-process result cache (default 300s TTL, 64 entries, `clear_cache()` helper,
  `PRIME_AGENT_WEBSEARCH_CACHE_TTL` to tune or disable) to protect provider quota
  during agent loops.
- Gemini endpoint failover across multiple endpoints, not just across keys.
- `backends()` now reports each credential's **source** (`$BRAVE_API_KEY`,
  `auth.json:tavily`, `models.json:<provider> + key-rotator (N pool keys)`), the
  supported recency values, and the cache state.
- `SECURITY.md` documenting the threat model.

### Changed

- Backends now take a normalized `SearchQuery` (text, result count, recency,
  domain filters) instead of a raw string.
- Redirect resolution no longer follows redirects: it reads `Location` with
  redirect following disabled, so a grounding link's target host is never
  contacted.
- `SearchResult` gained a `dropped` count for client-side filtering.

### Fixed

- DuckDuckGo rate limiting (`HTTP 202` with an empty body) was reported as
  "no results parsed"; it is now identified as a rate limit and fails over to the
  lite endpoint.
- Citation markers no longer land after a trailing space when a provider's segment
  offsets include it.
- Cache expiry now reads the clock through an injectable `_now()` helper. The TTL
  test previously forced expiry by writing an absolute `0.0` timestamp, which is
  wrong because `time.monotonic()` counts from an arbitrary origin (uptime on
  Linux): on a freshly booted CI runner the entry still looked fresh.

### Security

- Redirect targets are validated against loopback, private, link-local, multicast,
  reserved, and unspecified addresses (IPv4 and IPv6), plus `localhost`, `*.local`,
  `*.internal`, and cloud metadata hostnames. Non-`http(s)` schemes and URLs
  carrying credentials are refused.

## [0.1.0] - 2026-08-20

### Added

- Initial release: Python-backed `websearch` skill for Prime Agent with Gemini
  Google-Search grounding, Tavily, Brave, Serper, Exa, SearXNG, and keyless
  DuckDuckGo backends, auto-detected from `models.json`, `auth.json`,
  `key-rotator.json`, and environment variables.
- Credential redaction, `run()`/`search()`/`backends()` API, offline test suite.
