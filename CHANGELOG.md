# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-08-20

### Added

- Gemini **Files API** support for payloads over the ~18 MB inline request limit.
  `generateContent` carries inline data in the request body, so a large scanned PDF
  could not be transcribed at all before. Uploads use Google's documented resumable
  protocol (`start` with the upload headers, then `upload, finalize`), poll the file
  until it leaves `PROCESSING`, use it by URI on the same endpoint it was uploaded
  to, and delete it afterwards.
- Detection of endpoints that proxy only `generateContent`: a `404` on the files
  endpoints raises `FilesApiUnsupported`, that endpoint is skipped, and the final
  error suggests `max_pages` or local extraction. Both IBM ICA gateways behave this
  way; Google AI Studio exposes the API.

### Fixed

- `upload_base()` derived the upload prefix by splitting the raw URL string, so a
  base URL without a path (`https://example.com`) produced
  `https://upload/example.com` - a different host. It now rewrites only the path.

### Changed

- An oversized PDF is no longer refused outright; it goes through the upload path
  first and only fails when no endpoint can accept it.
- `generate()` was factored into a single-attempt helper so the upload path can pair
  one upload with one call on the same endpoint while keeping key and endpoint
  failover behaviour identical.

## [0.4.0] - 2026-08-20

### Added

- **Optional Gemini tiers in `webfetch`**, used only where local extraction cannot
  work. Endpoints, keys and key-rotator failover are reused from the `websearch`
  skill's discovery, so there is nothing extra to configure and no second source of
  truth. With no Gemini configured, every local capability keeps working and the
  tiers report why they are unavailable.
  - `prompt="..."` answers a question about a page through the `url_context` tool.
    Because Gemini fetches server-side, this also reads pages that need JavaScript
    or block scripted clients.
  - A YouTube link is read as a video (`fileData` part): content plus what is shown
    on screen.
  - A PDF with no text layer is transcribed by vision instead of returning an empty
    extraction - the deterministic-then-model tier chain used by pi-web-access.
  - A locally blocked or failed fetch falls back to `url_context` rather than
    returning an error.
  - `gemini=False` forces the local path, `gemini=True` forces the model path,
    `model=` pins the Gemini model.
- `Document.source` (`local`, `gemini-url-context`, `gemini-video`, `gemini-pdf`),
  `Document.answer`, `Document.retrieved_urls`, and `webfetch.gemini_available()`.

### Security

- A URL refused by the safety checks is never passed to the model either: unsafe-URL
  errors short-circuit before any tier, so `url_context` cannot be used to reach an
  address the local fetcher would refuse.
- Oversized PDFs are refused before the request when they exceed the inline payload
  limit for model extraction.

## [0.3.0] - 2026-08-20

### Added

- **`webfetch` skill**: fetch a URL and read it as markdown. The package now ships
  two skills and is renamed `prime-agent-web`.
  - HTML to markdown keeping headings, code blocks and link targets; boilerplate
    (`script`, `style`, `nav`, `header`, `footer`, `aside`, `form`, `iframe`) removed
    and content taken from `<main>`/`<article>` when present.
  - PDF text per page with `--- page N ---` markers, page count and metadata title;
    scanned PDFs are reported as having no text layer.
  - `mode="raw"` for exact bodies, `mode="text"` for plain text.
  - Binaries written to a temp file with the path reported.
  - `github.com/o/r/blob/…` rewritten to `raw.githubusercontent.com`; a repository
    root gets a `git clone` hint instead of scraped HTML.
  - `fetch()` returns `Document` objects with the full text and fetches lists of
    URLs concurrently; `run()` renders a bounded view and never raises.
- SSRF guards for fetching: http(s) only, no credentials in URLs, private, loopback,
  link-local, metadata and bare-internal targets refused, **DNS preflight** on every
  hostname, manual redirect following (max 5 hops) with re-validation of each hop.
- Size guards: `content-length` over the cap is rejected before download, a streaming
  guard cuts off unannounced bodies, and an oversized PDF fails with the exact
  `max_bytes=` value to retry with.
- `robots.txt` honoured by default for autonomous fetches, using the convention of
  the official MCP fetch server (Autonomous vs User-Specified user agents, `401`/`403`
  treated as a refusal), implemented with the standard library so it costs no
  dependency. `respect_robots=False` overrides it.
- `transport` injection point on `webfetch.fetch()` for tests and custom networking.
- Contract test that no module attribute shadows a sibling submodule, and that every
  skill in the package satisfies the Prime Agent detection contract.

### Changed

- Package renamed from `prime-agent-websearch` to `prime-agent-web`; both skills are
  versioned 0.3.0. GitHub redirects the old repository URL.
- A readability-style extractor was evaluated and **rejected** on measurements: on an
  API reference page it produced 1.7 KB with zero headings and zero links, against
  25 KB with 62 code blocks from the conservative pipeline that shipped.

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
