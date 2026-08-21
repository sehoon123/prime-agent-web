# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.3] - 2026-08-22

### Fixed

- Replaced the robots wildcard retry loop with ordered native substring searches.
  The old heuristic work counter could admit a short-looking pattern that then took
  seconds to reject; the simpler matcher preserves `*` and terminal `$` behavior
  without regex backtracking, while the aggregate work limit now reflects its actual
  linear scan cost.
- Corrected `run()` documentation: raw mode returns decoded, otherwise unprocessed
  text rather than the transfer bytes exactly as served.

### Changed

- Removed parser limits already implied by the bounded 1 MiB robots input, collapsed
  duplicated configuration-file and basic-auth parsing, and reduced cache fingerprints
  to one process-keyed BLAKE2 digest. Public APIs, cache boundaries, and health behavior
  are unchanged.

## [0.6.2] - 2026-08-21

### Fixed

- **websearch cache correctness.** The effective Gemini model is now part of the
  key, raw `search()` results cannot mutate cached entries, replacing an existing
  full-cache key no longer evicts an unrelated entry, and `clear_cache()` prevents
  an already-running search from repopulating the cleared cache.
- **websearch health ordering and output.** Health is recorded when each backend
  attempt actually completes, so an older `provider="all"` success cannot erase a
  newer concurrent failure. Turning cooldown off clears active deadlines; expired
  cooldowns are described as ready to retry rather than recovered; every
  last-resort attempt is removed from `cooled:`; and cache hits no longer report
  failures from the original network call. Backend names are not duplicated in
  formatted failures.
- **webfetch cache policy and invalidation.** `PRIME_AGENT_WEBFETCH_CACHE_TTL=0`
  now disables writes as well as reads. Manual (`respect_robots=False`) and
  autonomous fetches use separate entries. `clear_cache()` blocks late writes
  from requests that were already running, and replacing an existing full-cache
  key does not evict another document.
- Documents backed by a mutable `saved_path` are no longer cached; a later call
  refetches instead of returning a path the caller may have deleted or changed.
- Package metadata, module versions, both User-Agent families, and the root Pi/npm
  package now agree on `0.6.2`; contract tests keep them in sync. `npm test` now
  supplies both source paths and an isolated dependency environment on a clean
  checkout. The npm tarball now contains only skill source and required metadata,
  including the linked security notes, never test bytecode or build artifacts.
- **webfetch SSRF boundaries.** URL syntax and DNS safety now run before every local,
  video, prompt and Gemini fallback tier. Robots retrieval uses the guarded stream,
  redirected destinations get their own robots verdict, and native connections are
  pinned to vetted DNS addresses to prevent rebinding between validation and connect,
  while proxy/NO_PROXY routing is selected from the logical URL. Safe address failover
  remains, exact-origin redirect cookies work, and cookies never cross logical origins
  merely because hosts share an IP. NAT64-to-private, CGNAT, alternate numeric or
  percent/entity-encoded authorities, multicast, site-local and Unicode control targets
  are rejected.
- **Gemini confidentiality and availability.** Provider echoes are redacted across
  webfetch answers and errors, including short keys and decoded basic-auth values;
  retrieved URLs containing a known credential are removed rather than rewritten. AI Studio endpoints with no static model list use
  the documented fallback models. Resumable upload sessions cannot cross the
  configured origin or receive the API key, and partially processed files are deleted
  when validation, polling, timeout or cancellation fails. Credentials claimed by a
  custom `google`/`gemini` provider never fail over to the public Studio origin, and
  all Gemini/Files response bodies are bounded before JSON parsing.
- **Bounded document handling.** Exact-size bodies are no longer marked truncated;
  streaming retains at most one decoded byte over the cap, bounds transfer decoding,
  and wraps read failures. Binary bodies use atomically published mode-`0600`
  content-addressed temporary files without following or replacing existing paths.
  The PDF inline threshold accounts for base64 expansion, and
  `max_pages` also limits bytes sent to Gemini. Partial binary bodies are rejected,
  concatenated gzip members are preserved, compressed wire length is not confused with
  decoded length, and HTML/PDF extraction runs off the event loop.
- **Grounding correctness.** Gemini citation markers now retain provider part-relative
  byte offsets, survive output trimming, merge duplicate supports, and renumber after
  unsafe, duplicate or domain-filtered sources are removed. Only the exact Google
  grounding redirect hostname can receive a streamed `HEAD` request. Redirect candidate
  count, concurrency and aggregate time are bounded, and unresolved redirectors are
  dropped. Citation insertion is linear and work-capped, generated answers are capped,
  safety/domain filtering precedes the result cap, and unsupported out-of-scope answers
  are dropped.
- Process-keyed configuration fingerprints prevent cached SearXNG or Gemini results
  from crossing endpoint, model or credential rotations without retaining raw keys. Health reset now ignores old in-flight completions, and calling
  `health()` while cooldown is off clears stale deadlines.
- Per-URL unexpected webfetch failures become error Documents without losing sibling
  batch results. DNS, robots and file-processing waits obey caller timeouts, malformed
  ports are rejected, HTML meta charsets are honored, and user-facing `run()` calls
  keep their never-raise contract for invalid argument types. Robots parses the bounded
  RFC prefix and uses longest-match/Allow precedence, non-regex wildcards, anchors,
  and equivalent Unicode/UTF-8 percent normalization off the event loop.
  Explicit prompts or `gemini=True` now return a clear model error instead of caching
  a local dump that did not satisfy the request.

### Changed

- Refactored duplicated websearch health bookkeeping into the per-attempt path so
  sequential, fan-out, and last-resort calls share one completion-order rule.
  No new public API or capability was added.

## [0.6.1] - 2026-08-21

### Fixed

- **websearch: cooled backends are a last resort, not a blind spot (regression).**
  When every ready backend returned nothing, the automatic chain now retries the
  cooled backends once before failing - restoring the pre-0.6 guarantee that an
  automatic search only fails after all configured backends have seen the query.
  A last-resort success clears that backend's `cooled:` entry from the output and
  resets its health.
- **websearch: cache hits no longer claim `cooled:` skips.** A cache hit attempts
  nothing, so carrying the original call's skip list was stale information; hits
  now never set `deferred`.
- **websearch: truthful `health()` when the cooldown is disabled.** With
  `PRIME_AGENT_WEBSEARCH_COOLDOWN=0`, accumulated failures were rendered as
  "ok (recovered)"; they now read "N recent failure(s), cooldown off".
- **webfetch: cache entries are isolated from caller mutation.** Documents are
  copied on store and on hit (`notes`, `retrieved_urls` duplicated), so one
  caller's annotations can no longer leak into later hits or poison the cache.
- **User-Agent strings actually match the release.** The webfetch skill still
  sent `/0.3`; both skills now send `/0.6.1`.
- Single source for the cooldown base: `config.DEFAULT_COOLDOWN` aliases
  `_health.BASE_COOLDOWN` instead of duplicating the value.

## [0.6.0] - 2026-08-21

### Added

- **websearch: trajectory-driven backend health.** A backend whose call raises earns
  a session-scoped cooldown starting at 120s (`PRIME_AGENT_WEBSEARCH_COOLDOWN`, `0`
  disables) and doubling per consecutive failure, capped at x8; one success resets
  it. The automatic `auto` chain skips cooled backends, the output trailer lists
  them as `cooled: ...`, and new helpers expose the evidence:
  `await websearch.health()` renders per-backend state and reasons,
  `websearch.reset_health()` clears it. Explicit requests are never refused -
  named providers and `provider="all"` attempt everything regardless of health.
  An empty result set is nobody's fault and cools nothing down. State lives in
  the process only: every fresh session starts from zero assumptions.
- **webfetch: session document cache.** Successful fetches are reused within the
  kernel session under a content key (URL, mode, prompt, gemini/model choice,
  byte cap, page cap), default TTL 300s (`PRIME_AGENT_WEBFETCH_CACHE_TTL`, `0`
  disables), at most 32 entries, documents over ~2M chars not stored. Hits return
  a per-call copy annotated with a `from session cache` note; errors are never
  cached. Injected `transport`/`resolver` bypass the cache entirely, keeping test
  and custom-networking paths deterministic. New helper: `webfetch.clear_cache()`.

### Changed

- `backends()` now also reports the cooldown setting next to the cache line.
- User-Agent strings bumped to `/0.6` in both skills.

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
  - `mode="raw"` for unprocessed decoded text, `mode="text"` for plain text.
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
