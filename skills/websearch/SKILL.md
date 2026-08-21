---
name: websearch
description: Search the web and get an answer with real source URLs. Uses Gemini Google-Search grounding, Tavily, Brave, Serper, Exa, SearXNG, or keyless DuckDuckGo - whichever the host has configured. Supports recency and domain filters. Returns a grounded answer with [n] citations, numbered sources, and the searches that were run. Use for any question needing current information; fetch a page with httpx afterwards when full text is needed.
---

# Web Search

Multi-backend replacement for the Serper-only `websearch` skill bundled with
Prime Agent. Call the prepared import directly in the kernel:

```python
print(await websearch("prime agent latest release"))
```

## Arguments

```python
await websearch("gemini 3 pricing", num_results=8)          # 1-20 results
await websearch("cve-2026-1234", recency="week")            # day | week | month | year
await websearch("rust async", domains="github.com,-reddit.com")  # "-" excludes
await websearch("who won euro 2024", provider="ddg")        # force one backend
await websearch("rust async runtimes", provider="all")      # every backend, concurrently
await websearch("kernel panic", provider="gemini,ddg")      # explicit concurrent fan-out
help(websearch)                                             # full signature
```

Batching is plain Python — the kernel is the composition layer, so there is no
`queries` argument:

```python
import asyncio
answers = await asyncio.gather(*(websearch(q) for q in ["query a", "query b"]))
```

## Helpers

```python
await websearch.backends()      # which backends are usable here, and why
await websearch.health()        # per-backend cooldowns, with the evidence
websearch.reset_health()        # forget recorded failures; back to static order
websearch.clear_cache()         # drop the in-process result cache

results = await websearch.search("prime agent", provider="all")   # raw objects
for result in results:
    print(result.backend, result.detail, result.answer, result.dropped)
    for item in result.items:
        print(item.title, item.url, item.snippet)
```

## Backends

`provider="auto"` (the default) walks this order and returns the first backend
that produces results. Everything is optional and auto-detected; the output always
ends with which backends were used, unconfigured, or failed.

The chain also learns from this session's own trajectory: a backend that raises
earns a cooldown starting at 120s and doubling per consecutive failure (capped at
x8), and one success resets it. While cooling, it is skipped in `auto`, listed in
the output as `cooled: ...`, and explained by `websearch.health()` - evidence,
not guesswork. If the ready chain produces nothing at all, cooled backends get one
last try before the search fails. Naming providers (single or comma-separated
chain) or using `all` is an explicit request and is never refused;
`PRIME_AGENT_WEBSEARCH_COOLDOWN=0` turns this off.

| Backend | Credential | Notes |
|---|---|---|
| `gemini` | any `google-generative-ai` provider in `models.json` + key in `auth.json`, or `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `google` in `auth.json` | grounded answer, `[n]` citation markers, real publisher URLs |
| `tavily` | `TAVILY_API_KEY` or `tavily` in `auth.json` | answer + results, native date/domain filters |
| `brave` | `BRAVE_API_KEY` or `brave` in `auth.json` | result list, native freshness filter |
| `serper` | `SERPER_API_KEY` or `serper` in `auth.json` (`/login` -> MCP Connections) | knowledge graph + organic results |
| `exa` | `EXA_API_KEY` or `exa` in `auth.json` | neural search, native date/domain filters |
| `searxng` | `SEARXNG_URL` | self-hosted, free; the instance must allow `format=json` |
| `ddg` | none | always available fallback |

Gemini takes a key per endpoint from `auth.json`. If a
[pi-api-key-rotator](https://github.com/sehoon123/pi-api-key-rotator) style
`key-rotator.json` is present, its pool keys for that provider become failover
candidates, so a `401`/`429` on one key moves to the next key, then the next
endpoint, then the next backend.

Filters are mapped to each backend's native parameter where one exists
(`time_range`, `freshness`, `tbs`, `startPublishedDate`, `df`), expressed as
`site:` operators where it does not, and always re-checked client-side — removed
results are reported as a count.

## Environment overrides

- `PRIME_AGENT_WEBSEARCH_PROVIDER` - default backend (`auto`, a name, or `all`/a comma- or whitespace-separated concurrent fan-out)
- `PRIME_AGENT_WEBSEARCH_NUM_RESULTS` - default result count (default 5)
- `PRIME_AGENT_WEBSEARCH_TIMEOUT` - HTTP timeout in seconds (default 45)
- `PRIME_AGENT_WEBSEARCH_CACHE_TTL` - in-process cache TTL in seconds, `0` disables (default 300)
- `PRIME_AGENT_WEBSEARCH_COOLDOWN` - failure-cooldown base in seconds, `0` disables adaptive ordering (default 120)
- `PRIME_AGENT_WEBSEARCH_GEMINI_MODEL` - pin the grounding model
- `PRIME_AGENT_WEBSEARCH_KEY_ROTATOR` - path to a key-rotator config
- `SEARXNG_URL` - SearXNG instance base URL

## Notes

- If nothing is configured, `ddg` still answers, so the skill never hard-fails on
  a fresh install. DuckDuckGo's HTML endpoint rate-limits aggressively and reports
  it as `HTTP 202`; that is surfaced as a rate-limit message, not "no results".
- Identical repeated searches are served from a short-lived in-process cache to
  protect provider quota during agent loops.
- API keys and basic-auth passwords are redacted from all returned text and errors.
  A source URL containing a known bearer value is removed rather than leaked or
  corrupted. A key attached to
  a custom provider is never reused at a different origin.
- Provider response bodies, parsed entry counts, generated answers and citation work
  are bounded. Result URLs are safety/domain filtered before the output cap; an answer
  with no in-scope supporting URL is discarded.
- Grounding redirect links are resolved by reading `Location` without fetching the
  target. Candidate count, concurrency and aggregate time are bounded; unsafe or
  unresolved redirectors are discarded.
- `!command` credential references in `auth.json` are not executed from the
  kernel; export the value or use a literal for those providers.
