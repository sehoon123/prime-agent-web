---
name: websearch
description: Search the web and get an answer with real source URLs. Uses Gemini Google-Search grounding, Tavily, Brave, Serper, Exa, SearXNG, or keyless DuckDuckGo - whichever the host has configured. Takes one query and returns a grounded answer, numbered sources, and the searches that were run. Use for any question needing current information; fetch a page with httpx afterwards when full text is needed.
---

# Web Search

Multi-backend replacement for the Serper-only `websearch` skill bundled with
Prime Agent. Call the prepared import directly in the kernel:

```python
print(await websearch("prime agent latest release"))
```

Keyword arguments:

```python
await websearch("gemini 3 pricing", num_results=8)      # 1-20 results
await websearch("who won euro 2024", provider="ddg")    # force one backend
await websearch("rust async runtimes", provider="all")   # every backend, in sequence
await websearch("cve-2026-1234", provider="gemini,ddg") # a specific chain
await websearch.backends()                               # what is configured here
help(websearch)                                          # full signature
```

From a shell cell:

```bash
!websearch "prime agent latest release" --num-results 8
```

For programmatic use, `websearch.search(...)` returns `SearchResult` objects
(`backend`, `detail`, `answer`, `items[].{title,url,snippet}`, `queries`) instead
of rendered text.

## Backends

`provider="auto"` (the default) tries them in this order and returns the first
backend that produces results. Everything is optional and auto-detected; the
output always ends with which backends were used, unconfigured, or failed.

| Backend | Credential | Notes |
|---|---|---|
| `gemini` | any `google-generative-ai` provider in `models.json` + key in `auth.json`, or `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `google` in `auth.json` | grounded answer with real citations; resolves Google redirect links |
| `tavily` | `TAVILY_API_KEY` or `tavily` in `auth.json` | answer + results |
| `brave` | `BRAVE_API_KEY` or `brave` in `auth.json` | result list |
| `serper` | `SERPER_API_KEY` or `serper` in `auth.json` (`/login` -> MCP Connections) | knowledge graph + organic results |
| `exa` | `EXA_API_KEY` or `exa` in `auth.json` | neural search |
| `searxng` | `SEARXNG_URL` | self-hosted, free; the instance must allow `format=json` |
| `ddg` | none | always available fallback |

Gemini gets a key per endpoint from `auth.json`. If a
[pi-api-key-rotator](https://github.com/sehoon123/pi-api-key-rotator) style
`key-rotator.json` is present, its pool keys for that provider are added as
failover candidates, so a `401`/`429` on one key moves to the next instead of
failing the search.

## Environment overrides

- `PRIME_AGENT_WEBSEARCH_PROVIDER` - default backend (`auto`, `all`, a name, or a comma-separated chain)
- `PRIME_AGENT_WEBSEARCH_NUM_RESULTS` - default result count (default 5)
- `PRIME_AGENT_WEBSEARCH_TIMEOUT` - HTTP timeout in seconds (default 45)
- `PRIME_AGENT_WEBSEARCH_GEMINI_MODEL` - pin the grounding model (default: the endpoint's first `flash` model)
- `PRIME_AGENT_WEBSEARCH_KEY_ROTATOR` - path to a key-rotator config
- `SEARXNG_URL` - SearXNG instance base URL

## Notes

- If nothing is configured, `ddg` still answers, so the skill never hard-fails on
  a fresh install.
- API keys are redacted from all returned text and error messages.
- `!command` credential references in `auth.json` are not executed from the
  kernel; export the value or use a literal for those providers.
