# prime-agent-web

**Web access skills for [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent):
search and fetch.** Two Python-backed skills in one package:

| Skill | Call | What it does |
|---|---|---|
| `websearch` | `await websearch("query")` | Gemini Google-Search grounding, Tavily, Brave, Serper, Exa, self-hosted SearXNG, or keyless DuckDuckGo — auto-detected from the configuration your agent already has |
| `webfetch` | `await webfetch(url)` | reads a page as markdown (headings, code blocks and links kept), extracts PDF text per page, rewrites GitHub blob links to raw contents — through SSRF-guarded requests. With a prompt it answers questions about a page, reads YouTube videos, and transcribes scanned PDFs |

[![CI](https://github.com/sehoon123/prime-agent-web/actions/workflows/ci.yml/badge.svg)](https://github.com/sehoon123/prime-agent-web/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

Prime Agent's bundled search skill talks only to [Serper](https://serper.dev),
which needs a key that is free for 2,500 queries and prepaid after that. This
package is a drop-in superset: same `websearch` import, same call style, but it
uses whichever backend the host can actually reach — starting with Gemini
grounding through a Gemini provider you already pay for (or Google AI Studio's free
tier), and ending at keyless DuckDuckGo so it never hard-fails.

```python
print(await websearch("prime agent latest release"))
```

```text
# websearch: prime agent latest release

## gemini (ibm-ica-gemini/gemini-3.6-flash)

Prime Agent's latest release is v0.7.4.[1] It is built around the Recursive
Language Model and a Continual Harness.[1][2]

### Sources
1. github.com
   https://github.com/PrimeIntellect-ai/prime-agent/releases
2. primeintellect.ai
   https://www.primeintellect.ai/blog/prime-agent

### Searches run
prime agent latest release; primeintellect prime-agent releases
---
backends used: gemini · not configured: tavily, brave, exa, searxng
```

## Design

Prime Agent exposes exactly one tool (`ipython`), so this is **not** a new agent
tool. It is a Python-backed skill — a real package installed into the kernel venv
and called from Python — per the
[skills contract](https://github.com/PrimeIntellect-ai/prime-agent/blob/main/packages/coding-agent/docs/skills.md).

- **Zero configuration.** Backends are discovered from files a Prime Agent install
  already has (`models.json`, `auth.json`, optionally `key-rotator.json`) plus the
  usual environment variables. No new config file is introduced.
- **Never hard-fails.** With nothing configured at all, keyless DuckDuckGo answers.
- **Answers first.** `auto` prefers backends that return a grounded answer with
  citations over plain link lists.
- **Composition over parameters.** There is no `queries` argument: the kernel *is*
  the batching layer (`asyncio.gather`). There is no result store or
  `get_search_content` helper either — results are Python objects that stay in the
  kernel.
- **Model-readable output.** Markdown with `[n]` citation markers, ending with which
  backends were used, unconfigured, or failed.
- **Learns from this session's trajectory.** A backend that keeps failing sits out a
  doubling cooldown (`PRIME_AGENT_WEBSEARCH_COOLDOWN`, `0` disables); one success
  restores it. `await websearch.health()` shows the evidence behind the current
  ordering — Continual-Harness thinking applied at skill scale.
- **The session is the cache.** Successful fetches are reused inside the kernel
  session (TTL 300s, `PRIME_AGENT_WEBFETCH_CACHE_TTL`), so later retries can reuse
  the extracted Document — context as a variable, quota as a budget. Endpoint,
  model and credential rotations form cache boundaries. Concurrent duplicate calls
  remain independent.
- **Secrets stay out of output.** Credentials are redacted from returned text and
  error messages, including keys a provider echoes back in its own error body.

## Install

```bash
prime-agent package install git:github.com/sehoon123/prime-agent-web
```

That installs both skills.

Restart Prime Agent (a fresh session installs the Python package into the kernel
venv), then verify:

```python
print(await websearch.backends())
```

```text
# websearch backends

- ready  gemini - ibm-ica-gemini [models.json:ibm-ica-gemini + key-rotator (7 pool keys), 7 keys]
- off    tavily
           enable: set TAVILY_API_KEY, or store a tavily credential in auth.json
...
- ready  ddg - no credential required

auto order: gemini, tavily, brave, serper, exa, searxng, ddg
recency values: day, week, month, year
cache: 300s in-process
cooldown: 120s base, doubling per consecutive failure
```

Local checkout instead: `prime-agent package install /path/to/prime-agent-web`.
Remove with `prime-agent package remove git:github.com/sehoon123/prime-agent-web`.

### Replacing the built-in skill

A package skill overrides a built-in skill with the same name, so `websearch`
resolves here after install. The Python distribution deliberately reuses the
bundled skill's name (`prime-agent-skill-websearch`) so the kernel venv can never
hold two packages providing the `websearch` import. If the bundled one was already
installed, you can also pin it off:

```json
{
  "bundledSkills": { "websearch": false }
}
```

## Usage

```python
await websearch("gemini 3 pricing", num_results=8)               # 1-20 results
await websearch("cve-2026-1234", recency="week")                 # day|week|month|year
await websearch("rust async", domains="github.com,-reddit.com")  # "-" excludes
await websearch("who won euro 2024", provider="ddg")             # force one backend
await websearch("rust async runtimes", provider="all")           # all backends, concurrently
await websearch("kernel panic", provider="gemini,ddg")           # explicit concurrent fan-out
help(websearch)
```

Batching, because the kernel is the composition layer:

```python
import asyncio
answers = await asyncio.gather(*(websearch(q) for q in ["query a", "query b"]))
```

Raw objects when rendered text is not what you want:

```python
results = await websearch.search("prime agent", provider="all")
for result in results:
    print(result.backend, result.detail, result.answer, result.dropped)
    for item in result.items:
        print(item.title, item.url, item.snippet)
```

Shell cell: `!websearch "prime agent latest release" --num-results 8`

## Backends

`provider="auto"` (the default) walks this order and returns the first backend that
produces results.

| Backend | Credential source | Cost | Output |
|---|---|---|---|
| `gemini` | a `google-generative-ai` provider in `models.json` + its key in `auth.json`, or `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `google` in `auth.json` | free tier on AI Studio; free on a gateway you already pay for | grounded answer, `[n]` citations, resolved publisher URLs, the queries Google ran |
| `tavily` | `TAVILY_API_KEY` or `tavily` in `auth.json` | free tier | answer + results |
| `brave` | `BRAVE_API_KEY` / `BRAVE_SEARCH_API_KEY` or `brave` in `auth.json` | free tier | results |
| `serper` | `SERPER_API_KEY` or `serper` in `auth.json` | 2,500 free, then prepaid | knowledge graph + results |
| `exa` | `EXA_API_KEY` or `exa` in `auth.json` | free tier | neural search results |
| `searxng` | `SEARXNG_URL` | free, self-hosted | results |
| `ddg` | none | free | results |

### Filters

`recency` and `domains` are translated to each backend's native parameter where one
exists and expressed as query operators where it does not. Result URLs are always
re-checked client-side. A provider answer is discarded when no in-scope supporting
URL survives; mixed provider prose cannot be attributed clause by clause. Removed
results are reported as a count.

| Backend | Recency | Domains |
|---|---|---|
| `gemini` | prompt hint | `site:` / `-site:` operators |
| `tavily` | `time_range` | `include_domains` / `exclude_domains` |
| `brave` | `freshness` (`pd`/`pw`/`pm`/`py`) | `site:` operators, over-fetch + filter |
| `serper` | `tbs=qdr:d|w|m|y` | `site:` operators |
| `exa` | `startPublishedDate` | `includeDomains` / `excludeDomains` |
| `searxng` | `time_range` | `site:` operators |
| `ddg` | `df=d|w|m|y` | client-side filter |

### Gemini backend

Any provider in `models.json` whose `api` is `google-generative-ai` becomes a
search backend — a corporate Gemini gateway works exactly like Google AI Studio,
because grounding is requested through the standard `tools: [{google_search: {}}]`
field, with an automatic fallback to the older `google_search_retrieval` name for
gateways that still expect it. Answers carry `[n]` markers derived from
`groundingSupports`, so individual claims map to sources.

If a [pi-api-key-rotator](https://github.com/sehoon123/pi-api-key-rotator) style
`key-rotator.json` is present, the pool keys whose targets include that provider
become failover candidates: `401`/`403`/`429` moves to the next key, then the next
endpoint, then the next backend.

## webfetch

```python
print(await webfetch("https://docs.example.com/guide"))     # markdown
doc = await webfetch.fetch("https://arxiv.org/pdf/2605.09998", max_bytes=40_000_000)
doc.text[20_000:40_000]                                     # slice in the kernel, no refetch
docs = await webfetch.fetch([url_a, url_b])                 # concurrent
```

| Input | Result |
|---|---|
| HTML | markdown with headings, code blocks and link targets kept; `script`/`nav`/`header`/`footer`/`aside` dropped; content taken from `<main>`/`<article>` when present |
| PDF | per-page text with `--- page N ---` markers, page count, metadata title; scanned PDFs reported as having no text layer |
| JSON / YAML / text | decoded text (`mode="raw"` skips tidying) |
| images, archives | saved to a temp file, path reported |
| `github.com/o/r/blob/…` | rewritten to `raw.githubusercontent.com` |
| `github.com/o/r` | hint that `git clone` beats scraping HTML |

Arguments: `prompt`, `mode` (`markdown`/`text`/`raw`), `max_chars` (rendered cap,
default 20000), `max_pages`, `max_bytes` (default 10 MB), `respect_robots`,
`gemini`, `model`, `timeout`.

### Model tiers (optional)

Local extraction handles everything it can; Gemini is used only where it cannot.
The endpoints are the ones `websearch` already discovered, so there is nothing extra
to configure, and with no Gemini configured every local capability still works.

```python
await webfetch(url, prompt="Which auth methods does this API support?")
await webfetch("https://youtu.be/abc123", prompt="What libraries are shown?")
await webfetch("https://example.com/scan.pdf")   # no text layer -> transcribed
await webfetch(url, gemini=False)                # stay local, always
```

| Situation | Tier | Why |
|---|---|---|
| YouTube link | `gemini-video` | no local path exists |
| `prompt=`, or `gemini=True` | `gemini-url-context` | server-side fetch also reads JS-only and bot-blocked pages |
| local fetch blocked or failed | `gemini-url-context` | recovers content instead of returning an error |
| PDF with no text layer | `gemini-pdf` | a scan has nothing for pypdf to read |
| scan over ~14 MB raw | `gemini-pdf` via Files API | base64 would exceed the ~20 MB request limit |
| anything else | local | no model call, no token cost |

`doc.source` reports which tier answered; `doc.answer` holds the model text;
`webfetch.gemini_available()` says whether the tiers are usable. A URL refused by the
safety checks is never handed to the model either.

Documents over the ~14 MB raw inline limit are uploaded with the Gemini
[Files API](https://ai.google.dev/gemini-api/docs/files) (resumable protocol, state
polled until `ACTIVE`, file deleted afterwards). Google AI Studio exposes it; many
corporate gateways proxy only `generateContent` and answer `404`, which is detected
per endpoint and reported with a suggestion to use `max_pages` or local extraction.

**A readability-style extractor was measured and rejected.** On an API reference page
it returned 1.7 KB with zero headings and zero links, against 25 KB with 62 code
blocks from the conservative pipeline used here. Aggressive main-content extraction
is tuned for news articles and destroys developer documentation.

## Environment overrides

| Variable | Default | Purpose |
|---|---|---|
| `PRIME_AGENT_WEBSEARCH_PROVIDER` | `auto` | default backend; `all` or a comma/space list runs a concurrent fan-out |
| `PRIME_AGENT_WEBSEARCH_NUM_RESULTS` | `5` | default result count |
| `PRIME_AGENT_WEBSEARCH_TIMEOUT` | `45` | HTTP timeout in seconds |
| `PRIME_AGENT_WEBSEARCH_CACHE_TTL` | `300` | in-process cache TTL; `0` disables |
| `PRIME_AGENT_WEBSEARCH_COOLDOWN` | `120` | failure cooldown base in seconds; `0` disables |
| `PRIME_AGENT_WEBSEARCH_GEMINI_MODEL` | first `flash` model of the endpoint | pin the grounding model |
| `PRIME_AGENT_WEBSEARCH_KEY_ROTATOR` | `<agent dir>/key-rotator.json` | path to a key-rotator config |
| `SEARXNG_URL` / `PRIME_AGENT_WEBSEARCH_SEARXNG_URL` | unset | SearXNG instance base URL |
| `PRIME_AGENT_CODING_AGENT_DIR` / `PI_CODING_AGENT_DIR` | `~/.prime/agent` | authoritative config directory override; without one, `~/.pi/agent` is a fallback |
| `PRIME_AGENT_WEBFETCH_MAX_CHARS` | `20000` | webfetch rendered output cap; `0` disables |
| `PRIME_AGENT_WEBFETCH_MAX_BYTES` | `10485760` | webfetch body size cap |
| `PRIME_AGENT_WEBFETCH_TIMEOUT` | `45` | webfetch HTTP timeout in seconds |
| `PRIME_AGENT_WEBFETCH_RESPECT_ROBOTS` | `1` | `0` skips robots.txt checks |
| `PRIME_AGENT_WEBFETCH_CACHE_TTL` | `300` | session document-cache TTL; `0` disables |

Credential values follow Prime Agent's own rules: an environment variable name
resolves to that variable, otherwise the value is used literally. `!command`
references are **not** executed from inside the kernel — export the value or use a
literal for those providers.

## Security

See [SECURITY.md](SECURITY.md). Summary:

- Grounding redirect links are resolved by reading `Location` with redirect
  following **disabled**, so the target host is never contacted, and every hop is
  validated — loopback, private, link-local, multicast, reserved, metadata
  hostnames, non-`http(s)` schemes, encoded/alternate numeric authorities, Unicode
  controls, and URLs carrying credentials are refused. Candidate count, concurrency
  and aggregate redirect time are bounded; unresolved redirectors are dropped.
- Credentials are redacted from every returned string; source URLs containing a
  known bearer key or password are removed, and `backends()` shows only the
  credential's source.
- Search results are untrusted third-party text; treat snippets as data, not
  instructions.
- `webfetch` guards every request: http(s) only, no credentials or control bytes in
  URLs, non-global and metadata targets refused, DNS answers checked, and native
  connections pinned to vetted addresses with logical proxy routing and exact-origin
  redirect cookies. Redirects are followed manually and re-validated at every hop.
  Transfer decoding (including concatenated gzip), page bodies and Gemini/Files
  responses are bounded. `robots.txt` is honoured by default for autonomous fetches;
  its bounded parser and matcher run off the event loop.

## Compatibility

- Prime Agent 0.7+ (Python-backed skill contract with `run()`).
- Python 3.10+. Runtime dependencies `httpx` and `beautifulsoup4` ship in the Prime
  Agent kernel venv; DuckDuckGo parsing falls back to a regex parser when
  BeautifulSoup is unavailable, so the skill also works in a bare environment.
- No shell command is run. Paths use `pathlib`; secure temporary-file reuse falls
  back to a private random file where no-follow/hard-link primitives are unavailable.

## Development

```bash
git clone https://github.com/sehoon123/prime-agent-web
cd prime-agent-web
PYTHONPATH=skills/websearch/src:skills/webfetch/src python3 -m unittest discover -s tests -t .
```

The offline test suite needs no network or credentials (`httpx.MockTransport`),
covering backend discovery (`models.json`, `key-rotator.json`, credential
precedence and source reporting), every backend's request shape and response
parsing, filter mapping per backend, Gemini key/endpoint failover and legacy-tool
fallback, citation markers, redirect resolution including SSRF rejection,
DuckDuckGo rate-limit handling and both parsers, cache behaviour, rendering,
redaction, URL validation and DNS preflight, redirect and size guards, robots.txt
verdicts, HTML/PDF/binary extraction, Gemini tier routing and failover, the resumable
upload protocol including gateway `404` detection, and the Prime Agent skill contract
itself for
every skill in the package (including a reproduction of the kernel's module wrapper
and a guard against helpers shadowing submodules).

Standalone smoke test outside the kernel:

```bash
PYTHONPATH=skills/websearch/src python3 -m websearch            # list backends
PYTHONPATH=skills/websearch/src python3 -m websearch "a query"  # real search
PYTHONPATH=skills/webfetch/src  python3 -m webfetch https://example.com
```

## Credits

Design informed by two references:

- [pi-web-access](https://github.com/nicobailon/pi-web-access) by Nico Bailon — the
  equivalent capability for the pi agent. Source of the provider-precedence,
  filter-mapping, redirect-resolution, `content-length` pre-check and explicit
  size-limit error patterns adapted here. That extension is a tool-based design for
  a many-tool host; this package is a skill-based design for a one-tool host.
- The official [MCP fetch server](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch)
  — source of the robots.txt convention (autonomous vs user-specified user agents,
  `401`/`403` treated as a refusal) and the markdownify-based extraction approach.

## License

MIT
