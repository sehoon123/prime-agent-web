# prime-agent-websearch

Multi-backend web search skill for [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent).

Prime Agent ships one built-in search skill, and it only talks to
[Serper](https://serper.dev), which needs a paid-after-2,500-queries key. This
package is a drop-in superset: it keeps the same `websearch` import name and call
style, and picks whichever backend the host can actually reach.

```python
print(await websearch("prime agent latest release"))
```

```text
# websearch: prime agent latest release

## gemini (ibm-ica-gemini/gemini-3.6-flash)

Prime Agent's latest release is v0.7.4 ...

### Sources
1. github.com
   https://github.com/PrimeIntellect-ai/prime-agent/releases
2. primeintellect.ai
   https://www.primeintellect.ai/blog/rlm

### Searches run
prime agent latest release; primeintellect prime-agent releases
---
backends used: gemini · not configured: tavily, brave, exa, searxng
```

## Design

Prime Agent has exactly one tool (`ipython`), so this is **not** a new agent tool.
It is a Python-backed skill: a real package installed into the kernel venv and
called from Python, per the
[skills contract](https://github.com/PrimeIntellect-ai/prime-agent/blob/main/packages/coding-agent/docs/skills.md).

- **Zero configuration.** Backends are discovered from files a Prime Agent install
  already has (`models.json`, `auth.json`, optionally `key-rotator.json`) and from
  standard environment variables. No new config file is introduced.
- **Never hard-fails.** With nothing configured at all, keyless DuckDuckGo still
  answers.
- **Answers first.** `auto` prefers backends that return a grounded answer with
  citations (Gemini, Tavily) over plain link lists.
- **Model-readable output.** Markdown text, not JSON, ending with which backends
  were used, unconfigured, or failed.
- **Secrets stay out of output.** Credentials are redacted from returned text and
  from error messages, including keys echoed back by a provider.

## Install

```bash
prime-agent package install git:github.com/sehoon123/prime-agent-websearch
```

Then restart Prime Agent (a fresh session installs the Python package into the
kernel venv) and verify:

```python
print(await websearch.backends())
```

Local checkout instead:

```bash
prime-agent package install /path/to/prime-agent-websearch
```

Uninstall:

```bash
prime-agent package remove git:github.com/sehoon123/prime-agent-websearch
```

### Replacing the built-in skill

A package skill overrides a built-in skill with the same name, so `websearch`
resolves to this one after install. The Python distribution deliberately uses the
same name as the bundled skill (`prime-agent-skill-websearch`) so the kernel can
never end up with two packages providing the `websearch` import.

If the built-in one was already installed in the kernel venv before, force a clean
state once:

```json
{
  "bundledSkills": { "websearch": false }
}
```

## Usage

```python
await websearch("gemini 3 pricing", num_results=8)      # 1-20 results
await websearch("who won euro 2024", provider="ddg")    # force one backend
await websearch("rust async runtimes", provider="all")  # every configured backend
await websearch("cve-2026-1234", provider="gemini,ddg") # explicit chain
await websearch.backends()                              # what is usable here
help(websearch)                                         # full signature
```

Shell cell:

```bash
!websearch "prime agent latest release" --num-results 8
```

Programmatic use, when rendered text is not wanted:

```python
results = await websearch.search("prime agent", provider="all")
for result in results:
    print(result.backend, result.answer)
    for item in result.items:
        print(item.title, item.url, item.snippet)
```

## Backends

`provider="auto"` (the default) walks this order and returns the first backend
that produces results.

| Backend | Credential source | Cost | Output |
|---|---|---|---|
| `gemini` | a `google-generative-ai` provider in `models.json` + its key in `auth.json`, or `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `google` in `auth.json` | free tier on AI Studio; free on a corporate gateway you already pay for | grounded answer + resolved citations + the queries Google ran |
| `tavily` | `TAVILY_API_KEY` or `tavily` in `auth.json` | free tier | answer + results |
| `brave` | `BRAVE_API_KEY` / `BRAVE_SEARCH_API_KEY` or `brave` in `auth.json` | free tier | results |
| `serper` | `SERPER_API_KEY` or `serper` in `auth.json` | 2,500 free, then prepaid | knowledge graph + results |
| `exa` | `EXA_API_KEY` or `exa` in `auth.json` | free tier | neural search results |
| `searxng` | `SEARXNG_URL` | free, self-hosted | results |
| `ddg` | none | free | results |

### Gemini backend

Any provider in `models.json` whose `api` is `google-generative-ai` becomes a
search backend — a corporate Gemini gateway works exactly like Google AI Studio,
because grounding is requested through the standard `tools: [{google_search: {}}]`
field (with an automatic fallback to the older `google_search_retrieval` name for
gateways that still expect it).

Grounding returns `vertexaisearch.cloud.google.com` redirect links. This skill
resolves them to publisher URLs concurrently, and keeps the redirect link if
resolution fails, so a source is never lost.

If a [pi-api-key-rotator](https://github.com/sehoon123/pi-api-key-rotator) style
`key-rotator.json` is present, the pool keys whose targets include that provider
are added as failover candidates: a `401`/`403`/`429` on one key moves to the next
key, then to the next endpoint, then to the next backend.

## Environment overrides

| Variable | Default | Purpose |
|---|---|---|
| `PRIME_AGENT_WEBSEARCH_PROVIDER` | `auto` | default backend, `all`, or a comma-separated chain |
| `PRIME_AGENT_WEBSEARCH_NUM_RESULTS` | `5` | default result count |
| `PRIME_AGENT_WEBSEARCH_TIMEOUT` | `45` | HTTP timeout in seconds |
| `PRIME_AGENT_WEBSEARCH_GEMINI_MODEL` | first `flash` model of the endpoint | pin the grounding model |
| `PRIME_AGENT_WEBSEARCH_KEY_ROTATOR` | `<agent dir>/key-rotator.json` | path to a key-rotator config |
| `SEARXNG_URL` | unset | SearXNG instance base URL |
| `PRIME_AGENT_CODING_AGENT_DIR` | `~/.prime/agent` | config directory to read (`~/.pi/agent` is a fallback) |

Credential values follow Prime Agent's own rules: an environment variable name
resolves to that variable, otherwise the value is used literally. `!command`
references are **not** executed from inside the kernel — export the value or use a
literal for those providers.

## Compatibility

- Prime Agent 0.7+ (Python-backed skill contract with `run()`).
- Python 3.10+; runtime dependencies `httpx` and `beautifulsoup4` are already in
  the Prime Agent kernel venv. DuckDuckGo parsing falls back to a regex parser if
  BeautifulSoup is missing, so the skill also works in a bare environment.
- No OS-specific code: paths go through `pathlib`, and no shell command is ever run.

## Development

```bash
git clone https://github.com/sehoon123/prime-agent-websearch
cd prime-agent-websearch
PYTHONPATH=skills/websearch/src python3 -m unittest discover -s tests -t .
```

46 offline tests cover backend discovery (including `models.json` and
`key-rotator.json` parsing), every backend's request shape and response parsing,
Gemini key failover and legacy-tool fallback, DuckDuckGo redirect unwrapping with
both parsers, rendering, and credential redaction. They use `httpx.MockTransport`,
so no network access or credentials are required.

Standalone smoke test outside the kernel:

```bash
PYTHONPATH=skills/websearch/src python3 -m websearch            # list backends
PYTHONPATH=skills/websearch/src python3 -m websearch "a query"  # real search
```

## License

MIT
