---
name: webfetch
description: Fetch a URL and read it as markdown, keeping headings, code blocks and link targets. Handles HTML, PDF (per-page text), JSON and other text formats, rewrites GitHub blob links to raw file contents, and saves binaries to a temp file. With a prompt it answers questions about a page, reads YouTube videos, and transcribes scanned PDFs through Gemini. Use whenever a specific page, API reference, source file, paper or video must actually be read; use websearch first when the URL is unknown.
---

# Web Fetch

Fetch a URL as readable text in the kernel:

```python
print(await webfetch("https://docs.example.com/guide"))
```

`run()` renders a bounded view for reading. `fetch()` returns `Document` objects
with the **full** text, because slicing and storing belong in the kernel:

```python
doc = await webfetch.fetch("https://docs.example.com/guide")
doc.text[20_000:40_000]                  # slice it yourself, no second request
doc.kind, doc.title, doc.final_url, doc.content_type, doc.bytes_len, doc.pages
```

Several URLs are fetched concurrently, and one failure never loses the others:

```python
docs = await webfetch.fetch([url_a, url_b, url_c])
for doc in docs:
    print(doc.final_url, doc.ok, doc.error or len(doc.text))
```

Successful fetches live in a small session cache, so later retries and
reformulations can reuse the extracted Document. Hits come back with a
`from session cache` note, errors are never cached, and `webfetch.clear_cache()`
drops completed entries. Concurrent duplicate calls remain independent.

```python
webfetch.clear_cache()   # force the next fetch to hit the network again
```

## Arguments

```python
await webfetch(url)                          # markdown (default)
await webfetch(url, mode="text")             # plain text
await webfetch(url, mode="raw")              # decoded text without extraction; binaries go to a secure file
await webfetch(url, max_chars=0)             # do not truncate the rendered output
await webfetch(url, max_pages=5)             # PDFs: first 5 pages only
await webfetch(url, max_bytes=40_000_000)    # allow a large PDF (default 10 MB)
await webfetch(url, respect_robots=False)    # the user explicitly asked for this page
help(webfetch)
```

## Asking about a page, videos, scans

These need a Gemini endpoint (the same one `websearch` discovers). Without one,
everything above still works and these report why they cannot run.

```python
await webfetch(url, prompt="Which auth methods does this API support?")
await webfetch("https://youtu.be/abc123", prompt="What libraries are shown?")
await webfetch("https://example.com/scan.pdf")   # no text layer -> transcribed automatically
await webfetch(url, gemini=False)                # never leave the local path
await webfetch(url, gemini=True)                 # force the model path
```

A prompt routes through Gemini's `url_context` tool, which fetches server-side and
therefore also reads pages that need JavaScript or block scripted clients. An explicit
prompt or `gemini=True` returns a clear error if that model call fails; it never returns
or caches a local page dump that did not satisfy the request.

The model is used only where local extraction cannot work:

| Situation | Tier |
|---|---|
| YouTube link | `gemini-video` (there is no local path) |
| `prompt=` given, or `gemini=True` | `gemini-url-context` |
| local fetch blocked or failed | `gemini-url-context` fallback |
| PDF with no text layer | `gemini-pdf` transcription |
| everything else | local, no model call |

`doc.source` says which path produced the result, and `doc.answer` holds the model
text. `webfetch.gemini_available()` reports whether these tiers are usable.

Shell cell: `!webfetch https://example.com/page`

## What it does for you

| Input | Result |
|---|---|
| HTML | markdown with headings, code blocks and link targets kept; `script`, `style`, `nav`, `header`, `footer`, `aside`, `form`, `iframe` dropped; content taken from `<main>`/`<article>` when present |
| PDF | text per page with `--- page N ---` markers, page count, title from metadata; scanned PDFs are reported as having no text layer |
| JSON / YAML / plain text | returned as-is |
| Images and other binaries | written to a temp file, path reported (use the `attach-image` skill or PIL) |
| `github.com/o/r/blob/ref/path` | rewritten to `raw.githubusercontent.com` for exact file contents |
| `github.com/o/r` | fetched, plus a hint that `git clone` gives real file contents |

## Safety

Fetching URLs that came from search results or page content is an SSRF risk, so
every request is guarded:

- only `http(s)`; no credentials in the URL; `file:`, `data:`, `ftp:` refused
- `localhost`, `*.local`, `*.internal`, cloud metadata names and every non-global
  address (including CGNAT and IPv6 site-local ranges) are refused
- DNS records are checked and native connections are pinned to vetted addresses,
  closing the DNS-rebinding gap while retaining address failover and the logical URL's
  proxy routing; redirect cookies stay on their exact logical origin and are never
  reused merely because hosts share an IP
- redirects are followed manually, at most 5 hops, and **every hop is re-validated**
  before it is requested
- identity `content-length` over the cap is refused before download; compressed bodies
  are judged by bounded decoded size (including concatenated gzip members), unannounced
  bodies are cut off while streaming, partial binaries are errors, and Gemini/Files
  responses have their own cap
- `robots.txt` is honoured by default for autonomous fetches, following the
  convention of the official MCP fetch server, with `401`/`403` on robots.txt read
  as a refusal. Parsing is limited to a 1 MiB prefix; matching has a bounded work
  budget and denies conservatively if that budget is exhausted. Pass
  `respect_robots=False` when a human asked for the page. For
  Gemini's opaque server-side URL retrieval, this check covers the submitted URL;
  provider-internal redirects cannot be inspected locally.

## Environment overrides

- `PRIME_AGENT_WEBFETCH_MAX_CHARS` - rendered output cap (default 20000, `0` disables)
- `PRIME_AGENT_WEBFETCH_MAX_BYTES` - body size cap (default 10485760)
- `PRIME_AGENT_WEBFETCH_TIMEOUT` - HTTP timeout in seconds (default 45)
- `PRIME_AGENT_WEBFETCH_RESPECT_ROBOTS` - `0` to skip robots.txt checks
- `PRIME_AGENT_WEBFETCH_CACHE_TTL` - session document-cache TTL in seconds, `0` disables (default 300)

Gemini endpoints and keys come from the `websearch` skill's discovery
(`models.json`, `auth.json`, optional `key-rotator.json`, `GEMINI_API_KEY`), so
there is nothing extra to configure here. Pin the model with `model=` if needed.

## Notes

- `run()` returns a message for every error. `fetch()` turns each retrieval or
  extraction failure into a Document with `kind="error"`; invalid call-level
  arguments still raise `ValueError`.
- Documents backed by a mutable temporary file (`saved_path`) are not cached;
  refetching avoids returning a path the caller may have deleted or overwritten.
- A large PDF fails with the exact `max_bytes=` value to retry with, instead of a
  library parse error.
- A scanned PDF over the ~14 MB raw inline limit is uploaded through the Gemini
  Files API so base64 stays below the ~20 MB request cap, and is deleted afterwards. Endpoints that proxy only `generateContent`
  (common for corporate gateways) answer `404` there; that is detected and the error
  suggests `max_pages` or local extraction instead.
- For a whole repository, clone it (`git clone`) rather than fetching pages.
- A URL refused by the safety checks is never handed to the model either.
