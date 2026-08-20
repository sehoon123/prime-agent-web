---
name: webfetch
description: Fetch a URL and read it as markdown, keeping headings, code blocks and link targets. Handles HTML, PDF (per-page text), JSON and other text formats, rewrites GitHub blob links to raw file contents, and saves binaries to a temp file. Use whenever a specific page, API reference, source file or paper must actually be read; use websearch first when the URL is unknown.
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

## Arguments

```python
await webfetch(url)                          # markdown (default)
await webfetch(url, mode="text")             # plain text
await webfetch(url, mode="raw")              # body exactly as served (JSON, YAML, source)
await webfetch(url, max_chars=0)             # do not truncate the rendered output
await webfetch(url, max_pages=5)             # PDFs: first 5 pages only
await webfetch(url, max_bytes=40_000_000)    # allow a large PDF (default 10 MB)
await webfetch(url, respect_robots=False)    # the user explicitly asked for this page
help(webfetch)
```

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
- `localhost`, `*.local`, `*.internal`, cloud metadata names and literal private,
  loopback, link-local, multicast or reserved addresses are refused
- **DNS preflight**: a hostname whose records point into private space is refused
- redirects are followed manually, at most 5 hops, and **every hop is re-validated**
  before it is requested
- `content-length` over the cap is refused before the body is downloaded; unannounced
  bodies are cut off by a streaming guard
- `robots.txt` is honoured by default for autonomous fetches, following the
  convention of the official MCP fetch server, with `401`/`403` on robots.txt read
  as a refusal. Pass `respect_robots=False` when a human asked for the page.

## Environment overrides

- `PRIME_AGENT_WEBFETCH_MAX_CHARS` - rendered output cap (default 20000, `0` disables)
- `PRIME_AGENT_WEBFETCH_MAX_BYTES` - body size cap (default 10485760)
- `PRIME_AGENT_WEBFETCH_TIMEOUT` - HTTP timeout in seconds (default 45)
- `PRIME_AGENT_WEBFETCH_RESPECT_ROBOTS` - `0` to skip robots.txt checks

## Notes

- Errors never raise: `run()` returns a message and `fetch()` returns a Document
  with `kind="error"` and `error` set.
- A large PDF fails with the exact `max_bytes=` value to retry with, instead of a
  library parse error.
- For a whole repository, clone it (`git clone`) rather than fetching pages.
