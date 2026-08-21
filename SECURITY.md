# Security

## Reporting

Open a private security advisory on the GitHub repository, or a normal issue for
non-sensitive hardening suggestions.

## Threat model

These skills run **inside Prime Agent's IPython kernel**, with the same privileges
as the agent. They perform outbound HTTP requests and read credentials from the
agent's configuration. They never run shell commands or execute provider content.
`webfetch` can save a non-text response to a private temporary file when that is the
requested result; Gemini Files API uploads are deleted after use.

## What is protected

**Credential exposure.** Keys are read from `auth.json`, `models.json`, an optional
`key-rotator.json`, or environment variables. Known API-key values are redacted from
provider-controlled answers, source fields and errors before output or caching.
A source URL containing a known bearer key or password is removed rather than leaked
or rewritten. This check is deliberately fail-closed even for unusually short literal
credentials, which can suppress an innocent matching URL; use real provider secrets,
not one-character placeholders, for live calls. Basic-auth usernames are not treated
as bearer secrets.
`backends()` reports a credential source, not its value, and strips basic auth from
a configured SearXNG URL. Custom-provider credentials are never repurposed for the
public Google origin. A Gemini resumable upload URL must have the same origin as the
configured endpoint; the finalize request does not forward the API key.

**`!command` references are not executed.** Prime Agent supports command-backed
credentials, but these skills deliberately skip them inside the kernel. Export the
value or use a literal credential instead.

**Local-network SSRF in `webfetch`.** Every local page and robots request:

- accepts only `http(s)`, rejects URL credentials and known internal names, and
  rejects every non-global address, including private, loopback, link-local, CGNAT,
  site-local, multicast, reserved and unspecified ranges;
- checks every DNS answer and, for the native HTTP transport, connects to a vetted
  address while preserving the original Host header and TLS SNI. This closes the
  DNS-rebinding gap between validation and connection;
- follows redirects manually, validates and pins every hop, and checks the
  destination origin's robots policy before requesting the redirected path;
- caps redirect depth and streamed body size, including decompressed data.

The same URL and robots checks run before video or Gemini URL-context retrieval, so
a refused target is not handed to a model. An explicitly injected resolver or HTTP
transport is a caller-controlled networking boundary and is not cached.

**Search-result and grounding redirects.** Provider result URLs must be public
`http(s)` URLs without credentials before they are shown. Only the exact
`vertexaisearch.cloud.google.com` hostname may receive a grounding `HEAD` request.
Redirect following is disabled; each `Location` is validated, the publisher is
never contacted by the resolver, candidates and concurrency are capped, and all
redirect resolution for one provider response has a five-second aggregate budget.

**Denial of service.** Requests and DNS resolution have bounded timeouts and
connection limits. Search queries, result counts, generated answer text, citation work and provider
response bodies are bounded. Webfetch also bounds redirect depth, transfer-decoded
page bytes, Gemini and Files responses, and rendered output; both caches are bounded.
CPU-heavy HTML/PDF and robots parsing/matching run in worker threads. A one-byte
streaming look-ahead detects truncation without retaining an arbitrarily large decoded
chunk; robots matching denies when its aggregate work budget is exhausted.

**Cache and files.** Successful values are copied into bounded, TTL-based process
memory caches; credential rotation is part of each cache boundary. Errors, mutable
`saved_path` documents, credentials and call-only failure state are not retained. Binary downloads use mode-`0600` temporary files,
atomically reuse identical safe files, and never follow or replace an existing path.
The returned path belongs to the caller and is not cached.

## What is not protected

- **Content trust.** Search results and fetched pages are untrusted text and may
  contain prompt injection. Treat them as data, not instructions.
- **Configured endpoints.** Custom Gemini and SearXNG base URLs are trusted operator
  configuration. They receive the requests and credentials configured for them.
- **Server-side redirects.** For Gemini URL-context/video retrieval, the submitted
  URL is safety- and robots-checked locally, but redirects followed inside the
  provider are opaque to this process and cannot receive a local destination-policy
  check. Use `gemini=False` when that distinction matters.
- **Provider answer attribution.** Domain filters are re-applied to result URLs, and
  an answer with no surviving in-scope support is dropped. Mixed provider prose cannot
  be attributed clause by clause.
- **Provider terms.** DuckDuckGo HTML is not an official API and may rate-limit it;
  use a keyed backend for production volume.
- **Transport trust.** Standard system TLS verification and proxy configuration are
  used. There is no certificate pinning.
