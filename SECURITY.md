# Security

## Reporting

Open a private security advisory on the GitHub repository, or a normal issue for
non-sensitive hardening suggestions.

## Threat model

This skill runs **inside Prime Agent's IPython kernel**, with the same privileges
as the agent. It performs outbound HTTP requests to search providers and reads
credentials from the agent's own configuration. It never writes files, never runs
shell commands, and never executes content returned by a provider.

## What is protected

**Credential exposure.** Keys are read from `auth.json`, `models.json`, an
optional `key-rotator.json`, or environment variables, and are only ever sent to
the provider's own endpoint. Every string returned to the model — answers, source
lists, and error messages — is passed through a redactor, because providers do
sometimes echo a key back inside an error body. `backends()` reports only where a
credential came from (`$BRAVE_API_KEY`, `auth.json:tavily`), never its value.

**`!command` references are not executed.** Prime Agent itself supports
`"key": "!some-shell-command"` in `auth.json`. This skill deliberately does not
run those commands from inside the kernel; such credentials are skipped, and the
affected backend reports as unconfigured.

**SSRF via redirect following.** Gemini grounding returns
`vertexaisearch.cloud.google.com` redirect links. Resolving them naively would let
a provider response steer an authenticated client at an arbitrary address. Instead
the resolver:

- sends `HEAD` with redirect following **disabled** and reads `Location`, so the
  final target is never contacted;
- validates every hop with `is_public_http_url()`, rejecting non-`http(s)`
  schemes, URLs containing credentials, `localhost`, `*.local`, `*.internal`,
  cloud metadata hostnames, and any literal loopback, private, link-local,
  multicast, reserved, or unspecified IP address (IPv4 and IPv6);
- caps redirect chains at 5 hops and 10 seconds;
- keeps the original redirect link when resolution fails, rather than emitting an
  unvalidated URL.

**Denial of service.** Every request has a timeout (default 45s), connection
limits, a result cap (max 20), and a query length cap (2000 chars). One backend
raising an unexpected exception cannot break a call or the kernel — it is recorded
as a failure and the next backend runs.

**Cache.** Results are cached in process memory only, keyed by query and filters,
with a default 300s TTL and a 64-entry bound. Nothing is written to disk. Set
`PRIME_AGENT_WEBSEARCH_CACHE_TTL=0` to disable.

## What is not protected

- **Content trust.** Search results are untrusted third-party text and may contain
  prompt-injection attempts. Treat retrieved snippets as data, not instructions.
- **Provider terms.** The DuckDuckGo backend scrapes the public HTML endpoint. It
  is rate-limited by DuckDuckGo and is not an official API; use a keyed backend for
  production volume.
- **Transport.** Standard system TLS verification is used. No certificate pinning.
