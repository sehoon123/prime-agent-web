"""Backend discovery and request shaping for the websearch skill.

Everything is derived from files and environment variables that a Prime Agent
install already has. This module never writes anything and never returns a
secret in an error message.
"""

from __future__ import annotations

import html
import ipaddress
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import cached_property
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

# Single source of truth lives in _health.py; this alias keeps the settings
# layer from drifting away from what health() reports.
from ._health import BASE_COOLDOWN

GOOGLE_API = "google-generative-ai"
AI_STUDIO_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Fallback models for the public Google AI Studio endpoint, tried in order when
# the live model listing is unavailable and no model was pinned.
AI_STUDIO_FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)

DEFAULT_NUM_RESULTS = 5
DEFAULT_TIMEOUT = 45.0
DEFAULT_CACHE_TTL = 300.0

# Cooldown base for a backend that just failed (see _health.py): each
# consecutive failure doubles it. PRIME_AGENT_WEBSEARCH_COOLDOWN overrides; 0
# disables adaptive ordering entirely.
DEFAULT_COOLDOWN = BASE_COOLDOWN
MAX_NUM_RESULTS = 20
MAX_QUERY_CHARS = 2000

RECENCY_VALUES: tuple[str, ...] = ("day", "week", "month", "year")
RECENCY_DAYS = {"day": 1, "week": 7, "month": 31, "year": 365}

# Backend order used by provider="auto". Answer-producing backends come first.
AUTO_ORDER: tuple[str, ...] = ("gemini", "tavily", "brave", "serper", "exa", "searxng", "ddg")

# How to enable each backend, shown by backends() when one is unavailable.
ENABLE_HINTS: dict[str, str] = {
    "gemini": (
        "add a google-generative-ai provider to models.json with its key in auth.json, "
        "or set GEMINI_API_KEY / GOOGLE_API_KEY"
    ),
    "serper": "store a serper credential via /login -> MCP Connections, or set SERPER_API_KEY",
    "tavily": "set TAVILY_API_KEY, or store a tavily credential in auth.json",
    "brave": "set BRAVE_API_KEY, or store a brave credential in auth.json",
    "exa": "set EXA_API_KEY, or store an exa credential in auth.json",
    "searxng": "set SEARXNG_URL to a SearXNG instance that allows format=json",
    "ddg": "always available (no credential required)",
}

# Backends that support a native recency filter. Others get a query hint instead.
NATIVE_RECENCY: frozenset[str] = frozenset({"tavily", "brave", "serper", "exa", "searxng", "ddg"})
# Backends that support native domain include/exclude fields.
NATIVE_DOMAINS: frozenset[str] = frozenset({"tavily", "exa"})


def env_str(name: str) -> Optional[str]:
    value = os.environ.get(name, "").strip()
    return value or None


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ[name])
    except (KeyError, ValueError):
        return default
    return value if math.isfinite(value) else default


def agent_dirs() -> tuple[Path, ...]:
    """Return one coherent Prime Agent configuration root.

    An explicit override is authoritative. Without one, the current Prime root
    wins when it contains an agent configuration file; the legacy Pi root is
    used only when Prime has no such file. Empty files remain authoritative.
    Mixing files across roots can pair one installation's endpoint with another
    installation's credential.
    """
    for name in ("PRIME_AGENT_CODING_AGENT_DIR", "PI_CODING_AGENT_DIR"):
        raw = env_str(name)
        if raw:
            return (Path(raw).expanduser(),)

    home = Path.home()
    prime = home / ".prime" / "agent"
    legacy = home / ".pi" / "agent"
    config_names = ("auth.json", "models.json", "key-rotator.json")
    if any((prime / name).is_file() for name in config_names):
        return (prime,)
    if any((legacy / name).is_file() for name in config_names):
        return (legacy,)
    # With no authoritative file in either location, prefer the current root.
    return (prime,)


def read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if text.startswith("\ufeff"):
        text = text[1:]
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def read_first_json(filename: str) -> dict[str, Any]:
    """Read `filename` from the selected agent directory."""
    for directory in agent_dirs():
        path = directory / filename
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if text.startswith("\ufeff"):
            text = text[1:]
        try:
            data = json.loads(text)
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def resolve_secret(raw: Any) -> Optional[str]:
    """Resolve a credential the way Prime Agent does: env var name, else literal.

    `!command` references are intentionally not executed from inside the kernel;
    export the value or use a literal instead.
    """
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value or value.startswith("!"):
        return None
    return (os.environ.get(value) or value).strip() or None


def auth_credential(auth: dict[str, Any], credential_id: str) -> Optional[str]:
    entry = auth.get(credential_id)
    if not isinstance(entry, dict) or entry.get("type") != "api_key":
        return None
    return resolve_secret(entry.get("key"))


@dataclass(frozen=True)
class Credential:
    """A resolved credential plus where it came from (never the value itself)."""

    value: str
    source: str


def find_credential(
    auth: dict[str, Any],
    credential_ids: Sequence[str],
    env_names: Sequence[str],
) -> Optional[Credential]:
    """First credential found. Environment variables win over stored ones."""
    for name in env_names:
        value = env_str(name)
        if value:
            return Credential(value, f"${name}")
    for credential_id in credential_ids:
        value = auth_credential(auth, credential_id)
        if value:
            return Credential(value, f"auth.json:{credential_id}")
    return None


def credential(
    auth: dict[str, Any],
    credential_ids: Sequence[str],
    env_names: Sequence[str],
) -> Optional[str]:
    found = find_credential(auth, credential_ids, env_names)
    return found.value if found else None


def _dedupe(values: Iterable[Optional[str]]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


# --------------------------------------------------------------------------- #
# URL safety
# --------------------------------------------------------------------------- #

_PRIVATE_HOST_SUFFIXES = (".local", ".internal", ".localdomain", ".home.arpa")
_BLOCKED_HOSTS = frozenset({"localhost", "localhost.localdomain", "metadata", "metadata.google.internal"})


def safe_endpoint_label(url: str) -> str:
    """Describe a configured URL without exposing embedded basic-auth data."""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return "configured endpoint"
    host = parts.hostname or ""
    if not host:
        return "configured endpoint"
    display_host = f"[{host}]" if ":" in host else host
    return f"{parts.scheme}://{display_host}" + (f":{port}" if port is not None else "")


MAX_RESULT_URL_CHARS = 8192


def _looks_like_noncanonical_ipv4(host: str) -> bool:
    labels = host.split(".")
    if not 1 <= len(labels) <= 4 or any(not label for label in labels):
        return False
    return all(
        label.isdigit()
        or (
            label.lower().startswith("0x")
            and len(label) > 2
            and all(character in "0123456789abcdef" for character in label[2:].lower())
        )
        for label in labels
    )


def is_public_http_url(url: str) -> bool:
    """True when `url` is an http(s) URL that does not target a private host.

    Applied to URLs discovered in provider responses (notably redirect targets)
    before they are shown or followed, so a redirect can never point the agent at
    loopback, link-local metadata services, or private ranges.
    """
    from urllib.parse import urlsplit

    if not isinstance(url, str) or len(url) > MAX_RESULT_URL_CHARS:
        return False
    decoded_url = html.unescape(url)
    url = decoded_url
    if any(
        character.isspace()
        or character == "\\"
        or unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
        for character in decoded_url
    ):
        return False
    try:
        parts = urlsplit(url.strip())
        _ = parts.port
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    if "%" in (parts.netloc or ""):
        return False
    if "@" in (parts.netloc or ""):  # credentials in the authority
        return False

    host = parts.hostname.lower().rstrip(".")
    if host in _BLOCKED_HOSTS or host.endswith(_PRIVATE_HOST_SUFFIXES):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return (
            "." in host and not _looks_like_noncanonical_ipv4(host)
        )  # a real registered name, not a bare/internal or alternate numeric form
    return bool(
        getattr(address, "scope_id", None) is None
        and address.is_global
        and not address.is_reserved
        and not address.is_multicast
        and not getattr(address, "is_site_local", False)
    )


# --------------------------------------------------------------------------- #
# Query shaping
# --------------------------------------------------------------------------- #


def normalize_domain(raw: str) -> Optional[str]:
    """Turn user input into a bare hostname: '-https://Example.com/docs' -> 'example.com'."""
    from urllib.parse import urlsplit

    value = (raw or "").strip().lstrip("-").strip()
    if not value:
        return None
    if "://" in value:
        value = urlsplit(value).hostname or ""
    value = value.split("/")[0].split("@")[-1].strip().strip(".").lower()
    if not value or " " in value:
        return None
    return value if re.fullmatch(r"[a-z0-9.-]+", value) else None


def parse_domains(domains: Union[str, Sequence[str], None]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a domain filter into (include, exclude). A leading '-' excludes."""
    if domains is None:
        return (), ()
    raw_list = [part for part in re.split(r"[,\s]+", domains) if part] if isinstance(domains, str) else list(domains)
    include: list[str] = []
    exclude: list[str] = []
    for raw in raw_list:
        domain = normalize_domain(raw)
        if not domain:
            continue
        target = exclude if str(raw).strip().startswith("-") else include
        if domain not in target:
            target.append(domain)
    return tuple(include), tuple(exclude)


def parse_recency(recency: Optional[str]) -> Optional[str]:
    if recency is None:
        return None
    value = recency.strip().lower()
    if not value:
        return None
    aliases = {"d": "day", "w": "week", "m": "month", "y": "year"}
    value = aliases.get(value, value)
    if value not in RECENCY_VALUES:
        raise ValueError(f"recency must be one of {', '.join(RECENCY_VALUES)} (got {recency!r})")
    return value


def host_matches(hostname: str, domain: str) -> bool:
    host = (hostname or "").lower().rstrip(".")
    return host == domain or host.endswith(f".{domain}")


def recency_start_date(recency: str, now: Optional[datetime] = None) -> str:
    """ISO-8601 start date for backends that only accept an absolute date."""
    moment = now or datetime.now(timezone.utc)
    return (moment - timedelta(days=RECENCY_DAYS[recency])).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class SearchQuery:
    """One normalized search request, shared by every backend."""

    text: str
    num_results: int = DEFAULT_NUM_RESULTS
    recency: Optional[str] = None
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()

    @property
    def cache_key(self) -> tuple[Any, ...]:
        return (self.text, self.num_results, self.recency, self.include_domains, self.exclude_domains)

    def operator_text(self, *, with_recency_hint: bool = False) -> str:
        """Query string with `site:` operators for backends without native fields."""
        parts = [self.text]
        if self.include_domains:
            if len(self.include_domains) == 1:
                parts.append(f"site:{self.include_domains[0]}")
            else:
                parts.append("(" + " OR ".join(f"site:{domain}" for domain in self.include_domains) + ")")
        parts.extend(f"-site:{domain}" for domain in self.exclude_domains)
        if with_recency_hint and self.recency:
            parts.append(f"(published within the last {self.recency})")
        return " ".join(parts)

    def allows(self, url: str) -> bool:
        """Client-side domain filter, applied to every backend as a safety net."""
        if not self.include_domains and not self.exclude_domains:
            return True
        from urllib.parse import urlsplit

        try:
            hostname = urlsplit(url).hostname or ""
        except ValueError:
            return False
        if self.exclude_domains and any(host_matches(hostname, domain) for domain in self.exclude_domains):
            return False
        if self.include_domains and not any(host_matches(hostname, domain) for domain in self.include_domains):
            return False
        return True


# --------------------------------------------------------------------------- #
# Gemini endpoints
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeminiEndpoint:
    """One Gemini-compatible endpoint plus every key that may be tried on it."""

    label: str
    base_url: str
    models: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()
    source: str = ""

    def pick_model(self, pinned: Optional[str]) -> Optional[str]:
        if pinned:
            return pinned
        for model in self.models:
            if "flash" in model:
                return model
        return self.models[0] if self.models else None

    def with_models(self, models: Sequence[str]) -> "GeminiEndpoint":
        return GeminiEndpoint(self.label, self.base_url, tuple(models), self.keys, self.source)


@dataclass
class RotatorKeys:
    """Extra keys per provider id, collected from an optional key-rotator config."""

    by_provider: dict[str, list[str]] = field(default_factory=dict)

    def get(self, provider_id: str) -> tuple[str, ...]:
        return tuple(self.by_provider.get(provider_id, ()))


def _pool_keys(pool: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for entry in pool.get("keys") or []:
        if not isinstance(entry, dict):
            continue
        literal = entry.get("value")
        if isinstance(literal, str) and literal.strip():
            keys.append(literal.strip())
            continue
        env_name = entry.get("env")
        if isinstance(env_name, str):
            value = env_str(env_name)
            if value:
                keys.append(value)
    return keys


def _pool_providers(pool: dict[str, Any]) -> list[str]:
    providers: list[str] = []
    targets = pool.get("targets")
    if isinstance(targets, list):
        for target in targets:
            if isinstance(target, dict) and isinstance(target.get("provider"), str):
                providers.append(target["provider"])
    legacy = pool.get("provider")
    if isinstance(legacy, str):
        providers.append(legacy)
    return providers


def load_rotator_keys() -> RotatorKeys:
    """Collect keys from a pi-api-key-rotator style config, when present.

    Supports both the multi-pool (`{"pools": [...]}`) and single-pool shapes.
    Absent or unreadable configs simply yield no extra keys.
    """
    override = env_str("PRIME_AGENT_WEBSEARCH_KEY_ROTATOR")
    if override:
        data = read_json(Path(override).expanduser())
    else:
        data = read_first_json("key-rotator.json")
    if not data:
        return RotatorKeys()

    pools = data.get("pools")
    pool_list = [pool for pool in pools if isinstance(pool, dict)] if isinstance(pools, list) else [data]

    result = RotatorKeys()
    for pool in pool_list:
        keys = _pool_keys(pool)
        if not keys:
            continue
        for provider_id in _pool_providers(pool):
            bucket = result.by_provider.setdefault(provider_id, [])
            for key in keys:
                if key not in bucket:
                    bucket.append(key)
    return result


def _provider_uses_google_api(provider: dict[str, Any]) -> bool:
    if provider.get("api") == GOOGLE_API:
        return True
    models = provider.get("models")
    if isinstance(models, list):
        return any(isinstance(model, dict) and model.get("api") == GOOGLE_API for model in models)
    return False


def _provider_model_ids(provider: dict[str, Any]) -> tuple[str, ...]:
    models = provider.get("models")
    if not isinstance(models, list):
        return ()
    ids: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        model_api = model.get("api") or provider.get("api")
        if model_api != GOOGLE_API:
            continue
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id.strip():
            ids.append(model_id.strip())
    return tuple(ids)


def gemini_endpoints(
    models_json: Optional[dict[str, Any]] = None,
    auth: Optional[dict[str, Any]] = None,
    rotator: Optional[RotatorKeys] = None,
) -> tuple[GeminiEndpoint, ...]:
    """Every Gemini-compatible endpoint this host can reach.

    Custom providers declared in models.json come first (they are usually a
    corporate gateway with its own quota), then the public Google AI Studio
    endpoint if a generic key is available.
    """
    models_json = read_first_json("models.json") if models_json is None else models_json
    auth = read_first_json("auth.json") if auth is None else auth
    rotator = load_rotator_keys() if rotator is None else rotator

    endpoints: list[GeminiEndpoint] = []
    claimed_generic_ids: set[str] = set()
    providers = models_json.get("providers")
    if isinstance(providers, dict):
        for provider_id, provider in providers.items():
            if not isinstance(provider, dict) or not _provider_uses_google_api(provider):
                continue
            base_url = provider.get("baseUrl")
            if not isinstance(base_url, str) or not base_url.strip():
                continue
            if provider_id in ("google", "gemini"):
                claimed_generic_ids.add(provider_id)
            pool_keys = rotator.get(provider_id)
            keys = _dedupe(
                [
                    auth_credential(auth, provider_id),
                    resolve_secret(provider.get("apiKey")),
                    *pool_keys,
                ]
            )
            if not keys:
                continue
            source = f"models.json:{provider_id}"
            if pool_keys:
                source += f" + key-rotator ({len(pool_keys)} pool keys)"
            endpoints.append(
                GeminiEndpoint(
                    label=provider_id,
                    base_url=base_url.strip().rstrip("/"),
                    models=_provider_model_ids(provider),
                    keys=keys,
                    source=source,
                )
            )

    studio = find_credential(auth, (), ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    if studio is None:
        studio = find_credential(
            auth,
            tuple(
                provider_id
                for provider_id in ("google", "gemini")
                if provider_id not in claimed_generic_ids
            ),
            (),
        )
    if studio:
        endpoints.append(
            GeminiEndpoint(
                label="google-ai-studio",
                base_url=AI_STUDIO_BASE_URL,
                models=(),
                keys=(studio.value,),
                source=studio.source,
            )
        )
    return tuple(endpoints)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

_SIMPLE_CREDENTIALS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "serper": (("serper",), ("SERPER_API_KEY",)),
    "tavily": (("tavily",), ("TAVILY_API_KEY",)),
    "brave": (("brave", "brave-search"), ("BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY")),
    "exa": (("exa",), ("EXA_API_KEY",)),
}


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings for one search call."""

    num_results: int
    timeout: float
    order: tuple[str, ...]
    gemini_model: Optional[str]
    searxng_url: Optional[str]
    cache_ttl: float
    auth: dict[str, Any]
    cooldown_base: float = DEFAULT_COOLDOWN

    @cached_property
    def secrets(self) -> tuple[str, ...]:
        """Every secret that must never appear in returned text."""
        values: list[str] = []
        for endpoint in self.gemini_endpoints:
            values.extend(endpoint.keys)
            from urllib.parse import unquote, urlsplit

            try:
                parts = urlsplit(endpoint.base_url)
                if parts.password:
                    values.extend((parts.password, unquote(parts.password)))
            except ValueError:
                pass
        for name in _SIMPLE_CREDENTIALS:
            found = self.find_simple(name)
            if found:
                values.append(found.value)
        if self.searxng_url:
            from urllib.parse import unquote, urlsplit

            try:
                parts = urlsplit(self.searxng_url)
                if parts.password:
                    values.extend((parts.password, unquote(parts.password)))
            except ValueError:
                pass
        return _dedupe(values)

    @cached_property
    def gemini_endpoints(self) -> tuple[GeminiEndpoint, ...]:
        return gemini_endpoints(auth=self.auth)

    def find_simple(self, backend: str) -> Optional[Credential]:
        spec = _SIMPLE_CREDENTIALS.get(backend)
        return find_credential(self.auth, *spec) if spec else None

    def simple_key(self, backend: str) -> Optional[str]:
        found = self.find_simple(backend)
        return found.value if found else None

    def available(self, backend: str) -> bool:
        if backend == "ddg":
            return True
        if backend == "gemini":
            return bool(self.gemini_endpoints)
        if backend == "searxng":
            return bool(self.searxng_url)
        return bool(self.simple_key(backend))

    def describe(self, backend: str) -> str:
        """Where this backend's credential came from, for backends()."""
        if backend == "ddg":
            return "no credential required"
        if backend == "gemini":
            return ", ".join(
                f"{endpoint.label} [{endpoint.source}, {len(endpoint.keys)} key{'' if len(endpoint.keys) == 1 else 's'}]"
                for endpoint in self.gemini_endpoints
            )
        if backend == "searxng":
            return safe_endpoint_label(self.searxng_url) if self.searxng_url else ""
        found = self.find_simple(backend)
        return found.source if found else ""


def parse_order(provider: Optional[str], default_order: Sequence[str] = AUTO_ORDER) -> tuple[str, ...]:
    """Turn a provider argument into an ordered backend list.

    Accepts "auto", "all", a single backend name, or a comma-separated list.
    """
    raw = (provider or env_str("PRIME_AGENT_WEBSEARCH_PROVIDER") or "auto").strip().lower()
    if raw in ("auto", "all", ""):
        return tuple(default_order)
    names = [part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()]
    unknown = [name for name in names if name not in default_order]
    if unknown:
        raise ValueError(
            f"unknown search backend(s): {', '.join(unknown)}. "
            f"available: {', '.join(default_order)}, or auto/all"
        )
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def wants_every_backend(provider: Optional[str]) -> bool:
    raw = (provider or env_str("PRIME_AGENT_WEBSEARCH_PROVIDER") or "auto").strip().lower()
    if raw == "all":
        return True
    names = [part for part in raw.replace(" ", ",").split(",") if part.strip()]
    return len(names) > 1


def load_settings(
    num_results: Optional[int] = None,
    timeout: Optional[float] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Settings:
    count = (
        num_results
        if num_results is not None
        else env_int("PRIME_AGENT_WEBSEARCH_NUM_RESULTS", DEFAULT_NUM_RESULTS)
    )
    count = max(1, min(MAX_NUM_RESULTS, count))
    seconds = timeout if timeout is not None else env_float("PRIME_AGENT_WEBSEARCH_TIMEOUT", DEFAULT_TIMEOUT)
    if (
        not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds <= 0
    ):
        if timeout is not None:
            raise ValueError("timeout must be a positive finite number")
        seconds = DEFAULT_TIMEOUT
    searxng = env_str("SEARXNG_URL") or env_str("PRIME_AGENT_WEBSEARCH_SEARXNG_URL")
    return Settings(
        num_results=count,
        timeout=float(seconds),
        order=parse_order(provider),
        gemini_model=model or env_str("PRIME_AGENT_WEBSEARCH_GEMINI_MODEL"),
        searxng_url=searxng.rstrip("/") if searxng else None,
        cache_ttl=max(0.0, env_float("PRIME_AGENT_WEBSEARCH_CACHE_TTL", DEFAULT_CACHE_TTL)),
        cooldown_base=max(0.0, env_float("PRIME_AGENT_WEBSEARCH_COOLDOWN", DEFAULT_COOLDOWN)),
        auth=read_first_json("auth.json"),
    )
