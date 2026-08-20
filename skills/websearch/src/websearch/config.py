"""Backend discovery for the websearch skill.

Everything is derived from files and environment variables that a Prime Agent
install already has. This module never writes anything and never returns a
secret in an error message.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

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
MAX_NUM_RESULTS = 20
MAX_QUERY_CHARS = 2000

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


def env_str(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def agent_dirs() -> tuple[Path, ...]:
    """Config directories to read, most specific first.

    Honours the same overrides as Prime Agent itself, then falls back to the
    default Prime Agent directory and finally to a pi directory, so the skill
    also works on hosts that keep credentials there.
    """
    candidates: list[Path] = []
    for name in ("PRIME_AGENT_CODING_AGENT_DIR", "PI_CODING_AGENT_DIR"):
        raw = env_str(name)
        if raw:
            candidates.append(Path(raw).expanduser())
    home = Path.home()
    candidates.append(home / ".prime" / "agent")
    candidates.append(home / ".pi" / "agent")

    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return tuple(unique)


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
    """Read `filename` from the first agent directory that has a usable copy."""
    for directory in agent_dirs():
        data = read_json(directory / filename)
        if data:
            return data
    return {}


def resolve_secret(raw: Any) -> str | None:
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


def auth_credential(auth: dict[str, Any], credential_id: str) -> str | None:
    entry = auth.get(credential_id)
    if not isinstance(entry, dict) or entry.get("type") != "api_key":
        return None
    return resolve_secret(entry.get("key"))


def credential(auth: dict[str, Any], credential_ids: Sequence[str], env_names: Sequence[str]) -> str | None:
    """First credential found. Environment variables win over stored ones."""
    for name in env_names:
        value = env_str(name)
        if value:
            return value
    for credential_id in credential_ids:
        value = auth_credential(auth, credential_id)
        if value:
            return value
    return None


def _dedupe(values: Iterable[str | None]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


@dataclass(frozen=True)
class GeminiEndpoint:
    """One Gemini-compatible endpoint plus every key that may be tried on it."""

    label: str
    base_url: str
    models: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()

    def pick_model(self, pinned: str | None) -> str | None:
        if pinned:
            return pinned
        for model in self.models:
            if "flash" in model:
                return model
        return self.models[0] if self.models else None


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
    data: dict[str, Any] = {}
    if override:
        data = read_json(Path(override).expanduser())
    if not data:
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
    models_json: dict[str, Any] | None = None,
    auth: dict[str, Any] | None = None,
    rotator: RotatorKeys | None = None,
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
    providers = models_json.get("providers")
    if isinstance(providers, dict):
        for provider_id, provider in providers.items():
            if not isinstance(provider, dict) or not _provider_uses_google_api(provider):
                continue
            base_url = provider.get("baseUrl")
            if not isinstance(base_url, str) or not base_url.strip():
                continue
            keys = _dedupe(
                [
                    auth_credential(auth, provider_id),
                    resolve_secret(provider.get("apiKey")),
                    *rotator.get(provider_id),
                ]
            )
            if not keys:
                continue
            endpoints.append(
                GeminiEndpoint(
                    label=provider_id,
                    base_url=base_url.strip().rstrip("/"),
                    models=_provider_model_ids(provider),
                    keys=keys,
                )
            )

    studio_key = credential(auth, ("google", "gemini"), ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    if studio_key:
        endpoints.append(
            GeminiEndpoint(
                label="google-ai-studio",
                base_url=AI_STUDIO_BASE_URL,
                models=(),
                keys=(studio_key,),
            )
        )
    return tuple(endpoints)


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings for one search call."""

    num_results: int
    timeout: float
    order: tuple[str, ...]
    gemini_model: str | None
    searxng_url: str | None
    auth: dict[str, Any]

    @property
    def secrets(self) -> tuple[str, ...]:
        """Every secret that must never appear in returned text."""
        values: list[str] = []
        for endpoint in self.gemini_endpoints:
            values.extend(endpoint.keys)
        for name in ("serper", "tavily", "brave", "exa"):
            value = self.simple_key(name)
            if value:
                values.append(value)
        return _dedupe(values)

    @property
    def gemini_endpoints(self) -> tuple[GeminiEndpoint, ...]:
        return gemini_endpoints(auth=self.auth)

    def simple_key(self, backend: str) -> str | None:
        if backend == "serper":
            return credential(self.auth, ("serper",), ("SERPER_API_KEY",))
        if backend == "tavily":
            return credential(self.auth, ("tavily",), ("TAVILY_API_KEY",))
        if backend == "brave":
            return credential(self.auth, ("brave", "brave-search"), ("BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY"))
        if backend == "exa":
            return credential(self.auth, ("exa",), ("EXA_API_KEY",))
        return None

    def available(self, backend: str) -> bool:
        if backend == "ddg":
            return True
        if backend == "gemini":
            return bool(self.gemini_endpoints)
        if backend == "searxng":
            return bool(self.searxng_url)
        return bool(self.simple_key(backend))


def parse_order(provider: str | None, default_order: Sequence[str] = AUTO_ORDER) -> tuple[str, ...]:
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
    return tuple(names)


def wants_every_backend(provider: str | None) -> bool:
    raw = (provider or env_str("PRIME_AGENT_WEBSEARCH_PROVIDER") or "auto").strip().lower()
    if raw == "all":
        return True
    return "," in raw


def load_settings(
    num_results: int | None = None,
    timeout: float | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> Settings:
    count = num_results if num_results is not None else env_int("PRIME_AGENT_WEBSEARCH_NUM_RESULTS", DEFAULT_NUM_RESULTS)
    count = max(1, min(MAX_NUM_RESULTS, count))
    seconds = timeout if timeout is not None else env_float("PRIME_AGENT_WEBSEARCH_TIMEOUT", DEFAULT_TIMEOUT)
    searxng = env_str("SEARXNG_URL") or env_str("PRIME_AGENT_WEBSEARCH_SEARXNG_URL")
    return Settings(
        num_results=count,
        timeout=max(1.0, seconds),
        order=parse_order(provider),
        gemini_model=model or env_str("PRIME_AGENT_WEBSEARCH_GEMINI_MODEL"),
        searxng_url=searxng.rstrip("/") if searxng else None,
        auth=read_first_json("auth.json"),
    )
