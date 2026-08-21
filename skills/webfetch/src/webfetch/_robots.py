"""robots.txt handling for autonomous fetches.

Convention follows the official MCP fetch server
(github.com/modelcontextprotocol/servers, `src/fetch`): an agent fetching a URL on
its own initiative checks robots.txt first, identifies itself with an
"Autonomous" user agent, treats `401`/`403` on robots.txt as a refusal, and lets
the caller override when a human explicitly asked for that page.

Rules use RFC-style longest-match precedence, `*` wildcards and terminal `$`;
an equally specific `Allow` wins over `Disallow`.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

from ._safety import FetchError, Resolver, TooLargeError, UnsafeUrlError, guarded_get

ROBOTS_TIMEOUT = 10.0
MAX_ROBOTS_BYTES = 1_000_000
MAX_ROBOTS_MATCH_WORK = 250_000
_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)


def _normalize_percent(value: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(value):
        if index + 2 < len(value) and value[index] == "%":
            token = value[index + 1 : index + 3]
            try:
                character = chr(int(token, 16))
            except ValueError:
                pass
            else:
                output.append(character if character in _UNRESERVED else "%" + token.upper())
                index += 3
                continue
        character = value[index]
        if ord(character) > 127:
            output.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
        else:
            output.append(character)
        index += 1
    return "".join(output)


@dataclass(frozen=True)
class _Rule:
    pattern: str
    anchored: bool
    specificity: int
    allow: bool

    def matches(self, target: str) -> bool:
        """Match `*` with ordered native substring searches."""
        pattern = self.pattern
        if "*" not in pattern:
            return target == pattern if self.anchored else target.startswith(pattern)

        parts = pattern.split("*")
        position = 0
        if parts[0]:
            if not target.startswith(parts[0]):
                return False
            position = len(parts[0])

        literals = [part for part in parts[1:] if part]
        for index, literal in enumerate(literals):
            is_last = index == len(literals) - 1
            if self.anchored and is_last and not pattern.endswith("*"):
                start = len(target) - len(literal)
                if start < position or not target.endswith(literal):
                    return False
                position = len(target)
            else:
                found = target.find(literal, position)
                if found < 0:
                    return False
                position = found + len(literal)

        if self.anchored and not pattern.endswith("*") and not literals:
            return position == len(target)
        return True


@dataclass
class _Policy:
    groups: list[tuple[tuple[str, ...], tuple[_Rule, ...]]]

    def can_fetch(self, user_agent: str, url: str) -> Optional[bool]:
        product = user_agent.split("/", 1)[0].lower()
        best_agent = -1
        selected: list[_Rule] = []
        for agents, rules in self.groups:
            scores = [
                0 if agent == "*" else len(agent)
                for agent in agents
                if agent == "*" or product.startswith(agent)
            ]
            if not scores:
                continue
            score = max(scores)
            if score > best_agent:
                best_agent = score
                selected = list(rules)
            elif score == best_agent:
                selected.extend(rules)

        parts = urlsplit(url)
        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query
        target = _normalize_percent(target)
        matches: list[_Rule] = []
        work = 0
        for rule in selected:
            # The native matcher scans each literal and target at most linearly.
            work += len(rule.pattern)
            if "*" in rule.pattern:
                work += len(target)
            if work > MAX_ROBOTS_MATCH_WORK:
                return None
            if rule.matches(target):
                matches.append(rule)
        if not matches:
            return True
        longest = max(rule.specificity for rule in matches)
        return any(rule.allow for rule in matches if rule.specificity == longest)


def _compile_rule(value: str, allow: bool) -> Optional[_Rule]:
    if not value:
        return None
    anchored = value.endswith("$")
    core = _normalize_percent(value[:-1] if anchored else value)
    # Adjacent stars are equivalent and needlessly increase matching work.
    while "**" in core:
        core = core.replace("**", "*")
    return _Rule(
        pattern=core,
        anchored=anchored,
        specificity=len(re.sub(r"%[0-9A-F]{2}", "x", core.replace("*", ""))),
        allow=allow,
    )


def _parse_policy(text: str) -> _Policy:
    groups: list[tuple[tuple[str, ...], tuple[_Rule, ...]]] = []
    agents: list[str] = []
    rules: list[_Rule] = []
    saw_rule = False

    def finish() -> None:
        nonlocal agents, rules, saw_rule
        if agents:
            groups.append((tuple(agents), tuple(rules)))
        agents = []
        rules = []
        saw_rule = False

    for raw_line in text.lstrip("\ufeff").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field_name, value = line.split(":", 1)
        field_name = field_name.strip().lower()
        value = value.strip()
        if field_name == "user-agent":
            if saw_rule:
                finish()
            if value:
                agents.append(value.lower())
        elif field_name in ("allow", "disallow") and agents:
            saw_rule = True
            rule = _compile_rule(value, field_name == "allow")
            if rule is not None:
                rules.append(rule)
    finish()
    return _Policy(groups)


@dataclass
class RobotsVerdict:
    allowed: bool
    reason: str = ""


class RobotsDeniedError(FetchError):
    """A redirect target is disallowed by its own robots policy."""


@dataclass
class RobotsCache:
    """Per-origin robots.txt decisions, cached for the life of one fetch call."""

    user_agent: str
    resolver: Optional[Resolver] = None
    timeout: float = ROBOTS_TIMEOUT
    _origins: dict[str, Optional[_Policy]] = field(default_factory=dict)
    _refusals: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def robots_url(url: str) -> str:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))

    async def _load(self, client: httpx.AsyncClient, origin_url: str) -> Optional[_Policy]:
        robots_url = self.robots_url(origin_url)
        try:
            response = await guarded_get(
                client,
                robots_url,
                max_bytes=MAX_ROBOTS_BYTES,
                resolver=self.resolver,
                timeout=self.timeout,
                headers={"user-agent": self.user_agent},
                raise_for_status=False,
                reject_declared_oversize=False,
            )
        except UnsafeUrlError:
            # A policy lookup must never turn a refused target or redirect into
            # an SSRF request. Let the caller return the safety error.
            raise
        except TooLargeError:
            # RFC 9309 permits crawlers to ignore content beyond their parser cap.
            return None
        except FetchError:
            # Unreachable robots.txt is not a refusal: fetch and let the server decide.
            return None

        if response.status in (401, 403):
            self._refusals[robots_url] = (
                f"robots.txt at {robots_url} returned HTTP {response.status}, "
                "so autonomous fetching is treated as disallowed"
            )
            return None
        if response.status >= 400:
            return None
        # For an oversized file, parse the bounded RFC window and ignore an
        # incomplete final line plus all later content.
        policy_text = response.text
        if response.truncated:
            policy_text = policy_text.rsplit("\n", 1)[0] if "\n" in policy_text else ""
        return await asyncio.to_thread(_parse_policy, policy_text)

    async def check(self, client: httpx.AsyncClient, url: str) -> RobotsVerdict:
        origin = urlsplit(url)
        key = f"{origin.scheme}://{origin.netloc}"
        if key not in self._origins:
            self._origins[key] = await self._load(client, url)

        robots_url = self.robots_url(url)
        if robots_url in self._refusals:
            return RobotsVerdict(False, self._refusals[robots_url])

        parser = self._origins[key]
        if parser is None:
            return RobotsVerdict(True)
        allowed = await asyncio.to_thread(parser.can_fetch, self.user_agent, url)
        if allowed is True:
            return RobotsVerdict(True)
        if allowed is None:
            return RobotsVerdict(
                False,
                f"robots.txt matching at {robots_url} exceeded the local work budget, "
                "so autonomous fetching is conservatively refused. Pass "
                "respect_robots=False if the user explicitly asked for this page.",
            )
        return RobotsVerdict(
            False,
            f"robots.txt at {robots_url} disallows {self.user_agent} from fetching this path. "
            "Pass respect_robots=False if the user explicitly asked for this page.",
        )
