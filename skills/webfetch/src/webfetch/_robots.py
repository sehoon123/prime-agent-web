"""robots.txt handling for autonomous fetches.

Convention follows the official MCP fetch server
(github.com/modelcontextprotocol/servers, `src/fetch`): an agent fetching a URL on
its own initiative checks robots.txt first, identifies itself with an
"Autonomous" user agent, treats `401`/`403` on robots.txt as a refusal, and lets
the caller override when a human explicitly asked for that page.

Implemented with the standard library's `RobotFileParser`, so this costs no
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

ROBOTS_TIMEOUT = 10.0


@dataclass
class RobotsVerdict:
    allowed: bool
    reason: str = ""


@dataclass
class RobotsCache:
    """Per-origin robots.txt decisions, cached for the life of one fetch call."""

    user_agent: str
    _origins: dict[str, Optional[RobotFileParser]] = field(default_factory=dict)
    _refusals: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def robots_url(url: str) -> str:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))

    async def _load(self, client: httpx.AsyncClient, origin_url: str) -> Optional[RobotFileParser]:
        robots_url = self.robots_url(origin_url)
        try:
            response = await client.get(
                robots_url,
                follow_redirects=True,
                timeout=ROBOTS_TIMEOUT,
                headers={"user-agent": self.user_agent},
            )
        except httpx.HTTPError:
            # Unreachable robots.txt is not a refusal: fetch and let the server decide.
            return None

        if response.status_code in (401, 403):
            self._refusals[robots_url] = (
                f"robots.txt at {robots_url} returned HTTP {response.status_code}, "
                "so autonomous fetching is treated as disallowed"
            )
            return None
        if 400 <= response.status_code < 500:
            return None  # no robots.txt means everything is allowed

        parser = RobotFileParser()
        # Comments can contain sequences that confuse the parser; drop them first.
        parser.parse(
            [line for line in response.text.splitlines() if not line.strip().startswith("#")]
        )
        return parser

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
        if parser.can_fetch(self.user_agent, url):
            return RobotsVerdict(True)
        return RobotsVerdict(
            False,
            f"robots.txt at {robots_url} disallows {self.user_agent} from fetching this path. "
            "Pass respect_robots=False if the user explicitly asked for this page.",
        )
