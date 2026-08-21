"""Session-scoped backend health, refined from this session's own trajectory.

Prime Agent's Continual Harness improves the harness from what actually
happened instead of keeping scaffolding static. This module applies that idea
at skill scale: a backend that fails earns a cooldown which doubles with each
consecutive failure, and a single success clears it again. State lives only in
this process - every fresh session starts from zero assumptions, and
`websearch.health()` shows the evidence behind the current ordering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# One failure costs this many seconds; each consecutive failure doubles the
# previous cooldown, capped so one flaky backend can never disappear for long.
BASE_COOLDOWN = 120.0
MAX_MULTIPLIER = 8


def _now() -> float:
    """Monotonic clock, indirected so tests can control expiry exactly."""
    return time.monotonic()


@dataclass
class BackendHealth:
    """Evidence kept for one backend inside this session."""

    failures: int = 0
    """Consecutive failures without an intervening success."""
    until: float = 0.0
    """Monotonic timestamp the current cooldown ends at."""
    reason: str = ""
    """Most recent failure message, kept as evidence."""

    @property
    def cooling(self) -> bool:
        return self.until > _now()

    @property
    def multiplier(self) -> int:
        exponent = min(max(0, self.failures - 1), MAX_MULTIPLIER.bit_length() - 1)
        return min(MAX_MULTIPLIER, 2**exponent)


class HealthTracker:
    """Consecutive-failure cooldowns for named backends."""

    def __init__(self) -> None:
        self._state: dict[str, BackendHealth] = {}

    def record_success(self, name: str) -> None:
        """One success clears the evidence: the backend is trusted again."""
        state = self._state.get(name)
        if state is not None and (state.failures or state.until):
            state.failures = 0
            state.until = 0.0
            state.reason = ""

    def record_failure(self, name: str, reason: str, base: float) -> None:
        """Grow the cooldown exponentially; ``base`` <= 0 disables cooling."""
        state = self._state.setdefault(name, BackendHealth())
        state.failures += 1
        state.reason = reason
        state.until = _now() + base * state.multiplier if base > 0 else 0.0

    def reset(self) -> None:
        self._state.clear()

    def get(self, name: str) -> BackendHealth:
        return self._state.get(name) or BackendHealth()

    def partition(self, names: list[str], *, enabled: bool = True) -> tuple[list[str], list[str]]:
        """Split backends into (ready, cooling), preserving the given order.

        Disabling cooldowns also clears their deadlines so an old deadline
        cannot come back if the setting is re-enabled later in this session.
        Failure counts remain as diagnostic evidence until a success resets them.
        """
        if not enabled:
            for state in self._state.values():
                state.until = 0.0
            return list(names), []

        ready: list[str] = []
        cooling: list[str] = []
        for name in names:
            (cooling if self.get(name).cooling else ready).append(name)
        return ready, cooling


def render(tracker: HealthTracker, order: list[str], *, base: float) -> str:
    lines = ["# websearch health", ""]
    for name in order:
        state = tracker.get(name)
        if state.failures and base <= 0:
            detail = f"{state.failures} recent failure(s), cooldown off"
            if state.reason:
                detail += f": {state.reason}"
        elif state.cooling:
            remaining = max(1, round(state.until - _now()))
            detail = f"cooling {remaining}s left after {state.failures} consecutive failure(s)"
            if state.reason:
                detail += f": {state.reason}"
        elif state.failures:
            detail = f"ready; cooldown expired after {state.failures} consecutive failure(s)"
            if state.reason:
                detail += f": {state.reason}"
        else:
            detail = "ok"
        lines.append(f"- {name}: {detail}")
    lines += [
        "",
        f"cooldown: {'off' if base <= 0 else f'{base:g}s base, doubling per consecutive failure (cap x{MAX_MULTIPLIER})'}",
        "one success resets a backend; set PRIME_AGENT_WEBSEARCH_COOLDOWN=0 to disable",
    ]
    return "\n".join(lines)
