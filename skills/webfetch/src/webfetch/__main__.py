"""Standalone entry point: `python -m webfetch <url> [mode]`.

Inside Prime Agent's kernel the skill is called as `await webfetch(...)`; this
exists for smoke-testing outside the kernel, where the `rlm.skill:cli` console
script is unavailable.
"""

from __future__ import annotations

from . import cli

if __name__ == "__main__":
    cli()
