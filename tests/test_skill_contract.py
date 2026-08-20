"""Tests for the Prime Agent skill contract itself.

These lock the host-facing rules that are easy to break silently: the detection
contract (SKILL.md / pyproject / src layout / import name), the `run()` signature
that becomes both `help()` output and the tyro CLI, and the module wrapper the
kernel applies to a skill that defines `run()`.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import sys
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

import websearch

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = REPO_ROOT / "skills" / "websearch"
IMPORT_NAME = "websearch"


class DetectionContractTest(unittest.TestCase):
    """https://github.com/PrimeIntellect-ai/prime-agent -> docs/skills.md"""

    def test_required_files_exist(self) -> None:
        self.assertTrue((SKILL_DIR / "SKILL.md").is_file())
        self.assertTrue((SKILL_DIR / "pyproject.toml").is_file())
        self.assertTrue((SKILL_DIR / "src" / IMPORT_NAME / "__init__.py").is_file())

    def test_import_name_matches_skill_directory(self) -> None:
        self.assertEqual(SKILL_DIR.name.replace("-", "_"), IMPORT_NAME)
        self.assertTrue(IMPORT_NAME.isidentifier())

    def test_skill_md_frontmatter(self) -> None:
        text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---\n"), "SKILL.md must start with YAML frontmatter")
        frontmatter = text.split("---", 2)[1]
        name = re.search(r"^name:\s*(.+)$", frontmatter, re.M)
        description = re.search(r"^description:\s*(.+)$", frontmatter, re.M)
        self.assertIsNotNone(name)
        self.assertIsNotNone(description)
        assert name and description
        self.assertEqual(name.group(1).strip(), SKILL_DIR.name)
        # The description is always in the system prompt: keep it single-line and short.
        self.assertLessEqual(len(description.group(1).strip()), 500)

    def test_pyproject_declares_the_right_names(self) -> None:
        data = tomllib.loads((SKILL_DIR / "pyproject.toml").read_text(encoding="utf-8"))
        project = data["project"]
        # The console script name must match the import name exactly, underscores included.
        self.assertEqual(list(project["scripts"]), [IMPORT_NAME])
        self.assertEqual(project["scripts"][IMPORT_NAME], "rlm.skill:cli")
        # hatchling needs an explicit package path whenever the names differ.
        self.assertEqual(data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"], [f"src/{IMPORT_NAME}"])
        self.assertGreaterEqual(tuple(int(p) for p in project["requires-python"].lstrip(">=").split(".")), (3, 10))
        # prime-agent-runtime is bundled with the host and must never be declared.
        self.assertNotIn("prime-agent-runtime", " ".join(project["dependencies"]))

    def test_declared_dependencies_cover_third_party_imports(self) -> None:
        data = tomllib.loads((SKILL_DIR / "pyproject.toml").read_text(encoding="utf-8"))
        declared = {re.split(r"[<>=!\[ ]", dep, 1)[0].lower() for dep in data["project"]["dependencies"]}
        sources = " ".join(
            path.read_text(encoding="utf-8") for path in (SKILL_DIR / "src" / IMPORT_NAME).glob("*.py")
        )
        if "import httpx" in sources:
            self.assertIn("httpx", declared)
        if "from bs4" in sources or "import bs4" in sources:
            self.assertIn("beautifulsoup4", declared)

    def test_package_manifest_points_at_the_skill(self) -> None:
        manifest = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
        self.assertIn("pi-package", manifest["keywords"])
        skills = manifest["pi"]["skills"]
        self.assertTrue(skills)
        for entry in skills:
            self.assertTrue((REPO_ROOT / entry).is_dir(), f"{entry} does not exist")


class RunSignatureTest(unittest.TestCase):
    """`run()` is the skill's public API, its help() text, and its CLI."""

    def test_run_is_async_and_documented(self) -> None:
        self.assertTrue(inspect.iscoroutinefunction(websearch.run))
        doc = inspect.getdoc(websearch.run) or ""
        self.assertIn("Args:", doc)
        self.assertIn("Returns:", doc)

    def test_signature_is_cli_friendly(self) -> None:
        parameters = list(inspect.signature(websearch.run).parameters.values())
        self.assertEqual(parameters[0].name, "query")
        self.assertIs(parameters[0].default, inspect.Parameter.empty)
        for parameter in parameters:
            self.assertIsNot(parameter.annotation, inspect.Parameter.empty, f"{parameter.name} needs a type")
            self.assertNotEqual(parameter.kind, inspect.Parameter.VAR_KEYWORD, "tyro cannot map **kwargs")
        for parameter in parameters[1:]:
            self.assertIsNot(parameter.default, inspect.Parameter.empty, f"{parameter.name} needs a default")

    def test_helpers_are_exported(self) -> None:
        for name in ("run", "backends", "search"):
            self.assertIn(name, websearch.__all__)
            self.assertTrue(callable(getattr(websearch, name)))

    def test_backends_helper_does_not_shadow_the_private_module(self) -> None:
        # websearch.backends is the public listing helper; implementations live in
        # _backends so the two can never collide.
        self.assertTrue(inspect.iscoroutinefunction(websearch.backends))
        self.assertTrue((SKILL_DIR / "src" / IMPORT_NAME / "_backends.py").is_file())
        self.assertFalse((SKILL_DIR / "src" / IMPORT_NAME / "backends.py").exists())


class KernelWrapperTest(unittest.IsolatedAsyncioTestCase):
    """Reproduces prime-agent's `_prime_agent_wrap_skill_module` contract."""

    @staticmethod
    def wrap(module: types.ModuleType) -> types.ModuleType:
        run = getattr(module, "run", None)
        assert callable(run)

        class CallableSkillModule(types.ModuleType):
            async def __call__(self, *args: Any, **kwargs: Any) -> Any:
                return await self.run(*args, **kwargs)

        wrapped = CallableSkillModule(module.__name__)
        wrapped.__dict__.update(module.__dict__)
        wrapped.__signature__ = inspect.signature(run)  # type: ignore[attr-defined]
        if run.__doc__:
            wrapped.__doc__ = run.__doc__
        return wrapped

    async def test_module_becomes_awaitable_and_keeps_attributes(self) -> None:
        wrapped = self.wrap(websearch)

        async def fake_execute(query: str, *args: Any, **kwargs: Any) -> websearch.Outcome:
            from websearch import config

            with mock.patch.object(config, "read_first_json", return_value={}):
                settings = config.load_settings()
            return websearch.Outcome(query=query, settings=settings, results=[], failures=[])

        # Callable module: await websearch("...")
        with mock.patch.object(websearch, "_execute", fake_execute):
            with mock.patch.object(websearch, "_render", lambda outcome: f"rendered:{outcome.query}"):
                self.assertEqual(await wrapped("hello"), "rendered:hello")

        # Attribute access survives: await websearch.backends(), websearch.search(...)
        self.assertTrue(inspect.iscoroutinefunction(wrapped.backends))
        self.assertTrue(inspect.iscoroutinefunction(wrapped.search))
        self.assertIs(wrapped.Outcome, websearch.Outcome)
        self.assertEqual(wrapped.__doc__, websearch.run.__doc__)
        self.assertEqual(str(wrapped.__signature__), str(inspect.signature(websearch.run)))

    async def test_console_script_entry_point_resolves_run(self) -> None:
        # rlm.skill:cli imports "<script name>.run"; emulate that lookup.
        module = sys.modules[IMPORT_NAME]
        run = getattr(module, "run", None)
        self.assertTrue(callable(run))
        self.assertTrue(asyncio.iscoroutinefunction(run))


if __name__ == "__main__":
    unittest.main()
