"""Every W&B entry point must name its entity explicitly.

Left implicit, W&B resolves the entity from the account default, which varies between machines
and login sessions: one run landed under a personal entity while its siblings went to the shared
one, splitting a comparison across two workspaces and making `wandb.Api().run()` on a two-part
path fail to find runs that plainly exist.
"""
import ast
import re
import unittest
from pathlib import Path

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT, wandb_run_path

REPO = Path(__file__).resolve().parents[3]
SOURCES = sorted(
    p for p in list((REPO / "src").rglob("*.py")) + list((REPO / "scripts").rglob("*.py"))
    if "tests" not in p.parts)


class TestRunPath(unittest.TestCase):
    def test_is_fully_qualified(self):
        self.assertEqual(wandb_run_path("abc123"),
                         f"{WANDB_ENTITY}/{WANDB_PROJECT}/abc123")

    def test_two_part_paths_are_not_produced(self):
        """`wandb.Api().run` reads a two-part path as project/run under the default entity."""
        self.assertEqual(wandb_run_path("abc123").count("/"), 2)

    def test_overridable(self):
        self.assertEqual(wandb_run_path("abc123", "someone", "proj"), "someone/proj/abc123")


class TestCallSitesNameTheEntity(unittest.TestCase):
    def _calls_named(self, target):
        """Yield (path, lineno, kwargs) for every call to `target` in the shipped sources."""
        for path in SOURCES:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if ast.unparse(node.func).endswith(target):
                    yield path, node.lineno, {kw.arg for kw in node.keywords}

    def test_wandb_init_passes_entity(self):
        found = list(self._calls_named("wandb.init"))
        self.assertTrue(found, "no wandb.init call sites found -- has the scan path moved?")
        for path, lineno, kwargs in found:
            if not kwargs:
                # A bare wandb.init() inside a sweep agent inherits the agent's entity.
                continue
            self.assertIn("entity", kwargs,
                          f"{path.relative_to(REPO)}:{lineno} calls wandb.init without an entity")

    def test_wandb_agent_passes_entity(self):
        for path, lineno, kwargs in self._calls_named("wandb.agent"):
            self.assertIn("entity", kwargs,
                          f"{path.relative_to(REPO)}:{lineno} calls wandb.agent without an entity")

    def test_api_run_lookups_are_fully_qualified(self):
        """An f-string run path must carry three segments, not project/run."""
        pattern = re.compile(r"\.run\(\s*f?\"([^\"]*\{[^\"]*)\"")
        for path in SOURCES:
            for match in pattern.finditer(path.read_text()):
                self.assertGreaterEqual(
                    match.group(1).count("/"), 2,
                    f"{path.relative_to(REPO)} looks up a run by a path missing the entity: "
                    f"{match.group(1)!r} -- use wandb_run_path()")


if __name__ == "__main__":
    unittest.main()
