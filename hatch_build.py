"""Hatchling build hook: generate Wyckoff position mappings during package build.

This hook runs generate_wyckoff_mappings() so that
wyckoffs_enumerated_by_ss.json is always present in the installed package
without requiring a separate manual step.
"""
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        # scripts/ is not yet installed, so add it to sys.path manually.
        # generate_wyckoff_mappings() intentionally has no wyckoff_transformer
        # dependency, so it is safe to call here.
        root = Path(__file__).parent
        sys.path.insert(0, str(root / "scripts"))
        from preprocess_wychoffs import generate_wyckoff_mappings  # noqa: PLC0415
        output = root / "src" / "wyckoff_transformer" / "wyckoffs_enumerated_by_ss.json"
        generate_wyckoff_mappings(output_file=output)
