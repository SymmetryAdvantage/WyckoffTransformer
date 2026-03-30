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
        root = Path(__file__).parent
        sys.path.insert(0, str(root / "src"))
        from wyckoff_transformer.preprocess_wychoffs import (  # noqa: PLC0415
            generate_wyckoff_mappings, enumerate_wychoffs_by_ss)
        generate_wyckoff_mappings()
        enumerate_wychoffs_by_ss()
