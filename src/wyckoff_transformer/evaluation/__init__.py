"""Unified evaluation package combining core and dataset-level evaluators.

Re-exports are resolved lazily (PEP 562).  Importing this package eagerly used
to pull in ``core``, and through it torch, wandb, smact, sklearn and matminer --
so ``import wyckoff_transformer.evaluation.protocol`` dragged the whole training
stack in with it.  That matters for :mod:`wyckoff_transformer.cli.protocol`'s
score stage, which is meant to run under LeMat-GenBench's environment, where
torch is held at 2.6 by CUDA-specific torch-scatter wheels.  Attribute access is
unchanged; only the import cost moved to first use.
"""

# ruff: noqa: F401 -- the TYPE_CHECKING block below re-exports names that are
# resolved at runtime through __getattr__, so ruff cannot see them used and
# cannot match them against the dynamically built __all__.

from typing import TYPE_CHECKING

#: Exported name -> (submodule, attribute in that submodule).
_EXPORTS = {
    "DoFCounter": (".core", "DoFCounter"),
    "StatisticalEvaluator": (".core", "StatisticalEvaluator"),
    "count_unique": (".core", "count_unique"),
    "evaluate_and_log": (".core", "evaluate_and_log"),
    "generated_to_fingerprint": (".core", "generated_to_fingerprint"),
    "ks_to_dict": (".core", "ks_to_dict"),
    "record_to_augmented_fingerprints": (".core", "record_to_augmented_fingerprints"),
    "smac_validity_from_counter": (".core", "smac_validity_from_counter"),
    "smact_validity": (".core", "smact_validity"),
    "smact_validity_from_record": (".core", "smact_validity_from_record"),
    "smact_validity_optimised": (".core", "smact_validity_optimised"),
    "timed_smact_validity_from_record": (".core", "timed_smact_validity_from_record"),
    "wycryst_to_pyxtal_dict": (".core", "wycryst_to_pyxtal_dict"),
    "Crystal": (".cdvae_metrics", "Crystal"),
    "DATA_KEYS": (".generated_dataset", "DATA_KEYS"),
    "GeneratedDataset": (".generated_dataset", "GeneratedDataset"),
    "load_all_from_config": (".generated_dataset", "load_all_from_config"),
    "NoveltyFilter": (".novelty", "NoveltyFilter"),
    "filter_by_unique_structure": (".novelty", "filter_by_unique_structure"),
    "EnhancedStatisticalEvaluator": (".statistical_evaluator", "StatisticalEvaluator"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Import the owning submodule on first access to an exported name."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value  # cache, so later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:  # keep static analysis and IDEs working
    from .cdvae_metrics import Crystal
    from .core import (
        DoFCounter,
        StatisticalEvaluator,
        count_unique,
        evaluate_and_log,
        generated_to_fingerprint,
        ks_to_dict,
        record_to_augmented_fingerprints,
        smac_validity_from_counter,
        smact_validity,
        smact_validity_from_record,
        smact_validity_optimised,
        timed_smact_validity_from_record,
        wycryst_to_pyxtal_dict,
    )
    from .generated_dataset import DATA_KEYS, GeneratedDataset, load_all_from_config
    from .novelty import NoveltyFilter, filter_by_unique_structure
    from .statistical_evaluator import StatisticalEvaluator as EnhancedStatisticalEvaluator
