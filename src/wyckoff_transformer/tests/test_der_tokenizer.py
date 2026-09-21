"""``der_tokenizer_v1`` tokenises, and does so on the corrected augmentation.

The config is the replacement for the obsolete ones, so the thing worth pinning
is that its two novelties actually work end to end: ``site_symmetries`` is
augmented alongside ``sites_enumeration``, and ``site_symmetry_ops_id`` -- which
is keyed on the symbol a relabelling changes -- is recomputed per variant rather
than carried over from the original.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from wyckoff_transformer.evaluation.protocol import GeneFingerprinter
from wyckoff_transformer.tokenization import obsolete_reason
from wyckoff_transformer.wyckoff_processor import validate_augmented_token_fields

CONFIG = Path(__file__).resolve().parents[3] / "yamls" / "tokenisers" / "der_tokenizer_v1.yaml"

#: Space group 68 is one of the 26 where a relabelling changes the symbol.
EXPOSED_GENE = {"group": 68, "species": ["Li", "Be"], "numIons": [1, 1],
                "sites": [["1e"], ["1c"]]}


@pytest.fixture(scope="module")
def config():
    return OmegaConf.load(CONFIG)


def test_it_is_not_obsolete(config):
    assert obsolete_reason(config) is None, "the replacement must not be marked obsolete"


def test_it_passes_the_augmentation_guard(config):
    validate_augmented_token_fields(config)


def test_it_augments_the_symbol_with_the_index(config):
    augmented = list(config.augmented_token_fields)
    assert "sites_enumeration" in augmented and "site_symmetries" in augmented, (
        "the two name one Wyckoff position between them")


def test_it_recomputes_the_ops_per_variant(config):
    """``site_symmetry_ops_id`` is keyed on the symbol, so it is not invariant."""
    augmented_engineered = config.token_fields.get("augmented_engineered", {})
    assert "site_symmetry_ops_id" in augmented_engineered
    assert augmented_engineered.site_symmetry_ops_id.augmented_input == "site_symmetries"
    assert "multiplicity" not in augmented_engineered, (
        "multiplicity is a normaliser invariant; augmenting it would store one "
        "number once per variant")


def test_it_keeps_every_scalar_of_its_parents(config):
    scalars = set(config.sequence_fields.no_processing)
    for parent in ("lemat_bulk_fmax1_sg_multiplicity",
                   "lemat_bulk_ehull_sg_multiplicity_ssops"):
        parent_config = OmegaConf.load(CONFIG.parent / f"{parent}.yaml")
        assert set(parent_config.sequence_fields.no_processing) <= scalars, (
            f"{parent} passes a scalar this config drops")


def test_it_keeps_every_token_field_of_its_parents(config):
    fields = set(config.token_fields.pure_categorical)
    engineered = set(config.token_fields.engineered)
    for parent in ("lemat_bulk_fmax1_sg_multiplicity",
                   "lemat_bulk_ehull_sg_multiplicity_ssops"):
        parent_config = OmegaConf.load(CONFIG.parent / f"{parent}.yaml")
        assert set(parent_config.token_fields.pure_categorical) <= fields
        assert set(parent_config.token_fields.get("engineered", {})) <= engineered


def test_the_ops_actually_differ_between_variants_in_an_exposed_space_group():
    """The reason the ops have to be augmented, demonstrated rather than asserted.

    In space group 68 a relabelling turns a ``2..`` site into a ``.2.`` one, and
    those two symbols carry different operations -- so a description that reused
    the original's ops would be describing a different position.
    """
    from wyckoff_transformer.wyckoff_processor import (
        ENGINEERS_DIR,
        WyckoffProcessor,
        _SerialisedFeatureEngineer,
    )

    engineer = WyckoffProcessor._deserialise_feature_engineer(
        _SerialisedFeatureEngineer.model_validate_json(
            (ENGINEERS_DIR / "site_symmetry_ops_id.json").read_text(encoding="utf-8")))
    record = GeneFingerprinter().record(EXPOSED_GENE)
    space_group = record["spacegroup_number"]
    by_variant = {
        tuple(engineer.db.loc[(space_group, symbol)] for symbol in variant)
        for variant in record["site_symmetries_augmented"]
    }
    assert len(by_variant) > 1, (
        "the variants of this gene must not all share one set of operations")
