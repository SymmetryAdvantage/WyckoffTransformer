"""Tests for the cache migration that adds ``site_symmetries_augmented``.

The migration is run on another machine against a 5.3M-row cache, so the only
chance to find a mistake in it is here, on a frame small enough to check by
hand.  What it must get right: recompute the pair from ``wyckoff_letters``
alone, produce exactly what a fresh record would carry, and leave every other
column untouched.
"""
from __future__ import annotations

import pandas as pd
import pytest

from wyckoff_transformer.data import (
    AUGMENTED_SITE_SYMMETRIES,
    AUGMENTED_SITES_ENUMERATION,
)
from wyckoff_transformer.evaluation.novelty import record_to_augmented_fingerprint
from wyckoff_transformer.evaluation.protocol import GeneFingerprinter
from wyckoff_transformer.preprocess_wychoffs import get_augmentation_dict
from wyckoff_transformer.tokenization import load_wyckoff_mappings

# One from the 26 exposed space groups, one from outside them, and the pair from
# the audit that a relabelling does *not* relate.
GENES = (
    {"group": 68, "species": ["Li", "Be"], "numIons": [1, 1], "sites": [["1e"], ["1c"]]},
    {"group": 68, "species": ["Li", "Be"], "numIons": [1, 1], "sites": [["1e"], ["1d"]]},
    {"group": 225, "species": ["Na", "Cl"], "numIons": [4, 4], "sites": [["4a"], ["4b"]]},
    {"group": 16, "species": ["Be"], "numIons": [1], "sites": [["1t"]]},
)


@pytest.fixture(scope="module")
def fingerprinter():
    return GeneFingerprinter()


@pytest.fixture(scope="module")
def tables():
    mappings = load_wyckoff_mappings()
    return get_augmentation_dict(), mappings.enum_from_ss_letter, mappings.ss_from_letter


def _stale_frame(fingerprinter) -> pd.DataFrame:
    """A cache as it was written before the fix: no paired symmetries."""
    rows = []
    for gene in GENES:
        record = dict(fingerprinter.record(gene))
        record.pop(AUGMENTED_SITE_SYMMETRIES)
        record["immutable_id"] = f"{gene['group']}-{gene['sites']}"
        rows.append(record)
    return pd.DataFrame(rows)


def test_the_migration_reproduces_a_fresh_record(fingerprinter, tables):
    from wyckoff_transformer.evaluation.novelty import augmented_variants

    frame = _stale_frame(fingerprinter)
    with pytest.raises(KeyError, match=AUGMENTED_SITE_SYMMETRIES):
        augmented_variants(frame.iloc[0])

    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "migrate", Path(__file__).resolve().parents[3]
        / "scripts" / "migrate_augmented_site_symmetries.py")
    migrate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migrate)

    migrate.migrate_frame(frame, *tables)

    for position, gene in enumerate(GENES):
        fresh = fingerprinter.record(gene)
        assert tuple(frame.iloc[position][AUGMENTED_SITE_SYMMETRIES]) == tuple(
            fresh[AUGMENTED_SITE_SYMMETRIES])
        assert tuple(frame.iloc[position][AUGMENTED_SITES_ENUMERATION]) == tuple(
            fresh[AUGMENTED_SITES_ENUMERATION])
        assert record_to_augmented_fingerprint(frame.iloc[position]) == \
            record_to_augmented_fingerprint(fresh)


def test_the_migration_keeps_the_other_columns(fingerprinter, tables):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "migrate", Path(__file__).resolve().parents[3]
        / "scripts" / "migrate_augmented_site_symmetries.py")
    migrate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migrate)

    frame = _stale_frame(fingerprinter)
    before = frame.drop(columns=[AUGMENTED_SITES_ENUMERATION]).copy()
    migrate.migrate_frame(frame, *tables)
    pd.testing.assert_frame_equal(
        frame.drop(columns=[AUGMENTED_SITES_ENUMERATION, AUGMENTED_SITE_SYMMETRIES]),
        before)


def test_the_migration_separates_the_two_genes_it_used_to_merge(fingerprinter, tables):
    """The point of the exercise, checked end to end on a frame."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "migrate", Path(__file__).resolve().parents[3]
        / "scripts" / "migrate_augmented_site_symmetries.py")
    migrate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migrate)

    frame = _stale_frame(fingerprinter)
    migrate.migrate_frame(frame, *tables)
    a, b = record_to_augmented_fingerprint(frame.iloc[0]), \
        record_to_augmented_fingerprint(frame.iloc[1])
    assert a != b, "space group 68: no relabelling relates these two genes"


def test_a_frame_without_letters_cannot_be_migrated(fingerprinter, tables):
    """Then the augmentation is not recoverable and the cache has to be rebuilt."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "migrate", Path(__file__).resolve().parents[3]
        / "scripts" / "migrate_augmented_site_symmetries.py")
    migrate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migrate)

    frame = _stale_frame(fingerprinter).drop(columns=["wyckoff_letters"])
    with pytest.raises(KeyError):
        migrate.migrate_frame(frame, *tables)
