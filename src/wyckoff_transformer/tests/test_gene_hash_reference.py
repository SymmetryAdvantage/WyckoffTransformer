"""The tensor key table against the real LeMat-Bulk reference.

Marked ``needs_cache`` because it reads the 5.3M-row archive: run it with
``pytest --run-cache -k reference``.  The unit tests in ``test_gene_hash.py``
pin the encoding on synthetic genes; this pins it on the reference the protocol
actually judges novelty against, which is the one that matters and the one whose
size makes a subtle disagreement easy to miss.
"""
from __future__ import annotations

import pytest

from wyckoff_transformer.evaluation.gene_hash import (
    GeneKeyTable,
    default_key_table_path,
    gene_keys,
    keys_from_frame,
)
from wyckoff_transformer.evaluation.novelty import record_to_augmented_fingerprint
from wyckoff_transformer.evaluation.protocol import (
    DEFAULT_REFERENCE_CACHE,
    DEFAULT_REFERENCE_SPLITS,
    GeneFingerprinter,
    default_fingerprint_cache,
    load_reference_fingerprints,
)

pytestmark = pytest.mark.needs_cache


@pytest.fixture(scope="module")
def key_table():
    path = default_key_table_path(
        DEFAULT_REFERENCE_CACHE, DEFAULT_REFERENCE_SPLITS)
    if not path.is_file():
        pytest.skip(f"No key table at {path}; build it with gene_hash.build_reference_table")
    return GeneKeyTable.load(path)


def test_the_table_has_one_key_per_fingerprint_class(key_table):
    """The strongest cheap check: the two encodings must find the same number
    of distinct genes in the same archive.  A key that merged two fingerprints
    would show up here as a smaller table, and one that split a fingerprint as
    a larger one."""
    fingerprints = load_reference_fingerprints(
        DEFAULT_REFERENCE_CACHE, DEFAULT_REFERENCE_SPLITS,
        fingerprint_cache=default_fingerprint_cache(
            DEFAULT_REFERENCE_CACHE, DEFAULT_REFERENCE_SPLITS))
    assert len(key_table) == len(fingerprints), (
        f"{len(fingerprints)} fingerprint classes against {len(key_table)} key classes")


def test_a_sample_of_the_reference_is_in_its_own_table(key_table):
    """Every archive row must be found by the table built from it."""
    from wyckoff_transformer.dataset_cache import load_split  # noqa: PLC0415

    frame = load_split(DEFAULT_REFERENCE_CACHE, "val")
    sample = frame.sample(n=min(20000, len(frame)), random_state=0)
    assert key_table.contains(keys_from_frame(sample)).all()


def test_the_two_backends_agree_on_a_generated_cohort(key_table):
    """Novelty verdicts for a real cohort, both ways, against the same archive."""
    from wyckoff_transformer.evaluation.protocol import load_genes  # noqa: PLC0415
    from wyckoff_transformer.paths import runs_root  # noqa: PLC0415

    cohort_path = runs_root().parent / "roe" / "broadside" / "engaged_genes.json.gz"
    if not cohort_path.is_file():
        pytest.skip(f"No generated cohort at {cohort_path}")
    genes = load_genes(cohort_path)

    fingerprinter = GeneFingerprinter()
    records = [fingerprinter.record(gene) for gene in genes]
    reference = load_reference_fingerprints(
        DEFAULT_REFERENCE_CACHE, DEFAULT_REFERENCE_SPLITS,
        fingerprint_cache=default_fingerprint_cache(
            DEFAULT_REFERENCE_CACHE, DEFAULT_REFERENCE_SPLITS))

    expected = [record_to_augmented_fingerprint(r) in reference for r in records]
    got = key_table.contains(gene_keys(records)).tolist()
    assert got == expected
