"""Tests for the tensor-space gene key, its table, and the two checkers built on it.

The key claims to be *exactly* equivalent to
:func:`~wyckoff_transformer.evaluation.novelty.record_to_augmented_fingerprint`,
not a cheap approximation of it, and everything downstream -- gene novelty,
cohort uniqueness -- rests on that.  So the equivalence is tested as an identity
of the *partitions* the two induce over a few thousand random genes, not as a
handful of spot checks: two genes must share a key exactly when they share a
fingerprint, and a single false merge anywhere in the sample fails the test.
"""
from __future__ import annotations

import random
import subprocess
import sys

import pytest
import torch

from wyckoff_transformer.evaluation.gene_hash import (
    GENE_HASH_VERSION,
    GeneKeyTable,
    gene_key,
    gene_keys,
    unique_representatives,
)
from wyckoff_transformer.evaluation.novelty import record_to_augmented_fingerprint
from wyckoff_transformer.evaluation.protocol import GeneFingerprinter, screen_genes
from wyckoff_transformer.tokenization import load_wyckoff_mappings

#: Enough elements to make collisions between random genes rare and repeats common.
ELEMENTS = ["H", "Li", "Be", "C", "N", "O", "Na", "Mg", "Al", "Si", "P", "S",
            "Cl", "K", "Ca", "Ti", "Fe", "Cu", "Zn", "Ga", "Sr", "Y", "Zr", "Ba"]

NACL = {"group": 225, "species": ["Na", "Cl"], "numIons": [4, 4], "sites": [["4a"], ["4b"]]}
NACL_REORDERED = {
    "group": 225, "species": ["Cl", "Na"], "numIons": [4, 4], "sites": [["4b"], ["4a"]],
}
PEROVSKITE = {"group": 221, "species": ["Sr", "Ti", "O"], "numIons": [1, 1, 3],
              "sites": [["1a"], ["1b"], ["3c"]]}


@pytest.fixture(scope="module")
def fingerprinter():
    return GeneFingerprinter()


@pytest.fixture(scope="module")
def letters_by_space_group():
    return load_wyckoff_mappings().ss_from_letter


def random_gene(rng: random.Random, letters_by_space_group) -> dict:
    """A random formally legal gene.

    The multiplicity written into each site string is a placeholder: neither the
    fingerprint nor the key reads it (only the AFLOW variants do), and the point
    here is to exercise the (space group, element, site symmetry, enumeration)
    structure that both of them *do* read.
    """
    space_group = rng.randrange(1, 231)
    letters = sorted(letters_by_space_group[space_group])
    n_sites = rng.randint(1, min(4, len(letters)))
    by_species: dict[str, list[str]] = {}
    for _ in range(n_sites):
        # Letters may repeat: two occupied orbits of the same kind must not
        # collapse into one, which is the difference between a multiset and a set.
        letter = rng.choice(letters)
        element = rng.choice(ELEMENTS[: rng.randint(3, len(ELEMENTS))])
        by_species.setdefault(element, []).append(f"1{letter}")
    return {
        "group": space_group,
        "species": list(by_species),
        "numIons": [len(sites) for sites in by_species.values()],
        "sites": list(by_species.values()),
    }


@pytest.fixture(scope="module")
def random_records_with_genes(fingerprinter, letters_by_space_group):
    """(gene, record) pairs -- the gene is needed for the letter-orbit truth."""
    rng = random.Random(20260922)
    pairs = []
    while len(pairs) < 3000:
        gene = random_gene(rng, letters_by_space_group)
        try:
            pairs.append((gene, fingerprinter.record(gene)))
        except Exception:  # noqa: BLE001 - an illegal draw is not what is under test
            continue
    return pairs


@pytest.fixture(scope="module")
def random_records(random_records_with_genes):
    return [record for _, record in random_records_with_genes]


# --------------------------------------------------------------------------- #
# The equivalence claim
# --------------------------------------------------------------------------- #
def test_the_key_induces_the_same_partition_as_the_fingerprint(random_records):
    """Two genes share a key exactly when they share a fingerprint."""
    by_fingerprint: dict = {}
    by_key: dict = {}
    for index, record in enumerate(random_records):
        by_fingerprint.setdefault(record_to_augmented_fingerprint(record), []).append(index)
        by_key.setdefault(gene_key(record), []).append(index)

    assert len(by_key) == len(by_fingerprint), (
        f"{len(by_fingerprint)} fingerprint classes against {len(by_key)} key classes: "
        "the key merges or splits genes the fingerprint does not")
    assert sorted(by_key.values()) == sorted(by_fingerprint.values()), (
        "the two partitions have the same size but not the same members")


def test_the_sample_actually_contains_collisions_to_detect(random_records):
    """A partition test over genes that are all distinct would prove nothing."""
    classes = {record_to_augmented_fingerprint(record) for record in random_records}
    assert len(classes) < len(random_records), (
        "the random sample has no repeated genes, so the equivalence test above is "
        "not exercising the merge half of the claim")


def test_distinct_genes_do_not_collide(random_records):
    fingerprints = {record_to_augmented_fingerprint(r) for r in random_records}
    keys = {gene_key(r) for r in random_records}
    assert len(keys) == len(fingerprints)


# --------------------------------------------------------------------------- #
# The invariances the key must have
# --------------------------------------------------------------------------- #
def test_site_order_does_not_change_the_key(fingerprinter):
    assert gene_key(fingerprinter.record(NACL)) == gene_key(fingerprinter.record(NACL_REORDERED))


def test_the_key_survives_every_spelling_of_an_element(fingerprinter):
    from pymatgen.core import Element

    record = fingerprinter.record(NACL)
    baseline = gene_key(record)
    for rewrite in (str, repr, lambda e: Element(str(e))):
        variant = dict(record)
        variant["elements"] = [rewrite(element) for element in record["elements"]]
        assert gene_key(variant) == baseline, (
            f"{rewrite} changed the key; a record and a restored processor spell "
            "elements differently and must still agree")


def test_the_key_is_stable_across_processes():
    """No Python ``hash()`` anywhere in the encoding.

    A key cached in one process and queried in another must match, and Python
    randomises string and frozenset hashing per process by default.
    """
    program = (
        "from wyckoff_transformer.evaluation.protocol import GeneFingerprinter;"
        "from wyckoff_transformer.evaluation.gene_hash import gene_key;"
        f"print(gene_key(GeneFingerprinter().record({NACL!r})))"
    )
    outputs = set()
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}, check=True)
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"the key moved with PYTHONHASHSEED: {outputs}"


# --------------------------------------------------------------------------- #
# The distinctions the key must keep
# --------------------------------------------------------------------------- #
def test_the_space_group_is_part_of_the_key(fingerprinter):
    record = fingerprinter.record(NACL)
    other = dict(record, spacegroup_number=216)
    assert gene_key(record) != gene_key(other)


def test_the_elements_are_part_of_the_key(fingerprinter):
    swapped = {"group": 225, "species": ["K", "Cl"], "numIons": [4, 4],
               "sites": [["4a"], ["4b"]]}
    assert gene_key(fingerprinter.record(NACL)) != gene_key(fingerprinter.record(swapped))


def test_which_element_sits_on_which_orbit_is_part_of_the_key(fingerprinter):
    """NaCl with the two species exchanged between 4a and 4b is a different gene.

    It is also the case the fingerprint's augmentation makes subtle: in space
    group 225 the alternative settings may relabel a and b, so the two are only
    distinguishable if the augmentation orbit says so -- and whatever the
    fingerprint answers, the key must answer identically.
    """
    exchanged = {"group": 225, "species": ["Na", "Cl"], "numIons": [4, 4],
                 "sites": [["4b"], ["4a"]]}
    same_fingerprint = (
        record_to_augmented_fingerprint(fingerprinter.record(NACL))
        == record_to_augmented_fingerprint(fingerprinter.record(exchanged)))
    same_key = gene_key(fingerprinter.record(NACL)) == gene_key(fingerprinter.record(exchanged))
    assert same_key == same_fingerprint


def test_a_repeated_orbit_is_not_collapsed(fingerprinter):
    """Two sites of the same kind differ from one: a multiset, not a set."""
    once = {"group": 225, "species": ["Na"], "numIons": [4], "sites": [["4a"]]}
    twice = {"group": 225, "species": ["Na"], "numIons": [8], "sites": [["4a", "4a"]]}
    assert gene_key(fingerprinter.record(once)) != gene_key(fingerprinter.record(twice))
    assert (record_to_augmented_fingerprint(fingerprinter.record(once))
            != record_to_augmented_fingerprint(fingerprinter.record(twice)))


def test_a_different_gene_is_a_different_key(fingerprinter):
    assert gene_key(fingerprinter.record(NACL)) != gene_key(fingerprinter.record(PEROVSKITE))


# --------------------------------------------------------------------------- #
# Uniqueness
# --------------------------------------------------------------------------- #
def test_uniqueness_agrees_with_the_python_screen(random_records, fingerprinter,
                                                  letters_by_space_group):
    """The tensor checker must pick the same representatives as ``screen_genes``."""
    rng = random.Random(4242)
    genes, records = [], []
    while len(genes) < 400:
        gene = random_gene(rng, letters_by_space_group)
        try:
            record = fingerprinter.record(gene)
        except Exception:  # noqa: BLE001
            continue
        genes.append(gene)
        records.append(record)
    # Seed the pool with duplicates so uniqueness has something to do.
    genes += genes[:120]
    records += records[:120]

    screen = screen_genes(genes, reference=set(), fingerprinter=fingerprinter)
    representatives, counts = unique_representatives(gene_keys(records))

    assert representatives.tolist() == sorted(screen.counts)
    assert counts.tolist() == [screen.counts[i] for i in sorted(screen.counts)]
    assert int(counts.sum()) == len(genes), "every draw must be behind exactly one representative"


def test_uniqueness_on_an_empty_cohort():
    representatives, counts = unique_representatives(torch.empty((0, 2), dtype=torch.int64))
    assert representatives.numel() == 0 and counts.numel() == 0


def test_uniqueness_keeps_the_first_occurrence(fingerprinter):
    records = [fingerprinter.record(g) for g in (PEROVSKITE, NACL, NACL_REORDERED, PEROVSKITE)]
    representatives, counts = unique_representatives(gene_keys(records))
    assert representatives.tolist() == [0, 1]
    assert counts.tolist() == [2, 2]


# --------------------------------------------------------------------------- #
# Novelty
# --------------------------------------------------------------------------- #
def test_membership_agrees_with_the_python_set(random_records):
    """``contains`` must answer exactly as ``fingerprint in reference`` does."""
    rng = random.Random(7)
    held = [record for record in random_records if rng.random() < 0.4]
    reference = {record_to_augmented_fingerprint(record) for record in held}
    table = GeneKeyTable.from_keys(gene_keys(held))

    expected = [record_to_augmented_fingerprint(r) in reference for r in random_records]
    got = table.contains(gene_keys(random_records)).tolist()
    assert got == expected


def test_an_empty_table_holds_nothing(fingerprinter):
    table = GeneKeyTable.from_keys(torch.empty((0, 2), dtype=torch.int64))
    assert len(table) == 0
    assert table.contains(gene_keys([fingerprinter.record(NACL)])).tolist() == [False]


def test_a_query_below_and_above_every_key_is_not_a_hit():
    """``searchsorted`` puts an out-of-range query at 0 or at N; neither may hit."""
    table = GeneKeyTable.from_keys(torch.tensor([[5, 1], [9, 2]], dtype=torch.int64))
    probes = torch.tensor([[-(2 ** 62), 0], [2 ** 62, 0], [7, 1]], dtype=torch.int64)
    assert table.contains(probes).tolist() == [False, False, False]


def test_a_matching_low_word_with_a_different_high_word_is_not_a_hit():
    """The second word is the whole reason a 64-bit collision cannot be a false positive."""
    table = GeneKeyTable.from_keys(torch.tensor([[5, 1]], dtype=torch.int64))
    assert table.contains(torch.tensor([[5, 999]], dtype=torch.int64)).tolist() == [False]
    assert table.contains(torch.tensor([[5, 1]], dtype=torch.int64)).tolist() == [True]


def test_a_duplicated_low_word_is_refused_rather_than_hidden():
    with pytest.raises(RuntimeError, match="low word"):
        GeneKeyTable.from_keys(torch.tensor([[5, 1], [5, 2]], dtype=torch.int64))


def test_the_table_deduplicates_its_own_keys():
    table = GeneKeyTable.from_keys(torch.tensor([[5, 1], [5, 1], [9, 2]], dtype=torch.int64))
    assert len(table) == 2


def test_the_table_survives_a_round_trip(tmp_path, random_records):
    table = GeneKeyTable.from_keys(gene_keys(random_records), meta={"splits": ["train"]})
    reloaded = GeneKeyTable.load(table.save(tmp_path / "keys.npz"))
    assert len(reloaded) == len(table)
    assert torch.equal(reloaded.low, table.low) and torch.equal(reloaded.high, table.high)
    assert reloaded.meta["splits"] == ["train"]
    assert reloaded.contains(gene_keys(random_records)).all()


def test_a_table_from_another_encoding_is_refused(tmp_path, monkeypatch):
    import numpy as np

    path = tmp_path / "old.npz"
    np.savez_compressed(path, low=np.array([1]), high=np.array([2]),
                        version=np.array(GENE_HASH_VERSION + 1), meta=np.array("{}"))
    with pytest.raises(ValueError, match="Rebuild it"):
        GeneKeyTable.load(path)


def test_keys_must_be_two_columns():
    table = GeneKeyTable.from_keys(torch.tensor([[5, 1]], dtype=torch.int64))
    with pytest.raises(ValueError, match=r"\[N, 2\]"):
        table.contains(torch.tensor([5], dtype=torch.int64))


def test_membership_is_exact_on_a_pool_with_no_overlap(random_records, fingerprinter,
                                                       letters_by_space_group):
    """A reference that shares nothing with the query must return all False."""
    rng = random.Random(99)
    held, seen = [], {record_to_augmented_fingerprint(r) for r in random_records}
    while len(held) < 300:
        try:
            record = fingerprinter.record(random_gene(rng, letters_by_space_group))
        except Exception:  # noqa: BLE001
            continue
        if record_to_augmented_fingerprint(record) not in seen:
            held.append(record)
    table = GeneKeyTable.from_keys(gene_keys(held))
    assert not table.contains(gene_keys(random_records)).any()


# --------------------------------------------------------------------------- #
# The premise underneath any orbit argument
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def relabellings():
    from wyckoff_transformer.preprocess_wychoffs import get_augmentation_dict

    return get_augmentation_dict()


def test_the_wyckoff_relabellings_are_a_group(relabellings):
    """Identity, closure and inverses, in all 230 space groups.

    It holds today and it is a theorem -- the maps are the image of a
    homomorphism from the space group's normaliser quotient into the symmetric
    group on its Wyckoff positions, and the image of a homomorphism is a
    subgroup.  The guard is here because the maps come from an *undocumented*
    static table in PyXtal (`database/wyckoff_sets.json`, returned verbatim by
    `Group.get_alternatives`), so a future release could change them with
    nothing to notice.

    The group property is necessary and *not sufficient* for canonicalising a
    gene by the minimum over its orbit: that additionally needs the
    representation to be equivariant, and ours is not.  See
    ``docs/wyckoff_augmentation_audit.md``.
    """
    for space_group, maps in relabellings.items():
        letters = sorted(maps[0])
        keyed = {tuple(m[letter] for letter in letters) for m in maps}

        assert tuple(letters) in keyed, f"sg {space_group}: no identity"
        for image in keyed:
            assert sorted(image) == letters, f"sg {space_group}: {image} is not a permutation"

        index = {letter: position for position, letter in enumerate(letters)}
        for first in keyed:
            inverse = [None] * len(letters)
            for position, letter in enumerate(first):
                inverse[index[letter]] = letters[position]
            assert tuple(inverse) in keyed, f"sg {space_group}: {first} has no inverse"
            for second in keyed:
                composed = tuple(second[index[letter]] for letter in first)
                assert composed in keyed, (
                    f"sg {space_group}: {first} composed with {second} leaves the set")


def test_the_relabellings_do_not_preserve_the_oriented_site_symmetry_symbol(relabellings):
    """The reason min-over-orbit is not available, pinned as a fact.

    In 26 orthorhombic space groups an axis-permuting relabelling sends a letter
    to one with a different oriented site-symmetry symbol -- ``2..`` to ``.2.``.
    International Tables A1 makes the same point with ``I222`` (space group 23),
    whose ``4e,4f`` / ``4g,4h`` / ``4i,4j`` form one Wyckoff set on differently
    oriented axes.  Conjugation preserves the site-symmetry group up to
    conjugacy, which is its point-group type, not the oriented symbol.

    If this test ever fails, the audit's conclusion should be revisited: either
    the table changed, or ``pyxtal_notation_to_sites`` was fixed to relabel the
    symbol too, in which case this expectation is what is now stale.
    """
    from wyckoff_transformer.tokenization import load_wyckoff_mappings

    ss_from_letter = load_wyckoff_mappings().ss_from_letter
    affected = set()
    for space_group, maps in relabellings.items():
        symbols = ss_from_letter[space_group]
        for relabel in maps:
            for letter, symbol in symbols.items():
                image = relabel.get(letter)
                if image in symbols and symbols[image] != symbol:
                    affected.add(space_group)
    assert affected == {16, 17, 20, 21, 22, 23, 24, 25, 35, 42, 44, 47, 48, 49, 50,
                        59, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74}


def _letter_orbit(gene, relabellings):
    """The ground truth: which genes a relabelling actually relates.

    Built from the Wyckoff **letters**, which is what the relabellings act on,
    and therefore independent of how a record chooses to encode a position.
    Since the relabellings form a group this partitions the genes correctly.
    """
    from collections import Counter

    pairs = [(element, site[-1])
             for element, sites in zip(gene["species"], gene["sites"])
             for site in sites]
    return (gene["group"], frozenset(
        frozenset(Counter((element, relabel[letter]) for element, letter in pairs).items())
        for relabel in relabellings[gene["group"]]))


def test_the_fingerprint_separates_genes_no_relabelling_relates(fingerprinter,
                                                                relabellings):
    """Space group 68: the false merge that the audit found, now absent.

    ``Ccce``.  Every relabelling that swaps ``c`` and ``d`` also moves ``e`` to
    ``f``, so no map takes A to B.  Before the augmentation carried the oriented
    site-symmetry symbol alongside the enumeration index, they shared a
    fingerprint: ``f``'s index within ``.2.`` is 0 just as ``e``'s within ``2..``
    is, and the record kept ``e``'s symbol while taking ``f``'s index.
    """
    a = {"group": 68, "species": ["Li", "Be"], "numIons": [1, 1],
         "sites": [["1e"], ["1c"]]}
    b = {"group": 68, "species": ["Li", "Be"], "numIons": [1, 1],
         "sites": [["1e"], ["1d"]]}
    assert _letter_orbit(a, relabellings) != _letter_orbit(b, relabellings), (
        "no relabelling relates these two genes")
    assert (record_to_augmented_fingerprint(fingerprinter.record(a))
            != record_to_augmented_fingerprint(fingerprinter.record(b)))
    assert gene_key(fingerprinter.record(a)) != gene_key(fingerprinter.record(b))


def test_the_fingerprint_merges_genes_an_axis_permutation_relates(fingerprinter,
                                                                  relabellings):
    """Space group 16: the missed merge, now caught.

    ``P222``.  ``1t`` sits on ``..2`` and ``1o`` on ``.2.``; the axis-permuting
    alternative setting maps one to the other, so they are one gene.  Before the
    fix their recorded symbols differed and the fingerprint kept them apart.
    """
    t = {"group": 16, "species": ["Be"], "numIons": [1], "sites": [["1t"]]}
    o = {"group": 16, "species": ["Be"], "numIons": [1], "sites": [["1o"]]}
    assert _letter_orbit(t, relabellings) == _letter_orbit(o, relabellings), (
        "an axis permutation does relate these two genes")
    assert (record_to_augmented_fingerprint(fingerprinter.record(t))
            == record_to_augmented_fingerprint(fingerprinter.record(o)))
    assert gene_key(fingerprinter.record(t)) == gene_key(fingerprinter.record(o))


def test_the_fingerprint_agrees_with_the_letter_orbit_truth(random_records_with_genes,
                                                            relabellings):
    """The test that would have caught the defect: fingerprint == letter orbit.

    Both partition the same genes; they must induce the *same* partition.  The
    old fingerprint failed this in the 26 space groups where a relabelling
    changes the oriented site-symmetry symbol, in both directions -- merging
    genes no map relates, and splitting genes a map does relate.
    """
    by_fingerprint, by_truth = {}, {}
    for index, (gene, record) in enumerate(random_records_with_genes):
        by_fingerprint.setdefault(record_to_augmented_fingerprint(record), []).append(index)
        by_truth.setdefault(_letter_orbit(gene, relabellings), []).append(index)
    assert sorted(by_fingerprint.values()) == sorted(by_truth.values()), (
        f"{len(by_truth)} true classes against {len(by_fingerprint)} fingerprint classes")


def test_the_key_agrees_with_the_letter_orbit_truth(random_records_with_genes,
                                                    relabellings):
    """And so must the tensor key, independently of the fingerprint."""
    by_key, by_truth = {}, {}
    for index, (gene, record) in enumerate(random_records_with_genes):
        by_key.setdefault(gene_key(record), []).append(index)
        by_truth.setdefault(_letter_orbit(gene, relabellings), []).append(index)
    assert sorted(by_key.values()) == sorted(by_truth.values())


def test_the_augmented_columns_stay_aligned(fingerprinter):
    """One symmetry tuple per enumeration tuple, each the length of the gene."""
    gene = {"group": 47, "species": ["Ba", "Ti", "O"], "numIons": [1, 1, 3],
            "sites": [["1a"], ["1b"], ["1c", "1d", "1e"]]}
    record = fingerprinter.record(gene)
    symmetries = record["site_symmetries_augmented"]
    enumerations = record["sites_enumeration_augmented"]
    assert len(symmetries) == len(enumerations) > 1, "space group 47 has 48 relabellings"
    for symmetry, enumeration in zip(symmetries, enumerations):
        assert len(symmetry) == len(enumeration) == len(record["site_symmetries"])


def test_the_augmentation_order_does_not_depend_on_the_hash_seed():
    """The variants are sorted, so a cache built twice is the same cache.

    They are derived from a ``frozenset`` of relabellings, whose iteration order
    moves with ``PYTHONHASHSEED``; before they were sorted, the *set* was stable
    but the order was not, and the order is what a paired representation and a
    tokenised cache are indexed by.
    """
    import subprocess
    import sys

    program = (
        "from wyckoff_transformer.evaluation.protocol import GeneFingerprinter;"
        "r = GeneFingerprinter().record({'group': 47, 'species': ['Ba', 'O'],"
        "'numIons': [1, 1], 'sites': [['1a'], ['1c']]});"
        "print(r['site_symmetries_augmented'], r['sites_enumeration_augmented'])"
    )
    outputs = set()
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}, check=True)
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"the augmentation order moved with the hash seed: {outputs}"


def test_a_record_without_the_paired_symmetries_is_refused(fingerprinter):
    """An unmigrated cache must fail loudly, not be mis-fingerprinted quietly."""
    record = dict(fingerprinter.record(
        {"group": 68, "species": ["Li"], "numIons": [1], "sites": [["1e"]]}))
    del record["site_symmetries_augmented"]
    for compute in (record_to_augmented_fingerprint, gene_key):
        with pytest.raises(KeyError, match="site_symmetries_augmented"):
            compute(record)


def test_a_structure_record_agrees_with_the_mappings_table(relabellings):
    """PyXtal's oriented symbol and the package table must name positions alike.

    ``structure_to_sites`` takes ``site_symmetries`` from PyXtal
    (``wp.site_symm``) while the augmented variants come from the package's
    ``ss_from_letter``.  The protocol compares a *sampled gene's* fingerprint
    with a *relaxed structure's* against one reference, so if the two
    conventions ever diverged, novelty would silently stop matching -- and since
    the fingerprint is now built from the table alone, the divergence would be
    invisible in the fingerprint itself.
    """
    from pymatgen.core import Lattice, Structure

    from wyckoff_transformer.data import structure_to_sites
    from wyckoff_transformer.tokenization import load_wyckoff_mappings

    mappings = load_wyckoff_mappings()
    structure = Structure.from_spacegroup(
        "Pm-3m", Lattice.cubic(4.0), ["Sr", "Ti", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]])
    record = structure_to_sites(
        structure, mappings.enum_from_ss_letter, relabellings)

    from_table = tuple(
        mappings.ss_from_letter[record["spacegroup_number"]][letter]
        for letter in record["wyckoff_letters"])
    assert tuple(record["site_symmetries"]) == from_table, (
        "pyxtal's site_symm and ss_from_letter disagree; gene and structure "
        "fingerprints would stop matching")
    assert from_table in record["site_symmetries_augmented"], (
        "the identity relabelling must be among the variants")
