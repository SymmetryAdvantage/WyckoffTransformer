"""Tests for template-matching starts.

The index and the ranking are pure bookkeeping and are tested on a synthetic
frame.  The rebuild needs PyXtal symmetry detection, so it is exercised on a
structure this module builds itself -- rocksalt NaCl -- rather than on
LeMat-Bulk, which is not present in CI.
"""
import unittest

import numpy as np
import pandas as pd
import pytest

from wyckoff_transformer.cryspr.template import (
    INDEX_COLUMNS,
    TemplateIndex,
    TemplateQuery,
    anonymous_hash,
    composition_distance,
    composition_key,
    element_distance,
    gene_letters,
    letters_key,
    parse_composition,
    single_template,
    template_atoms,
)

NACL_GENE = {
    "group": 225,
    "species": ["Na", "Cl"],
    "numIons": [4, 4],
    "sites": [["4a"], ["4b"]],
}

#: The same orbits with a different chemistry, which is the case the method
#: exists for: KF is a template for NaCl, not a match for it.
KF_GENE = {
    "group": 225,
    "species": ["K", "F"],
    "numIons": [4, 4],
    "sites": [["4a"], ["4b"]],
}


def rocksalt(cation: str, anion: str, a: float = 5.6):
    """A rocksalt conventional cell: Fm-3m, the cation on 4a and the anion on 4b."""
    from pymatgen.core import Lattice, Structure

    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(a), [cation, anion],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def caesium_chloride(cation: str, anion: str, a: float = 4.1):
    """Pm-3m, 1a and 1b: the same two species on orbits NaCl's gene does not have."""
    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice.cubic(a), [cation, anion], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #
class TestKeys(unittest.TestCase):
    def test_anonymous_hash_is_stable_across_frozenset_order(self):
        first = (225, frozenset({frozenset({(("m-3m", 0), 1), (("-43m", 1), 1)})}))
        second = (225, frozenset({frozenset({(("-43m", 1), 1), (("m-3m", 0), 1)})}))
        self.assertEqual(anonymous_hash(first), anonymous_hash(second))

    def test_anonymous_hash_separates_different_orbits(self):
        first = (225, frozenset({frozenset({(("m-3m", 0), 1)})}))
        second = (225, frozenset({frozenset({(("m-3m", 0), 2)})}))
        self.assertNotEqual(anonymous_hash(first), anonymous_hash(second))

    def test_anonymous_hash_separates_space_groups(self):
        orbits = frozenset({frozenset({(("m-3m", 0), 1)})})
        self.assertNotEqual(anonymous_hash((225, orbits)), anonymous_hash((221, orbits)))

    def test_anonymous_hash_fits_in_uint64(self):
        value = anonymous_hash((225, frozenset({frozenset({(("m-3m", 0), 1)})})))
        self.assertIsInstance(value, int)
        self.assertTrue(0 <= value < 2**64)

    def test_letters_key_is_order_independent(self):
        self.assertEqual(letters_key(["d", "a", "d"]), letters_key(["d", "d", "a"]))
        self.assertNotEqual(letters_key(["a", "b"]), letters_key(["a", "a"]))

    def test_composition_key_round_trips(self):
        from collections import Counter

        formula = composition_key(Counter({"O": 8, "Rb": 3, "Ho": 1, "P": 2}))
        self.assertEqual(formula, "Ho1O8P2Rb3")
        self.assertEqual(
            parse_composition(formula), {"Ho": 1, "O": 8, "P": 2, "Rb": 3}
        )

    def test_gene_letters_reads_multiplicity_free_letters(self):
        self.assertEqual(gene_letters(NACL_GENE), ["a", "b"])
        self.assertEqual(
            gene_letters({"sites": [["12k", "2a"], ["6h"]]}), ["k", "a", "h"]
        )


# --------------------------------------------------------------------------- #
# Chemical distance
# --------------------------------------------------------------------------- #
class TestChemicalDistance(unittest.TestCase):
    def test_same_element_is_zero(self):
        self.assertEqual(element_distance("Fe", "Fe"), 0.0)

    def test_group_neighbours_are_closer_than_period_neighbours(self):
        # Na is one row below Li in group 1; Be is next to Li in period 2.
        self.assertLess(element_distance("Li", "Na"), element_distance("Li", "Be"))

    def test_distance_is_bounded(self):
        for first in ("H", "Cs", "F", "U"):
            for second in ("H", "Cs", "F", "U"):
                self.assertTrue(0.0 <= element_distance(first, second) <= 1.0)

    def test_unknown_element_is_maximally_distant(self):
        self.assertEqual(element_distance("Na", "Xx"), 1.0)

    def test_identical_compositions_are_zero(self):
        self.assertEqual(composition_distance({"Na": 4, "Cl": 4}, {"Cl": 4, "Na": 4}), 0.0)

    def test_partial_overlap_only_pays_for_the_difference(self):
        # Three of four atoms already agree, so at most a quarter of the cost.
        distance = composition_distance({"Na": 3, "K": 1}, {"Na": 3, "Rb": 1})
        self.assertGreater(distance, 0.0)
        self.assertLess(distance, 0.25)

    def test_chemically_closer_template_scores_lower(self):
        gene = {"Na": 4, "Cl": 4}
        close = composition_distance(gene, {"K": 4, "Br": 4})
        far = composition_distance(gene, {"U": 4, "Fe": 4})
        self.assertLess(close, far)

    def test_assignment_beats_the_naive_pairing(self):
        # Naively pairing in dict order would match Na->F and Cl->K; the optimal
        # assignment matches Na->K and Cl->F, which is far cheaper.
        distance = composition_distance({"Na": 1, "Cl": 1}, {"F": 1, "K": 1})
        self.assertLess(distance, 0.1)

    def test_different_atom_counts_are_rejected(self):
        with pytest.raises(ValueError, match="atoms"):
            composition_distance({"Na": 4}, {"Na": 8})


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #
def _index(rows) -> TemplateIndex:
    frame = pd.DataFrame(
        rows, columns=("immutable_id",) + INDEX_COLUMNS
    ).set_index("immutable_id")
    frame["anon_hash"] = frame["anon_hash"].astype("uint64")
    return TemplateIndex(frame)


class TestTemplateIndex(unittest.TestCase):
    def setUp(self):
        self.index = _index([
            ("exact", 7, "a b", "Cl4Na4", 0.0),
            ("close", 7, "a b", "Br4K4", 0.0),
            ("far", 7, "a b", "Fe4U4", 0.0),
            ("stabler-duplicate", 7, "a b", "Cl4Na4", -0.01),
            ("other-orbits", 7, "a a", "Cl4Na4", 0.0),
            ("other-fingerprint", 9, "a b", "Cl4Na4", 0.0),
        ])
        self.query = TemplateQuery(anon_hash=7, letters="a b",
                                   composition={"Na": 4, "Cl": 4})

    def test_missing_column_is_rejected(self):
        with pytest.raises(ValueError, match="missing column"):
            TemplateIndex(pd.DataFrame({"anon_hash": [1]}))

    def test_candidates_need_both_the_fingerprint_and_the_orbits(self):
        ids = set(self.index.candidates(self.query).index)
        self.assertEqual(ids, {"exact", "close", "far", "stabler-duplicate"})

    def test_unknown_fingerprint_has_no_candidates(self):
        query = TemplateQuery(anon_hash=999, letters="a b", composition={"Na": 8})
        self.assertEqual(len(self.index.candidates(query)), 0)
        self.assertEqual(self.index.select(query), [])

    def test_ranking_is_by_formula_then_hull_distance(self):
        ranked = [match.immutable_id for match in self.index.select(self.query, k=4)]
        self.assertEqual(ranked[:2], ["stabler-duplicate", "exact"])
        self.assertEqual(ranked[2:], ["close", "far"])

    def test_exclusion_removes_the_named_entries(self):
        ranked = [
            match.immutable_id
            for match in self.index.select(
                self.query, exclude=["exact", "stabler-duplicate"], k=4
            )
        ]
        self.assertEqual(ranked, ["close", "far"])

    def test_excluding_everything_yields_no_match(self):
        self.assertEqual(
            self.index.select(
                self.query,
                exclude=["exact", "close", "far", "stabler-duplicate"],
            ),
            [],
        )

    def test_k_bounds_the_result(self):
        self.assertEqual(len(self.index.select(self.query, k=2)), 2)

    def test_distance_is_reported(self):
        by_id = {m.immutable_id: m for m in self.index.select(self.query, k=4)}
        self.assertEqual(by_id["exact"].distance, 0.0)
        self.assertGreater(by_id["far"].distance, by_id["close"].distance)

    def test_prefilter_keeps_the_best_shared_atom_counts(self):
        # With room for one candidate only, the one sharing every atom survives.
        ranked = self.index.select(self.query, k=1, prefilter=1)
        self.assertEqual(ranked[0].composition, "Cl4Na4")


# --------------------------------------------------------------------------- #
# The rebuild
# --------------------------------------------------------------------------- #
class TestTemplateAtoms(unittest.TestCase):
    def test_rebuild_puts_the_genes_elements_on_the_templates_geometry(self):
        atoms = template_atoms(NACL_GENE, rocksalt("K", "F", a=5.6))
        self.assertEqual(
            dict(zip(*np.unique(atoms.get_chemical_symbols(), return_counts=True))),
            {"Cl": 4, "Na": 4},
        )
        # The cell is the template's, untouched.
        self.assertAlmostEqual(atoms.get_volume(), 5.6**3, places=6)

    def test_assignment_follows_chemistry_not_gene_order(self):
        # The template has K on 4a and F on 4b.  Na is the alkali metal, so it
        # takes 4a even though the gene lists Na first and F is the 4b species.
        atoms = template_atoms(NACL_GENE, rocksalt("K", "F", a=5.6))
        scaled = atoms.get_scaled_positions()
        origin = [
            symbol
            for symbol, position in zip(atoms.get_chemical_symbols(), scaled)
            if np.allclose(position, 0.0, atol=1e-6)
        ]
        self.assertEqual(set(origin), {"Na"})

    def test_identical_chemistry_is_a_no_op_on_the_species(self):
        atoms = template_atoms(NACL_GENE, rocksalt("Na", "Cl", a=5.6))
        self.assertEqual(sorted(set(atoms.get_chemical_symbols())), ["Cl", "Na"])

    def test_a_template_on_other_orbits_is_rejected(self):
        gene = dict(NACL_GENE, sites=[["4a"], ["8c"]], numIons=[4, 8],
                    species=["Na", "Cl"])
        with pytest.raises(ValueError, match="no symmetry tolerance"):
            template_atoms(gene, rocksalt("K", "F"))

    def test_a_template_in_another_space_group_is_rejected(self):
        with pytest.raises(ValueError, match="no symmetry tolerance"):
            template_atoms(NACL_GENE, caesium_chloride("K", "F"))


class TestSingleTemplate(unittest.TestCase):
    def _match(self, immutable_id):
        from wyckoff_transformer.cryspr.template import TemplateMatch

        return TemplateMatch(immutable_id, "F4K4", 0.05, 0.0)

    def test_no_candidate_reports_why(self):
        atoms, match, error = single_template(NACL_GENE, [], {})
        self.assertIsNone(atoms)
        self.assertIsNone(match)
        self.assertIn("fingerprint", error)

    def test_first_usable_candidate_is_taken(self):
        structures = {"bad": caesium_chloride("K", "F"),  # wrong space group
                      "good": rocksalt("K", "F")}
        atoms, match, error = single_template(
            NACL_GENE, [self._match("bad"), self._match("good")], structures
        )
        self.assertIsNotNone(atoms)
        self.assertEqual(match.immutable_id, "good")
        self.assertIsNone(error)

    def test_all_candidates_unusable_reports_the_last_reason(self):
        structures = {"bad": caesium_chloride("K", "F")}
        atoms, match, error = single_template(
            NACL_GENE, [self._match("bad")], structures
        )
        self.assertIsNone(atoms)
        self.assertIsNone(match)
        self.assertIn("bad", error)

    def test_a_candidate_with_no_geometry_is_skipped(self):
        atoms, match, error = single_template(
            NACL_GENE, [self._match("absent")], {}
        )
        self.assertIsNone(atoms)
        self.assertIn("could be read", error)


if __name__ == "__main__":
    unittest.main()
