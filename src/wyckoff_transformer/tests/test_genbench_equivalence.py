"""Pin our reimplementations against LeMat-GenBench's own.

These are the tests that justify not importing LeMat-GenBench.  Each one runs
our implementation and the benchmark's over the same structures and asserts they
agree exactly.  They are the only place that project is referenced, and they
skip when it is not available -- so they gate the port without making it a
dependency.

Point them at a checkout with::

    LEMAT_GENBENCH_PATH=/path/to/lemat-genbench uv run pytest -k equivalence

Structures come from a previous CrySPR run when one is present, so the
comparison covers real generated output -- collapsed cells, odd stoichiometries
and all -- rather than only textbook crystals.
"""
import logging
import os
import sys
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

# How many CIFs from a real run to compare.  Charge-neutrality guessing is slow,
# so this trades coverage against a test suite that still finishes.
N_REAL_STRUCTURES = 12

FALLBACK_STRUCTURES = [
    Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    Structure(Lattice.cubic(3.61), ["Cu"], [[0, 0, 0]]),
    Structure(
        Lattice.cubic(3.905), ["Sr", "Ti", "O", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    ),
]


def _genbench_root() -> Path | None:
    for candidate in (
        os.environ.get("LEMAT_GENBENCH_PATH"),
        "/home/kna/sun-forest/external/lemat-genbench",
        "../lemat-genbench",
    ):
        if candidate and (Path(candidate) / "src").is_dir():
            return Path(candidate)
    return None


@pytest.fixture(scope="module")
def genbench():
    """Import LeMat-GenBench, or skip the whole module."""
    root = _genbench_root()
    if root is None:
        pytest.skip("No LeMat-GenBench checkout; set LEMAT_GENBENCH_PATH")
    sys.path.insert(0, str(root / "src"))
    # lemat_genbench's __init__ eagerly imports its CLI and every benchmark,
    # reaching material_hasher.similarity, which tries to import fairchem. When
    # fairchem is absent that ImportError is caught, but the handler calls
    # logging.warning(msg, arg) with no format placeholder in msg and raises
    # TypeError. Silencing logging for the duration keeps the record from being
    # emitted, so the intended except-branch completes.
    logging.disable(logging.CRITICAL)
    try:
        import lemat_genbench  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"LeMat-GenBench not importable: {exc}")
    finally:
        logging.disable(logging.NOTSET)
    return root


@pytest.fixture(scope="module")
def structures():
    """Relaxed CIFs from a real run, falling back to reference crystals."""
    found = []
    for run in (Path("generated/upi73i4k/final_cifs"),):
        if run.is_dir():
            for path in sorted(run.glob("*.cif"))[:N_REAL_STRUCTURES]:
                try:
                    found.append(Structure.from_file(path))
                except Exception:
                    continue
    return found or FALLBACK_STRUCTURES


class TestBawlEquivalence:
    """Our BAWL hashes must equal material-hasher's, string for string."""

    @staticmethod
    def _reference(shorten):
        pytest.importorskip("material_hasher")
        from material_hasher.hasher.bawl import BAWLHasher
        from pymatgen.analysis.local_env import EconNN

        return BAWLHasher(
            graphing_algorithm="WL",
            bonding_algorithm=EconNN,
            bonding_kwargs={"tol": 0.2, "cutoff": 10, "use_fictive_radius": True},
            include_composition=True,
            symmetry_labeling="SPGLib",
            shorten_hash=shorten,
        )

    @pytest.mark.parametrize("shorten", [True, False])
    def test_fingerprints_match(self, genbench, structures, shorten):
        from wyckoff_transformer.evaluation.bawl import BawlFingerprinter

        ours = BawlFingerprinter(shorten=shorten)
        theirs = self._reference(shorten)
        for structure in structures:
            assert ours(structure) == theirs.get_material_hash(structure), (
                f"BAWL mismatch for {structure.composition.reduced_formula}"
            )


class TestValidityEquivalence:
    """Our validity verdict must match OverallValidityMetric's."""

    def test_overall_validity_matches(self, genbench, structures):
        from lemat_genbench.metrics.validity_metrics import OverallValidityMetric

        from wyckoff_transformer.evaluation.structure_validity import is_valid

        for structure in structures:
            # compute_structure mutates structure.properties, so give each
            # implementation its own copy.
            theirs = OverallValidityMetric.compute_structure(structure.copy()) >= 1.0
            assert is_valid(structure.copy()) == theirs, (
                f"validity mismatch for {structure.composition.reduced_formula}"
            )

    def test_charge_deviation_matches(self, genbench, structures):
        from lemat_genbench.metrics.validity_metrics import ChargeNeutralityMetric
        from pymatgen.analysis.bond_valence import BVAnalyzer

        from wyckoff_transformer.evaluation.structure_validity import charge_deviation

        for structure in structures:
            theirs = ChargeNeutralityMetric.compute_structure(
                structure.copy(), tolerance=0.1, strict=False, bv_analyzer=BVAnalyzer()
            )
            assert charge_deviation(structure.copy()) == pytest.approx(theirs), (
                f"charge deviation mismatch for {structure.composition.reduced_formula}"
            )


class TestHullEnergyEquivalence:
    """Our e_above_hull must match get_energy_above_hull to numerical noise."""

    MLIP = "orb_conserv_inf"

    def test_energy_above_hull_matches(self, genbench, structures):
        from lemat_genbench.preprocess.reference_energies import get_energy_above_hull

        from wyckoff_transformer.evaluation.hull_energy import HullEnergyCalculator

        ours = HullEnergyCalculator(self.MLIP)
        checked = 0
        for structure in structures:
            # The energy value is arbitrary: both sides receive the same one,
            # and we are testing the hull construction, not the potential.
            energy = -5.0 * len(structure)
            try:
                theirs = get_energy_above_hull(
                    energy, structure.composition, hull_type=self.MLIP
                )
            except Exception:
                continue  # composition outside the hull for both; nothing to compare
            assert ours.energy_above_hull(energy, structure.composition) == pytest.approx(
                theirs, abs=1e-9
            ), f"e_above_hull mismatch for {structure.composition.reduced_formula}"
            checked += 1
        assert checked, "no composition was inside the hull; the test proved nothing"

    def test_subspace_selection_matches(self, genbench, structures):
        """The compositional filter is the only part we reimplemented.

        Note there are two incompatible ``filter_df``/``one_hot_encode_composition``
        pairs in LeMat-GenBench: ``preprocess.reference_energies`` encodes into
        119 slots by direct atomic number, and ``fingerprinting.encode_compositions``
        into 118 by ``Z - 1``.  Only the former matches the stored composition
        matrices -- the latter raises a shape error against them.  The hull path
        uses the former, so that is what we compare against.  We keep our own
        118-wide encoding, which is self-consistent because we build the matrix
        as well as the query vector.
        """
        from lemat_genbench.preprocess.reference_energies import (
            _retrieve_df,
            _retrieve_matrix,
            filter_df,
        )

        from wyckoff_transformer.evaluation.hull_energy import HullEnergyCalculator

        ours = HullEnergyCalculator(self.MLIP)
        reference_df = _retrieve_df(self.MLIP, 0.001)
        reference_matrix = _retrieve_matrix(self.MLIP, 0.001)
        for structure in structures:
            theirs = filter_df(reference_df, reference_matrix, structure.composition)
            assert len(ours.subspace(structure.composition)) == len(theirs), (
                f"subspace size differs for {structure.composition.reduced_formula}"
            )
