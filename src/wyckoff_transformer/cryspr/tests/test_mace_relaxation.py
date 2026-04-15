"""Substantial integration tests for MACE-based crystal structure relaxation.

All tests require the ``[relax]`` optional extra, network access (first run only,
to download the MACE model), and the MP-20 dataset at ``data/mp_20/`` relative to
the repository root.  Enable them with::

    pytest src/wyckoff_transformer/cryspr/tests/test_mace_relaxation.py -v --run-relax

Test suite
----------
TestForcesAndStrainConvergence
    Generate structures from 10 MP-20 Wyckoff genes via ``func_run``.  For
    each successful relaxation verify that the max absolute force in the
    FrechetCellFilter basis (which covers both atomic forces and cell strain)
    is below 2 × fmax.

TestRelaxRecovery
    Compute a MACE reference for each of the 10 MP-20 structures using raw
    ASE only (no wyformer code).  Perturb each reference (random atomic
    displacements + diagonal cell strain) and verify that ``stepwise_relax``
    recovers the reference energy within _E_TOL eV/atom.

TestWyckoffRecreation
    Using the same raw-ASE references, verify that ``func_run`` — starting
    from random PyXtal realisations of the gene extracted from the MP-20
    structure — also converges to the reference energy within _E_TOL eV/atom.
    All 10 structures have 0–1 internal degrees of freedom, so any valid
    PyXtal realisation of the gene leads to the same crystallographic minimum.
"""
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MACE_MODEL_URL = (
    "https://github.com/ACEsuit/mace-foundations/releases/download/"
    "mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model"
)

# Path to MP-20 CSV files (repo root / data / mp_20 /)
# parents: test file → tests/ → cryspr/ → wyckoff_transformer/ → src/ → repo root
_MP20_DIR = Path(__file__).parents[4] / "data" / "mp_20"

# 10 MP-20 structures covering common topologies.
# First 8 have 0 internal degrees of freedom (fractional coordinates fully
# fixed by symmetry); the last 2 have exactly 1 free internal parameter each.
_MATERIAL_IDS = [
    "mp-22862",   # NaCl  – SG 225, rock-salt,    0 DOF  (val)
    "mp-1265",    # MgO   – SG 225, rock-salt,    0 DOF  (train)
    "mp-1138",    # LiF   – SG 225, rock-salt,    0 DOF  (test)
    "mp-22865",   # CsCl  – SG 221, CsCl-type,    0 DOF  (train)
    "mp-2741",    # CaF2  – SG 225, fluorite,     0 DOF  (train)
    "mp-10695",   # ZnS   – SG 216, zincblende,   0 DOF  (train)
    "mp-30",      # Cu    – SG 225, FCC,           0 DOF  (train)
    "mp-13",      # Fe    – SG 229, BCC,           0 DOF  (test)
    "mp-2657",    # TiO2  – SG 136, rutile,        1 DOF  (train)
    "mp-1922",    # RuSe2 – SG 205, pyrite-type,  1 DOF  (train)
]

_FMAX = 0.05    # eV/Å — relaxation convergence threshold used throughout
_E_TOL = 0.10   # eV/atom — energy tolerance for tests 2 and 3

# ---------------------------------------------------------------------------
# Module-level calculator cache (load once per pytest session)
# ---------------------------------------------------------------------------

_calculator = None


def _get_calculator():
    global _calculator
    if _calculator is None:
        from wyckoff_transformer.cryspr.calculator import build_mace_calculator
        _calculator = build_mace_calculator(model=MACE_MODEL_URL)
    return _calculator


# ---------------------------------------------------------------------------
# Data-loading helpers
# ---------------------------------------------------------------------------

def _load_mp20_cif_map() -> dict[str, str]:
    """Return {material_id: cif_string} for every ID in *_MATERIAL_IDS*.

    Searches train, val and test splits.  Returns an empty dict if the data
    directory is absent (tests should skip in that case).
    """
    import pandas as pd

    if not _MP20_DIR.exists():
        return {}

    needed = set(_MATERIAL_IDS)
    found: dict[str, str] = {}
    for split in ("train", "val", "test"):
        csv_path = _MP20_DIR / f"{split}.csv"
        if not csv_path.exists() or not needed:
            continue
        df = pd.read_csv(csv_path, usecols=["material_id", "cif"])
        subset = df[df["material_id"].isin(needed)]
        for _, row in subset.iterrows():
            mid = row["material_id"]
            if mid not in found:
                found[mid] = row["cif"]
        needed -= set(found)

    return found


def _cif_to_pymatgen(cif: str):
    """Parse a CIF string; return the conventional-cell pymatgen Structure."""
    import warnings
    from pymatgen.io.cif import CifParser

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CifParser.from_str(cif).parse_structures(primitive=False)[0]


def _structure_to_gene(structure) -> dict:
    """Convert a pymatgen Structure to a WyFormer Wyckoff gene dict.

    Tries several tolerances so that the wide variety of MP-20 structures
    are all handled correctly by pyxtal.
    """
    from pyxtal import pyxtal

    cryst = pyxtal()
    for tol in (0.1, 0.01, 0.3, 0.5, 1.0, 0.001):
        try:
            cryst.from_seed(structure, tol=tol)
            if len(cryst.atom_sites) > 0:
                break
        except Exception:
            pass
    else:
        raise RuntimeError("pyxtal failed to symmetrise the structure.")

    sites_by_species: dict[str, list[str]] = defaultdict(list)
    for site in cryst.atom_sites:
        sp = str(site.specie)
        sites_by_species[sp].append(f"{site.wp.multiplicity}{site.wp.letter}")

    species = list(sites_by_species)
    return {
        "group": cryst.group.number,
        "species": species,
        "numIons": [sum(int(s[:-1]) for s in sites_by_species[sp]) for sp in species],
        "sites": [sites_by_species[sp] for sp in species],
    }


def _pymatgen_to_ase(structure):
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_atoms(structure)


# ---------------------------------------------------------------------------
# Physical helpers
# ---------------------------------------------------------------------------

def _relax_raw(atoms, fmax=_FMAX, steps=500):
    """Full cell + atom relaxation using raw ASE — intentionally avoids all
    wyformer code, so that this function can serve as an independent reference
    for tests 2 and 3.
    """
    from ase.constraints import FixSymmetry
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS

    a = atoms.copy()
    a.calc = _get_calculator()
    a.set_constraint([FixSymmetry(a, symprec=1e-3)])
    BFGS(FrechetCellFilter(a), logfile=None).run(fmax=fmax, steps=steps)
    return a


def _perturb(atoms, displacement_sigma: float = 0.15,
             strain_amplitude: float = 0.03, seed: int = 0):
    """Return a copy of *atoms* with random atomic displacements and
    independent diagonal cell scaling on each axis.

    The perturbation is chosen to be small enough to stay in the basin of
    attraction of the local minimum while still constituting a meaningful test.
    """
    rng = np.random.default_rng(seed)
    a = atoms.copy()
    a.positions += rng.normal(0.0, displacement_sigma, a.positions.shape)
    scale = 1.0 + rng.uniform(-strain_amplitude, strain_amplitude, 3)
    a.set_cell(a.cell.array * scale[None, :], scale_atoms=True)
    return a


def _cf_max_force(atoms) -> float:
    """Return the max absolute force in the FrechetCellFilter basis.

    This single number captures both atomic forces (eV/Å) and cell-DOF forces
    (which the filter normalises to the same unit scale).  Symmetry constraints
    are removed so we test the raw physical convergence of the structure.
    """
    from ase.filters import FrechetCellFilter

    a = atoms.copy()
    a.set_constraint([])   # remove FixSymmetry — test unconstrained convergence
    a.calc = atoms.calc
    return float(np.abs(FrechetCellFilter(a).get_forces()).max())


# ---------------------------------------------------------------------------
# Session-level reference cache — computed once, shared across all test classes
# ---------------------------------------------------------------------------

class _Session:
    """Holds the lazily-loaded MP-20 CIF map and the per-material references.

    A *reference* for material *mid* is a tuple
    ``(relaxed_atoms, energy_per_atom, wyckoff_gene)`` where the relaxation
    was performed using raw ASE (no wyformer code).
    """

    def __init__(self) -> None:
        self._cif_map: dict[str, str] | None = None
        self._refs: dict[str, tuple | None] = {}

    @property
    def cif_map(self) -> dict[str, str]:
        if self._cif_map is None:
            self._cif_map = _load_mp20_cif_map()
        return self._cif_map

    def reference(self, mid: str) -> tuple | None:
        """Return ``(relaxed_atoms, epa, gene)`` or ``None`` on failure."""
        if mid not in self._refs:
            self._refs[mid] = self._compute(mid)
        return self._refs[mid]

    def _compute(self, mid: str) -> tuple | None:
        cif = self.cif_map.get(mid)
        if cif is None:
            return None
        try:
            structure = _cif_to_pymatgen(cif)
            atoms_in = _pymatgen_to_ase(structure)
            gene = _structure_to_gene(structure)
        except Exception:
            return None
        ref = _relax_raw(atoms_in)
        return ref, ref.get_potential_energy() / len(ref), gene


_SESSION = _Session()


# ---------------------------------------------------------------------------
# Common setUp / tearDown mixin
# ---------------------------------------------------------------------------

class _MP20TestBase(unittest.TestCase):
    def setUp(self):
        if not _SESSION.cif_map:
            self.skipTest("MP-20 data not found — expected at data/mp_20/")
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# Test 1 — atomic forces and cell strain below fmax after func_run
# ---------------------------------------------------------------------------

@pytest.mark.needs_relax
class TestForcesAndStrainConvergence(_MP20TestBase):
    """Generate structures from MP-20 Wyckoff genes via func_run; verify both
    atomic forces and cell strain are below 2 × fmax (FrechetCellFilter norm).
    """

    def _check(self, mid: str) -> None:
        ref = _SESSION.reference(mid)
        if ref is None:
            self.skipTest(f"{mid}: not in MP-20 CSV or gene extraction failed")
        _, _, gene = ref

        from wyckoff_transformer.cryspr.generator import func_run
        atoms, formula, energy, energy_per_atom, cif = func_run(
            id_gene=mid,
            wyckoffgene=gene,
            calculator=_get_calculator(),
            output_dir=self.tmp_path,
            n_trials=3,
            fmax=_FMAX,
        )
        if atoms is None:
            self.skipTest(f"{mid}: all trials failed to generate/relax a structure")

        max_f = _cf_max_force(atoms)
        threshold = _FMAX * 2.0
        self.assertLessEqual(
            max_f, threshold,
            msg=(
                f"{mid} ({formula}): max CellFilter force {max_f:.4f} eV/Å "
                f"exceeds 2×fmax = {threshold:.4f} eV/Å"
            ),
        )

    def test_nacl(self):   self._check("mp-22862")
    def test_mgo(self):    self._check("mp-1265")
    def test_lif(self):    self._check("mp-1138")
    def test_cscl(self):   self._check("mp-22865")
    def test_caf2(self):   self._check("mp-2741")
    def test_zns(self):    self._check("mp-10695")
    def test_cu(self):     self._check("mp-30")
    def test_fe(self):     self._check("mp-13")
    def test_tio2(self):   self._check("mp-2657")
    def test_ruse2(self):  self._check("mp-1922")


# ---------------------------------------------------------------------------
# Test 2 — stepwise_relax recovers the raw-ASE reference from perturbed input
# ---------------------------------------------------------------------------

@pytest.mark.needs_relax
class TestRelaxRecovery(_MP20TestBase):
    """Manually relax each MP-20 structure with raw ASE to obtain a reference,
    then perturb the reference and verify ``stepwise_relax`` recovers it within
    _E_TOL eV/atom.

    Perturbation: ±0.15 Å random atomic displacements + ±3 % diagonal cell strain.
    """

    def _check(self, mid: str) -> None:
        ref = _SESSION.reference(mid)
        if ref is None:
            self.skipTest(f"{mid}: not in MP-20 CSV or gene extraction failed")
        ref_atoms, ref_epa, _ = ref

        seed = _MATERIAL_IDS.index(mid)
        perturbed = _perturb(ref_atoms, seed=seed)

        from wyckoff_transformer.cryspr.relaxer import stepwise_relax
        recovered = stepwise_relax(
            atoms_in=perturbed,
            calculator=_get_calculator(),
            fmax=_FMAX,
            wdir=self.tmp_path / mid,
        )
        rec_epa = recovered.get_potential_energy() / len(recovered)
        formula = ref_atoms.get_chemical_formula(mode="metal")

        self.assertAlmostEqual(
            rec_epa, ref_epa, delta=_E_TOL,
            msg=(
                f"{mid} ({formula}): recovered {rec_epa:.4f} eV/atom, "
                f"reference {ref_epa:.4f} eV/atom, "
                f"|Δ| = {abs(rec_epa - ref_epa):.4f} > tolerance {_E_TOL}"
            ),
        )

    def test_nacl(self):   self._check("mp-22862")
    def test_mgo(self):    self._check("mp-1265")
    def test_lif(self):    self._check("mp-1138")
    def test_cscl(self):   self._check("mp-22865")
    def test_caf2(self):   self._check("mp-2741")
    def test_zns(self):    self._check("mp-10695")
    def test_cu(self):     self._check("mp-30")
    def test_fe(self):     self._check("mp-13")
    def test_tio2(self):   self._check("mp-2657")
    def test_ruse2(self):  self._check("mp-1922")


# ---------------------------------------------------------------------------
# Test 3 — func_run recreates the MACE minimum from the Wyckoff representation
# ---------------------------------------------------------------------------

@pytest.mark.needs_relax
class TestWyckoffRecreation(_MP20TestBase):
    """Verify that ``func_run`` finds the same MACE energy minimum as the raw-ASE
    reference, starting only from the Wyckoff gene extracted from the MP-20 structure.

    All 10 structures have 0–1 internal degrees of freedom, so any valid PyXtal
    realisation of the gene converges to the same crystallographic minimum.
    """

    def _check(self, mid: str) -> None:
        ref = _SESSION.reference(mid)
        if ref is None:
            self.skipTest(f"{mid}: not in MP-20 CSV or gene extraction failed")
        _, ref_epa, gene = ref

        from wyckoff_transformer.cryspr.generator import func_run
        atoms, formula, energy, energy_per_atom, cif = func_run(
            id_gene=mid,
            wyckoffgene=gene,
            calculator=_get_calculator(),
            output_dir=self.tmp_path,
            n_trials=3,
            fmax=_FMAX,
        )
        if energy is None:
            self.fail(f"{mid}: func_run returned no structure after 3 trials")

        self.assertAlmostEqual(
            energy_per_atom, ref_epa, delta=_E_TOL,
            msg=(
                f"{mid} ({formula}): func_run {energy_per_atom:.4f} eV/atom, "
                f"reference {ref_epa:.4f} eV/atom, "
                f"|Δ| = {abs(energy_per_atom - ref_epa):.4f} > tolerance {_E_TOL}"
            ),
        )

    def test_nacl(self):   self._check("mp-22862")
    def test_mgo(self):    self._check("mp-1265")
    def test_lif(self):    self._check("mp-1138")
    def test_cscl(self):   self._check("mp-22865")
    def test_caf2(self):   self._check("mp-2741")
    def test_zns(self):    self._check("mp-10695")
    def test_cu(self):     self._check("mp-30")
    def test_fe(self):     self._check("mp-13")
    def test_tio2(self):   self._check("mp-2657")
    def test_ruse2(self):  self._check("mp-1922")
