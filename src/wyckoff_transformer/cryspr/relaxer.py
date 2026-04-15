"""ASE-based structure relaxation with optional symmetry and cell constraints."""
import logging
from pathlib import Path
from typing import Optional

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixSymmetry
from ase.filters import FrechetCellFilter as CellFilter
from ase.io import write
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer
from ase.spacegroup import get_spacegroup
from pymatgen.io.ase import AseAtomsAdaptor

logger = logging.getLogger(__name__)


def run_ase_relaxer(
        atoms_in: Atoms,
        calculator: Calculator,
        optimizer: type[Optimizer] = BFGS,
        cell_filter=None,
        fix_symmetry: bool = True,
        fix_fractional: bool = False,
        hydrostatic_strain: bool = False,
        symprec: float = 1e-3,
        fmax: float = 0.05,
        steps_limit: int = 500,
        wdir: Path = Path("."),
        logfile: Optional[Path] = None,
) -> Atoms:
    """Run a single ASE relaxation pass on *atoms_in*.

    Args:
        atoms_in: Input structure; not modified in place.
        calculator: ASE Calculator to attach to the atoms.
        optimizer: Local optimisation algorithm class (default :class:`~ase.optimize.BFGS`).
        cell_filter: ASE filter class for cell relaxation; ``None`` keeps the
            cell fixed.
        fix_symmetry: Apply a :class:`~ase.constraints.FixSymmetry` constraint.
        fix_fractional: Fix all atomic positions (ions immobile).
        hydrostatic_strain: Restrict cell filter to isotropic strain only.
        symprec: Symmetry tolerance in Å used by :func:`~ase.spacegroup.get_spacegroup`
            and :class:`~ase.constraints.FixSymmetry`.
        fmax: Force convergence criterion in eV/Å.
        steps_limit: Maximum number of optimisation steps.
        wdir: Directory for the output CIF file.
        logfile: Path to append optimiser output; ``None`` writes to *stderr*.

    Returns:
        Relaxed :class:`~ase.Atoms` object.
    """
    atoms = atoms_in.copy()
    full_formula = atoms.get_chemical_formula(mode="metal")
    reduced_formula = atoms.get_chemical_formula(mode="metal", empirical=True)
    atoms.calc = calculator

    if fix_fractional:
        atoms.set_constraint([FixAtoms(indices=list(range(len(atoms))))])
    spg0 = get_spacegroup(atoms, symprec=symprec)
    if fix_symmetry:
        atoms.set_constraint([FixSymmetry(atoms, symprec=symprec)])
    target = cell_filter(atoms, hydrostatic_strain=hydrostatic_strain) if cell_filter is not None else atoms

    E0 = atoms.get_potential_energy()
    logger.info(
        "Start relaxation: E₀ = %.5f eV, symmetry = %s (%d), fix_sym = %s, relax_cell = %s",
        E0, spg0.symbol, spg0.no, fix_symmetry, cell_filter is not None,
    )

    log_arg = str(logfile) if logfile is not None else "-"
    opt = optimizer(atoms=target, logfile=log_arg)
    opt.run(fmax=fmax, steps=steps_limit)

    cif_stem = f"{reduced_formula}_{full_formula}"
    if cell_filter is None:
        cif_path = wdir / f"{cif_stem}_fix-cell.cif"
    else:
        cif_path = wdir / f"{cif_stem}_cell+pos.cif"
    write(filename=str(cif_path), images=atoms, format="cif")

    E1 = atoms.get_potential_energy()
    spg1 = get_spacegroup(atoms, symprec=symprec)
    cell_diff = (atoms.cell.cellpar() / atoms_in.cell.cellpar() - 1.0) * 100
    logger.info(
        "End relaxation: E₁ = %.5f eV, symmetry = %s (%d), max|F| = %.4f eV/Å",
        E1, spg1.symbol, spg1.no, abs(atoms.get_forces()).max(),
    )
    logger.debug("Cell diff (%%): %s", cell_diff)

    return atoms


def stepwise_relax(
        atoms_in: Atoms,
        calculator: Calculator,
        optimizer: type[Optimizer] = BFGS,
        fix_symmetry: bool = True,
        hydrostatic_strain: bool = False,
        symprec: float = 1e-3,
        fmax: float = 0.05,
        steps_limit: int = 500,
        wdir: Path = Path("."),
        logfile_prefix: str = "",
        logfile_postfix: str = "",
) -> Atoms:
    """Two-stage relaxation: fix-cell first, then full cell + atomic positions.

    Args:
        atoms_in: Input structure.
        calculator: ASE Calculator.
        optimizer: Optimisation algorithm class.
        fix_symmetry: Apply symmetry constraints during both stages.
        hydrostatic_strain: Restrict cell relaxation to isotropic strain.
        symprec: Symmetry tolerance in Å.
        fmax: Force convergence criterion in eV/Å.
        steps_limit: Maximum optimisation steps per stage.
        wdir: Directory for output CIF and log files.
        logfile_prefix: Prefix for log file names.
        logfile_postfix: Postfix for log file names.

    Returns:
        Relaxed :class:`~ase.Atoms` after both stages.
    """
    wdir = Path(wdir)
    wdir.mkdir(parents=True, exist_ok=True)

    atoms = atoms_in.copy()
    full_formula = atoms.get_chemical_formula(mode="metal")
    reduced_formula = atoms.get_chemical_formula(mode="metal", empirical=True)

    structure0 = AseAtomsAdaptor.get_structure(atoms)
    structure0.to(
        filename=str(wdir / f"{reduced_formula}_{full_formula}_0_initial_symmetrized.cif"),
        symprec=symprec,
    )

    # Stage 1: fix cell, relax atomic positions
    parts1 = [p for p in [logfile_prefix, "fix-cell", logfile_postfix] if p]
    logfile1 = wdir / ("_".join(parts1) + ".log")
    atoms1 = run_ase_relaxer(
        atoms_in=atoms,
        calculator=calculator,
        optimizer=optimizer,
        fix_symmetry=fix_symmetry,
        cell_filter=None,
        hydrostatic_strain=hydrostatic_strain,
        symprec=symprec,
        fmax=fmax,
        steps_limit=steps_limit,
        wdir=wdir,
        logfile=logfile1,
    )
    structure1 = AseAtomsAdaptor.get_structure(atoms1)
    structure1.to(
        filename=str(wdir / f"{reduced_formula}_{full_formula}_1_fix-cell_symmetrized.cif"),
        symprec=symprec,
    )

    # Stage 2: relax both cell and atomic positions
    parts2 = [p for p in [logfile_prefix, "cell+positions", logfile_postfix] if p]
    logfile2 = wdir / ("_".join(parts2) + ".log")
    atoms2 = run_ase_relaxer(
        atoms_in=atoms1,
        calculator=calculator,
        optimizer=optimizer,
        fix_symmetry=fix_symmetry,
        cell_filter=CellFilter,
        hydrostatic_strain=hydrostatic_strain,
        symprec=symprec,
        fmax=fmax,
        steps_limit=steps_limit,
        wdir=wdir,
        logfile=logfile2,
    )
    structure2 = AseAtomsAdaptor.get_structure(atoms2)
    structure2.to(
        filename=str(wdir / f"{reduced_formula}_{full_formula}_2_cell+pos_symmetrized.cif"),
        symprec=symprec,
    )

    return atoms2
