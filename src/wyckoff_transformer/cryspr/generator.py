"""Crystal structure generation and relaxation orchestration."""
import logging
import os
from pathlib import Path
from typing import Optional

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer
from pyxtal import pyxtal
from pyxtal.tolerance import Tol_matrix

from wyckoff_transformer.cryspr.relaxer import stepwise_relax

logger = logging.getLogger(__name__)

_DEFAULT_IADM = Tol_matrix(prototype="atomic", factor=1.3)


def single_pyxtal(
        wyckoffgene: dict,
        iadm: Tol_matrix = _DEFAULT_IADM,
        nlimit: int = 20,
        wdir: Path = Path("."),
) -> Optional[Atoms]:
    """Generate a random crystal from a Wyckoff gene using PyXtal.

    Args:
        wyckoffgene: Dict with keys ``group`` (int), ``species`` (list of str),
            ``numIons`` (list of int), and ``sites`` (list of list of str).
        iadm: PyXtal tolerance matrix controlling minimum inter-atomic distances.
        nlimit: Maximum number of PyXtal generation attempts.
        wdir: Directory in which to save the raw generated CIF.

    Returns:
        An :class:`~ase.Atoms` object, or ``None`` if generation failed.
    """
    spg: int = wyckoffgene["group"]
    species: list[str] = wyckoffgene["species"]
    num_ions: list[int] = wyckoffgene["numIons"]
    sites: list[list] = wyckoffgene["sites"]

    try:
        candidate = pyxtal()
        candidate.from_random(
            dim=3,
            group=spg,
            species=species,
            numIons=num_ions,
            sites=sites,
            tm=iadm,
            max_count=nlimit,
        )
        atoms: Atoms = candidate.to_ase()
        formula = atoms.get_chemical_formula(mode="metal")
        wdir = Path(wdir)
        wdir.mkdir(parents=True, exist_ok=True)
        candidate.to_file(str(wdir / f"pyxtal_generated_{formula}.cif"))
        return atoms
    except Exception as exc:
        logger.warning("PyXtal generation failed: %s", exc)
        return None


def func_run(
        id_gene: int | str,
        wyckoffgene: dict,
        calculator: Calculator,
        output_dir: Path = Path("."),
        model_name: str = "model",
        n_trials: int = 6,
        fix_symmetry: bool = True,
        fmax: float = 0.01,
        optimizer: type[Optimizer] = BFGS,
) -> tuple[Optional[Atoms], Optional[str], Optional[float], Optional[float]]:
    """Generate and relax crystal structures for one Wyckoff gene.

    Runs *n_trials* independent PyXtal generation + MACE relaxation cycles.
    The trial with the lowest final energy is returned.  All trial directories
    and output CIF files are written under ``output_dir / str(id_gene) /``.

    Args:
        id_gene: Identifier for this Wyckoff gene, used as a sub-directory name.
        wyckoffgene: Wyckoff gene dictionary (see :func:`single_pyxtal`).
        calculator: Shared ASE Calculator instance (not recreated per trial).
        output_dir: Root directory for all output files.
        model_name: Label used in log messages and output filenames.
        n_trials: Number of random generation + relaxation trials.
        fix_symmetry: Apply symmetry constraints during relaxation.
        fmax: Force convergence criterion in eV/Å.
        optimizer: ASE local optimisation algorithm class.

    Returns:
        Tuple ``(atoms, formula, energy, energy_per_atom)`` for the
        lowest-energy successful trial, or ``(None, None, None, None)``
        when all trials fail.
    """
    output_dir = Path(output_dir)
    gene_dir = output_dir / str(id_gene)
    gene_dir.mkdir(parents=True, exist_ok=True)

    atoms_by_trial: dict[str, Atoms] = {}
    energy_by_trial: dict[str, float] = {}
    formula: Optional[str] = None

    for i_trial in range(n_trials):
        trial_key = f"trial-{i_trial}"
        trial_dir = gene_dir / trial_key
        trial_dir.mkdir(parents=True, exist_ok=True)

        atoms_in = single_pyxtal(wyckoffgene=wyckoffgene, nlimit=30, wdir=trial_dir)
        if atoms_in is None:
            continue

        formula = atoms_in.get_chemical_formula(mode="metal")
        logger.info("[%s-%s %s] Starting relaxation", model_name, id_gene, trial_key)

        try:
            atoms_relaxed = stepwise_relax(
                atoms_in=atoms_in,
                calculator=calculator,
                optimizer=optimizer,
                fix_symmetry=fix_symmetry,
                fmax=fmax,
                wdir=trial_dir,
                logfile_prefix=formula,
                logfile_postfix="relax",
            )
            atoms_by_trial[trial_key] = atoms_relaxed
            energy_by_trial[trial_key] = atoms_relaxed.get_potential_energy()
            logger.info(
                "[%s-%s %s] Done, E = %.5f eV",
                model_name, id_gene, trial_key, energy_by_trial[trial_key],
            )
        except Exception as exc:
            logger.warning("[%s-%s %s] Relaxation failed: %s", model_name, id_gene, trial_key, exc)

    if not atoms_by_trial:
        logger.warning(
            "[%s-%s] All %d trials failed or produced no structure.",
            model_name, id_gene, n_trials,
        )
        return None, None, None, None

    lowest_key = min(energy_by_trial, key=energy_by_trial.__getitem__)

    symlink_lowest = gene_dir / "trial-lowest"
    if not symlink_lowest.exists():
        os.symlink(lowest_key, symlink_lowest, target_is_directory=True)

    lowest_dir = gene_dir / lowest_key
    cell_pos_cifs = list(lowest_dir.glob("*_cell+pos.cif"))
    if cell_pos_cifs:
        symlink_cif = gene_dir / "min_e_strc.cif"
        if not symlink_cif.exists():
            # Relative symlink so it survives directory moves
            cif_rel = Path("trial-lowest") / cell_pos_cifs[0].name
            os.symlink(cif_rel, symlink_cif)

    atoms = atoms_by_trial[lowest_key]
    energy = energy_by_trial[lowest_key]
    energy_per_atom = energy / len(atoms)

    return atoms, formula, energy, energy_per_atom
