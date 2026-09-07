"""Crystal structure generation and relaxation orchestration."""
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer
from pyxtal import pyxtal
from pyxtal.tolerance import Tol_matrix

from wyckoff_transformer.cryspr.relaxer import (
    RATTLE_ACCEPT_EV_PER_ATOM,
    RATTLE_STDEV,
    RATTLE_STRAIN_STDEV,
    stepwise_relax,
)

logger = logging.getLogger(__name__)

#: PyXtal's inter-atomic distance tolerances, scaled by 1.3.
#:
#: The factor is a *floor* on how close two atoms may start, as a multiple of
#: the covalent-radius sum, and 1.3 rather than PyXtal's 1.0 is what the oracle
#: studies support for a *free* lattice, which is how genes are sampled here:
#: it draws a loose cell (median 1.68x the target volume) that the variable-cell
#: relaxation then contracts to 1.34x, and that compressive annealing gives
#: 50.8% recovery at 5 trials with only 1.2% generation failures
#: (docs/cryspr_oracle_relaxed_cell_report.md).  The 39.8% generation choking
#: that motivated trying 1.0 appears only when the cell is pinned to the
#: ground-truth lattice, and 1.0 in that setup jams the atoms instead: 19.0%
#: recovery against the loose baseline's 50.8%.  It is also what every
#: measurement the ranking protocol rests on used, so changing it would
#: invalidate the trial schedule those numbers set.
_DEFAULT_IADM = Tol_matrix(prototype="atomic", factor=1.3)

# Slightly more permissive tolerance used to *reject* collapsed structures after
# relaxation.  Set below the generation factor (1.3) so that the few real
# materials whose short metallic contacts graze the generation floor are not
# discarded, while still catching the energy-lowering MACE collapse artifacts
# (verified to sit comfortably below this floor).
_CLASH_IADM = Tol_matrix(prototype="atomic", factor=1.1)

#: Top-level package of the MACE ASE calculators, used to recognise a MACE
#: calculator without importing MACE (an optional dependency, and the backend
#: environments are mutually exclusive -- see
#: :mod:`wyckoff_transformer.cryspr.mlips`).
_MACE_PACKAGE = "mace"


def is_mace_calculator(calculator: Calculator) -> bool:
    """Return ``True`` if *calculator* is backed by MACE.

    Matched on the calculator's class hierarchy rather than on a model name,
    which is a free-form label, and rather than by importing MACE, which need
    not be installed in the environment running a different backend.  The MRO
    is walked because :func:`~wyckoff_transformer.cryspr.calculator.build_mace_calculator`
    hands back a local subclass of ``MACECalculator``, so the concrete class is
    defined in this project, not in MACE.

    Args:
        calculator: The ASE calculator used for relaxation.

    Returns:
        ``True`` if any class in the MRO comes from the ``mace`` package.
    """
    for klass in type(calculator).__mro__:
        module = getattr(klass, "__module__", "") or ""
        if module == _MACE_PACKAGE or module.startswith(_MACE_PACKAGE + "."):
            return True
    return False


def has_atomic_clash(atoms: Atoms, iadm: Tol_matrix = _CLASH_IADM) -> bool:
    """Return ``True`` if any pair of atoms is closer than the tolerance.

    Each interatomic distance (including periodic images) is compared against
    the species-pair minimum from *iadm*.  A relaxed structure that has
    collapsed into overlapping atoms (a known artifact of MACE-MP-0's spurious
    short-range energy basins, which produce unphysically low energies) is thus
    rejected on a purely physical, geometry-based criterion.

    The default *iadm* (:data:`_CLASH_IADM`, factor 1.1) is marginally more
    permissive than the generation tolerance (:data:`_DEFAULT_IADM`, factor 1.3)
    so that real structures whose contacts graze the generation floor are not
    falsely discarded.

    The collapse this guards against is a MACE pathology, so :func:`func_run`
    engages the guard only for MACE calculators; this function itself is
    potential-agnostic and can be called on any structure.

    Args:
        atoms: Structure to check.
        iadm: PyXtal tolerance matrix defining per-species-pair minimum distances.

    Returns:
        ``True`` if at least one interatomic distance is below tolerance.
    """
    numbers = atoms.numbers
    if len(numbers) == 0:
        return False
    unique = sorted({int(n) for n in numbers})
    cutoff = max(iadm.get_tol(a, b) for a in unique for b in unique)
    if cutoff <= 0:
        return False
    i_idx, j_idx, dists = neighbor_list("ijd", atoms, cutoff)
    tols = np.array([iadm.get_tol(int(numbers[i]), int(numbers[j]))
                     for i, j in zip(i_idx, j_idx)])
    return bool(np.any(dists < tols))


#: Suffix of the CIF holding the structure a trial actually kept.  The rattle
#: stage is accepted on energy, so which stage's CIF is the final one is not
#: known until it has run; this one always is.
KEPT_CIF_SUFFIX = "_kept.cif"


def _trial_seed(id_gene: int | str, i_trial: int) -> int:
    """A stable seed per (gene, trial), so a rerun rattles the same way.

    Hashed rather than derived arithmetically: the rattle of gene 1 trial 2 and
    that of gene 2 trial 1 should be independent draws.
    """
    digest = hashlib.sha256(f"{id_gene}_{i_trial}".encode()).hexdigest()
    return int(digest[:8], 16)


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
        release_symmetry: bool = True,
        rattle: bool = True,
        rattle_stdev: float = RATTLE_STDEV,
        strain_stdev: float = RATTLE_STRAIN_STDEV,
        rattle_accept: float = RATTLE_ACCEPT_EV_PER_ATOM,
        clash_guard: Optional[bool] = None,
        fmax: float = 0.01,
        optimizer: type[Optimizer] = BFGS,
) -> tuple[Optional[Atoms], Optional[str], Optional[float], Optional[float], Optional[str]]:
    """Generate and relax crystal structures for one Wyckoff gene.

    Runs *n_trials* independent PyXtal generation + MLIP relaxation cycles.
    The trial with the lowest final energy is returned.  All trial directories
    and output CIF files are written under ``output_dir / str(id_gene) /``.

    Args:
        id_gene: Identifier for this Wyckoff gene, used as a sub-directory name.
        wyckoffgene: Wyckoff gene dictionary (see :func:`single_pyxtal`).
        calculator: Shared ASE Calculator instance (not recreated per trial).
        output_dir: Root directory for all output files.
        model_name: Label used in log messages and output filenames.
        n_trials: Number of random generation + relaxation trials.
        fix_symmetry: Run the symmetry-constrained step of the relaxation.
        release_symmetry: Run the unconstrained step before the rattle.
            ``False`` drops it; see
            :func:`~wyckoff_transformer.cryspr.relaxer.stepwise_relax`.
        rattle: Run the rattle stage, which is the only one that can leave a
            symmetric stationary point.
        rattle_stdev: Per-atom displacement of the perturbation, Å.
        strain_stdev: Cell strain of the perturbation, dimensionless.
        rattle_accept: Energy the rattle must win to be kept, eV/atom.
        clash_guard: Discard a relaxed trial whose atoms have collapsed into
            each other (:func:`has_atomic_clash`).  ``None``, the default,
            engages the guard for MACE calculators only, which is where the
            collapse comes from; ``True`` or ``False`` forces it either way.
        fmax: Force convergence criterion in eV/Å.
        optimizer: ASE local optimisation algorithm class.

    Returns:
        Tuple ``(atoms, formula, energy, energy_per_atom, cif)`` for the
        lowest-energy successful trial, where *cif* is the text of the
        structure that trial kept.
        Returns ``(None, None, None, None, None)`` when all trials fail.
    """
    output_dir = Path(output_dir)
    gene_dir = output_dir / str(id_gene)
    gene_dir.mkdir(parents=True, exist_ok=True)

    # Resolved once: the calculator is shared across trials, and an unguarded
    # run should say so in the log rather than be inferred from its absence.
    if clash_guard is None:
        clash_guard = is_mace_calculator(calculator)
        logger.debug(
            "[%s-%s] Clash guard %s (auto: calculator %s MACE-backed)",
            model_name, id_gene,
            "on" if clash_guard else "off",
            "is" if clash_guard else "is not",
        )

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
                release_symmetry=release_symmetry,
                rattle=rattle,
                rattle_stdev=rattle_stdev,
                strain_stdev=strain_stdev,
                rattle_accept=rattle_accept,
                seed=_trial_seed(id_gene, i_trial),
                fmax=fmax,
                wdir=trial_dir,
                logfile_prefix=formula,
                logfile_postfix="relax",
            )
            energy = atoms_relaxed.get_potential_energy()
            if clash_guard and has_atomic_clash(atoms_relaxed):
                logger.warning(
                    "[%s-%s %s] Relaxed structure has atomic clashes "
                    "(E = %.5f eV); discarding as unphysical.",
                    model_name, id_gene, trial_key, energy,
                )
                continue
            atoms_by_trial[trial_key] = atoms_relaxed
            energy_by_trial[trial_key] = energy
            # The kept structure, whichever stage produced it.
            write(
                filename=str(trial_dir / f"{formula}{KEPT_CIF_SUFFIX}"),
                images=atoms_relaxed,
                format="cif",
            )
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
        return None, None, None, None, None

    lowest_key = min(energy_by_trial, key=energy_by_trial.__getitem__)

    symlink_lowest = gene_dir / "trial-lowest"
    if not symlink_lowest.exists():
        os.symlink(lowest_key, symlink_lowest, target_is_directory=True)

    lowest_dir = gene_dir / lowest_key
    # The kept CIF, not a stage CIF: with the rattle stage the last stage to
    # run is not necessarily the one whose structure was kept.
    final_cifs = sorted(lowest_dir.glob(f"*{KEPT_CIF_SUFFIX}"))
    if final_cifs:
        symlink_cif = gene_dir / "min_e_strc.cif"
        if not symlink_cif.exists():
            # Relative symlink so it survives directory moves
            cif_rel = Path("trial-lowest") / final_cifs[0].name
            os.symlink(cif_rel, symlink_cif)

    atoms = atoms_by_trial[lowest_key]
    energy = energy_by_trial[lowest_key]
    energy_per_atom = energy / len(atoms)

    cif_content: Optional[str] = final_cifs[0].read_text() if final_cifs else None

    return atoms, formula, energy, energy_per_atom, cif_content
