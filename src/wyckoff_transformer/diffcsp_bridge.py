"""Checks and the rattle for structures DiffCSP++ made from Wyckoff genes.

DiffCSP++ runs in its own environment (``/home/kna/DiffCSPNew``) through its
order-preserving bench harness; ``scripts/alex_bench/write_benchset.py`` and
``export_predictions.py`` carry genes there and structures back. What comes back is
judged and perturbed here, by the protocol's ``starts`` stage and by the submission
assembler alike, so that the structures evaluated and the structures submitted have been
through exactly the same code.

The rattle is the one that :func:`~wyckoff_transformer.cryspr.relaxer.perturb` applies in
CrySPR's rattle stage. DiffCSP++ places atoms exactly on their Wyckoff orbits, and a
structure at a symmetric stationary point stays there under gradient descent; the
perturbation is what lets an unconstrained relaxation -- ours, or a benchmark's -- leave it.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
from ase import Atoms

#: Closest approach below which a start is refused outright, in Å. Far below any bond:
#: it catches atoms placed on top of each other, not merely short contacts, which the
#: relaxation is there to resolve.
MIN_DISTANCE = 0.5


def gene_composition(gene: dict) -> Counter:
    """Atoms per element in the conventional cell the gene describes."""
    composition: Counter = Counter()
    for element, count in zip(gene["species"], gene["numIons"]):
        composition[str(element)] += int(count)
    return composition


@dataclass
class StartCheck:
    """What :func:`check_start` found; ``error`` is ``None`` when the start passes."""
    error: Optional[str]
    n_atoms: int
    min_distance: float
    spacegroup: Optional[int]
    volume_per_atom: float


def check_start(atoms: Atoms, gene: dict, symprec: float = 0.1) -> StartCheck:
    """Judge one generated structure against the gene it was made for.

    Refused: a composition other than the gene's (per element, in the conventional cell
    DiffCSP++ builds), or two atoms closer than :data:`MIN_DISTANCE`. Recorded but not
    refused: the spglib space group at *symprec*, which a relaxation may legitimately
    change, and the volume per atom.
    """
    import spglib

    n_atoms = len(atoms)
    if n_atoms > 1:
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        min_distance = float(distances.min())
    else:
        min_distance = float("inf")
    try:
        dataset = spglib.get_symmetry_dataset(
            (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers), symprec=symprec)
        spacegroup = int(dataset.number) if dataset is not None else None
    except Exception:  # noqa: BLE001 - a diagnostic, never a verdict
        spacegroup = None
    volume_per_atom = float(atoms.get_volume() / n_atoms) if n_atoms else float("nan")

    error = None
    have = Counter(atoms.get_chemical_symbols())
    want = gene_composition(gene)
    if have != want:
        error = f"composition {dict(have)} is not the gene's {dict(want)}"
    elif min_distance < MIN_DISTANCE:
        error = f"atoms {min_distance:.3f} A apart"
    return StartCheck(error, n_atoms, min_distance, spacegroup, volume_per_atom)


def rattle_start(atoms: Atoms, gene_index: int, trial: int) -> Atoms:
    """CrySPR's rattle, seeded per (gene, trial) so a rerun perturbs identically."""
    from wyckoff_transformer.cryspr.generator import _trial_seed
    from wyckoff_transformer.cryspr.relaxer import (
        RATTLE_STDEV,
        RATTLE_STRAIN_STDEV,
        perturb,
    )

    rattled = perturb(atoms, RATTLE_STDEV, RATTLE_STRAIN_STDEV,
                      seed=_trial_seed(gene_index, trial))
    rattled.info = dict(atoms.info)
    return rattled
