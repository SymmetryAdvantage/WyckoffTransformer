"""Structure, chemistry and calculation descriptors for residual diagnostics."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from math import gcd, pi

import numpy as np
from ase.data import covalent_radii
from pymatgen.core import Element, Structure
from pyxtal.symmetry import Group


MP_U_ELEMENTS = {"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"}


def composition_keys(elements, multiplicities):
    """Cell-independent reduced formula and unordered chemical-system keys."""
    counts = Counter()
    for element, n in zip(elements, multiplicities):
        z = int(element.Z) if hasattr(element, "Z") else Element(str(element)).Z
        counts[z] += int(n)
    divisor = 0
    for n in counts.values():
        divisor = gcd(divisor, n)
    if divisor == 0:
        raise ValueError("Empty composition")
    return (
        ";".join(f"{z}:{n // divisor}" for z, n in sorted(counts.items())),
        "-".join(map(str, sorted(counts))),
    )


@lru_cache(maxsize=230)
def group_info(number):
    group = Group(int(number))
    # PyXtal's crystal-system label determines the independent lattice metrics.
    return group, group.lattice_type


def structure_features(structure: Structure, gene: dict) -> dict:
    """Use actual volume/contacts and the saved final gene, without energy labels."""
    number = int(gene["group"])
    group, system = group_info(number)
    lattice_dof = dict(triclinic=6, monoclinic=4, orthorhombic=3,
                       tetragonal=2, trigonal=2, hexagonal=2, cubic=1)[system]
    elements, mults, dofs = [], [], []
    for symbol, sites in zip(gene["species"], gene["sites"]):
        for site in sites:
            elements.append(Element(symbol))
            mults.append(int(site[:-1]))
            dofs.append(int(group[site[-1]].get_dof()))
    formula, chemsys = composition_keys(elements, mults)
    n_gene = sum(mults)
    comp = structure.composition.get_el_amt_dict()
    expected = Counter()
    for element, mult in zip(elements, mults):
        expected[element.symbol] += mult
    total = sum(comp.values())
    if set(comp) != set(expected) or any(abs(comp[e] / total - expected[e] / n_gene) > 1e-8 for e in comp):
        raise ValueError("Saved final gene and geometry have different compositions")
    species = [Element(e) for e in sorted(comp)]
    fractions = np.array([comp[e.symbol] / total for e in species])
    z = np.array([e.Z for e in species])
    radii = np.array([float(covalent_radii[e.Z]) for e in species])
    x = np.array([float(e.X) if e.X is not None else np.nan for e in species])
    finite_x = np.isfinite(x)
    mean_z = float(fractions @ z)
    result = dict(
        formula_key=formula, chemsys_key=chemsys, reduced_formula=structure.composition.reduced_formula,
        space_group=number, crystal_system=system, n_atoms=len(structure),
        n_gene_atoms=n_gene, n_orbits=len(mults), n_elements=len(species),
        positional_dof=sum(dofs), lattice_dof=lattice_dof,
        positional_dof_per_orbit=sum(dofs) / len(dofs),
        positional_dof_per_gene_atom=sum(dofs) / n_gene,
        fraction_atoms_free_orbits=sum(m for m, d in zip(mults, dofs) if d > 0) / n_gene,
        volume=float(structure.volume), volume_per_atom=float(structure.volume / len(structure)),
        density=float(structure.density), lattice_anisotropy=float(max(structure.lattice.abc) / min(structure.lattice.abc)),
        covalent_sphere_fraction=float(total * np.sum(fractions * 4 * pi * radii**3 / 3) / structure.volume),
        mean_atomic_number=mean_z, std_atomic_number=float(np.sqrt(fractions @ (z - mean_z)**2)),
        mean_electronegativity=float(np.sum(fractions[finite_x] * x[finite_x]) / fractions[finite_x].sum()) if finite_x.any() else np.nan,
        electronegativity_range=float(np.ptp(x[finite_x])) if finite_x.any() else np.nan,
        mean_covalent_radius=float(fractions @ radii),
        fraction_transition_metal=float(sum(f for e, f in zip(species, fractions) if e.is_transition_metal)),
        fraction_lanthanide=float(sum(f for e, f in zip(species, fractions) if e.is_lanthanoid)),
        fraction_mp_u_element=float(sum(f for e, f in zip(species, fractions) if e.symbol in MP_U_ELEMENTS)),
        mp_u_chemistry_proxy=int(bool(set(comp) & MP_U_ELEMENTS) and bool(set(comp) & {"O", "F"})),
    )
    for atomic_number in range(1, 119):
        result[f"element_fraction_{atomic_number}"] = float(fractions[z == atomic_number].sum())
    # Include self-images and periodic neighbors: a one-atom cell has a real
    # nearest-neighbor distance even though its distance matrix is zero.
    centers, neighbors, _, distances = structure.get_neighbor_list(r=5.)
    if not len(distances):
        centers, neighbors, _, distances = structure.get_neighbor_list(r=12.)
    if len(distances):
        site_radii = np.array([covalent_radii[site.specie.Z] for site in structure])
        ratios = distances / (site_radii[centers] + site_radii[neighbors])
        nearest = np.full(len(structure), np.inf)
        np.minimum.at(nearest, centers, ratios)
        result.update(
            min_pair_distance=float(distances.min()), min_covalent_distance_ratio=float(ratios.min()),
            mean_nearest_covalent_ratio=float(nearest[np.isfinite(nearest)].mean()),
            fraction_short_contacts=float((nearest < .75).mean()),
        )
    return result


def calculation_features(entry: dict | None, comp: dict) -> dict:
    """Extract recorded settings; missing data never means 'no Hubbard U'."""
    if not entry:
        return dict(dft_metadata_available=0, dft_uses_hubbard=np.nan,
                    dft_hubbard_per_atom=np.nan, dft_run_type="unrecorded")
    params = entry.get("parameters", {})
    hubbards = params.get("hubbards")
    is_hubbard = params.get("is_hubbard")
    result = dict(
        dft_metadata_available=1,
        dft_uses_hubbard=float(is_hubbard) if is_hubbard is not None else np.nan,
        dft_run_type=params.get("run_type", "unrecorded"),
        dft_hubbard_per_atom=(sum(float(u) * comp.get(e, 0) for e, u in hubbards.items()) / sum(comp.values())) if hubbards is not None else np.nan,
    )
    for spec in params.get("potcar_spec", []):
        title = spec.get("titel", "")
        if len(title.split()) >= 2:
            result["potcar_" + title.split()[1]] = 1.
    return result
