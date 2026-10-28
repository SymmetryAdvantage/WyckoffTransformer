"""Electrochemical windows, interface reactions and the join flag of a DFT entry.

The method is Richards, Miara, Wang, Kim and Ceder, Chem. Mater. 28, 266 (2016)
(``docs/archive/acs.chemmater.5b04082.pdf``):

* **window** (:func:`stability_window`): the range of mu_Li over which a compound is on
  the hull of the grand potential Phi = E - n_Li mu_Li of a system open to Li. A
  compound above the hull is first placed on it (:func:`placed_on_hull`), as the paper
  did for metastable electrolytes. Its window is then the range in which its whole
  decomposition assemblage coexists, which shrinks to a single potential when that
  assemblage spans a facet.
* **reaction energies** (:func:`chemical_reaction`, :func:`electrochemical_reaction`):
  eq. 2, the most negative energy of x c_electrode + (1 - x) c_electrolyte ->
  equilibrium in a closed system, per atom; eq. 4, the same open to Li at the
  electrode's potential, per atom other than Li, with the electrode at its
  grand-potential equilibrium and the electrolyte on the hull; eq. 5, the x = 0 term
  of eq. 4, the electrolyte's decomposition by the Li reservoir alone.
* **joins** (:func:`pseudo_binary_joins`): the line from a compound's Li binary to the
  rest of its formula, on which the plan's Phase 6 join flag counts known compounds.

Energies are LeMat's raw PBE/PBE+U, or MP2020-corrected (:func:`mp2020_energy`) as the
paper's MP-compatible energies were.
"""
from __future__ import annotations

from functools import cache

import numpy as np
from pymatgen.analysis.phase_diagram import (
    GrandPotentialPhaseDiagram,
    PDEntry,
    PhaseDiagram,
)
from pymatgen.core import Composition, Element


@cache
def _mp2020():
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

    return MaterialsProject2020Compatibility(check_potcar=False)


def mp_run_parameters(composition: Composition) -> dict:
    """MP's run type and U values for *composition*.

    MPRelaxSet applies U by the most electronegative element: the U elements (Co, Cr,
    Fe, Mn, Mo, Ni, V, W) get it in oxides and fluorides only. LeMat-Bulk's
    ``compatible_pbe`` follows the same scheme (``docs/lemat_bulk_source_databases.md``).
    """
    elements = [element for element in composition.elements if composition[element] > 0]
    settings = _mp2020().u_settings.get(max(elements, key=lambda e: e.X).symbol, {})
    hubbards = {e.symbol: float(settings[e.symbol]) for e in elements if settings.get(e.symbol)}
    return {"run_type": "GGA+U" if hubbards else "GGA", "hubbards": hubbards}


def mp2020_energy(structure, energy: float, entry_id: str = "") -> float | None:
    """*energy* (eV per cell of *structure*) with MP2020's corrections, None if refused.

    The structure is needed: it tells a sulfate from a sulfide and a peroxide from an
    oxide.
    """
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    entry = ComputedStructureEntry(structure, energy, entry_id=entry_id,
                                   parameters=mp_run_parameters(structure.composition))
    processed = _mp2020().process_entry(entry)
    return None if processed is None else float(processed.energy)


def placed_on_hull(pd: PhaseDiagram, composition: Composition, name: str,
                   margin: float = 1e-4) -> PDEntry:
    """An entry *margin* eV/atom below *pd*'s hull at *composition*, *name* its attribute.

    The margin makes it a vertex; with it the window is wider than the assemblage's by
    about margin * atoms / Li atoms in eV. Entries are told apart by ``attribute``:
    pymatgen's interface code reads ``name`` as a formula.
    """
    energy = pd.get_hull_energy(composition) - margin * composition.num_atoms
    return PDEntry(composition, energy, attribute=name)


def stability_window(pd: PhaseDiagram, name: str, element: str = "Li"
                     ) -> tuple[float, float] | None:
    """(min, max) of mu_element - mu_element(reference) over which the entry whose
    ``attribute`` is *name* is on the hull, or None if it is not a vertex.

    A vertex's stability region in chemical-potential space is the polytope whose
    vertices are the chemical potentials of the facets that hold it, so its extent in
    mu_element is their minimum and maximum.
    """
    open_element = Element(element)
    found = [i for i, entry in enumerate(pd.qhull_entries) if entry.attribute == name]
    if not found:
        return None
    mus = [pd._get_facet_chempots(facet)[open_element]
           for facet in pd.facets if found[0] in facet]
    if not mus:
        return None
    reference = pd.el_refs[open_element].energy_per_atom
    return min(mus) - reference, max(mus) - reference


def without(composition: Composition, element: str) -> Composition:
    return Composition({e: n for e, n in composition.items() if e.symbol != element})


def phases(decomposition: dict) -> str:
    """``a A + b B`` from a pymatgen decomposition, largest first.

    The amounts are pymatgen's: fractions of the atoms, of the atoms other than the
    open element in a grand-potential diagram.
    """
    def formula(entry):
        return getattr(entry, "original_entry", entry).composition.reduced_formula

    parts = sorted(decomposition.items(), key=lambda item: -item[1])
    return " + ".join(f"{amount:.2f} {formula(entry)}" for entry, amount in parts)


def grand_diagram(pd: PhaseDiagram, mu: float, element: str = "Li"
                  ) -> GrandPotentialPhaseDiagram:
    """*pd*'s entries open to *element* at *mu* relative to its reference."""
    open_element = Element(element)
    absolute = mu + pd.el_refs[open_element].energy_per_atom
    return GrandPotentialPhaseDiagram(pd.all_entries, {open_element: absolute})


def decomposition_at(pd: PhaseDiagram, composition: Composition, mu: float,
                     element: str = "Li") -> str:
    """The phases *composition* forms in contact with an *element* reservoir at *mu*."""
    grand = grand_diagram(pd, mu, element)
    return phases(grand.get_decomposition(without(composition, element).fractional_composition))


def _mixing_ratio(point: Composition, first: Composition, second: Composition) -> float:
    """x with point = x first + (1 - x) second, all fractional compositions."""
    elements = sorted(set(first) | set(second) | set(point), key=lambda e: e.Z)
    a = np.array([first[e] for e in elements]) - np.array([second[e] for e in elements])
    b = np.array([point[e] for e in elements]) - np.array([second[e] for e in elements])
    return float(np.clip(a @ b / (a @ a), 0.0, 1.0))


def chemical_reaction(pd: PhaseDiagram, electrolyte: Composition, electrode: Composition
                      ) -> dict:
    """Richards' eq. 2: the most negative reaction energy (eV/atom) of the two, both on
    the hull, over mixing ratios x of the electrode, and its products."""
    from pymatgen.analysis.interface_reactions import InterfacialReactivity

    reactivity = InterfacialReactivity(electrode, electrolyte, pd, norm=True,
                                       use_hull_energy=True)
    _, x, energy, reaction, _ = min(reactivity.get_kinks(), key=lambda kink: kink[2])
    return {"energy": float(energy), "x": float(x), "reaction": str(reaction)}


def electrochemical_reaction(pd: PhaseDiagram, electrolyte: Composition,
                             electrode: Composition, mu: float, element: str = "Li"
                             ) -> dict:
    """Richards' eqs. 4 and 5 at mu_element = *mu* relative to its reference.

    Energies are per atom other than *element*. ``no_mixing`` (eq. 5) is the
    electrolyte's own decomposition by the reservoir; ``energy`` (eq. 4) adds mixing
    with the electrode, held at its grand-potential equilibrium. The electrolyte is on
    the hull, as in the paper.
    """
    grand = grand_diagram(pd, mu, element)
    absolute = mu + pd.el_refs[Element(element)].energy_per_atom
    first = without(electrode, element).fractional_composition
    second = without(electrolyte, element).fractional_composition
    own = (pd.get_hull_energy(electrolyte) - electrolyte[element] * absolute) \
        / without(electrolyte, element).num_atoms
    no_mixing = grand.get_hull_energy(second) - own
    result = {"no_mixing": float(no_mixing), "energy": float(no_mixing), "x": 0.0,
              "products": phases(grand.get_decomposition(second))}
    if not len(first):          # the electrode is the reservoir itself, Li metal
        return result
    electrode_phi = grand.get_hull_energy(first)
    for point in grand.get_critical_compositions(first, second):
        point = point.fractional_composition
        x = _mixing_ratio(point, first, second)
        energy = grand.get_hull_energy(point) - x * electrode_phi - (1 - x) * own
        if energy < result["energy"] - 1e-9:
            result.update(energy=float(energy), x=x,
                          products=phases(grand.get_decomposition(point)))
    return result


def pseudo_binary_joins(composition: Composition, oxidation_states: dict[str, float],
                        mobile: str = "Li") -> list[tuple[str, Composition, Composition]]:
    """(anion, Li binary, rest) for each anion that can balance all of Li's charge.

    Li4GeSe4 lies on the join Li2Se-GeSe2, LiYTe2 on Li2Te-Y2Te3. The rest need not be
    a compound: Li4B4Sb2SO16's is B4Sb2SO14.
    """
    joins = []
    n_mobile = composition[mobile]
    for anion, state in oxidation_states.items():
        if anion == mobile or state >= 0 or Element(anion) not in composition:
            continue
        taken = n_mobile / -state
        if composition[anion] < taken - 1e-8:
            continue
        rest = Composition({e: n - (n_mobile if e.symbol == mobile else 0)
                            - (taken if e.symbol == anion else 0)
                            for e, n in composition.items()})
        rest = Composition({e: n for e, n in rest.items() if n > 1e-8})
        if len(rest):
            joins.append((anion, Composition({mobile: -state, anion: 1}),
                          Composition(rest.get_integer_formula_and_factor()[0])))
    return joins


def position_on_join(composition: Composition, binary: Composition, rest: Composition,
                     tol: float = 1e-6) -> float | None:
    """lambda with composition = lambda binary + (1 - lambda) rest (fractional), if the
    composition lies on the join, else None. End members give 0 or 1."""
    point = composition.fractional_composition
    first, second = binary.fractional_composition, rest.fractional_composition
    if set(point) - (set(first) | set(second)):
        return None
    elements = sorted(set(first) | set(second), key=lambda e: e.Z)
    a = np.array([first[e] - second[e] for e in elements])
    b = np.array([point[e] - second[e] for e in elements])
    position = float(a @ b / (a @ a))
    if np.abs(b - position * a).max() > tol or not -tol <= position <= 1 + tol:
        return None
    return position
