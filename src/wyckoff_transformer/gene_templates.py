"""Building Wyckoff genes that carry a chosen motif, at three levels of specification.

1. **Complete genes**, by decorating a prototype: :func:`decorate` enumerates every
   charge-balanced assignment of elements to the slots of a template, e.g. the 2H
   hexagonal perovskite (P6_3/mmc: A 2c, B 2a, X 6h) with a d0 B cation.
2. **Forced prefixes**: :func:`forced_prefix_tokens` fixes some sites of a gene
   (element and Wyckoff position) and lets the model complete the rest. WyFormer is
   trained on sites in a uniformly random order, so the completion is drawn from
   p(rest | prefix), not from a distribution the model never saw.
3. **Site-symmetry masks**: :func:`site_symmetry_mask` restricts where chosen elements
   may sit, e.g. Cr only on sites whose point group admits no octahedron. Every Cr the
   model places then satisfies the rule; whether Cr appears at all is up to
   `required_element_set`.

Levels 2 and 3 go through ``WyckoffGenerator.generate_tensors`` (``forced_tokens``,
``conditional_masks``) and ``wyformer-generate`` (``--force-sites``, ``--site-rule``).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from wyckoff_transformer.evaluation.coordination_motifs import ALLOWED_BY, POLAR_POINT_GROUPS
from wyckoff_transformer.evaluation.gene_motifs import _site_point_groups

# --- level 1: prototype decoration ------------------------------------------------------


@dataclass(frozen=True)
class Template:
    """A prototype with element slots.

    ``slots`` maps a slot name to its Wyckoff positions (one entry per orbit, e.g.
    ``["2a"]``); ``palettes`` maps each slot to candidate (element, oxidation state)
    pairs. Slots are filled with distinct elements.
    """
    name: str
    space_group: int
    slots: Dict[str, Tuple[str, ...]]
    palettes: Dict[str, Tuple[Tuple[str, int], ...]]


def _multiplicity(site: str) -> int:
    return int(site[:-1])


def decorate(template: Template) -> List[dict]:
    """Every charge-balanced decoration of ``template``, as pyxtal-notation genes."""
    names = list(template.slots)
    genes = []
    for choice in product(*(template.palettes[n] for n in names)):
        elements = [el for el, _ in choice]
        if len(set(elements)) != len(elements):
            continue
        charge = sum(z * sum(_multiplicity(s) for s in template.slots[n])
                     for n, (_, z) in zip(names, choice))
        if charge != 0:
            continue
        genes.append({
            "group": template.space_group,
            "species": elements,
            "sites": [list(template.slots[n]) for n in names],
            "numIons": [sum(_multiplicity(s) for s in template.slots[n]) for n in names],
            "template": template.name,
            "oxidation_states": {el: z for el, z in choice},
        })
    return genes


_ALKALI = (("Li", 1), ("Na", 1), ("K", 1), ("Rb", 1), ("Cs", 1), ("Tl", 1), ("Ag", 1))
_ALKALINE_EARTH = (("Mg", 2), ("Ca", 2), ("Sr", 2), ("Ba", 2), ("Pb", 2), ("Sn", 2),
                   ("Eu", 2), ("Cd", 2))
_D0_4 = (("Ti", 4), ("Zr", 4), ("Hf", 4))
_D0_5 = (("V", 5), ("Nb", 5), ("Ta", 5))

#: 2H hexagonal perovskite (BaNiO3 type): B octahedra share faces in infinite chains
#: along c. The d0 version is absent from all 5.3M LeMat-Bulk genes.
TEMPLATE_2H_D0 = Template(
    name="2H_d0_chain",
    space_group=194,
    slots={"A": ("2c",), "B": ("2a",), "X": ("6h",)},
    palettes={"A": _ALKALI + _ALKALINE_EARTH, "B": _D0_4 + _D0_5, "X": (("O", -2),)},
)

TEMPLATES = {t.name: t for t in (TEMPLATE_2H_D0,)}


# --- level 2: forced prefixes -----------------------------------------------------------


def parse_sites(spec: str) -> List[Tuple[str, str]]:
    """``"Cr:2d,O:6h"`` -> [("Cr", "d"), ("O", "h")]; the multiplicity is optional."""
    out = []
    for item in spec.split(","):
        element, site = item.strip().split(":")
        out.append((element.strip(), site.strip().lstrip("0123456789")))
    return out


def forced_prefix_tokens(trainer, space_group: int, sites: Sequence[Tuple[str, str]],
                         n_rows: int) -> torch.Tensor:
    """[n_rows, len(sites), len(cascade_order)] token ids for a forced prefix; -1 = free."""
    from wyckoff_transformer.tokenization import load_wyckoff_mappings

    mappings = load_wyckoff_mappings(trainer.run_path)
    ss_from_letter = mappings.ss_from_letter[space_group]
    enum_from_letter = mappings.enum_from_ss_letter[space_group]
    tokens = torch.full((len(sites), len(trainer.cascade_order)), -1, dtype=torch.int64)
    for i, (element, letter) in enumerate(sites):
        if letter not in ss_from_letter:
            raise ValueError(f"Space group {space_group} has no Wyckoff position {letter!r}")
        values = {"elements": element, "site_symmetries": ss_from_letter[letter],
                  "sites_enumeration": enum_from_letter[letter]}
        for j, field in enumerate(trainer.cascade_order):
            if field in values and trainer.cascade_is_target.get(field, False):
                tokeniser = trainer.tokenisers[field]
                if values[field] not in tokeniser:
                    raise KeyError(f"{field} value {values[field]!r} is not in the model's vocabulary")
                tokens[i, j] = tokeniser[values[field]]
    return tokens.unsqueeze(0).expand(n_rows, -1, -1).clone()


# --- level 3: site-symmetry masks -------------------------------------------------------

#: Named rules on the unoriented site point group.
SITE_RULES = {
    "no-octahedron": lambda pg: pg not in ALLOWED_BY["octahedron"],
    "no-square-planar": lambda pg: pg not in ALLOWED_BY["square_planar"],
    "no-square-planar-no-octahedron": lambda pg: (pg not in ALLOWED_BY["square_planar"]
                                                  and pg not in ALLOWED_BY["octahedron"]),
    "non-polar": lambda pg: pg not in POLAR_POINT_GROUPS,
}


def symbol_point_groups() -> Dict[str, str]:
    """Oriented site-symmetry symbol -> unoriented point group, over all 230 groups."""
    from wyckoff_transformer.tokenization import load_wyckoff_mappings

    ss_from_letter = load_wyckoff_mappings().ss_from_letter
    out: Dict[str, str] = {}
    for sg in range(1, 231):
        pgs = _site_point_groups(sg)
        for letter, symbol in ss_from_letter[sg].items():
            pg = pgs[letter]
            if out.setdefault(symbol, pg) != pg:
                raise ValueError(f"site symmetry {symbol!r} maps to both {out[symbol]} and {pg}")
    return out


def parse_rules(spec: str) -> Dict[str, str]:
    """``"Cr:no-octahedron,Pd:no-square-planar"`` -> {"Cr": "no-octahedron", ...}."""
    out = {}
    for item in spec.split(","):
        element, rule = item.strip().split(":")
        if rule not in SITE_RULES:
            raise ValueError(f"unknown site rule {rule!r}; known: {sorted(SITE_RULES)}")
        out[element.strip()] = rule
    return out


def site_symmetry_mask(trainer, rules: Dict[str, str]) -> Tuple[str, str, torch.Tensor]:
    """A (source, target, table) triple for ``conditional_masks``: listed elements may
    only take site symmetries their rule allows; every other element is unrestricted."""
    elements = trainer.tokenisers["elements"]
    symmetries = trainer.tokenisers["site_symmetries"]
    pgs = symbol_point_groups()
    width = max(symmetries.values()) + 1
    table = torch.ones((max(elements.values()) + 1, width), dtype=torch.bool)
    for element, rule in rules.items():
        if element not in elements:
            raise KeyError(f"element {element!r} is not in the model's vocabulary")
        row = torch.zeros(width, dtype=torch.bool)
        for symbol, token in symmetries.items():
            if symbol in pgs and SITE_RULES[rule](pgs[symbol]):
                row[token] = True
        if not bool(row.any()):
            raise ValueError(f"no site symmetry satisfies {rule!r}")
        table[elements[element]] = row
    return "elements", "site_symmetries", table


def space_groups_with_site(rule: str) -> List[int]:
    """Space groups having at least one Wyckoff position whose point group passes ``rule``."""
    return [sg for sg in range(1, 231)
            if any(SITE_RULES[rule](pg) for pg in _site_point_groups(sg).values())]
