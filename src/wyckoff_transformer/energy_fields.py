"""What an energy number *is*: the machine-readable provenance of energy fields.

A column name cannot carry the meaning of an energy, and in this repository it
never did: ``energy_above_hull`` is raw PBE against LeMat-Bulk's own hull in one
dataset and MP2020-corrected against MP's hull in another, and LeMat's
``energy_corrected`` is not corrected at all.  An :class:`EnergyField` says what
a column holds, in the terms two numbers must share before they may be compared:

- the **quantity** (a total energy, a formation energy, a distance to a hull...)
  and its **extent** (per cell or per atom);
- the **source** of the number -- DFT at some settings with some correction, or
  an MLIP, whose scale is that of the DFT data it was trained on
  (:class:`EnergySource`);
- for everything but a bare energy, the **reference** entry set the elemental
  references and the convex hull are taken from.

Every value is drawn from a closed registry below, and each registry entry says
what it means and on what evidence -- so an error message can print it, and a
new label has to be argued for here before a dataset may use it.

:func:`check_compatible` is the one place that decides whether two fields may be
combined.  ``docs/energy_fields.md`` is the prose version of this module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

#: Bumped when the serialised form changes incompatibly.
FORMAT_VERSION = 1

#: What kind of energy a field holds.
QUANTITIES = {
    "energy": "The energy of the structure itself, relative to nothing.",
    "formation_energy": (
        "Energy relative to the elemental references of the reference entry set: "
        "E - sum_i n_i mu_i, with mu_i the lowest energy per atom of element i."),
    "energy_above_hull": (
        "Distance to the convex hull of the reference entry set at this composition. "
        "Non-negative by construction when the reference set contains the entry "
        "itself; genuinely negative only for a structure outside it."),
    "hull_formation_energy": (
        "The hull's own formation energy at a composition -- a property of the "
        "composition and the reference set, not of any one structure."),
    "delta_e_polymorph": (
        "Energy above the lowest-energy polymorph of the same reduced formula within "
        "the reference entry set."),
}

#: Per cell, or normalised by the number of atoms.  Units follow.
EXTENTS = {"total": "eV", "per_atom": "eV/atom"}

#: How the number was obtained.
METHODS = ("dft", "mlip")

#: DFT settings.  An MLIP's scale is that of the DFT data it was trained on, so an
#: MLIP carries one of these too.
DFT_SETTINGS = {
    "PBE_MP": (
        "Materials Project GGA/GGA+U settings (MPRelaxSet: PBE, MP's Hubbard U values "
        "and PAW pseudopotentials).  Shared by Materials Project, MPtrj, and LeMat-Bulk "
        "'compatible_pbe', which admits Alexandria and OQMD rows as MP-compatible.  "
        "Checked 2026-09-26 on 2.15M LeMat-Bulk rows that MP2020 leaves uncorrected: "
        "MACE-MP-0b3 (trained on MPtrj) sits at median +0.003 (MP rows), -0.010 "
        "(Alexandria) and -0.035 eV/atom (OQMD) from LeMat's DFT.  The OQMD offset "
        "cannot be told apart from the model extrapolating to OQMD's prototypes."),
    "PBE_OMat24": (
        "OMat24's settings.  Not the MP scale: ORB-v3-omat, UMA-omat and MACE-OMAT-0, "
        "three architectures, all sit +0.05 to +0.07 eV/atom above LeMat-Bulk's DFT on "
        "every source (2026-09-26), which is the training data's offset, not a model's."),
}

#: Energy corrections applied on top of the raw number.
CORRECTIONS = {
    "none": "The raw total energy, as the calculation (or the MLIP) produced it.",
    "MaterialsProject2020Compatibility": (
        "pymatgen's MaterialsProject2020Compatibility: fitted anion corrections plus "
        "the GGA/GGA+U mixing corrections for oxides and fluorides of Co, Cr, Fe, Mn, "
        "Mo, Ni, V, W.  About -0.7 eV/atom on a typical +U oxide."),
}


@dataclass(frozen=True)
class MlipModel:
    """One exact MLIP checkpoint, and the scale of the data it was trained on."""
    dft_settings: str
    correction: str
    description: str


#: Exact checkpoints.  The training-data scale is a property of the checkpoint,
#: so a field naming a model must repeat it -- validation checks that it does.
MLIP_MODELS = {
    "orb-v3-conservative-inf-omat-20250404": MlipModel(
        "PBE_OMat24", "none", "ORB v3 conservative, infinite neighbours, trained on OMat24."),
    "orb-v3-direct-20-omat-20250404": MlipModel(
        "PBE_OMat24", "none", "ORB v3 direct, 20 neighbours, trained on OMat24."),
    "uma-s-1/omat": MlipModel(
        "PBE_OMat24", "none", "fairchem UMA small v1, omat task head."),
    "mace-mp-0b3-medium": MlipModel(
        "PBE_MP", "none",
        "MACE-MP-0b3 medium, trained on MPtrj's uncorrected energies (the slope of "
        "MLIP-DFT against the MP2020 correction is -0.002, measured 2026-09-26)."),
    "MACE-OMAT-0-medium": MlipModel(
        "PBE_OMat24", "none", "MACE-OMAT-0 medium, trained on OMat24."),
}

#: Hull splits of HF ``LeMaterial/LeMat-Bulk-MLIP-Hull`` at the pinned revision, by
#: the MLIP each was computed with.
_MLIP_HULL_REVISION = "70d505bb"
_MLIP_HULL_SPLITS = {
    "orb_conserv_inf": "orb-v3-conservative-inf-omat-20250404",
    "orb_direct_20": "orb-v3-direct-20-omat-20250404",
    "uma": "uma-s-1/omat",
    "mace_mp": "mace-mp-0b3-medium",
    "mace_omat": "MACE-OMAT-0-medium",
    "dft": None,
}


def mlip_hull_reference(hull_type: str) -> str:
    """The reference id of one LeMat-Bulk-MLIP-Hull split."""
    return f"lemat_bulk_mlip_hull/{hull_type}@{_MLIP_HULL_REVISION}"


def mlip_hull_energy_field(hull_type: str) -> "EnergyField":
    """What an ``e_above_hull`` scored against one LeMat-Bulk-MLIP-Hull split means.

    The structure's energy comes from the split's own potential (or, for ``dft``,
    from DFT), and the hull from the same potential's energies of the archive.
    """
    model = _MLIP_HULL_SPLITS[hull_type]
    if model is None:
        source = EnergySource("dft", "PBE_MP", "none")
    else:
        spec = MLIP_MODELS[model]
        source = EnergySource("mlip", spec.dft_settings, spec.correction, model)
    field = EnergyField("energy_above_hull", "per_atom", source, mlip_hull_reference(hull_type))
    field.validate()
    return field


#: Reference entry sets.  Elemental references and the hull both come from the set,
#: so one id covers both.
REFERENCES = {
    "lemat_bulk_pbe": (
        "All 5,335,299 LeMat-Bulk 'compatible_pbe' rows (MP, Alexandria, OQMD) at their "
        "raw PBE/PBE+U energies, no correction.  Elemental references: the lowest energy "
        "per atom of each element's entries.  Hull: one pymatgen PhaseDiagram per "
        "chemical system (formula_energy/hull_table.py).  The pre-2026-09-07 "
        "scripts/compute_e_hull.py used the same set: it left 589,250 rows (Z >= 84, Yb) "
        "unlabelled, and the 4,746,049 labels it did produce are reproduced bit for bit "
        "(docs/e_hull_definitions.md)."),
    "lemat_bulk_pbe_mp_oqmd": (
        "The MP (ICSD and theoretical) and OQMD rows of LeMat-Bulk only -- no Alexandria "
        "(formula_energy/answer_key.py SHALLOW_SOURCES) -- with elemental references and "
        "hull recomputed from that subset.  It stands for what was known before "
        "Alexandria's substitution campaign.  Formation energies differ from "
        "lemat_bulk_pbe wherever an elemental reference came from Alexandria."),
    "mp_gga_gga_u_2026.04.13": (
        "Materials Project release 2026.04.13, GGA_GGA+U thermo documents: MP's own "
        "phase diagram over its GGA/GGA+U entries, MP2020-corrected."),
    "mp_cdvae_2021": (
        "The Materials Project hull at the time CDVAE pulled MP-20 (snapshot date "
        "unknown, about 2021), MP2020-corrected.  Not reproducible from anything here."),
    **{
        mlip_hull_reference(hull_type): (
            f"HF LeMaterial/LeMat-Bulk-MLIP-Hull, split {hull_type!r} at revision "
            f"{_MLIP_HULL_REVISION}: LeMat-Bulk structures within 1 meV/atom of the hull "
            f"built from {model or 'the DFT (true_energy)'} energies of the whole archive.")
        for hull_type, model in _MLIP_HULL_SPLITS.items()
    },
}

#: A reference that is a dataset's own rows, e.g. the polymorphs ``delta_e_polymorph``
#: is measured against.  Any dataset name may follow the prefix.
DATASET_REFERENCE_PREFIX = "dataset:"

#: Statistics taken over rows before the value was stored.
AGGREGATES = {
    "none": "One row's own value.",
    "gene_min": "Minimum over all rows sharing the Wyckoff gene (gene_energy.py).",
    "formula_min": "Minimum over all rows sharing the reduced formula.",
}

#: The canonical field id for a quantity at an extent.  Dataset columns carry these
#: names and nothing else; see :func:`canonical_id`.
_CANONICAL_BASE = {
    ("energy", "total"): "energy",
    ("energy", "per_atom"): "energy_per_atom",
    ("formation_energy", "total"): "formation_energy",
    ("formation_energy", "per_atom"): "formation_energy_per_atom",
    ("energy_above_hull", "per_atom"): "energy_above_hull",
    ("energy_above_hull", "total"): "energy_above_hull_total",
    ("hull_formation_energy", "per_atom"): "hull_formation_energy_per_atom",
    ("delta_e_polymorph", "per_atom"): "delta_e_polymorph",
    ("delta_e_polymorph", "total"): "delta_e_polymorph_total",
}
_AGGREGATE_PREFIX = {"none": "", "gene_min": "gene_min_", "formula_min": "min_"}

#: Quantities that are meaningless without a reference entry set.
_NEEDS_REFERENCE = frozenset(QUANTITIES) - {"energy"}


class EnergyFieldError(ValueError):
    """A field's provenance is malformed or uses an unregistered label."""


class IncompatibleEnergyFieldError(ValueError):
    """Two energy fields that were about to be combined do not mean the same thing."""


def _require(value: Any, allowed: Mapping | Tuple, what: str) -> None:
    if value not in allowed:
        raise EnergyFieldError(
            f"Unknown {what} {value!r}; registered: {sorted(allowed)}.  "
            f"Add it to wyckoff_transformer.energy_fields with its meaning first.")


def reference_description(reference: str) -> str:
    """The registered meaning of a reference id."""
    if reference.startswith(DATASET_REFERENCE_PREFIX):
        return (f"The rows of dataset {reference[len(DATASET_REFERENCE_PREFIX):]!r} "
                f"themselves.")
    return REFERENCES[reference]


@dataclass(frozen=True)
class EnergySource:
    """How an energy was computed.

    DFT -> settings -> correction; or MLIP -> exact checkpoint -> the settings and
    correction of the DFT labels it was trained on.  A correction applied to an
    MLIP's output after inference is not this: it would make a different field.
    """
    method: str
    dft_settings: str
    correction: str = "none"
    mlip_model: Optional[str] = None

    def validate(self) -> None:
        _require(self.method, METHODS, "energy method")
        _require(self.dft_settings, DFT_SETTINGS, "DFT settings")
        _require(self.correction, CORRECTIONS, "energy correction")
        if self.method == "dft":
            if self.mlip_model is not None:
                raise EnergyFieldError(
                    f"A DFT energy names an MLIP model ({self.mlip_model!r}).")
            return
        if self.mlip_model is None:
            raise EnergyFieldError("An MLIP energy must name its exact checkpoint.")
        _require(self.mlip_model, MLIP_MODELS, "MLIP checkpoint")
        model = MLIP_MODELS[self.mlip_model]
        if (model.dft_settings, model.correction) != (self.dft_settings, self.correction):
            raise EnergyFieldError(
                f"MLIP {self.mlip_model!r} was trained on {model.dft_settings} with "
                f"correction {model.correction!r}, not {self.dft_settings} with "
                f"{self.correction!r}.")

    def to_dict(self) -> dict:
        out = {"method": self.method, "dft_settings": self.dft_settings,
               "correction": self.correction}
        if self.mlip_model is not None:
            out["mlip_model"] = self.mlip_model
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> "EnergySource":
        unknown = set(data) - {"method", "dft_settings", "correction", "mlip_model"}
        if unknown:
            raise EnergyFieldError(f"Unknown energy source keys: {sorted(unknown)}")
        source = cls(method=data["method"], dft_settings=data["dft_settings"],
                     correction=data.get("correction", "none"),
                     mlip_model=data.get("mlip_model"))
        source.validate()
        return source

    def describe(self) -> str:
        if self.method == "mlip":
            return (f"MLIP {self.mlip_model} (trained on {self.dft_settings}, "
                    f"correction {self.correction})")
        return f"DFT {self.dft_settings}, correction {self.correction}"


@dataclass(frozen=True)
class EnergyField:
    """What an energy column means."""
    quantity: str
    extent: str
    source: EnergySource
    reference: Optional[str] = None
    aggregate: str = "none"

    @property
    def units(self) -> str:
        return EXTENTS[self.extent]

    def validate(self) -> None:
        _require(self.quantity, QUANTITIES, "energy quantity")
        _require(self.extent, EXTENTS, "extent")
        _require(self.aggregate, AGGREGATES, "aggregate")
        if (self.quantity, self.extent) not in _CANONICAL_BASE:
            raise EnergyFieldError(
                f"{self.quantity} has no {self.extent} form.")
        self.source.validate()
        if self.quantity in _NEEDS_REFERENCE:
            if self.reference is None:
                raise EnergyFieldError(
                    f"A {self.quantity} is meaningless without its reference entry set.")
            if not self.reference.startswith(DATASET_REFERENCE_PREFIX):
                _require(self.reference, REFERENCES, "reference entry set")
        elif self.reference is not None:
            raise EnergyFieldError(
                f"A bare {self.quantity} has no reference, but {self.reference!r} is given.")

    def to_dict(self) -> dict:
        out = {"quantity": self.quantity, "extent": self.extent,
               "source": self.source.to_dict()}
        if self.reference is not None:
            out["reference"] = self.reference
        if self.aggregate != "none":
            out["aggregate"] = self.aggregate
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> "EnergyField":
        unknown = set(data) - {"quantity", "extent", "source", "reference", "aggregate"}
        if unknown:
            raise EnergyFieldError(f"Unknown energy field keys: {sorted(unknown)}")
        field = cls(quantity=data["quantity"], extent=data["extent"],
                    source=EnergySource.from_dict(data["source"]),
                    reference=data.get("reference"),
                    aggregate=data.get("aggregate", "none"))
        field.validate()
        return field

    def describe(self) -> str:
        parts = [f"{self.quantity} ({self.units})", self.source.describe()]
        if self.reference is not None:
            parts.append(f"reference {self.reference}")
        if self.aggregate != "none":
            parts.append(f"aggregate {self.aggregate}")
        return ", ".join(parts)


def canonical_id(field: EnergyField, qualifier: Optional[str] = None) -> str:
    """The column name a field must carry.

    A name states the quantity, its extent and any aggregate -- never a model, a
    tokeniser or a data source; those are what the provenance is for.  A dataset
    holding two definitions of one quantity tells them apart by a ``qualifier``
    naming the difference in scale (``energy_per_atom_uncorrected``).
    """
    name = _AGGREGATE_PREFIX[field.aggregate] + _CANONICAL_BASE[(field.quantity, field.extent)]
    return f"{name}_{qualifier}" if qualifier else name


def differences(expected: Optional[EnergyField],
                actual: Optional[EnergyField]) -> List[Tuple[str, Any, Any]]:
    """The parts in which two fields differ, as (part, expected, actual).

    ``None`` stands for unknown provenance, which matches nothing -- not even
    another unknown.  ``aggregate`` is left out: it is a choice of the consumer
    (a gene-minimum regressor may be screened against per-row energies), not a
    difference of scale.
    """
    if expected is None or actual is None:
        return [("provenance",
                 "unknown" if expected is None else expected.describe(),
                 "unknown" if actual is None else actual.describe())]
    pairs = [
        ("quantity", expected.quantity, actual.quantity),
        ("extent", expected.extent, actual.extent),
        ("source.method", expected.source.method, actual.source.method),
        ("source.dft_settings", expected.source.dft_settings, actual.source.dft_settings),
        ("source.correction", expected.source.correction, actual.source.correction),
        ("source.mlip_model", expected.source.mlip_model, actual.source.mlip_model),
        ("reference", expected.reference, actual.reference),
    ]
    return [pair for pair in pairs if pair[1] != pair[2]]


def check_compatible(expected: Optional[EnergyField], actual: Optional[EnergyField],
                     context: str, allow: bool = False) -> List[str]:
    """Refuse to combine two energy fields that do not mean the same thing.

    Args:
        expected: The field the consumer was built for, e.g. a model's target.
        actual: The field it is about to be given, e.g. a reference table's column.
        context: What is being combined, for the message.
        allow: Downgrade the error to a warning -- the ``--allow-incompatible-energy``
            escape hatch.  The differences are still returned, so the caller can
            record in its output that the combination was forced.

    Returns:
        The differences, one human-readable line each; empty when compatible.

    Raises:
        IncompatibleEnergyFieldError: The fields differ and ``allow`` is False.
    """
    lines = [f"  {part}: {left!r} (expected) != {right!r} (given)"
             for part, left, right in differences(expected, actual)]
    if not lines:
        return []
    message = (f"Incompatible energy fields ({context}):\n" + "\n".join(lines))
    for reference in {getattr(expected, "reference", None), getattr(actual, "reference", None)}:
        if reference is not None:
            message += f"\n  {reference}: {reference_description(reference)}"
    if not allow:
        raise IncompatibleEnergyFieldError(
            message + "\nPass --allow-incompatible-energy to combine them anyway.")
    logger.warning("%s\nCombining them anyway, as --allow-incompatible-energy asks.", message)
    return lines
