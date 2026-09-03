"""MLIPs for which LeMat-Bulk publishes a convex hull.

``e_above_hull = E_structure - E_hull(composition)`` is only meaningful when both
terms come from the same potential: MLIPs sit on model-specific absolute energy
scales, and mixing them contaminates the difference.  Measured over the 204976
structures shared by every split of ``LeMaterial/LeMat-Bulk-MLIP-Hull``, the
per-atom offset between ``mace_mp`` and any of the OMat24-family models is
0.058 eV/atom (sd 0.104), against the ~0.023 eV/atom effect sizes this protocol
is built to resolve.  LeMat-GenBench calls this the *self-consistent hull* and
pairs each model with its own.

This module therefore admits exactly the hull names published in that dataset
and refuses anything else, so a caller cannot silently evaluate against a hull
that was never computed for their potential.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class UnsupportedHullMlip(ValueError):
    """Raised for an MLIP that LeMat-Bulk publishes no convex hull for."""


@dataclass(frozen=True)
class HullMlipSpec:
    """One entry of ``LeMaterial/LeMat-Bulk-MLIP-Hull``.

    Attributes:
        hull_type: Split name in the published dataset, and the value
            LeMat-GenBench's ``get_energy_above_hull`` expects as ``hull_type``.
        energy_column: Column holding this model's total energy in every split
            of the published parquet.
        builder: Constructs the ASE calculator, or ``None`` when the split is
            not a runnable potential.
        checkpoint: Human-readable identity of the weights, for the manifest.
        note: Anything a caller needs to know before trusting the pairing.
    """

    hull_type: str
    energy_column: str
    builder: Optional[str]
    checkpoint: str
    note: str = ""

    @property
    def is_runnable(self) -> bool:
        return self.builder is not None


#: ORB v3 conservative, unlimited neighbours, OMat24.  The only one of the three
#: LeMat-GenBench ensemble members whose checkpoint is identifiable from source:
#: ``orb-models`` hard-codes the weights URL as a default argument, and it is
#: byte-identical from v0.5.1 through v0.7.0.
ORB_CONSERV_INF_CHECKPOINT = (
    "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/"
    "forcefields/orb-v3/orb-v3-conservative-inf-omat-20250404.ckpt"
)
ORB_DIRECT_20_CHECKPOINT = (
    "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/"
    "forcefields/orb-v3/orb-v3-direct-20-omat-20250404.ckpt"
)

#: LeMat-GenBench's ORB calculator defaults, reproduced so that energies land on
#: the same scale as the published hull.
ORB_PRECISION = "float32-high"

_MACE_MP_NOTE = (
    "UNIDENTIFIED CHECKPOINT. LeMat-GenBench built this hull by calling "
    "mace_mp() with no 'model' argument, whose meaning changed in mace-torch "
    "0.3.10: before, it resolved to 'medium' (MACE-MP-0a-medium, MP-only); "
    "after, to 'medium-mpa-0' (MACE-MPA-0-medium, MPtrj+Alexandria). The paper "
    "describes MACE-MP as 'trained exclusively on Materials Project' (the "
    "former); the project's lock file pins mace-torch 0.3.13 (the latter). "
    "Neither reproduces the published mace_mp_energy: measured on 4 LeMat-Bulk "
    "structures, MACE-MP-0a-medium is off by mean +0.0010 eV/atom (max 0.0223) "
    "and MACE-MPA-0-medium by mean +0.0046 (max 0.0140), with float32 and "
    "float64 agreeing to 1e-6 so dtype is not the cause. For scale, ORB "
    "reproduces its own hull through the identical code path to within "
    "9e-5 eV/atom. Treat any e_above_hull from this hull as carrying a few "
    "meV/atom of unexplained systematic error, and prefer orb_conserv_inf."
)

#: Every split published in ``LeMaterial/LeMat-Bulk-MLIP-Hull``.
HULL_MLIPS: dict[str, HullMlipSpec] = {
    "orb_conserv_inf": HullMlipSpec(
        hull_type="orb_conserv_inf",
        energy_column="orb_conserv_inf_energy",
        builder="orb",
        checkpoint=ORB_CONSERV_INF_CHECKPOINT,
    ),
    "orb_direct_20": HullMlipSpec(
        hull_type="orb_direct_20",
        energy_column="orb_direct_20_energy",
        builder="orb",
        checkpoint=ORB_DIRECT_20_CHECKPOINT,
        note="Published hull, but not part of the LeMat-GenBench ensemble.",
    ),
    "uma": HullMlipSpec(
        hull_type="uma",
        energy_column="uma_energy",
        builder="uma",
        checkpoint="uma-s-1 (task=omat)",
        note=(
            "The LeMat-GenBench paper names uma-s1p1 while the code default is "
            "uma-s-1; fairchem resolves names through a registry rather than a "
            "weights URL, so the pairing is unverified."
        ),
    ),
    "mace_mp": HullMlipSpec(
        hull_type="mace_mp",
        energy_column="mace_mp_energy",
        builder="mace",
        # Resolves to the same weights file as mace-torch's 'medium' alias, but
        # names it, so the identity does not depend on the installed version.
        checkpoint="MACE-MP-0a-medium",
        note=_MACE_MP_NOTE,
    ),
    "mace_omat": HullMlipSpec(
        hull_type="mace_omat",
        energy_column="mace_omat_energy",
        builder="mace",
        checkpoint="MACE-OMAT-0-medium",
        note="Published hull, but not part of the LeMat-GenBench ensemble.",
    ),
    "dft": HullMlipSpec(
        hull_type="dft",
        energy_column="true_energy",
        builder=None,
        checkpoint="PBE (LeMat-Bulk reference)",
        note="A hull, not a potential: there is nothing to run locally.",
    ),
}

#: ORB is the default because it is the only ensemble member whose checkpoint is
#: pinned by a URL literal in its own source, and it tracks the three-model
#: ensemble mean twice as closely as MACE does (residual sd 0.036 vs 0.071
#: eV/atom, after removing each model's constant offset).
DEFAULT_HULL_MLIP = "orb_conserv_inf"


def resolve_hull_mlip(name: str) -> HullMlipSpec:
    """Look up an MLIP, refusing any without a published LeMat-Bulk hull.

    Args:
        name: A hull split name, e.g. ``"orb_conserv_inf"``.

    Returns:
        The matching :class:`HullMlipSpec`.

    Raises:
        UnsupportedHullMlip: If *name* is not published, or names a split that
            is a hull but not a runnable potential (``"dft"``).
    """
    spec = HULL_MLIPS.get(name)
    if spec is None:
        raise UnsupportedHullMlip(
            f"{name!r} has no convex hull in LeMaterial/LeMat-Bulk-MLIP-Hull, so "
            f"e_above_hull computed with it would mix energy scales. "
            f"Available: {', '.join(sorted(HULL_MLIPS))}."
        )
    if not spec.is_runnable:
        raise UnsupportedHullMlip(
            f"{name!r} is a published hull but not a runnable potential: {spec.note}"
        )
    if spec.note:
        logger.warning("%s: %s", name, spec.note)
    return spec


def build_hull_calculator(name: str, device: str = "cpu"):
    """Build the ASE calculator whose energies match *name*'s published hull.

    The construction mirrors LeMat-GenBench's own calculators, since any
    difference in how the model is loaded shows up as an energy offset against
    the hull it is paired with.

    Args:
        name: A runnable hull split name; see :data:`HULL_MLIPS`.
        device: Torch device string, e.g. ``"cpu"`` or ``"cuda:0"``.

    Returns:
        An ASE ``Calculator``.

    Raises:
        UnsupportedHullMlip: If *name* has no published hull.
        ImportError: If the backend's dependency set is missing.
    """
    spec = resolve_hull_mlip(name)

    if spec.builder == "orb":
        return _build_orb(spec, device)
    if spec.builder == "mace":
        return _build_mace(spec, device)
    if spec.builder == "uma":
        return _build_uma(spec, device)
    raise UnsupportedHullMlip(f"No builder registered for {name!r}")


#: HuggingFace dataset holding the published per-model energies and hulls.
HULL_REPO_ID = "LeMaterial/LeMat-Bulk-MLIP-Hull"

#: Local LeMat-Bulk export carrying ``immutable_id`` and ``cif``.  The hull
#: parquet has no geometry, so verification needs structures from here.
DEFAULT_LEMAT_CIF_CSV = Path("data/lemat-bulk/lemat_pbe.csv.gz")


def verify_hull_energies(
    mlip: str,
    n_structures: int = 20,
    lemat_cif_csv: Path = DEFAULT_LEMAT_CIF_CSV,
    hull_parquet: Optional[Path] = None,
    device: str = "cpu",
    chunksize: int = 50_000,
):
    """Check that our calculator reproduces the published per-structure energies.

    This is the only way to establish which checkpoint built a given hull:
    LeMat-GenBench pins no model versions, and for ``mace_mp`` the identity is
    genuinely ambiguous (see :data:`HULL_MLIPS`).  A checkpoint mismatch shows up
    as a large systematic offset, which would otherwise propagate silently into
    every ``e_above_hull`` derived from that hull.

    Energies are single points on the LeMat-Bulk geometry with no relaxation,
    matching how the published values were produced (``relax_structures=False``).

    Sensitivity.  The CIF round-trip through *lemat_cif_csv* is not the limiting
    factor: ORB reproduces its published energies through exactly this path to
    within 90 ueV/atom (mean -7e-6 eV/atom over 8 structures), which both
    confirms the ORB pairing and shows the geometry survives the export.  A
    residual of milli-eV/atom or more therefore indicates a real difference in
    the model -- a different checkpoint, or a different dtype -- rather than
    noise.

    Args:
        mlip: A runnable hull split name.
        n_structures: How many structures to check.  A dozen suffices: a wrong
            checkpoint is off by tens of meV/atom, not by noise.
        lemat_cif_csv: Local LeMat-Bulk export with ``immutable_id`` and ``cif``.
        hull_parquet: The published parquet.  Downloaded from
            :data:`HULL_REPO_ID` when omitted.
        device: Torch device for the calculator.
        chunksize: Rows per CSV chunk while searching for matching ids.

    Returns:
        A ``DataFrame`` with one row per structure: ``immutable_id``,
        ``n_atoms``, ``published`` and ``ours`` total energies, and their
        per-atom difference ``delta_per_atom``.  Agreement to well under
        1 meV/atom means the checkpoint matches.
    """
    import pandas as pd
    from ase.io import read as ase_read

    spec = resolve_hull_mlip(mlip)

    if hull_parquet is None:
        from huggingface_hub import hf_hub_download

        hull_parquet = hf_hub_download(
            repo_id=HULL_REPO_ID,
            filename=f"data/{spec.hull_type}-00000-of-00001.parquet",
            repo_type="dataset",
        )
    published = pd.read_parquet(
        hull_parquet, columns=["immutable_id", "nsites", spec.energy_column]
    ).dropna(subset=[spec.energy_column])
    wanted = dict(
        zip(published["immutable_id"], published[spec.energy_column])
    )

    # The CIF export is ~1 GB; stop as soon as enough ids have matched.
    found: list[tuple[str, str, float]] = []
    for chunk in pd.read_csv(
        lemat_cif_csv, chunksize=chunksize, usecols=["immutable_id", "cif"]
    ):
        hits = chunk[chunk["immutable_id"].isin(wanted)]
        for _, row in hits.iterrows():
            found.append((row["immutable_id"], row["cif"], wanted[row["immutable_id"]]))
            if len(found) >= n_structures:
                break
        if len(found) >= n_structures:
            break

    if not found:
        raise RuntimeError(
            f"No overlap between {lemat_cif_csv} and the {spec.hull_type} hull."
        )

    calculator = build_hull_calculator(mlip, device=device)
    rows = []
    for immutable_id, cif, published_energy in found:
        import io

        atoms = ase_read(io.StringIO(cif), format="cif")
        atoms.calc = calculator
        ours = float(atoms.get_potential_energy())
        rows.append(
            {
                "immutable_id": immutable_id,
                "n_atoms": len(atoms),
                "published": published_energy,
                "ours": ours,
                "delta_per_atom": (ours - published_energy) / len(atoms),
            }
        )

    frame = pd.DataFrame(rows)
    delta = frame["delta_per_atom"]
    logger.info(
        "%s vs published %s: mean delta %.6f eV/atom, max |delta| %.6f over %d structures",
        spec.checkpoint, spec.energy_column, delta.mean(), delta.abs().max(), len(frame),
    )
    return frame


def _build_orb(spec: HullMlipSpec, device: str):
    """Build an ORB calculator across the 0.5/0.6 and 0.7 APIs.

    The weights are stable -- the same date-stamped URL from orb-models 0.5.1
    through 0.7.0 -- but the API around them is not.  Up to 0.6 the loader lived
    in ``orb_models.forcefield.calculator`` and returned a model; from 0.7 it is
    in ``orb_models.forcefield.inference.calculator`` and returns a
    ``(model, atoms_adapter)`` pair.  Both are supported so that this pairs with
    LeMat-GenBench's locked 0.5.4 as well as current releases.
    """
    try:
        from orb_models.forcefield import pretrained
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "orb-models is required for the ORB hulls: pip install 'orb-models>=0.5.1'"
        ) from exc

    try:  # orb-models >= 0.7
        from orb_models.forcefield.inference.calculator import ORBCalculator
    except ImportError:  # orb-models < 0.7
        from orb_models.forcefield.calculator import ORBCalculator

    model_func = {
        "orb_conserv_inf": "orb_v3_conservative_inf_omat",
        "orb_direct_20": "orb_v3_direct_20_omat",
    }[spec.hull_type]
    # weights_path is left at its default: it is a date-stamped URL literal in
    # orb-models, and overriding it would break the pairing with the hull.
    loaded = getattr(pretrained, model_func)(
        device=device, precision=ORB_PRECISION, compile=False
    )
    # Dispatch on what the loader returned rather than on a version number, so
    # this keeps working if the API moves again.
    if isinstance(loaded, tuple):
        model, atoms_adapter = loaded
        return ORBCalculator(model.eval(), atoms_adapter, device=device)
    return ORBCalculator(loaded.eval(), device=device)


def _build_mace(spec: HullMlipSpec, device: str):
    from wyckoff_transformer.cryspr.calculator import build_mace_calculator

    return build_mace_calculator(model=spec.checkpoint, device=device)


def _build_uma(spec: HullMlipSpec, device: str):
    try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "fairchem-core is required for the UMA hull: pip install 'fairchem-core>=2.3.0'"
        ) from exc

    predictor = pretrained_mlip.get_predict_unit("uma-s-1", device=device)
    return FAIRChemCalculator(predict_unit=predictor, task_name="omat")
