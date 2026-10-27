"""The labelled sets of the transport benchmark: OBELiX and Kahle 2020.

OBELiX (Therrien et al., NRC-Mila/OBELiX) lists 599 measured Li SEs with their
room-temperature conductivity; 321 have a CIF, 249 of them with partial occupancies.
Conductivities below the measurable range are written as strings such as ``<1E-10``.

Kahle, Marcolongo and Marzari (Energy Environ. Sci. 13, 928, 2020) ran first-principles
MD for ~120 Li compounds.  Their Materials Cloud record (doi:10.24435/materialscloud:vg-ya)
is a 21 GB AiiDA export whose first zip member, ``data.json``, holds every node's
attributes: the diffusion coefficients, the trajectories' thermostat temperatures and
the supercells.  Only that member is needed, and it is fetched by a range request.
"""
from __future__ import annotations

import json
import logging
import math
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Elementary charge (C) and Boltzmann constant (J/K).
E_CHARGE = 1.602176634e-19
K_B = 1.380649e-23

_CENSORED = re.compile(r"^\s*([<>])\s*([0-9.eE+-]+)\s*$")


def parse_conductivity(value) -> tuple[float, str]:
    """An OBELiX conductivity cell as ``(sigma S/cm, censoring)``.

    *censoring* is ``""`` for a measured value, ``"<"`` for an upper bound such as
    ``<1E-10`` and ``">"`` for a lower one.  Unparseable cells give ``(nan, "")``.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan, ""
    if isinstance(value, (int, float)):
        return float(value), ""
    text = str(value).strip()
    match = _CENSORED.match(text)
    if match:
        return float(match.group(2)), match.group(1)
    try:
        return float(text), ""
    except ValueError:
        return math.nan, ""


def load_obelix(table: Path, cifs_tar: Path | None = None) -> pd.DataFrame:
    """OBELiX entries with their conductivity and, where there is one, the CIF text.

    Columns: ``ID``, ``formula`` (the reduced composition as written), ``sigma``,
    ``censoring``, ``family``, ``space_group``, ``cif`` (``None`` without a CIF).
    """
    frame = pd.read_csv(table)
    parsed = frame["Ionic conductivity (S cm-1)"].map(parse_conductivity)
    out = pd.DataFrame({
        "ID": frame["ID"].astype(str),
        "formula": frame["Reduced Composition"].astype(str),
        "true_composition": frame["True Composition"].astype(str),
        "sigma": [p[0] for p in parsed],
        "censoring": [p[1] for p in parsed],
        "family": frame["Family"].astype(str),
        "space_group": frame["Space group #"],
    })
    cifs: dict[str, str] = {}
    if cifs_tar is not None:
        with tarfile.open(cifs_tar) as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.endswith(".cif"):
                    handle = archive.extractfile(member)
                    cifs[Path(member.name).stem] = handle.read().decode()
    out["cif"] = out["ID"].map(cifs)
    return out


# --------------------------------------------------------------------------- #
# Kahle 2020
# --------------------------------------------------------------------------- #
def _structure_from_aiida(attributes: dict):
    """A pymatgen Structure from a StructureData node's attributes."""
    from pymatgen.core import Lattice, Structure

    kinds = {kind["name"]: kind for kind in attributes["kinds"]}
    species, coords = [], []
    for site in attributes["sites"]:
        kind = kinds[site["kind_name"]]
        if len(kind["symbols"]) != 1:
            raise ValueError(f"alloy kind {kind['name']} in a Kahle structure")
        species.append(kind["symbols"][0])
        coords.append(site["position"])
    return Structure(Lattice(attributes["cell"]), species, coords, coords_are_cartesian=True)


def nernst_einstein(diffusion_cm2_s: float, n_carriers: int, volume_a3: float,
                    temperature_k: float, charge: int = 1) -> float:
    """Conductivity in S/cm from a tracer diffusion coefficient, no Haven ratio."""
    n_m3 = n_carriers / (volume_a3 * 1e-30)
    d_m2_s = diffusion_cm2_s * 1e-4
    sigma_s_m = n_m3 * (charge * E_CHARGE) ** 2 * d_m2_s / (K_B * temperature_k)
    return sigma_s_m / 100.0


def load_kahle(data_json: Path, mobile: str = "Li") -> pd.DataFrame:
    """One row per (structure, temperature) of Kahle et al.'s first-principles MD.

    Each diffusion-coefficient node of the export's ``diffusion_coefficients`` group
    is traced back through its calculation to the trajectory (for the thermostat
    temperature) and the supercell (for the carrier density), and from the supercell
    to the unit cell it was built from, which is returned as a CIF.

    Columns: ``label`` (the supercell's formula), ``structure_uuid``,
    ``temperature``, ``D_cm2_s``, ``D_sem_cm2_s``, ``sigma`` and ``sigma_sem`` (S/cm,
    Nernst-Einstein), ``n_mobile``, ``volume``, ``source_db``, ``source_id``, ``cif``.
    """
    with open(data_json, encoding="utf-8") as handle:
        data = json.load(handle)
    nodes = data["export_data"]["Node"]
    attributes = data["node_attributes"]
    extras = data["node_extras"]
    pk_of = {node["uuid"]: pk for pk, node in nodes.items()}
    inputs: dict[str, list[dict]] = defaultdict(list)
    for link in data["links_uuid"]:
        inputs[link["output"]].append(link)
    groups = {group["label"]: group["uuid"]
              for group in data["export_data"]["Group"].values()}
    members = data["groups_uuid"]

    def parent(uuid: str, link_type: str, label: str | None = None) -> str | None:
        for link in inputs[uuid]:
            if link["type"] == link_type and (label is None or link["label"] == label):
                return link["input"]
        return None

    rows = []
    for uuid in members[groups["diffusion_coefficients"]]:
        pk = pk_of[uuid]
        result = attributes[pk].get(mobile)
        if not result:
            continue
        calc = parent(uuid, "create")
        supercell = parent(calc, "input_calc", "structure") if calc else None
        trajectory = parent(calc, "input_calc", "trajectory") if calc else None
        if supercell is None or trajectory is None:
            logger.warning("Kahle node %s: no supercell or trajectory upstream", uuid)
            continue
        unit_calc = parent(supercell, "create")
        unit = parent(unit_calc, "input_calc", "structure") if unit_calc else None
        cell = _structure_from_aiida(attributes[pk_of[supercell]])
        temperature = float(attributes[pk_of[trajectory]]["temperature_thermostat"])
        n_mobile = sum(1 for site in cell if site.specie.symbol == mobile)
        d_mean = float(result["diffusion_mean_cm2_s"])
        d_sem = float(result["diffusion_sem_cm2_s"])
        unit_structure = _structure_from_aiida(attributes[pk_of[unit]]) if unit else cell
        source = extras.get(pk_of[unit], {}).get("source", {}) if unit else {}
        rows.append({
            "label": nodes[pk_of[supercell]]["label"] or cell.formula.replace(" ", ""),
            "structure_uuid": unit or supercell,
            "temperature": temperature,
            "D_cm2_s": d_mean,
            "D_sem_cm2_s": d_sem,
            "sigma": nernst_einstein(d_mean, n_mobile, cell.volume, temperature),
            "sigma_sem": nernst_einstein(d_sem, n_mobile, cell.volume, temperature),
            "n_mobile": n_mobile,
            "volume": cell.volume,
            "source_db": source.get("db_name"),
            "source_id": source.get("id") or source.get("uri"),
            "reduced_formula": unit_structure.composition.reduced_formula,
            "cif": unit_structure.to(fmt="cif"),
        })
    frame = pd.DataFrame(rows)
    logger.info("Kahle 2020: %d diffusion records, %d structures, temperatures %s",
                len(frame), frame["structure_uuid"].nunique(),
                sorted(frame["temperature"].unique()))
    return frame


def gaussian_label_weights(value: np.ndarray, error: np.ndarray, positive: float,
                           negative: float, log10: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Maevskiy et al.'s smoothed labels: the weight above *positive*, below *negative*.

    Each value is taken as a Gaussian with its error as the standard deviation; the
    positive weight is the probability mass above *positive* and the negative weight
    the mass below *negative*.  A zero error gives hard 0/1 weights.  With *log10* the
    Gaussian lives in log10 space and *error* is a relative error.
    """
    from scipy.stats import norm

    value = np.asarray(value, dtype=float)
    error = np.asarray(error, dtype=float)
    if log10:
        centre = np.log10(np.clip(value, 1e-30, None))
        width = np.log10(1.0 + np.clip(error, 0.0, None))
        positive, negative = math.log10(positive), math.log10(negative)
    else:
        centre, width = value, np.clip(error, 0.0, None)
    hard = width <= 0
    width = np.where(hard, 1.0, width)
    w_pos = np.where(hard, (centre >= positive).astype(float),
                     norm.sf(positive, loc=centre, scale=width))
    w_neg = np.where(hard, (centre <= negative).astype(float),
                     norm.cdf(negative, loc=centre, scale=width))
    return w_pos, w_neg


__all__ = [
    "gaussian_label_weights", "load_kahle", "load_obelix", "nernst_einstein",
    "parse_conductivity",
]
