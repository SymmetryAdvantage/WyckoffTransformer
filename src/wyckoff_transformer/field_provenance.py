"""What a trained model's input and output fields meant: recorded with its weights.

A model conditions on some fields and may predict one.  Their names are in its
config, but a name is not a meaning -- ``energy_above_hull`` is on a different
energy scale in each dataset that has one.  At training time every field the
model uses is resolved through its dataset's manifest
(:mod:`wyckoff_transformer.dataset_manifest`), and the result is recorded next
to the weights as a *field provenance*::

    {"format": 1, "dataset": "mp_20", "recorded": true, "cache_build": {...},
     "fields": {"condition:energy_above_hull": {"requested": "energy_above_hull",
                                                "field": "energy_above_hull",
                                                "quantity": "energy_above_hull",
                                                "energy": {...EnergyField...}},
                "target": {...}}}

It lives in ``field_provenance.json`` in the run directory and in the W&B run's
config under :data:`CONFIG_KEY` -- deliberately not in ``config.yaml``, which a
resumed run must reproduce exactly.  A model trained before this existed has
none; :func:`load_field_provenance` then *infers* it from its dataset's current
manifest and says so (``"recorded": false``).

:func:`check_against_dataset` and :func:`energy_field` are what a consumer uses
to refuse combining the model with data, or another model, of a different
energy definition.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

from wyckoff_transformer.dataset_manifest import (
    ManifestError,
    ManifestNotFound,
    load_manifest,
)
from wyckoff_transformer.energy_fields import EnergyField, check_compatible

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
PROVENANCE_FILENAME = "field_provenance.json"
#: Where the W&B run's config holds it.  Outside the training config proper.
CONFIG_KEY = "field_provenance"
TARGET_ROLE = "target"
CONDITION_PREFIX = "condition:"


def _plain(config: Any) -> Any:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    return config


def requested_fields(config: Mapping | DictConfig) -> Dict[str, str]:
    """The fields a model config uses, by role: ``condition:<name>`` and ``target``."""
    from wyckoff_transformer.trainer import normalise_condition_features  # noqa: PLC0415

    args = _plain(config)["model"].get("WyckoffTrainer_args", {}) or {}
    roles = {f"{CONDITION_PREFIX}{name}": name
             for name in normalise_condition_features(args.get("condition_feature"))}
    if args.get("target") == "Scalar" and args.get("target_name"):
        roles[TARGET_ROLE] = str(args["target_name"])
    return roles


def _entry(dataset: Optional[str], requested: str) -> dict:
    """One role's record, resolved through the dataset's manifest if it has one."""
    entry = {"requested": requested, "field": None, "quantity": None, "energy": None}
    if dataset is None:
        return entry
    try:
        field = load_manifest(dataset).resolve(requested)
    except ManifestError:  # no manifest, or it does not declare the field
        return entry
    entry.update(field=field.name, **field.to_record())
    return entry


def build_field_provenance(config: Mapping | DictConfig,
                           cache_build: Optional[Mapping] = None,
                           recorded: bool = True) -> dict:
    """Resolve every field a model config uses through its dataset's manifest.

    Args:
        cache_build: The ``build`` record of the cache trained on, kept alongside so
            the provenance names the exact data as well as its definitions.
        recorded: False when this is inferred for a model trained without one.
    """
    dataset = _plain(config).get("dataset")
    fields = {role: _entry(dataset, name) for role, name in requested_fields(config).items()}
    return {"format": FORMAT_VERSION, "dataset": dataset, "recorded": recorded,
            "cache_build": _plain(cache_build), "fields": fields}


def require_resolved(config: Mapping | DictConfig) -> None:
    """Refuse to train with a field the dataset's manifest does not declare.

    Only for a dataset that has a manifest: training on one without is already
    refused as obsolete unless forced, and forcing it means there is nothing to
    resolve against.
    """
    dataset = _plain(config).get("dataset")
    try:
        manifest = load_manifest(dataset)
    except ManifestNotFound:
        return
    for role, name in requested_fields(config).items():
        manifest.resolve(name)


def write_field_provenance(provenance: Mapping, run_path: Path) -> Path:
    path = Path(run_path) / PROVENANCE_FILENAME
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    return path


def recorded_field_provenance(config: Mapping | DictConfig,
                              run_path: Optional[Path] = None) -> Optional[dict]:
    """The provenance recorded for a trained model, or None if it predates recording."""
    recorded = _plain(config).get(CONFIG_KEY)
    if recorded is None and run_path is not None:
        path = Path(run_path) / PROVENANCE_FILENAME
        if path.is_file():
            recorded = json.loads(path.read_text())
    return None if recorded is None else dict(_plain(recorded))


def alias_stored_columns(dataset: Optional[str], names, splits) -> None:
    """Let a model find each field it asks for under whatever name the tensors hold it.

    A tensor cache built before a dataset's columns were renamed holds a field under
    its old name (``e_above_hull``); one built after, under the canonical one.  The
    manifest says both are the same field, so the name asked for is made to point at
    whichever is there -- in place, in every split dict.
    """
    if dataset is None:
        return
    try:
        manifest = load_manifest(dataset)
    except ManifestNotFound:
        return
    for name in names:
        try:
            field = manifest.resolve(name)
        except ManifestError:
            continue
        for split in splits:
            if split is None or name in split:
                continue
            try:
                stored = field.column_in(split.keys())
            except ManifestError:
                continue
            logger.info("Dataset %r stores %r as %r", dataset, name, stored)
            split[name] = split[stored]


def load_field_provenance(config: Mapping | DictConfig,
                          run_path: Optional[Path] = None) -> dict:
    """A trained model's field provenance: recorded if it has one, inferred if not.

    Looks in the config (a W&B run's config carries :data:`CONFIG_KEY`), then in
    ``run_path/field_provenance.json``.  A model trained before provenance was
    recorded gets it inferred from its dataset's current manifest, with a warning:
    right as long as the dataset's labels have not been redefined since.
    """
    plain = _plain(config)
    recorded = recorded_field_provenance(plain, run_path)
    if recorded is not None:
        return recorded
    inferred = build_field_provenance(plain, recorded=False)
    if inferred["fields"]:
        unknown = sorted(role for role, entry in inferred["fields"].items()
                         if entry["field"] is None)
        logger.warning(
            "Model trained on %r records no field provenance; inferred it from the dataset's "
            "current manifest%s.", inferred["dataset"],
            f" -- {unknown} could not be resolved and have unknown provenance" if unknown else "")
    return inferred


def energy_field(provenance: Mapping, role: str) -> Optional[EnergyField]:
    """The energy definition of one role, or None if unknown or not an energy."""
    entry = provenance.get("fields", {}).get(role)
    if not entry or entry.get("energy") is None:
        return None
    return EnergyField.from_dict(entry["energy"])


def target_field(provenance: Mapping) -> Optional[EnergyField]:
    return energy_field(provenance, TARGET_ROLE)


def check_against_dataset(provenance: Mapping, dataset: str, context: str,
                          allow: bool = False) -> List[str]:
    """Refuse a dataset whose fields mean something else than the model's did.

    Every role is resolved in *dataset* by the name the model asked for.  Energy
    fields must be compatible (:func:`~wyckoff_transformer.energy_fields.check_compatible`);
    anything else must at least be the same quantity.

    Returns:
        The differences, when ``allow`` let them through.
    """
    lines: List[str] = []
    for role, entry in provenance.get("fields", {}).items():
        other = _entry(dataset, entry["requested"])
        where = f"{context}: {role} of a model trained on {provenance.get('dataset')!r} vs {dataset!r}"
        if entry.get("energy") is not None or other.get("energy") is not None \
                or entry.get("quantity") is None:
            lines += check_compatible(
                energy_field(provenance, role),
                None if other["energy"] is None else EnergyField.from_dict(other["energy"]),
                where, allow=allow)
        elif entry["quantity"] != other["quantity"]:
            lines += check_compatible(None, None, where, allow=allow)
    return lines
