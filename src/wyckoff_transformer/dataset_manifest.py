"""Dataset manifests: what each dataset's fields mean, and whether it may be used.

One tracked yaml per dataset, ``yamls/datasets/<name>.yaml``, is the source of
truth for three things the data files cannot say about themselves:

- **status** -- ``current`` (use it), ``source`` (a raw input other datasets are
  built from, not trained on directly) or ``obsolete`` (kept only to analyse runs
  already trained on it).  A dataset with no manifest counts as obsolete: new
  work has to be argued for with one.
- **fields** -- the canonical name of each column, the name it had in the source
  files, older names it may still be asked for by, and for an energy its
  :class:`~wyckoff_transformer.energy_fields.EnergyField`.
- **lineage** -- ``parent``, whose fields a derived dataset inherits.

A Wyckoff dataset has one table, its splits (``fields:`` at the top level).  A
dataset of several files, such as ``formula_energy``, has a ``tables:`` block.
``docs/energy_fields.md`` describes the format.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field as dataclass_field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml

from wyckoff_transformer.energy_fields import (
    QUANTITIES,
    EnergyField,
    EnergyFieldError,
    canonical_id,
)

logger = logging.getLogger(__name__)

MANIFEST_DIR = Path(__file__).resolve().parents[2] / "yamls" / "datasets"

#: The table of a Wyckoff dataset: its train/val/test splits, which share columns.
SPLITS_TABLE = "splits"

STATUSES = ("current", "source", "obsolete")

_TOP_KEYS = {"name", "status", "reason", "description", "parent", "defaults",
             "fields", "tables", "drop", "duplicates", "built_by"}
_TABLE_KEYS = {"description", "defaults", "fields", "drop", "duplicates", "path"}
_FIELD_KEYS = {"column", "aliases", "primary", "qualifier", "description",
               "quantity", "energy"}


class ManifestError(ValueError):
    """A manifest is malformed, or a field was asked for that it does not declare."""


class ManifestNotFound(ManifestError):
    """There is no manifest for this dataset."""


class ObsoleteDatasetError(ValueError):
    """New work was about to start on a dataset that must not be used for it."""


class CacheManifestMismatch(ValueError):
    """A cache was built under field definitions other than its manifest's."""


@dataclass(frozen=True)
class DatasetField:
    """One column of a dataset, by its canonical name.

    Attributes:
        name: The canonical id -- the column's name in a cache built from the
            dataset, and the name configs should ask for.
        column: The column's name in the dataset's source files.
        aliases: Older names it may still be asked for by; resolving one warns.
        quantity: For an energy, ``energy.quantity``; otherwise a free label
            (``band_gap``, ``max_force``).
        energy: The provenance of an energy field; ``None`` for anything else.
        primary: The field a bare quantity resolves to when several share it.
    """
    name: str
    column: str
    quantity: str
    energy: Optional[EnergyField] = None
    aliases: Tuple[str, ...] = ()
    primary: bool = False
    qualifier: Optional[str] = None
    description: str = ""

    @property
    def names(self) -> Tuple[str, ...]:
        """Every name this field may be stored under, canonical first."""
        seen = [self.name]
        for name in (self.column, *self.aliases):
            if name not in seen:
                seen.append(name)
        return tuple(seen)

    def to_record(self) -> dict:
        """The part of a field a cache or a model records: what it means."""
        return {"quantity": self.quantity,
                "energy": None if self.energy is None else self.energy.to_dict()}

    def column_in(self, available: Iterable[str]) -> str:
        """The name this field is stored under among ``available``.

        A cache built before the rename carries the source column name; one built
        after carries the canonical one.  Both mean the same field.
        """
        available = set(available)
        for name in self.names:
            if name in available:
                return name
        raise ManifestError(
            f"Field {self.name!r} is not among the columns present; looked for "
            f"{list(self.names)}.")


@dataclass(frozen=True)
class DatasetManifest:
    """A parsed ``yamls/datasets/<name>.yaml``."""
    name: str
    status: str
    reason: Optional[str]
    description: str
    parent: Optional[str]
    tables: Mapping[str, Mapping[str, DatasetField]]
    drop: Mapping[str, Tuple[str, ...]] = dataclass_field(default_factory=dict)
    duplicates: Mapping[str, Mapping[str, str]] = dataclass_field(default_factory=dict)
    #: Each named table's file, relative to the dataset's data directory.
    paths: Mapping[str, str] = dataclass_field(default_factory=dict)

    def fields(self, table: str = SPLITS_TABLE) -> Mapping[str, DatasetField]:
        """The fields of one table; an unlabelled dataset's splits declare none."""
        if table == SPLITS_TABLE and not self.tables:
            return {}
        try:
            return self.tables[table]
        except KeyError:
            raise ManifestError(
                f"Dataset {self.name!r} has no table {table!r}; it has "
                f"{sorted(self.tables)}.") from None

    def resolve(self, requested: str, table: str = SPLITS_TABLE) -> DatasetField:
        """The field a config asks for by canonical id, older alias or bare quantity.

        Raises:
            ManifestError: Nothing matches, or a quantity matches several fields
                and none of them is marked ``primary``.
        """
        fields = self.fields(table)
        if requested in fields:
            return fields[requested]
        by_alias = [f for f in fields.values() if requested in f.names]
        if len(by_alias) == 1:
            logger.warning(
                "Dataset %r: %r is a superseded name of field %r; ask for the canonical "
                "name.", self.name, requested, by_alias[0].name)
            return by_alias[0]
        by_quantity = [f for f in fields.values() if f.quantity == requested]
        if len(by_quantity) == 1:
            return by_quantity[0]
        primary = [f for f in by_quantity if f.primary]
        if len(primary) == 1:
            return primary[0]
        # A bare quantity means the row's own value: not a gene or formula minimum,
        # not a second definition told apart by a qualifier.
        plain = [f for f in by_quantity if f.qualifier is None
                 and (f.energy is None or f.energy.aggregate == "none")]
        if len(plain) == 1:
            return plain[0]
        if by_quantity:
            raise ManifestError(
                f"Dataset {self.name!r}: quantity {requested!r} matches "
                f"{sorted(f.name for f in by_quantity)} and none is marked primary; "
                f"ask for one by name.")
        raise ManifestError(
            f"Dataset {self.name!r} declares no field {requested!r} in table {table!r}; "
            f"it declares {sorted(fields)}.  Add it to yamls/datasets/{self.name}.yaml "
            f"with its provenance before using it.")

    def fields_record(self, table: str = SPLITS_TABLE) -> Dict[str, dict]:
        """What a cache records about its fields, for :func:`check_cache_matches`."""
        return {name: f.to_record() for name, f in sorted(self.fields(table).items())}

    @property
    def obsolete_reason(self) -> Optional[str]:
        return self.reason if self.status == "obsolete" else None


def _merge(base: Mapping, override: Mapping) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_field(dataset: str, name: str, raw: Mapping, defaults: Mapping) -> DatasetField:
    unknown = set(raw) - _FIELD_KEYS
    if unknown:
        raise ManifestError(f"{dataset}: field {name!r} has unknown keys {sorted(unknown)}")
    qualifier = raw.get("qualifier")
    aliases = tuple(raw.get("aliases", ()))
    column = raw.get("column", name)
    if "energy" in raw:
        spec = _merge({"source": defaults.get("source"), "reference": defaults.get("reference")},
                      raw["energy"])
        if spec.get("quantity") == "energy":
            spec.pop("reference", None)
        if spec.get("source") is None:
            raise ManifestError(f"{dataset}: energy field {name!r} names no source")
        try:
            energy = EnergyField.from_dict({k: v for k, v in spec.items() if v is not None})
        except (EnergyFieldError, KeyError) as error:
            raise ManifestError(f"{dataset}: field {name!r}: {error}") from error
        expected = canonical_id(energy, qualifier)
        if name != expected:
            raise ManifestError(
                f"{dataset}: field {name!r} holds {energy.describe()}, whose canonical "
                f"name is {expected!r}.  Name the field that and keep {name!r} as its "
                f"'column' or an alias.")
        if "quantity" in raw and raw["quantity"] != energy.quantity:
            raise ManifestError(f"{dataset}: field {name!r} names two quantities")
        quantity = energy.quantity
    else:
        energy = None
        quantity = raw.get("quantity")
        if quantity is None:
            raise ManifestError(
                f"{dataset}: field {name!r} has neither an energy block nor a quantity")
        if quantity in QUANTITIES:
            raise ManifestError(
                f"{dataset}: field {name!r} is an energy ({quantity}) without provenance; "
                f"give it an 'energy' block.")
    return DatasetField(name=name, column=column, quantity=quantity, energy=energy,
                        aliases=aliases, primary=bool(raw.get("primary", False)),
                        qualifier=qualifier, description=str(raw.get("description", "")))


def _parse_table(dataset: str, raw: Mapping, defaults: Mapping) -> Dict[str, DatasetField]:
    fields = {name: _parse_field(dataset, name, spec or {}, defaults)
              for name, spec in (raw or {}).items()}
    seen: Dict[str, str] = {}
    for f in fields.values():
        for alias in f.names:
            if alias in seen and seen[alias] != f.name:
                raise ManifestError(
                    f"{dataset}: name {alias!r} is claimed by both {seen[alias]!r} "
                    f"and {f.name!r}")
            seen[alias] = f.name
    return fields


def parse_manifest(raw: Mapping, parent: Optional[DatasetManifest] = None) -> DatasetManifest:
    """Validate one manifest's yaml, with its parent's fields for inheritance."""
    name = raw.get("name")
    if not name:
        raise ManifestError("A manifest must give its dataset's name")
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        raise ManifestError(f"{name}: unknown manifest keys {sorted(unknown)}")
    status = raw.get("status")
    if status not in STATUSES:
        raise ManifestError(f"{name}: status must be one of {STATUSES}, not {status!r}")
    reason = raw.get("reason")
    if status == "obsolete" and not reason:
        raise ManifestError(f"{name}: an obsolete dataset must say why ('reason')")
    defaults = raw.get("defaults") or {}

    tables: Dict[str, Dict[str, DatasetField]] = {}
    drop: Dict[str, Tuple[str, ...]] = {}
    duplicates: Dict[str, Dict[str, str]] = {}
    if parent is not None:
        tables = {t: dict(f) for t, f in parent.tables.items()}
        drop = dict(parent.drop)
        duplicates = {t: dict(d) for t, d in parent.duplicates.items()}
    if "fields" in raw and "tables" in raw:
        raise ManifestError(f"{name}: give 'fields' (one table of splits) or 'tables', not both")
    if "fields" in raw or "drop" in raw or "duplicates" in raw:
        tables[SPLITS_TABLE] = _parse_table(name, raw.get("fields"), defaults)
        drop[SPLITS_TABLE] = tuple(raw.get("drop", ()))
        duplicates[SPLITS_TABLE] = dict(raw.get("duplicates") or {})
    paths: Dict[str, str] = dict(parent.paths) if parent is not None else {}
    for table, spec in (raw.get("tables") or {}).items():
        extra = set(spec) - _TABLE_KEYS
        if extra:
            raise ManifestError(f"{name}: table {table!r} has unknown keys {sorted(extra)}")
        if "path" in spec:
            paths[table] = str(spec["path"])
        tables[table] = _parse_table(name, spec.get("fields"),
                                     _merge(defaults, spec.get("defaults") or {}))
        drop[table] = tuple(spec.get("drop", ()))
        duplicates[table] = dict(spec.get("duplicates") or {})
    for table, pairs in duplicates.items():
        for copy, original in pairs.items():
            if original not in {f.column for f in tables.get(table, {}).values()}:
                raise ManifestError(
                    f"{name}: duplicate {copy!r} names {original!r}, which is no field's "
                    f"source column")
    return DatasetManifest(
        name=name, status=status, reason=reason,
        description=str(raw.get("description", "")), parent=raw.get("parent"),
        tables=tables, drop=drop, duplicates=duplicates, paths=paths)


def manifest_path(name: str) -> Path:
    return MANIFEST_DIR / f"{name}.yaml"


def dataset_name(dataset: str | Path) -> str:
    """A dataset's name, from its name or from a path to its data or cache directory."""
    return Path(str(dataset).rstrip("/")).name


@lru_cache(maxsize=None)
def load_manifest(dataset: str | Path) -> DatasetManifest:
    """The manifest of a dataset, validated, with its parent's fields merged in.

    Raises:
        ManifestNotFound: No ``yamls/datasets/<name>.yaml`` exists.
        ManifestError: It exists and is malformed.
    """
    name = dataset_name(dataset)
    path = manifest_path(name)
    if not path.exists():
        raise ManifestNotFound(
            f"No manifest for dataset {name!r} at {path}.  A dataset without one counts "
            f"as obsolete; see docs/energy_fields.md to add one.")
    with open(path, encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if raw.get("name") != name:
        raise ManifestError(f"{path} names dataset {raw.get('name')!r}, not {name!r}")
    parent = load_manifest(raw["parent"]) if raw.get("parent") else None
    return parse_manifest(raw, parent=parent)


def all_manifests() -> Dict[str, DatasetManifest]:
    return {path.stem: load_manifest(path.stem) for path in sorted(MANIFEST_DIR.glob("*.yaml"))}


def obsolete_reason(dataset: str | Path) -> Optional[str]:
    """Why a dataset must not be used for new work, or None if it may be."""
    try:
        manifest = load_manifest(dataset)
    except ManifestNotFound:
        return (f"it has no manifest in yamls/datasets/, so nothing records what its "
                f"fields mean or that it is current")
    return manifest.obsolete_reason


def refuse_if_obsolete_dataset(dataset: str | Path, context: str = "",
                               allow: bool = False) -> None:
    """Raise rather than start new work -- training, caching, tokenising -- on it.

    A ``source`` dataset is refused too: it is the raw input other datasets are
    built from, not something to train on directly.

    Args:
        allow: The ``--allow-obsolete-dataset`` escape hatch: warn and carry on.
    """
    name = dataset_name(dataset)
    reason = obsolete_reason(name)
    if reason is None and load_manifest(name).status == "source":
        reason = "it is a source dataset: build a training dataset from it instead"
    if reason is None:
        return
    where = f" ({context})" if context else ""
    if allow:
        logger.warning("Dataset %r is obsolete%s, used anyway as asked: %s",
                       name, where, reason)
        return
    raise ObsoleteDatasetError(
        f"Dataset {name!r} must not be used for new work{where}: {reason}.  Pass "
        f"--allow-obsolete-dataset to use it anyway.")


_warned: set = set()


def warn_if_obsolete_dataset(dataset: str | Path, context: str = "") -> Optional[str]:
    """Log, once per process and dataset, that an evaluation is reading an obsolete one.

    For the paths that must keep working -- scoring an old run, re-reading a
    reference -- where refusing would make earlier results unreproducible.
    """
    name = dataset_name(dataset)
    reason = obsolete_reason(name)
    if reason is None or name in _warned:
        return reason
    _warned.add(name)
    logger.warning("Dataset %r is obsolete%s: %s", name,
                   f" ({context})" if context else "", reason)
    return reason


def check_cache_matches(dataset: str | Path, recorded: Optional[Mapping[str, Any]],
                        table: str = SPLITS_TABLE) -> None:
    """Refuse a cache whose recorded field definitions differ from its manifest's.

    Args:
        recorded: The ``fields`` a cache build recorded in its ``build`` options, or
            ``None`` for a cache built before they were recorded -- which only warns.
    """
    name = dataset_name(dataset)
    try:
        manifest = load_manifest(name)
    except ManifestNotFound:
        return
    if recorded is None:
        logger.info("Cache of %r records no field definitions; trusting its manifest.", name)
        return
    expected = manifest.fields_record(table)
    # Only what the cache holds: a manifest may declare fields a build did not carry.
    changed = sorted(k for k, v in recorded.items() if expected.get(k) != _plain(v))
    if changed:
        raise CacheManifestMismatch(
            f"The cache of {name!r} was built when its manifest said something else about "
            f"{changed}.  Rebuild the cache, or restore the manifest it was built under.")


def _plain(value: Any) -> Any:
    """A record read back from JSON or OmegaConf, as plain dicts for comparison."""
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    return value


def resolve_fields(dataset: str | Path, requested: Sequence[str],
                   table: str = SPLITS_TABLE) -> Dict[str, DatasetField]:
    """Resolve several names at once, keyed by what was asked for."""
    manifest = load_manifest(dataset)
    return {name: manifest.resolve(name, table) for name in requested}



def table_for_file(path: str | Path) -> Optional[Tuple[DatasetManifest, str]]:
    """Which manifest table a data file is, or None if no manifest names it.

    How a consumer handed a file -- a hull reference CSV, a formula table -- finds
    out what its energies mean, rather than trusting its file name.
    """
    from wyckoff_transformer.paths import data_path, resolve_store_path  # noqa: PLC0415

    target = Path(resolve_store_path(Path(path))).resolve()
    for manifest in all_manifests().values():
        for table, relative in manifest.paths.items():
            try:
                candidate = (Path(data_path(manifest.name)) / relative).resolve()
            except (FileNotFoundError, KeyError, ValueError):
                continue
            if candidate == target:
                return manifest, table
    return None


def formation_energy_field(manifest: DatasetManifest, table: str = SPLITS_TABLE):
    """The table's formation energy definition -- its row values or its formula floor.

    Of the fields holding a formation energy, the first without a qualifier: the
    aggregate does not change the scale, and a qualified field is a second
    definition kept alongside the main one.
    """
    for field in manifest.fields(table).values():
        if (field.energy is not None and field.energy.quantity == "formation_energy"
                and field.qualifier is None):
            return field.energy
    return None


def file_formation_energy_field(path: str | Path):
    """The formation-energy definition of a data file, or None if it is not labelled."""
    found = table_for_file(path)
    if found is None:
        return None
    return formation_energy_field(*found)
