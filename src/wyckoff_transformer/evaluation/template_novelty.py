"""Anonymous Wyckoff template novelty, independent of chemical substitution.

Template representatives are the first valid structures in gene-index order.
They are selected before inspecting hull energy, and missing energies remain
in the denominator. Unresolved symmetry never falls back to the sampled gene.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
from collections import Counter

import numpy as np
import pandas as pd

from wyckoff_transformer.dataset_cache import (
    build_info,
    iter_splits,
    provenance,
    resolve_cache,
)
from wyckoff_transformer.evaluation.novelty import record_to_anonymous_fingerprint
from wyckoff_transformer.evaluation.protocol import (
    DEFAULT_REFERENCE_CACHE,
    DEFAULT_REFERENCE_SPLITS,
    METASTABLE_THRESHOLD,
    STABLE_THRESHOLD,
    reference_identity,
)
from wyckoff_transformer.paths import cache_root

logger = logging.getLogger(__name__)
FORMAT_VERSION = 1
GENE_TEMPLATES_FILE = "gene_templates.csv"
REPRESENTATIVE_RULE = "first valid structure in ascending sampled gene index; independent of hull energy"
DEFINITION = "anonymous augmented Wyckoff fingerprint: space group and multiset of occupied orbits; elements and species partition dropped"


def anonymize_fingerprint(fingerprint: tuple) -> tuple:
    """Drop elements from an augmented fingerprint, merging orbit counts."""
    variants = set()
    for variant in fingerprint[1]:
        counts = Counter()
        for (_, symmetry, enumeration), count in variant:
            counts[symmetry, enumeration] += count
        variants.add(frozenset(counts.items()))
    return int(fingerprint[0]), frozenset(variants)


def template_id(fingerprint: tuple) -> str:
    """Stable SHA256 identifier of an *anonymous* fingerprint, across processes."""
    variants = sorted(
        sorted((str(symmetry), int(enumeration), int(count))
               for (symmetry, enumeration), count in variant)
        for variant in fingerprint[1]
    )
    encoded = json.dumps([int(fingerprint[0]), variants], separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def template_id_from_augmented(fingerprint: tuple) -> str:
    return template_id(anonymize_fingerprint(fingerprint))


def load_template_reference(cache=DEFAULT_REFERENCE_CACHE, splits=DEFAULT_REFERENCE_SPLITS):
    """Load/build template IDs through dataset_cache, reading only three columns.

    Persist a portable JSON cache with the source's build records. No dataset
    cache is rewritten. A changed source build record invalidates this index.
    """
    cache = resolve_cache(cache)
    splits = tuple(splits)
    source = {"identity": reference_identity(cache, splits),
              "build": {split: build_info(cache, split) for split in splits}}
    # Dataset caches may be immutable; derived indices live outside them.
    index_dir = cache_root() / "template_novelty"
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / f"{cache.name}_v{FORMAT_VERSION}_{'+'.join(splits)}.json.gz"
    if path.is_file():
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle)
        if payload["source"] == source and payload["format_version"] == FORMAT_VERSION:
            payload["provenance"]["options"].setdefault(
                "template_set_sha256", hashlib.sha256("\n".join(sorted(payload["ids"])).encode()).hexdigest(),
            )
            logger.info("Loaded %d reference templates from %s", len(payload["ids"]), path)
            return set(payload["ids"]), payload["provenance"]
    columns = ("spacegroup_number", "site_symmetries_augmented", "sites_enumeration_augmented")
    fingerprints = set()
    rows = 0
    for split, frame in iter_splits(cache, splits, columns=columns):
        for values in frame.itertuples(index=False, name=None):
            fingerprints.add(record_to_anonymous_fingerprint(dict(zip(columns, values))))
        rows += len(frame)
        logger.info("Reference %s: %d rows, %d templates", split, len(frame), len(fingerprints))
        del frame
    ids = {template_id(fp) for fp in fingerprints}
    record = provenance("template_novelty_reference", definition=DEFINITION,
                        format_version=FORMAT_VERSION, source=source, rows=rows,
                        unique_templates=len(ids),
                        template_set_sha256=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest())
    payload = {"format_version": FORMAT_VERSION, "source": source,
               "provenance": record, "ids": sorted(ids)}
    temporary = path.with_suffix(".tmp.gz")
    with gzip.open(temporary, "wt") as handle:
        json.dump(payload, handle)
    temporary.replace(path)
    return ids, record


def label_templates(indices, ids: dict, reference: set[str], eligible=None, errors=None):
    """Label every row; deduplicate eligible rows without using energy."""
    frame = pd.DataFrame(index=pd.Index(indices, name="index"))
    frame["template_id"] = pd.Series(ids, dtype="string")
    frame["template_resolved"] = frame.template_id.notna()
    frame["template_novel"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    resolved = frame.template_resolved
    frame.loc[resolved, "template_novel"] = ~frame.loc[resolved, "template_id"].isin(reference)
    frame["template_error"] = pd.Series(errors or {}, dtype="string")
    eligible = (pd.Series(True, index=frame.index) if eligible is None
                else eligible.reindex(frame.index).fillna(False).astype(bool))
    selected = frame.loc[resolved & eligible].sort_index()
    representatives = selected.drop_duplicates("template_id").index
    first = selected.drop_duplicates("template_id").reset_index().set_index("template_id")["index"]
    frame["template_representative_index"] = frame.template_id.map(first).astype("Int64")
    frame["unique_template"] = frame.index.isin(representatives)
    frame["novel_unique_template"] = frame.unique_template & frame.template_novel.fillna(False)
    return frame


def template_metrics(labels: pd.DataFrame, structures: pd.DataFrame | None = None) -> dict:
    """Counts per unique template and conditional stability fractions (unweighted)."""
    selected = labels.novel_unique_template
    n = int(selected.sum())
    result = {
        "rows": len(labels), "resolved": int(labels.template_resolved.sum()),
        "unresolved": int((~labels.template_resolved).sum()),
        "novel": int(labels.template_novel.fillna(False).sum()),
        "unique": int(labels.unique_template.sum()), "novel_unique": n,
    }
    if structures is not None:
        present = structures.has_structure.fillna(False).astype(bool)
        valid = present & structures.valid_structure.fillna(False).astype(bool)
        result["no_structure"] = int((~present).sum())
        result["invalid_structure"] = int((present & ~valid).sum())
        result["unresolved_valid_structure"] = int((valid & ~labels.template_resolved).sum())
        energy = pd.to_numeric(structures.e_above_hull, errors="raise").reindex(labels.index)
        finite = pd.Series(np.isfinite(energy), index=labels.index)
        result["no_hull_energy_novel_unique"] = int((selected & ~finite).sum())
        for name, threshold in (("metastable", METASTABLE_THRESHOLD), ("stable", STABLE_THRESHOLD)):
            count = int((selected & finite & energy.le(threshold)).sum())
            result[f"{name}_novel_unique"] = count
            result[f"{name}_fraction_novel_unique"] = count / n if n else None
    return result


def label_genes(genes, reference, fingerprinter):
    """Include all sampled genes, including duplicates and invalid genes."""
    ids, errors = {}, {}
    for index, gene in enumerate(genes):
        try:
            ids[index] = template_id_from_augmented(fingerprinter.fingerprint(gene))
        except Exception as exc:  # noqa: BLE001 - invalid genes must retain a diagnostic row
            errors[index] = f"{type(exc).__name__}: {exc}"
    return label_templates(range(len(genes)), ids, reference, errors=errors)


def label_structures(structures, ids, reference, errors=None):
    """Judge all structures, but count only valid structures as representatives."""
    eligible = structures.has_structure.fillna(False).astype(bool) & structures.valid_structure.fillna(False).astype(bool)
    return label_templates(structures.index, ids, reference, eligible, errors)


def attach_labels(structures, labels, gene_labels):
    for column in labels:
        structures[column] = labels[column]
    for column in ("template_id", "template_resolved", "template_novel"):
        structures[f"gene_{column}"] = gene_labels[column].reindex(structures.index)


def template_summary(report: dict, prefix: str) -> dict:
    """W&B summary keys including explicit nulls for undefined fractions.

    Keeping nulls clears a previous cohort's defined rate when a subsequent
    evaluation has no novel templates. The general funnel flattener omits them.
    """
    return {
        f"{prefix}{section}/template_novelty/{key}": value
        for section in ("gene", "free", "fixed_symmetry")
        for key, value in report.get(section, {}).get("template_novelty", {}).items()
    }


def measurement_provenance(reference):
    return provenance("protocol_template_novelty", format_version=FORMAT_VERSION,
                      definition=DEFINITION, reference=reference,
                      representative_rule=REPRESENTATIVE_RULE,
                      denominator="valid structures with novel unique relaxed templates, including missing hull energies",
                      metastable_threshold=METASTABLE_THRESHOLD, stable_threshold=STABLE_THRESHOLD,
                      units="eV/atom", symmetry_tol=0.1, symmetry_angle_tol=5.0)
