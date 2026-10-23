"""Backfill template novelty from the latest alias of every protocol_eval collection.

Uses saved genes, CIFs, validity and hull energies. Publishes a separate analysis
artifact and updates only template_novelty summary keys on the source runs.
Training runs are never resumed. Run with --no-upload for a local dry run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
from wyckoff_transformer.dataset_cache import as_cache_dir, dataset_cache_dir
from wyckoff_transformer.evaluation.protocol import (
    GeneFingerprinter,
    load_genes,
    reference_identity,
    same_reference,
)
from wyckoff_transformer.evaluation.template_novelty import (
    FORMAT_VERSION,
    GENE_TEMPLATES_FILE,
    attach_labels,
    label_genes,
    label_structures,
    load_template_reference,
    measurement_provenance,
    template_id_from_augmented,
    template_metrics,
)
from wyckoff_transformer.paths import resolve_store_path, runs_root, wandb_dir

logger = logging.getLogger(__name__)
TRACKS = {"free": ("structures.csv", "cifs"),
          "fixed_symmetry": ("structures_fixed_symmetry.csv", "cifs_fixed_symmetry")}
_fingerprinter = None


def _init_worker():
    global _fingerprinter
    _fingerprinter = GeneFingerprinter()


def fingerprint_cif(task):
    """Cache by CIF bytes, not run or row number: arms often reuse geometry."""
    path, cache_root = map(Path, task)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    cached = cache_root / f"v{FORMAT_VERSION}_{digest}.json"
    if cached.is_file():
        return json.loads(cached.read_text())
    from pymatgen.core import Structure
    try:
        structure = Structure.from_str(data.decode(), fmt="cif")
        result = {"template_id": template_id_from_augmented(
            _fingerprinter.fingerprint_structure(structure)), "error": None}
    except Exception as exc:  # noqa: BLE001 - unresolved symmetry is an explicit measurement outcome
        result = {"template_id": None, "error": f"{type(exc).__name__}: {exc}"}
    # Duplicate CIFs can be processed concurrently; atomic replacement keeps
    # readers from seeing a partial JSON file.
    import os
    temporary = cached.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result))
    temporary.replace(cached)
    return result


def _matches(path, entry):
    if not path.is_file():
        return False
    from wandb.sdk.lib.hashutil import md5_file_b64, xxh128_file_b64
    digest = xxh128_file_b64(path) if entry.extra.get("alg") == "XXH128" else md5_file_b64(path)
    return digest == entry.digest


def fetch_entry(entry, directory, local=None):
    """Local reuse requires the exact digest from the immutable W&B artifact."""
    target = directory / entry.path
    if _matches(target, entry):
        return target
    candidate = local / entry.path if local is not None else None
    if candidate is not None and _matches(candidate, entry):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate, target)
    else:
        entry.download(root=str(directory))
    if not _matches(target, entry):
        raise ValueError(f"Artifact digest mismatch: {target}")
    return target


def source_destination(collection, run_ids):
    """Resolve standard arm names and the historical _ehull005 collection."""
    suffix = collection.removeprefix("protocol_")
    for run_id in sorted(run_ids, key=len, reverse=True):
        if suffix == run_id:
            return run_id, "protocol/"
        if suffix.startswith(run_id + "."):
            return run_id, f"protocol_{suffix[len(run_id) + 1:]}/"
        if suffix == run_id + "_ehull005":
            return run_id, "protocol_ehull005/"
    return None, None


def analyse_artifact(artifact, root, pool, downloader, references, fingerprinter):
    collection = artifact.name.rsplit(":", 1)[0]
    source = f"{artifact.entity}/{artifact.project}/{collection}:{artifact.version}"
    directory = root / "inputs" / collection / artifact.version
    out = root / "results" / collection / artifact.version
    out.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    entries = {name: artifact.get_entry(name) for name in artifact.manifest.entries}
    for name in ("manifest.json", "screen.json", "wyckoff_genes.json.gz"):
        if name not in entries:
            raise FileNotFoundError(f"{source}: missing {name}")
        fetch_entry(entries[name], directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    local = Path(manifest["input"]).parent if manifest.get("input") else None
    if local is not None and not local.is_absolute():
        local = resolve_store_path(local)
    genes = load_genes(directory / "wyckoff_genes.json.gz")
    screen = json.loads((directory / "screen.json").read_text())
    if len(genes) != screen["n_sampled"]:
        raise ValueError(f"{source}: cohort size differs from screen")
    from wyckoff_transformer.cli.protocol import genes_digest
    recorded = manifest.get("lineage", {}).get("screen.json", {})
    if recorded.get("parent") and recorded["parent"] != genes_digest(genes):
        raise ValueError(f"{source}: screen belongs to another gene cohort")
    reference_cache = manifest.get("reference_cache")
    if reference_cache is None:
        raise ValueError(f"{source}: novelty reference not recorded")
    splits = manifest.get("reference_splits", ["train", "val", "test"])
    splits = tuple(splits.split(",") if isinstance(splits, str) else splits)
    dataset = as_cache_dir(reference_cache).name
    cache = dataset_cache_dir(dataset)
    identity = reference_identity(cache, splits)
    if recorded.get("reference") and not same_reference(recorded["reference"], identity):
        raise ValueError(f"{source}: screen and score references disagree")
    key = (dataset, splits)
    if key not in references:
        references[key] = load_template_reference(cache, splits)
    reference, reference_record = references[key]
    gene_labels = label_genes(genes, reference, fingerprinter)
    gene_labels.to_csv(out / GENE_TEMPLATES_FILE)
    report = {"gene": {"template_novelty": template_metrics(gene_labels)}}
    missing = []
    for track, (csv, cif_dir) in TRACKS.items():
        if csv not in entries:
            missing.append(csv)
            continue
        fetch_entry(entries[csv], directory, local)
        frame = pd.read_csv(directory / csv, index_col="index")
        if not frame.index.is_unique or not frame.index.isin(range(len(genes))).all():
            raise ValueError(f"{source}/{csv}: invalid gene indices")
        for column in ("has_structure", "valid_structure"):
            if not frame[column].dropna().isin([True, False]).all():
                raise ValueError(f"{source}/{csv}: nonboolean {column}")
        needed = []
        indices = []
        errors = {}
        for index in frame.index[frame.has_structure.fillna(False).astype(bool)]:
            name = f"{cif_dir}/{index}.cif"
            if name not in entries:
                errors[index] = "CIF missing from source artifact"
                missing.append(name)
            else:
                needed.append(entries[name])
                indices.append(index)
        # The SDK's bulk downloader fetches signed URLs in batches, avoiding
        # an authenticated redirect for each remote CIF. Keep exact-digest
        # local reuse for evaluations whose original output directory exists.
        if len(needed) >= 128 and (local is None or not (local / cif_dir).is_dir()):
            artifact.download(root=str(directory), path_prefix=f"{cif_dir}/")
        paths = list(downloader.map(lambda entry: fetch_entry(entry, directory, local), needed))
        results = pool.map(fingerprint_cif, ((str(path), str(root / "fingerprints")) for path in paths), chunksize=8)
        ids = {}
        for index, result in zip(indices, results):
            if result["template_id"] is not None:
                ids[index] = result["template_id"]
            else:
                errors[index] = result["error"]
        labels = label_structures(frame, ids, reference, errors)
        attach_labels(frame, labels, gene_labels)
        frame.to_csv(out / csv)
        report[track] = {"template_novelty": template_metrics(labels, frame)}
        logger.info("%s %s: %s", source, track, report[track]["template_novelty"])
    record = measurement_provenance(reference_record)
    record.update({"source_artifact": source, "source_artifact_id": artifact.id,
                   "source_artifact_digest": artifact.digest, "source_manifest": manifest,
                   "missing_data": missing,
                   "energy_field": manifest.get("hull", {}).get("energy_field"),
                   "artifact_selection": "latest alias of every protocol_eval collection"})
    (out / "template_novelty.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return out, report, record


def publish_result(run, api, artifact, out, report, record, run_ids, project):
    """Keep results in W&B; only then attach measurement summaries to the source."""
    import wandb

    from wyckoff_transformer.cli.protocol_wandb import flatten_funnel
    collection = artifact.name.rsplit(":", 1)[0]
    run.use_artifact(record["source_artifact"])
    name = f"template_novelty_{collection}"
    result = wandb.Artifact(name=name, type="protocol_template_novelty", metadata=record)
    result.add_dir(str(out))
    logged = run.log_artifact(result)
    logged.wait()
    run_id, prefix = source_destination(collection, run_ids)
    if run_id is not None:
        source_run = api.run(f"{project}/{run_id}")
        values = flatten_funnel(report, prefix)
        # Preserve explicit undefined fractions, and remove stale measurements
        # for tracks no longer present. All unrelated summary keys are retained.
        for section in ("gene", "free", "fixed_symmetry"):
            base = f"{prefix}{section}/template_novelty/"
            for key in list(source_run.summary.keys()):
                if key.startswith(base):
                    del source_run.summary[key]
            for key, value in report.get(section, {}).get("template_novelty", {}).items():
                values[f"{base}{key}"] = value
        values[f"{prefix}template_novelty/source_artifact"] = record["source_artifact"]
        values[f"{prefix}template_novelty/result_artifact"] = f"{project}/{name}:{logged.version}"
        source_run.summary.update(values)
    return run_id, f"{project}/{name}:{logged.version}"


def main():
    import wandb
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-entity", default=WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--download-workers", type=int, default=32)
    parser.add_argument("--collection", action="append", help="Restrict to these collection names (repeatable)")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = (resolve_store_path(args.output_dir) if args.output_dir else
            runs_root() / "protocol_template_novelty" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    root.mkdir(parents=True, exist_ok=True)
    (root / "fingerprints").mkdir(exist_ok=True)
    project = f"{args.wandb_entity}/{args.wandb_project}"
    api = wandb.Api(timeout=180)
    collections = list(api.artifact_type("protocol_eval", project=project).collections())
    if args.collection:
        requested = set(args.collection)
        collections = [c for c in collections if c.name in requested]
        if {c.name for c in collections} != requested:
            raise ValueError("Some requested collections do not exist")
    run_ids = {r.id for r in api.runs(project)}
    logger.info("Found %d protocol evaluation collections", len(collections))
    analysis = None
    if not args.no_upload:
        analysis = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                              dir=wandb_dir(), job_type="protocol-template-novelty",
                              name=f"template-novelty-{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
                              config={"selection": "latest alias per protocol_eval collection",
                                      "collections": [c.name for c in collections],
                                      "format_version": FORMAT_VERSION})
        logger.info("Analysis run: %s", analysis.url)
    inventory, rows, references = [], [], {}
    fingerprinter = GeneFingerprinter()
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 mp_context=multiprocessing.get_context("spawn")) as pool, \
                ThreadPoolExecutor(max_workers=args.download_workers) as downloader:
            # Fetch manifests concurrently; consume in collection order so
            # per-source outputs and summaries are deterministic.
            artifacts = [downloader.submit(api.artifact, f"{project}/{c.name}:latest") for c in collections]
            for collection, future in zip(collections, artifacts):
                item = {"collection": collection.name}
                try:
                    artifact = future.result()
                    item["version"] = artifact.version
                    out, report, record = analyse_artifact(artifact, root, pool, downloader, references, fingerprinter)
                    item.update(status="partial" if record["missing_data"] else "complete",
                                source=record["source_artifact"], missing_data=record["missing_data"])
                    for track, section in report.items():
                        rows.append({"source_artifact": record["source_artifact"], "track": track,
                                     "reference_cache": record["source_manifest"].get("reference_cache"),
                                     "mlip": record["source_manifest"].get("mlip"),
                                     **section["template_novelty"]})
                    if analysis is not None:
                        source_run, result_artifact = publish_result(analysis, api, artifact, out, report, record, run_ids, project)
                        item.update(source_run=source_run, result_artifact=result_artifact)
                except Exception as exc:
                    logger.exception("Could not measure %s", collection.name)
                    item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                inventory.append(item)
                (root / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
                pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
                logger.info("Progress: %d/%d collections", len(inventory), len(collections))
        if analysis is not None:
            aggregate = wandb.Artifact(f"template_novelty_backfill_{analysis.id}", type="protocol_template_novelty_report")
            aggregate.add_file(str(root / "inventory.json"))
            aggregate.add_file(str(root / "summary.csv"))
            # A dirty git revision alone cannot reproduce new untracked code.
            # Preserve the measurement implementation with the report as well.
            from wyckoff_transformer.paths import repo_root
            checkout = repo_root()
            if checkout is not None:
                for name in ("evaluation/template_novelty.py", "cli/template_novelty.py",
                             "cli/protocol.py", "cli/protocol_wandb.py"):
                    aggregate.add_file(str(checkout / "src/wyckoff_transformer" / name),
                                       name=f"code/{name}")
            analysis.log_artifact(aggregate).wait()
            analysis.summary.update({"collections": len(inventory),
                                     "complete": sum(x["status"] == "complete" for x in inventory),
                                     "partial": sum(x["status"] == "partial" for x in inventory),
                                     "failed": sum(x["status"] == "failed" for x in inventory)})
    finally:
        if analysis is not None:
            analysis.finish()
    print(f"OUTPUT {root}")
    if any(x["status"] == "failed" for x in inventory):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
