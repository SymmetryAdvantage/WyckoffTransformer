"""Evaluate the published energy critic using existing ORB and DFT labels only.

Stages are resumable. No potential or electronic-structure calculator is loaded.
Run from the repository with the platform interpreter; see --help.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import logging
import multiprocessing
from pathlib import Path
import subprocess
import warnings
import zipfile

import numpy as np
import pandas as pd

from wyckoff_transformer.paths import runs_root, data_path, wandb_dir
from wyckoff_transformer.dataset_cache import dataset_cache_dir, load_split

MODEL = "SymmetryAdvantage/WyFormer-LeMat-Energy"
PROJECT = "symmetry-advantage/WyckoffTransformer"
FP = None


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def gene_from_record(r):
    species, sites = [], []
    for e, m, w in zip(r["elements"], r["multiplicity"], r["wyckoff_letters"]):
        e = str(e)
        if e not in species:
            species.append(e)
            sites.append([])
        sites[species.index(e)].append(f"{int(m)}{w}")
    return dict(group=int(r["spacegroup_number"]), species=species, sites=sites)


def init_worker():
    global FP
    from wyckoff_transformer.evaluation.protocol import GeneFingerprinter

    FP = GeneFingerprinter()
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")


def symmetry(task):
    """Content-addressed symmetry recovery, retaining explicit failed outcomes."""
    source, destination, kind = task
    destination = Path(destination)
    if destination.exists():
        return json.loads(destination.read_text())
    from pymatgen.core import Structure
    from wyckoff_transformer.data import structure_to_sites
    from wyckoff_transformer.evaluation.gene_hash import gene_key

    try:
        raw = Path(source).read_text()
        s = (
            Structure.from_dict(json.loads(raw))
            if kind == "json"
            else Structure.from_str(raw, fmt="cif")
        )
        r = structure_to_sites(
            s, FP.enum_from_ss_letter, FP.augmentation, sort_by_letter=True
        )
        result = dict(
            gene=gene_from_record(r),
            key=list(gene_key(r)),
            error=None,
            composition=s.composition.get_el_amt_dict(),
            n_atoms=len(s),
        )
    except Exception as exc:
        result = dict(gene=None, key=None, error=f"{type(exc).__name__}: {exc}")
    dump(destination, result)
    return result


def prepare(root, args):
    import wandb
    import requests
    from huggingface_hub import HfApi, snapshot_download
    from wyckoff_transformer.cli.template_novelty import fetch_entry
    from wyckoff_transformer.evaluation.protocol import load_genes
    from wyckoff_transformer.cli.protocol import genes_digest

    api = wandb.Api(timeout=120)
    model_revision = args.model_revision or HfApi().model_info(MODEL).sha
    model_path = snapshot_download(MODEL, revision=model_revision)
    dump(
        root / "model.json", dict(repo=MODEL, revision=model_revision, path=model_path)
    )
    inventory = []
    if args.source_inventory:
        from types import SimpleNamespace

        collections = [
            SimpleNamespace(
                name=item["source"].rsplit("/", 1)[-1].rsplit(":", 1)[0],
                pinned_version=item["source"].rsplit(":", 1)[-1],
            )
            for item in json.loads(args.source_inventory.read_text())
        ]
    else:
        collections = list(
            api.artifact_type("protocol_eval", project=PROJECT).collections()
        )
    previous = runs_root() / "protocol_template_novelty/20261023/inputs"

    def fetch(collection):
        a = api.artifact(
            f"{PROJECT}/{collection.name}:{getattr(collection, 'pinned_version', 'latest')}"
        )
        directory = root / "protocol" / collection.name / a.version
        directory.mkdir(parents=True, exist_ok=True)
        local = previous / collection.name / a.version
        # Verify all reused bytes against the selected immutable artifact.
        required = [
            "manifest.json",
            "screen.json",
            "wyckoff_genes.json.gz",
            "structures.csv",
            "structures_fixed_symmetry.csv",
        ]
        entries = a.manifest.entries
        for name in required:
            if name not in entries:
                return dict(
                    source=f"{PROJECT}/{collection.name}:{a.version}",
                    digest=a.digest,
                    id=a.id,
                    error=f"missing {name}",
                )
            fetch_entry(a.get_entry(name), directory, local)
        manifest = json.loads((directory / "manifest.json").read_text())
        genes = load_genes(directory / "wyckoff_genes.json.gz")
        lineage = manifest.get("lineage", {})
        for name in ["screen.json", "pyxtal.csv"]:
            parent = lineage.get(name, {}).get("parent")
            if parent and parent != genes_digest(genes):
                raise ValueError(f"{a.name}: {name} cohort lineage mismatch")
        if not local.exists():
            for prefix in ["cifs/", "cifs_fixed_symmetry/"]:
                if any(name.startswith(prefix) for name in entries):
                    a.download(root=str(directory), path_prefix=prefix)
        else:
            for name in entries:
                if name.startswith(("cifs/", "cifs_fixed_symmetry/")):
                    fetch_entry(a.get_entry(name), directory, local)
        return dict(
            source=f"{PROJECT}/{collection.name}:{a.version}",
            digest=a.digest,
            id=a.id,
            directory=str(directory),
            manifest=manifest,
            error=None,
        )

    if args.reuse_inventory:
        inventory = json.loads((root / "protocol_inventory.json").read_text())
        for item in inventory:
            if not item.get("error"):
                d = Path(item["directory"])
                item["source"] = f"{PROJECT}/{d.parent.name}:{d.name}"
    else:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for i, item in enumerate(pool.map(fetch, collections)):
                inventory.append(item)
                dump(root / "protocol_inventory.json", inventory)
                print(
                    "prepared protocol",
                    i + 1,
                    len(collections),
                    item["source"],
                    flush=True,
                )

    # The archive preserves folders, which disambiguate its repeated data.csv.gz names.
    archive = root / "figshare_v6.zip"
    metadata = requests.get(
        "https://api.figshare.com/v2/articles/29145101/versions/6", timeout=120
    ).json()
    dump(root / "figshare_metadata.json", metadata)
    if not archive.exists():
        response = requests.get(
            "https://figshare.com/ndownloader/articles/29145101/versions/6",
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        with archive.open("wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
    tables, tasks = [], {}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if "/DFT" not in name or not name.endswith(".csv.gz"):
                continue
            frame = pd.read_csv(io.BytesIO(z.read(name)), compression="gzip")
            energy_column = next(
                (c for c in ["dft_e_uncorrected", "energy_dft"] if c in frame), None
            )
            for i, r in frame.iterrows():
                text = r["structure"]
                digest = hashlib.sha256(text.encode()).hexdigest()
                path = root / "geometry" / f"{digest}.json"
                path.write_text(text)
                tasks[digest] = (
                    str(path),
                    str(root / "symmetry" / f"{digest}.json"),
                    "json",
                )
                tables.append(
                    dict(
                        dataset="c",
                        cohort=name,
                        index=i,
                        material_id=r["material_id"],
                        geometry=digest,
                        energy=float(r[energy_column]) if energy_column else None,
                        source_energy_column=energy_column,
                        target_error=None
                        if energy_column
                        else "Only formation energy with unrecorded reference/corrections; no raw total DFT energy",
                    )
                )
            print("figshare", name, len(frame), flush=True)
    dump(root / "c_rows.json", tables)

    # Protocol rows are per unique initial gene, not trial or sampled multiplicity.
    rows, genes_by_cohort = [], {}
    for item in inventory:
        if item["error"]:
            continue
        d = Path(item["directory"])
        m = item["manifest"]
        if m.get("mlip") != "orb_conserv_inf":
            item["excluded_reason"] = "not orb_conserv_inf"
            continue
        genes = load_genes(d / "wyckoff_genes.json.gz")
        genes_by_cohort[item["source"]] = genes
        for track, filename, cifdir in [
            ("free", "structures.csv", "cifs"),
            ("fixed_symmetry", "structures_fixed_symmetry.csv", "cifs_fixed_symmetry"),
        ]:
            frame = pd.read_csv(d / filename)
            for _, r in frame.iterrows():
                index = int(r["index"])
                row = r.to_dict()
                row.update(
                    dataset="b",
                    cohort=item["source"],
                    track=track,
                    reference=Path(
                        str(m.get("reference_cache", "unknown")).removesuffix(
                            "/data.pkl.gz"
                        )
                    ).name,
                    initial_gene=genes[index],
                    geometry=None,
                )
                candidates = [
                    d / cifdir / f"{index}.cif",
                    d / cifdir / f"gene_{index}.cif",
                ]
                path = next((p for p in candidates if p.exists()), None)
                if path is None:
                    # Saved protocol uses zero-padded gene identifiers on some versions.
                    matches = list((d / cifdir).glob(f"*{index:06d}*.cif"))
                    if len(matches) == 1:
                        path = matches[0]
                if (
                    path
                    and r.get("has_structure") == True
                    and np.isfinite(r.get("energy", float("nan")))
                ):
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    row["geometry"] = digest
                    tasks[digest] = (
                        str(path),
                        str(root / "symmetry" / f"{digest}.json"),
                        "cif",
                    )
                rows.append(row)
    dump(root / "b_rows.json", rows)
    dump(root / "symmetry_tasks.json", list(tasks.values()))
    dump(root / "protocol_inventory.json", inventory)
    print("symmetry tasks", len(tasks), "protocol rows", len(rows), flush=True)


def recover(root, args):
    if (root / "symmetry_tasks.json").exists():
        tasks = json.loads((root / "symmetry_tasks.json").read_text())
    else:
        tasks = {}
        for p in (runs_root() / "protocol_template_novelty/20261023/inputs").glob(
            "*/*/cifs*/*.cif"
        ):
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            tasks[digest] = (str(p), str(root / "symmetry" / f"{digest}.json"), "cif")
        tasks = list(tasks.values())
    pending = [t for t in tasks if not Path(t[1]).exists()]
    print("symmetry pending", len(pending), "of", len(tasks), flush=True)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
    ) as pool:
        for i, result in enumerate(pool.map(symmetry, pending, chunksize=16)):
            if i % 1000 == 0:
                print("symmetry", i + 1, "/", len(pending), flush=True)


def archive_sample(root, args):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    revision = (
        args.data_revision
        or HfApi().dataset_info("LeMaterial/LeMat-Bulk-MLIP-Hull-All").sha
    )
    files = HfApi().list_repo_files(
        "LeMaterial/LeMat-Bulk-MLIP-Hull-All", repo_type="dataset", revision=revision
    )
    columns = [
        "immutable_id",
        "nsites",
        "species_at_sites",
        "orb_conserv_inf_energy",
        "true_energy",
    ]
    raw = pd.concat(
        [
            pq.read_table(
                hf_hub_download(
                    "LeMaterial/LeMat-Bulk-MLIP-Hull-All",
                    f,
                    repo_type="dataset",
                    revision=revision,
                ),
                columns=columns,
            ).to_pandas()
            for f in files
            if f.endswith(".parquet")
        ],
        ignore_index=True,
    ).set_index("immutable_id")
    rows = []
    cols = [
        "spacegroup_number",
        "elements",
        "multiplicity",
        "wyckoff_letters",
        "formation_energy_per_atom",
        "gene_min_formation_energy_per_atom",
    ]
    for split in ["train", "val", "test"]:
        frame = load_split(
            dataset_cache_dir("lemat_bulk_fixed_fmax1_stress"), split, columns=cols
        )
        eligible = frame.index.intersection(raw.index)
        chosen = (
            pd.Series(eligible)
            .sample(min(args.sample_size, len(eligible)), random_state=args.seed)
            .tolist()
        )
        selected_raw = raw.loc[chosen]
        if len(selected_raw) != len(chosen) or not selected_raw.index.is_unique:
            raise ValueError(
                "Sample contains an ambiguous immutable_id in the published ORB table"
            )
        for index, r in frame.loc[chosen].iterrows():
            rr = selected_raw.loc[index]
            rows.append(
                dict(
                    dataset="a",
                    cohort=split,
                    index=str(index),
                    initial_gene=gene_from_record(r),
                    energy=float(rr.orb_conserv_inf_energy),
                    n_atoms=int(rr.nsites),
                    species=list(rr.species_at_sites),
                    true_energy=float(rr.true_energy),
                    formation_energy_per_atom=float(r.formation_energy_per_atom),
                    gene_min_formation_energy_per_atom=float(
                        r.gene_min_formation_energy_per_atom
                    ),
                )
            )
        print("archive sampled", split, len(chosen), flush=True)
        del frame
    dump(root / "a_rows.json", rows)
    dump(
        root / "archive_provenance.json",
        dict(
            repo="LeMaterial/LeMat-Bulk-MLIP-Hull-All",
            revision=revision,
            seed=args.seed,
            per_split=args.sample_size,
            eligible="IDs in fixed regressor dataset and published ORB table; sample before finite-label filtering",
        ),
    )


def infer(root, args):
    import torch
    from wyckoff_transformer.trainer import WyckoffTrainer
    from wyckoff_transformer.cli.gene_screen import predict_gene_scalars
    from wyckoff_transformer.evaluation.protocol import GeneFingerprinter
    from wyckoff_transformer.evaluation.gene_hash import GeneKeyTable, gene_key

    torch.set_num_threads(4)
    fp = GeneFingerprinter()
    model_info = json.loads((root / "model.json").read_text())
    model = WyckoffTrainer.from_huggingface(
        MODEL, revision=model_info["revision"], device=torch.device(args.device)
    )
    dump(root / "field_provenance.json", model.field_provenance)
    keys = {}

    def register(gene):
        if gene is None:
            return None
        r = fp.record(gene)
        k = ":".join(map(str, gene_key(r)))
        keys[k] = gene
        return k

    rows = []
    inventory = json.loads((root / "protocol_inventory.json").read_text())
    source_references = {
        item["source"]: Path(
            str(item["manifest"].get("reference_cache", "unknown")).removesuffix(
                "/data.pkl.gz"
            )
        ).name
        for item in inventory
        if not item.get("error")
    }
    for part in ["a", "b", "c"]:
        if not (root / f"{part}_rows.json").exists():
            print("input not ready; rerun infer after preparation:", part, flush=True)
            continue
        for r in json.loads((root / f"{part}_rows.json").read_text()):
            if part == "b":
                r["reference"] = source_references[r["cohort"]]
            if part == "a":
                r["initial_key"] = register(r["initial_gene"])
                r["final_key"] = r["initial_key"]
                r["composition"] = {e: r["species"].count(e) for e in set(r["species"])}
            else:
                if part == "b":
                    r["initial_key"] = register(r["initial_gene"])
                if part == "b" and (
                    r.get("has_structure") != True
                    or not np.isfinite(r.get("energy", float("nan")))
                ):
                    r["geometry"] = None
                geo = r.get("geometry")
                p = root / "symmetry" / f"{geo}.json"
                result = (
                    json.loads(p.read_text())
                    if geo and p.exists()
                    else dict(gene=None, error="missing geometry")
                )
                r["final_key"] = register(result["gene"])
                r["symmetry_error"] = result["error"]
                if result["gene"]:
                    if (
                        part == "b"
                        and r.get("n_atoms")
                        and np.isfinite(r["n_atoms"])
                        and int(r["n_atoms"]) != result["n_atoms"]
                    ):
                        raise ValueError(
                            f"Energy/CIF atom-count mismatch: {r['cohort']} {r['index']}"
                        )
                    r["composition"] = result["composition"]
                    r["n_atoms"] = result["n_atoms"]
                    if part == "b":
                        from collections import Counter

                        expected = Counter()
                        record = fp.record(r["initial_gene"])
                        for element, count in zip(
                            record["elements"], record["multiplicity"]
                        ):
                            expected[str(element)] += int(count)
                        observed = result["composition"]
                        total = sum(expected.values())
                        actual = sum(observed.values())
                        if set(expected) != set(observed) or any(
                            abs(expected[e] / total - observed[e] / actual) > 1e-8
                            for e in expected
                        ):
                            r["lineage_error"] = "Gene/structure composition mismatch"
                if part == "c":
                    r["initial_key"] = r["final_key"]
                if part == "b" and r.get("initial_key") and r.get("final_key"):
                    r["same_gene"] = r["initial_key"] == r["final_key"]
            rows.append(r)
    bad_cohorts = sorted({r["cohort"] for r in rows if r.get("lineage_error")})
    dump(
        root / "lineage_audit.json",
        dict(
            excluded_cohorts=bad_cohorts,
            mismatches=[
                dict(
                    cohort=r["cohort"],
                    index=r["index"],
                    track=r["track"],
                    error=r["lineage_error"],
                )
                for r in rows
                if r.get("lineage_error")
            ],
        ),
    )
    for r in rows:
        if r["cohort"] in bad_cohorts:
            r["evaluation_error"] = (
                "Cohort excluded: gene/structure composition mismatch in saved artifact"
            )
    dump(root / "gene_registry.json", keys)
    # Exact vectorised gene membership, under both protocol references and the model's.
    refs = ["lemat_bulk_fixed_fmax1_stress", "lemat_bulk_fmax1_stress"]
    key_order = list(keys)
    tensor = torch.tensor(
        [[int(v) for v in k.split(":")] for k in key_order], dtype=torch.int64
    )
    membership = {}
    for ref in refs:
        table = GeneKeyTable.load(dataset_cache_dir(ref) / "gene_keys.npz")
        known = table.contains(tensor).tolist()
        membership[ref] = dict(zip(key_order, known))
        dump(root / f"membership_{ref}.json", membership[ref])
        dump(root / f"reference_{ref}.json", table.meta)
    predicted = {}
    predpath = root / "predictions.json"
    if predpath.exists():
        predicted = json.loads(predpath.read_text())
    pending = [k for k in key_order if k not in predicted]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for i in range(0, len(pending), 256):
        batch = pending[i : i + 256]
        values = predict_gene_scalars(
            [keys[k] for k in batch], model, augmentation_samples=args.augmentations
        )
        predicted.update(
            {
                k: dict(
                    prediction=float(values.iloc[j].prediction),
                    reason=str(values.iloc[j].reason),
                )
                for j, k in enumerate(batch)
            }
        )
        if i % 4096 == 0:
            dump(predpath, predicted)
            print("predicted", i + len(batch), "/", len(pending), flush=True)
    dump(predpath, predicted)
    for r in rows:
        for which in ["initial", "final"]:
            k = r.get(which + "_key")
            r[which + "_prediction"] = predicted.get(k, {}).get("prediction")
            r[which + "_gene_novel_fixed"] = not membership[refs[0]][k] if k else None
            ref = r.get("reference", refs[0])
            ref = ref if ref in membership else refs[0]
            r[which + "_gene_novel_reference"] = not membership[ref][k] if k else None
    dump(root / "evaluated_rows.json", rows)


def novelty(root, args):
    """StructureMatcher novelty for Figshare, against model and benchmark data."""
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from wyckoff_transformer.evaluation.protocol import GeneFingerprinter
    from wyckoff_transformer.evaluation.structure_novelty import build_novelty_reference
    from wyckoff_transformer.evaluation.novelty import NoveltyFilter

    fp = GeneFingerprinter()
    rows = json.loads((root / "c_rows.json").read_text())
    geometries = sorted({r["geometry"] for r in rows})
    structures = {}
    fingerprints = {}
    for g in geometries:
        result = json.loads((root / "symmetry" / f"{g}.json").read_text())
        if result["gene"]:
            structures[g] = Structure.from_dict(
                json.loads((root / "geometry" / f"{g}.json").read_text())
            )
            fingerprints[g] = fp.fingerprint(result["gene"])
    reference = build_novelty_reference(
        fingerprints.values(),
        cache=dataset_cache_dir("lemat_bulk_fixed_fmax1_stress"),
        lemat_cif_csv=data_path("lemat-bulk-fixed") / "lemat_pbe.csv.gz",
    )
    filt = NoveltyFilter(reference)
    labels = {
        g: dict(
            novel_structure_fixed=filt.is_novel(
                pd.Series(dict(fingerprint=fingerprints[g], structure=s))
            )
        )
        for g, s in structures.items()
    }
    print(
        "DFT novelty fixed",
        len(reference),
        "candidates",
        len(labels),
        "structures",
        flush=True,
    )
    # Original MP-20 benchmark novelty: compare against every split by composition.
    from collections import defaultdict

    grouped = defaultdict(list)
    wanted = {s.composition.reduced_formula for s in structures.values()}
    for split in ["train", "val", "test"]:
        frame = pd.read_csv(data_path("mp_20") / f"{split}.csv", usecols=["cif"])
        for text in frame.cif:
            s = Structure.from_str(text, fmt="cif")
            k = s.composition.reduced_formula
            if k in wanted:
                grouped[k].append(s)
    matcher = StructureMatcher()
    for g, s in structures.items():
        labels[g]["novel_structure_mp20"] = not any(
            matcher.fit(s, ref) for ref in grouped[s.composition.reduced_formula]
        )
    dump(
        root / "c_novelty.json",
        dict(
            labels=labels,
            reference_rows=len(reference),
            reference="lemat_bulk_fixed_fmax1_stress/all splits; fingerprint then default StructureMatcher",
            benchmark_reference="MP-20/all splits; composition then default StructureMatcher",
        ),
    )


def statistics(frame):
    x = pd.to_numeric(frame.prediction, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame.truth, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    e = x - y
    out = dict(n_rows=len(frame), n=int(mask.sum()), excluded=int((~mask).sum()))
    if len(e):
        out.update(
            mae=float(np.mean(abs(e))),
            rmse=float(np.sqrt(np.mean(e**2))),
            bias=float(np.mean(e)),
            median_absolute_error=float(np.median(abs(e))),
            residual_std=float(np.std(e)),
            median_error=float(np.median(e)),
            p90_absolute_error=float(np.quantile(abs(e), 0.9)),
            within_50meV=float(np.mean(abs(e) <= 0.05)),
            within_100meV=float(np.mean(abs(e) <= 0.1)),
            pearson=float(np.corrcoef(x, y)[0, 1])
            if len(e) > 1 and np.std(x) > 0 and np.std(y) > 0
            else None,
            spearman=float(pd.Series(x).corr(pd.Series(y), method="spearman"))
            if len(e) > 1 and np.std(x) > 0 and np.std(y) > 0
            else None,
        )
    return out


def write_study(root, summary, frame, url, provenance):
    """Render the measured tables; the full intersections remain in summary.csv."""
    summary = summary.copy()
    summary["value"] = summary["value"].astype(str)

    def table(headers, rows):
        return "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
            + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
        )

    def number(value):
        return f"{value:.4f}" if pd.notna(value) else "—"

    inventory = json.loads((root / "protocol_inventory.json").read_text())
    audit = json.loads((root / "lineage_audit.json").read_text())
    verification = json.loads((root / "verification.json").read_text())
    test = summary[
        (summary.description == "a: ORB common PBE elemental baseline")
        & (summary.cohort == "test")
    ].iloc[0]
    minimum = summary[
        (summary.description == "a: fixed DFT gene_min_formation_energy_per_atom")
        & (summary.cohort == "test")
    ].iloc[0]
    known = summary[
        (summary.description == "c: novelty split")
        & (summary.axis == "novel_structure_fixed")
        & (summary.value == "False")
    ].iloc[0]
    novel = summary[
        (summary.description == "c: novelty split")
        & (summary.axis == "novel_structure_fixed")
        & (summary.value == "True")
    ].iloc[0]
    body = [
        "# Energy regressor precision on archive and generated structures",
        "",
        f"Measured 2026-10-28 at git `{provenance['git_commit']}`. "
        f"[W&B analysis run]({url}). Model: `{MODEL}` at "
        f"`{provenance['model']['revision']}`; training run "
        "[`min_energy_adamw_wsd_h7x_xl-20261015-185731`](https://wandb.ai/"
        "symmetry-advantage/WyckoffTransformer/runs/min_energy_adamw_wsd_h7x_xl-20261015-185731). "
        "The analysis code and all derived tables are preserved in the run's "
        "`energy_regressor_evaluation` artifact. No new ORB or DFT calculation was run.",
        "",
        f"Archive test MAE is **{1000 * minimum.mae:.1f} meV/atom** against the intended DFT "
        f"gene-minimum target and **{1000 * test.mae:.1f} meV/atom** against ORB. On generated "
        f"DFT structures, MAE is **{1000 * known.mae:.1f} meV/atom** for structures known to "
        f"LeMat-Bulk-fixed and **{1000 * novel.mae:.1f} meV/atom** for novel structures. "
        "Precision measured on the archive therefore does not transfer directly to novel generated structures.",
        "",
        "The model predicts the lowest observed **DFT formation energy of a gene**, "
        "rather than the energy of every geometry realising it. Errors against individual "
        "structures therefore include the energy gap between that structure and the gene's "
        "observed minimum. ORB errors additionally include the difference between ORB's "
        "OMat24 scale and the model's MP PBE/PBE+U scale.",
        "",
        "## Energy normalization and coverage",
        "",
        "All energies and errors below are in **eV/atom**. Raw total energies are divided "
        "by the atom count of the saved structure, then referenced to the same elemental "
        "minima as the model: LeMat-Bulk-fixed `compatible_pbe@bf8f0b31`. This common "
        "subtraction preserves the residual against raw total energy per atom exactly. "
        "No MP2020 correction, fitted offset, hull-distance substitution or clipping is "
        "applied. Bias is prediction minus observation. ORB comparisons are explicitly "
        "cross-scale; `energy_fields.check_compatible` records the differences in "
        "`provenance.json`. Predictions average 16 random equivalent descriptions.",
        "",
        f"The W&B inventory contains {len(inventory)} collections; "
        f"{sum(not x.get('error') for x in inventory)} provide the required protocol files. "
        "One collection (`se-families-hera-20261022-protocol`) lacks `manifest.json` and "
        "is listed as unavailable. Every source artifact is pinned by version, ID and digest "
        "in `protocol_inventory.json`. Tables retain failed rows as exclusions. "
        "Stale CIFs attached to rows without a successful saved structure and finite energy "
        "are ignored. Successful pairs are checked for equal composition and atom count.",
        "",
        f"Composition audit: {len(audit['mismatches'])} mismatched successful pairs; "
        f"{len(audit['excluded_cohorts'])} excluded cohorts. Unsupported model vocabulary "
        "and unresolved symmetry remain explicit missing predictions; denominators below "
        "count finite prediction–target pairs.",
        "",
        "## A Archive ORB energies",
        "",
        "Source: [LeMat-Bulk-MLIP-Hull-All](https://huggingface.co/datasets/LeMaterial/"
        "LeMat-Bulk-MLIP-Hull-All), pinned in `archive_provenance.json`. Uniform samples "
        "of 5,000 IDs from each fixed-dataset train/validation/test split, seed 20261028, "
        "were selected before filtering labels. Training membership here means membership "
        "in the regressor's split; ORB's own training membership has not been established. "
        "The gene-minimum labels were built over all three splits, as recorded by the model, "
        "so a held-out row is not necessarily a held-out gene.",
        "",
    ]
    rows = []
    for split in ["train", "val", "test"]:
        f = summary[(summary.dataset == "a") & (summary.cohort == split)]
        orb = f[f.description == "a: ORB common PBE elemental baseline"].iloc[0]
        minimum = f[
            f.description == "a: fixed DFT gene_min_formation_energy_per_atom"
        ].iloc[0]
        observed = f[f.description == "a: fixed DFT formation_energy_per_atom"].iloc[0]
        baseline = f[f.description == "a: fixed DFT versus ORB baseline"].iloc[0]
        rows.append(
            [
                split,
                int(orb.n),
                number(orb.mae),
                number(orb.rmse),
                number(orb.bias),
                number(minimum.mae),
                number(observed.mae),
                number(baseline.mae),
            ]
        )
    body += [
        table(
            [
                "Split",
                "N",
                "ORB MAE",
                "ORB RMSE",
                "ORB bias",
                "DFT gene minimum MAE",
                "DFT structure MAE",
                "DFT–ORB MAE baseline",
            ],
            rows,
        ),
        "",
        "## B Saved protocol ORB energies",
        "",
        "The unit is the saved best structure for each unique initial gene in each track, "
        "rather than each relaxation trial or sampled duplicate. Fixed-symmetry and free "
        "tracks may choose different trials. Initial-gene predictions measure screening "
        "precision before relaxation; final-gene predictions measure inference from the "
        "gene recovered from the saved geometry. `same_gene` uses the augmentation-invariant "
        "gene identity. Final genes are novel only if absent from all splits of the "
        "protocol's recorded reference.",
        "",
        "Structure novelty uses the saved protocol's `novel_structure` label. Initial and "
        "final gene novelty are recomputed against that protocol reference. "
        f"{verification['initial_gene_novelty_changes_from_saved_labels']} initial-gene labels "
        "differ from the saved labels under the current augmented identity. The sources "
        "use upstream LeMat-Bulk (76 cohorts) or LeMat-Bulk-fixed (3 cohorts); reference-specific "
        "and per-source results are included in `summary.csv`. The pooled tables describe "
        "the available evaluations and are not an estimate over independent draws: some "
        "arms share cohorts or geometries. Deduplicated summaries are supplied separately.",
        "",
    ]
    f = summary[summary.description == "b: pooled source rows"]
    body += [
        table(
            ["Track", "Predictor gene", "N", "Excluded", "MAE", "RMSE", "Bias"],
            [
                [
                    r.track,
                    r.predictor_gene,
                    int(r.n),
                    int(r.excluded),
                    number(r.mae),
                    number(r.rmse),
                    number(r.bias),
                ]
                for r in f.itertuples()
            ],
        ),
        "",
        "### Marginal splits",
        "",
    ]
    f = summary[
        (summary.description == "b: marginal split")
        & (summary.predictor_gene == "initial")
    ]
    body += [
        table(
            ["Track", "Axis", "Value", "N", "MAE", "Bias"],
            [
                [
                    r.track,
                    r.axis.replace("_", " "),
                    r.value,
                    int(r.n),
                    number(r.mae),
                    number(r.bias),
                ]
                for r in f.itertuples()
            ],
        ),
        "",
        "`True` means novel for the novelty axes and unchanged for `same_gene`; `nan` "
        "means unavailable. The 6,003 fixed-symmetry and 6,010 free finite-energy rows "
        "without a structure-novelty label fail the saved structural-validity check; "
        "they remain a separate category. Restricting to valid structures gives initial-gene "
        "MAEs 0.1691 (fixed) and 0.1302 (free), and final-gene free MAE 0.1190. "
        "All joint intersections of initial gene novelty, structure "
        "novelty, final gene novelty and gene preservation, with track and reference, are "
        "in `summary.csv` (`b: joint split` and `b: per source joint split`). Both initial "
        "and final predictions are tabulated for each cell, including counts, MAE, RMSE, "
        "bias, median and 90th-percentile absolute error, correlations, and fractions within "
        "50 and 100 meV/atom. `rows.csv.gz` preserves all row labels.",
        "",
        "## C Figshare DFT energies",
        "",
        "Source: [Figshare 29145101 version 6](https://figshare.com/articles/dataset/"
        "Generated_crystals_for_WyFormer_DiffCSP_DiffCSP_WyCryst_SymmCD_CrystalFormer_MiAD/29145101/6). "
        "All 24 DFT cohorts are inventoried. Raw `dft_e_uncorrected` or `energy_dft` "
        "is used with the saved final structure. The 94-row `DFT-PBE-54` cohort lacks raw "
        "total energies and a recoverable formation-energy convention, so it has no numeric "
        "target here. Other rows without readable model vocabulary are also excluded explicitly.",
        "",
        "Structure novelty is computed both against LeMat-Bulk-fixed (all regressor-dataset "
        "splits, augmented fingerprint then default `StructureMatcher`) and against MP-20 "
        "(all benchmark splits, composition then default `StructureMatcher`). The former "
        "asks whether a structure is known to the regressor's corpus; the latter follows "
        "the generated-data benchmark. Symmetry recovery uses the training tolerances "
        "0.1 Å and 5°. Fingerprint-gated novelty inherits the protocol's candidate-gating "
        "limitation: matches with inconsistent detected fingerprints may be missed.",
        "",
    ]
    f = summary[summary.description == "c: novelty split"]
    body += [
        table(
            ["Novelty reference", "Novel", "N", "Excluded", "MAE", "RMSE", "Bias"],
            [
                [
                    r.axis.replace("novel_structure_", ""),
                    r.value,
                    int(r.n),
                    int(r.excluded),
                    number(r.mae),
                    number(r.rmse),
                    number(r.bias),
                ]
                for r in f.itertuples()
            ],
        ),
        "",
        "### DFT cohorts split by novelty against MP-20",
        "",
    ]
    f = summary[
        (summary.description == "c: per source novelty split")
        & (summary.axis == "novel_structure_mp20")
    ]
    rows = []
    for cohort, ff in f.groupby("cohort"):
        values = []
        for novel in ["False", "True"]:
            cell = ff[ff.value == novel]
            values += (
                [int(cell.iloc[0].n), number(cell.iloc[0].mae)]
                if len(cell)
                else [0, "—"]
            )
        rows.append([cohort.removesuffix("/data.csv.gz"), *values])
    body += [
        table(["Cohort", "Known N", "Known MAE", "Novel N", "Novel MAE"], rows),
        "",
        "## Reproduction and saved outputs",
        "",
        "Use the platform's documented interpreter from the checkout root. The stages are "
        "`prepare`, `sample`, `symmetry`, `infer`, `novelty`, and `report` in "
        "`scripts/evaluate_energy_regressor.py`; finish `sample` and `symmetry` before the "
        "final `infer`. Pin the model and ORB dataset revisions from the saved provenance. "
        "`prepare --source-inventory protocol_inventory.json` selects the saved exact artifact versions; "
        "`prepare --reuse-inventory` resumes local assembly. No stage loads an MLIP "
        "calculator or launches DFT.",
        "",
        "The W&B artifact contains the source inventory, archive sample IDs and labels, "
        "Figshare metadata, common elemental references, model field provenance, gene "
        "registry, individual predictions, recovered symmetry, `rows.csv.gz`, `summary.csv`, "
        "the error-distribution figure (PNG/PDF), verification record, this report and code. "
        "The local run directory is a working copy; W&B holds the recoverable results. "
        "Large JSON files are gzip-compressed in the artifact.",
        "",
    ]
    (root / "study.md").write_text("\n".join(body))


def report(root, args):
    from datetime import datetime, UTC
    from dataclasses import replace
    from pymatgen.core import Composition
    from wyckoff_transformer.formula_energy.hf_reference import load_hf_reference
    from wyckoff_transformer.energy_fields import EnergySource, check_compatible
    from wyckoff_transformer.field_provenance import target_field
    import wandb

    field = target_field(json.loads((root / "field_provenance.json").read_text()))
    reference = load_hf_reference("kazeevn/LeMat-Bulk-fixed")
    unary = reference[~reference.chemsys.str.contains("-")]
    elemental = {}
    for r in unary.itertuples():
        c = Composition(r.full_formula)
        element = str(c.elements[0])
        e = r.energy_corrected / c.num_atoms
        elemental[element] = min(elemental.get(element, float("inf")), e)
    dump(root / "elemental_references.json", elemental)
    dft = replace(field, aggregate="none")
    orb = replace(
        dft,
        source=EnergySource(
            method="mlip",
            dft_settings="PBE_OMat24",
            correction="none",
            mlip_model="orb-v3-conservative-inf-omat-20250404",
        ),
    )
    # User explicitly requests this cross-scale comparison. Record every difference.
    # Common PBE elemental references preserve raw total-energy residuals exactly.
    differences = check_compatible(
        field, orb, "requested cross-scale ORB evaluation", allow=True
    )
    check_compatible(field, dft, "raw DFT with model elemental references")
    rows = json.loads((root / "evaluated_rows.json").read_text())
    if {r["dataset"] for r in rows} != {"a", "b", "c"}:
        raise ValueError(
            "Report requires all three datasets; finish sample and rerun infer"
        )
    labels = json.loads((root / "c_novelty.json").read_text())["labels"]
    for r in rows:
        comp = r.get("composition")
        shift = (
            sum(n * elemental[e] for e, n in comp.items()) / sum(comp.values())
            if comp and all(e in elemental for e in comp)
            else None
        )
        r["elemental_energy_per_atom"] = shift
        # Report formation energies on one explicit common elemental baseline.
        r["truth"] = (
            r["energy"] / r["n_atoms"] - shift
            if shift is not None
            and r.get("n_atoms")
            and r.get("energy") is not None
            and not r.get("evaluation_error")
            else None
        )
        if r["dataset"] == "c":
            r.update(labels.get(r["geometry"], {}))
        r.pop("initial_gene", None)
        r.pop("composition", None)
        r.pop("species", None)
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "rows.csv.gz", index=False)
    summaries = []

    def add(f, description, which="initial", **extra):
        f = f.copy()
        f["prediction"] = f[which + "_prediction"]
        summaries.append(
            dict(
                description=description, predictor_gene=which, **extra, **statistics(f)
            )
        )

    a = frame[frame.dataset == "a"]
    for split, f in a.groupby("cohort"):
        add(f, "a: ORB common PBE elemental baseline", dataset="a", cohort=split)
        baseline = f.copy()
        baseline["initial_prediction"] = baseline["formation_energy_per_atom"]
        add(baseline, "a: fixed DFT versus ORB baseline", dataset="a", cohort=split)
        for column in [
            "formation_energy_per_atom",
            "gene_min_formation_energy_per_atom",
        ]:
            add(
                f.assign(truth=f[column]),
                "a: fixed DFT " + column,
                dataset="a",
                cohort=split,
            )
    b = frame[frame.dataset == "b"]
    # Retain failed rows as exclusions in overall metrics.
    axes = [
        "initial_gene_novel_reference",
        "novel_structure",
        "final_gene_novel_reference",
        "same_gene",
    ]
    for track, f in b.groupby("track"):
        for which in ["initial", "final"]:
            add(f, "b: pooled source rows", which, dataset="b", track=track)
            add(
                f[f.valid_structure == True],
                "b: valid structures",
                which,
                dataset="b",
                track=track,
            )
            unique = f.drop_duplicates(["geometry", which + "_key"])
            add(
                unique,
                "b: unique structure and predictor gene",
                which,
                dataset="b",
                track=track,
            )
            for ref, ff in f.groupby("reference"):
                add(
                    ff,
                    "b: per novelty reference",
                    which,
                    dataset="b",
                    track=track,
                    reference=ref,
                )
            for axis in axes if track == "free" else axes[:2]:
                for value, ff in f.groupby(axis, dropna=False):
                    add(
                        ff,
                        "b: marginal split",
                        which,
                        dataset="b",
                        track=track,
                        axis=axis,
                        value=str(value),
                    )
            used_axes = ["reference"] + (axes if track == "free" else axes[:2])
            for values, ff in f.groupby(used_axes, dropna=False):
                add(
                    ff,
                    "b: joint split",
                    which,
                    dataset="b",
                    track=track,
                    **{k: str(v) for k, v in zip(used_axes, values)},
                )
            for cohort, ff in f.groupby("cohort"):
                add(
                    ff,
                    "b: per source artifact",
                    which,
                    dataset="b",
                    track=track,
                    cohort=cohort,
                )
                for values, fff in ff.groupby(used_axes, dropna=False):
                    add(
                        fff,
                        "b: per source joint split",
                        which,
                        dataset="b",
                        track=track,
                        cohort=cohort,
                        **{k: str(v) for k, v in zip(used_axes, values)},
                    )
    c = frame[frame.dataset == "c"]
    add(c, "c: all source rows", dataset="c")
    add(c.drop_duplicates("geometry"), "c: unique DFT geometries", dataset="c")
    for axis in ["novel_structure_fixed", "novel_structure_mp20"]:
        for value, f in c.groupby(axis, dropna=False):
            add(f, "c: novelty split", dataset="c", axis=axis, value=str(value))
            add(
                f.drop_duplicates("geometry"),
                "c: unique geometry novelty split",
                dataset="c",
                axis=axis,
                value=str(value),
            )
        for (cohort, value), f in c.groupby(["cohort", axis], dropna=False):
            add(
                f,
                "c: per source novelty split",
                dataset="c",
                axis=axis,
                value=str(value),
                cohort=cohort,
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(root / "summary.csv", index=False)
    verified = []
    for track in ["free", "fixed_symmetry"]:
        for which in ["initial", "final"]:
            parent = summary[
                (summary.description == "b: pooled source rows")
                & (summary.track == track)
                & (summary.predictor_gene == which)
            ].iloc[0]
            cells = summary[
                (summary.description == "b: joint split")
                & (summary.track == track)
                & (summary.predictor_gene == which)
            ]
            assert cells.n.sum() == parent.n
            assert cells.n_rows.sum() == parent.n_rows
            verified.append(
                f"Joint cells partition {track}/{which}: {int(parent.n)} finite pairs"
            )
    for split in ["train", "val", "test"]:
        assert len(a[a.cohort == split]) == args.sample_size
    assert not frame[frame.dataset == "a"]["index"].duplicated().any()
    verified.append("Archive sample has distinct IDs and requested split sizes")
    assert set(b.reference) <= {
        "lemat_bulk_fmax1_stress",
        "lemat_bulk_fixed_fmax1_stress",
    }
    verified.append("Every protocol novelty reference is explicitly supported")
    recorded = b[b.gene_novel.notna() & b.initial_gene_novel_reference.notna()]
    changed = int((recorded.gene_novel != recorded.initial_gene_novel_reference).sum())
    dump(
        root / "verification.json",
        dict(
            checks=verified,
            initial_gene_novelty_changes_from_saved_labels=changed,
            note="Structure novelty is the saved source label; gene novelty uses current augmented identity",
        ),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True, layout="constrained")

    def curve(ax, f, column, label):
        error = (
            (
                pd.to_numeric(f[column], errors="coerce")
                - pd.to_numeric(f.truth, errors="coerce")
            )
            .abs()
            .dropna()
            .sort_values()
            .to_numpy()
        )
        error = error[np.isfinite(error)]
        if len(error):
            ax.plot(
                error,
                np.arange(1, len(error) + 1) / len(error),
                label=f"{label} (n={len(error):,})",
            )

    for split, f in a.groupby("cohort"):
        curve(axs[0], f, "initial_prediction", split)
    for track, f in b.groupby("track"):
        curve(axs[1], f, "initial_prediction", track + " initial gene")
        if track == "free":
            curve(axs[1], f, "final_prediction", "free final gene")
    for value, f in c.groupby("novel_structure_fixed"):
        curve(axs[2], f, "initial_prediction", "novel" if value else "known")
    for ax, title in zip(
        axs,
        [
            "a  Archive ORB energies",
            "b  Generated ORB energies",
            "c  Generated DFT energies",
        ],
    ):
        ax.set(title=title, xlabel="Absolute error (eV/atom)", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")
    axs[0].set_ylabel("Fraction within error")
    fig.savefig(root / "precision_ecdf.png", dpi=180)
    fig.savefig(root / "precision_ecdf.pdf")
    plt.close(fig)
    provenance = dict(
        date=datetime.now(UTC).isoformat(),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        model=json.loads((root / "model.json").read_text()),
        cross_scale_differences=differences,
        units="eV/atom",
        error="prediction - truth",
        augmentations=args.augmentations,
        protocol_selection="latest alias of all protocol_eval collections; ORB conservative inf only",
        target="minimum observed DFT formation energy by gene, not arbitrary structure energy",
        normalization="all raw total energies divided by number of atoms, then subtract same model PBE elemental reference; no fitted offset or correction",
        no_new_orb_or_dft=True,
        arguments=vars(args),
    )
    dump(root / "provenance.json", provenance)
    provenance = json.loads((root / "provenance.json").read_text())
    with wandb.init(
        entity="symmetry-advantage",
        project="WyckoffTransformer",
        job_type="energy_regressor_evaluation",
        name="HF energy regressor precision 2026-10-28",
        dir=wandb_dir(),
        config=provenance,
    ) as run:
        write_study(root, summary, frame, run.url, provenance)
        artifact = wandb.Artifact(
            f"energy_regressor_precision_{run.id}",
            type="energy_regressor_evaluation",
            metadata={"model_revision": provenance["model"]["revision"]},
        )
        for path in root.glob("*.json"):
            if path.stat().st_size > 1_000_000:
                import shutil

                compressed = path.with_suffix(path.suffix + ".gz")
                with (
                    path.open("rb") as source,
                    gzip.open(compressed, "wb", compresslevel=3) as target,
                ):
                    shutil.copyfileobj(source, target)
                artifact.add_file(str(compressed))
            else:
                artifact.add_file(str(path))
        for name in ["rows.csv.gz", "summary.csv"]:
            artifact.add_file(str(root / name))
        for name in ["precision_ecdf.png", "precision_ecdf.pdf"]:
            artifact.add_file(str(root / name))
        artifact.add_file(str(root / "study.md"))
        with gzip.open(root / "symmetry.jsonl.gz", "wt", compresslevel=3) as f:
            for path in sorted((root / "symmetry").glob("*.json")):
                f.write(
                    json.dumps(dict(geometry=path.stem, **json.loads(path.read_text())))
                    + "\n"
                )
        artifact.add_file(str(root / "symmetry.jsonl.gz"))
        artifact.add_file(__file__, name="code/evaluate_energy_regressor.py")
        run.log_artifact(artifact).wait()
        run.summary.update(
            {
                "source_rows": len(frame),
                "summary_rows": len(summary),
                "model_revision": provenance["model"]["revision"],
            }
        )
        for r in summary[
            summary.description.isin(
                [
                    "a: ORB common PBE elemental baseline",
                    "b: pooled source rows",
                    "c: novelty split",
                ]
            )
        ].to_dict("records"):
            parts = [
                str(r[k])
                for k in [
                    "dataset",
                    "cohort",
                    "track",
                    "predictor_gene",
                    "axis",
                    "value",
                ]
                if pd.notna(r.get(k))
            ]
            run.summary["/".join(parts)] = {
                k: r[k]
                for k in [
                    "n",
                    "excluded",
                    "mae",
                    "rmse",
                    "bias",
                    "within_50meV",
                    "within_100meV",
                ]
            }
        dump(
            root / "wandb_result.json",
            dict(url=run.url, id=run.id, artifact=artifact.qualified_name),
        )
        print("W&B", run.url, flush=True)
    print(
        summary[
            summary.description.isin(
                [
                    "a: ORB common PBE elemental baseline",
                    "b: pooled source rows",
                    "c: all source rows",
                ]
            )
        ].to_string(index=False),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=["prepare", "symmetry", "sample", "infer", "novelty", "report"]
    )
    p.add_argument(
        "--output", type=Path, default=runs_root() / "regressor_precision_20261028"
    )
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sample-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--augmentations", type=int, default=16)
    p.add_argument("--model-revision")
    p.add_argument("--data-revision")
    p.add_argument(
        "--source-inventory",
        type=Path,
        help="Replay exact artifact versions from a saved protocol_inventory.json",
    )
    p.add_argument(
        "--reuse-inventory",
        action="store_true",
        help="Resume assembly from previously pinned W&B inventory",
    )
    args = p.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    (root / "geometry").mkdir(exist_ok=True)
    (root / "symmetry").mkdir(exist_ok=True)
    {
        "prepare": prepare,
        "symmetry": recover,
        "sample": archive_sample,
        "infer": infer,
        "novelty": novelty,
        "report": report,
    }[args.stage](root, args)


if __name__ == "__main__":
    main()
