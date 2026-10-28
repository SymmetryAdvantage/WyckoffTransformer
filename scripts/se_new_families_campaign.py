"""Family-blind generative search for new Li solid-electrolyte families.

The plan is docs/se_new_families_campaign_plan.md; everything the campaign does is set in
yamls/campaigns/se_new_families_hera_20261025.yaml.  Phase 1 benchmarks a transport
screen on known compounds before anything is generated:

    PY=.venv/bin/python S=scripts/se_new_families_campaign.py
    $PY $S data                     # verify the pinned inputs, parse Kahle 2020
    $PY $S bench-set                # palette compounds + labelled structures, with CIFs
    $PY $S bench-relax              # the candidates' ORB arm on every one of them
    $PY $S bench-frameworks         # Li-free frameworks, matched to OBELiX's -> labels
    $PY $S bench-describe           # coarse descriptors and BVSE barriers (CPU)
    $PY $S bench-pes --pes-kinds frozen   # Xi on the labelled, best-BVSE and control sets
    $PY $S bench-pes --pes-kinds empty    # empty-framework barriers of every structure
    $PY $S bench-evaluate           # pAUC on both label sets, rediscovery, gate G1
    $PY $S xi-check                 # ORB Xi against the values Maevskiy et al. printed
    $PY $S bvse-check               # our BVSE barriers against LiTraj's BVEL13k
    $PY $S matcher-calibrate        # gate G1b: 20 known-SE genes through the protocol
    $PY $S md-calibrate             # gate G1c: ORB MD on the fast and poor references

Phase 2 mines the palette compounds outside the families campaign's systems:

    $PY $S mine-gap                 # MP gap, else the gene-level regressor's
    $PY $S coordination-reference   # LeMat's cation-environment table (CPU)
    $PY $S mine-screen              # gap, anion roles, coordination, known-SE frameworks
    $PY $S mine-pes                 # the G1 descriptor where the benchmark did not scan
    $PY $S mine-rank                # survivors in the control's top 10%, by framework
    $PY $S mine-md                  # Phase 7 MD on the top clusters' best members
    $PY $S mine-dimension           # framework dimension: what E_2D's line selects

Phase 8 takes the clusters whose MD reached a tier:

    $PY $S mine-md-members          # Phase 7's MD on up to 3 structures per such cluster
    $PY $S phase8-entries           # LeMat-Bulk-fixed rows of their systems, MP2020 (CPU)
    $PY $S phase8-window            # windows, interface reactions, join flags (CPU)
    $PY $S phase8-check             # ORB structure vs DFT structure, polymorphs, ICSD
    $PY $S phase8-prototypes        # structure types among LeMat's ICSD-backed rows
    $PY $S phase8-report            # the plan's Phase 8 fields per member
    $PY $S upload                   # the benchmark and the gate runs as W&B artifacts

Outputs live in the runs store under ``<campaign_id>/``.  Every stage resumes:
``bench-relax`` keeps a row log and skips what it already relaxed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import sys
import time
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from wyckoff_transformer.dataset_cache import dataset_cache_dir, load_split
from wyckoff_transformer.paths import data_path, runs_root

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "yamls/campaigns/se_new_families_hera_20261025.yaml"
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Campaign definition
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def campaign_dir(config: dict, out: Path | None) -> Path:
    return out if out is not None else runs_root() / config["campaign_id"]


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.rename(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned(config_entry: dict, key: str, path: Path) -> Path:
    """*path*, after checking it against the sha256 the config pins, if any."""
    expected = config_entry.get(f"{key}_sha256")
    if expected:
        found = sha256(path)
        if found != expected:
            raise ValueError(f"{path} has sha256 {found}, the campaign pins {expected}")
    return path


def families_campaign_systems(config: dict) -> set[str]:
    """The 126 chemical systems of the families campaign."""
    sys.path.insert(0, str(REPO / "scripts"))
    from solid_electrolyte_campaign import expand_systems
    from solid_electrolyte_campaign import load_config as load_families_config

    return set(expand_systems(load_families_config(REPO / config["families_campaign"]))["system"])


def element_set(composition) -> frozenset[str]:
    return frozenset(str(element) for element in composition)


def canonical_system(elements) -> str:
    return "-".join(sorted(set(elements)))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def run_data(args, config: dict) -> None:
    """Check every pinned input and write the parsed label tables."""
    from wyckoff_transformer.transport.sources import load_kahle, load_obelix

    root = campaign_dir(config, args.out)
    spec = config["data"]
    obelix = load_obelix(
        pinned(spec["obelix"], "table", data_path(spec["obelix"]["table"])),
        pinned(spec["obelix"], "cifs", data_path(spec["obelix"]["cifs"])),
    )
    write_parquet(obelix, root / "data" / "obelix.parquet")
    logger.info("OBELiX: %d entries, %d with a CIF", len(obelix), obelix["cif"].notna().sum())

    kahle_json = data_path(spec["kahle2020"]["data_json"])
    digest = sha256(kahle_json)
    if spec["kahle2020"].get("data_json_sha256") not in (None, digest):
        raise ValueError(f"{kahle_json} has sha256 {digest}, the campaign pins "
                         f"{spec['kahle2020']['data_json_sha256']}")
    kahle = load_kahle(kahle_json)
    write_parquet(kahle, root / "data" / "kahle2020.parquet")
    pinned(spec["bvel13k"], "zip", data_path(spec["bvel13k"]["zip"]))
    (root / "data" / "inputs.json").write_text(json.dumps({
        "obelix_table_sha256": sha256(data_path(spec["obelix"]["table"])),
        "obelix_cifs_sha256": sha256(data_path(spec["obelix"]["cifs"])),
        "kahle2020_data_json_sha256": digest,
        "bvel13k_zip_sha256": sha256(data_path(spec["bvel13k"]["zip"])),
    }, indent=2))
    print(f"OBELiX {len(obelix)} entries, Kahle {len(kahle)} records -> {root / 'data'}")


# --------------------------------------------------------------------------- #
# bench-set
# --------------------------------------------------------------------------- #
def palette_compounds(config: dict) -> pd.DataFrame:
    """The palette's ionic Li compounds within the e_hull cut, from every split.

    Reproduces the plan's counts: 49,148 ionic Li compounds within 0.1 eV/atom, of
    which 12,383 have every element in the palette.
    """
    spec = config["data"]["lemat"]
    palette = set(config["palette"])
    anions = set(config["ionic_anions"])
    columns = ["composition", "energy_above_hull", "spacegroup_number", "multiplicity", "dof"]
    frames = []
    for split in SPLITS:
        frame = load_split(dataset_cache_dir(spec["cache"]), split, columns=columns)
        frame["split"] = split
        frames.append(frame)
    table = pd.concat(frames)
    elements = table["composition"].map(element_set)
    ionic = (table["energy_above_hull"] <= spec["max_e_hull"]) & elements.map(
        lambda found: config["mobile_ion"] in found and bool(found & anions))
    in_palette = ionic & elements.map(lambda found: found <= palette)
    logger.info("Ionic %s compounds within %.2f eV/atom: %d; in the palette: %d",
                config["mobile_ion"], spec["max_e_hull"], int(ionic.sum()), int(in_palette.sum()))
    chosen = table[in_palette].copy()
    chosen["system"] = elements[in_palette].map(canonical_system)
    chosen["formula_cache"] = chosen["composition"].map(
        lambda c: "".join(f"{e}{n}" for e, n in sorted((str(k), v) for k, v in c.items())))
    chosen["atoms_conventional"] = chosen["multiplicity"].map(sum)
    chosen["free_coordinates"] = chosen["dof"].map(sum)
    icsd = pd.concat([
        load_split(dataset_cache_dir(spec["icsd_cache"]), split, columns=["icsd_backed"])
        for split in SPLITS])
    chosen["icsd_backed"] = icsd["icsd_backed"].reindex(chosen.index).astype(bool)
    chosen.index.name = "immutable_id"
    return chosen.drop(columns=["composition", "multiplicity", "dof"]).reset_index()


def lemat_cifs(path: Path, ids: set[str]) -> dict[str, tuple[str, str]]:
    """``immutable_id -> (full_formula, cif)`` for *ids*, streamed from LeMat's CSV."""
    found: dict[str, tuple[str, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for chunk in pd.read_csv(handle, usecols=["immutable_id", "full_formula", "cif"],
                                 chunksize=200_000):
            hit = chunk[chunk["immutable_id"].isin(ids)]
            for row in hit.itertuples(index=False):
                found[row.immutable_id] = (row.full_formula, row.cif)
            if len(found) == len(ids):
                break
    return found


def same_composition(cif: str, reduced_formula: str) -> bool:
    from pymatgen.core import Composition, Structure

    structure = Structure.from_str(cif, fmt="cif")
    return structure.composition.reduced_formula == Composition(reduced_formula).reduced_formula


def to_primitive_cif(cif: str) -> tuple[str, int]:
    """The CIF's primitive cell (spglib at 1e-3 A) and its site count."""
    from pymatgen.core import Structure

    structure = Structure.from_str(cif, fmt="cif")
    primitive = structure.get_primitive_structure(tolerance=1e-3)
    if len(primitive) < len(structure):
        structure = primitive
    return structure.to(fmt="cif"), len(structure)


def run_bench_set(args, config: dict) -> None:
    """One table of every structure Phase 1 relaxes: palette, Kahle and ordered OBELiX."""
    from pymatgen.core import Structure

    warnings.filterwarnings("ignore", module="pymatgen")
    root = campaign_dir(config, args.out)
    palette = palette_compounds(config)
    palette["in_families_campaign"] = palette["system"].isin(families_campaign_systems(config))
    cifs = lemat_cifs(data_path(config["data"]["lemat"]["structures"]),
                      set(palette["immutable_id"]))
    missing = sorted(set(palette["immutable_id"]) - set(cifs))
    if missing:
        logger.warning("%d palette ids have no CIF in LeMat's CSV: %s", len(missing), missing[:10])
    palette["cif"] = palette["immutable_id"].map(lambda key: cifs.get(key, (None, None))[1])
    palette = palette[palette["cif"].notna()].copy()
    agree = [same_composition(cif, formula)
             for cif, formula in zip(palette["cif"], palette["formula_cache"])]
    palette["cif_agrees"] = agree
    if not all(agree):
        logger.warning("%d CIFs disagree with the cache's composition (structure replaced "
                       "in LeMat-Bulk-fixed); dropped: %s", len(agree) - sum(agree),
                       palette.loc[~palette["cif_agrees"], "immutable_id"].tolist()[:20])
    palette = palette[palette["cif_agrees"]].drop(columns="cif_agrees")
    palette["source"] = "lemat"
    palette["structure_id"] = palette["immutable_id"]

    kahle = pd.read_parquet(root / "data" / "kahle2020.parquet")
    kahle_structures = (kahle.sort_values("temperature")
                        .drop_duplicates("structure_uuid", keep="last")
                        [["structure_uuid", "reduced_formula", "cif"]]
                        .rename(columns={"structure_uuid": "structure_id"}))
    kahle_structures["source"] = "kahle2020"

    obelix = pd.read_parquet(root / "data" / "obelix.parquet")
    ordered = []
    for row in obelix[obelix["cif"].notna()].itertuples(index=False):
        structure = Structure.from_str(row.cif, fmt="cif")
        if structure.is_ordered:
            ordered.append({"structure_id": f"obelix-{row.ID}", "source": "obelix",
                            "reduced_formula": structure.composition.reduced_formula,
                            "cif": structure.to(fmt="cif")})
    obelix_structures = pd.DataFrame(ordered)

    bench = pd.concat([palette, kahle_structures, obelix_structures], ignore_index=True)
    if config["benchmark"]["relax"].get("primitive", True):
        reduced = [to_primitive_cif(cif) for cif in bench["cif"]]
        bench["cif"] = [cif for cif, _ in reduced]
        bench["n_atoms"] = [n for _, n in reduced]
    else:
        bench["n_atoms"] = [len(Structure.from_str(cif, fmt="cif")) for cif in bench["cif"]]
    bench["reduced_formula"] = [Structure.from_str(cif, fmt="cif").composition.reduced_formula
                                for cif in bench["cif"]]
    bench.insert(0, "index", np.arange(len(bench)))
    write_parquet(bench, root / "bench" / "structures.parquet")
    counts = bench["source"].value_counts().to_dict()
    print(f"{len(bench)} structures ({counts}), {int(bench['n_atoms'].sum())} atoms "
          f"in the relaxed cells -> {root / 'bench' / 'structures.parquet'}")


# --------------------------------------------------------------------------- #
# bench-relax
# --------------------------------------------------------------------------- #
def run_bench_relax(args, config: dict) -> None:
    """The protocol's ORB relaxation of every benchmark structure.

    The task function and worker initialiser are the protocol's own
    (:func:`wyckoff_transformer.cli.protocol._relax_one`), so a benchmark structure is
    relaxed exactly as a generated candidate is: fix-cell warm-up and symmetric stage
    under FixSymmetry, then the free stage and the rattle, each trial seeded by its key.
    """
    from ase.io import read as ase_read

    from wyckoff_transformer.cli import protocol
    from wyckoff_transformer.cli.worker_pool import run_supervised

    root = campaign_dir(config, args.out)
    spec = config["benchmark"]["relax"]
    bench = pd.read_parquet(root / "bench" / "structures.parquet")
    if args.limit is not None:
        bench = bench.head(args.limit)
    out = root / "bench" / "relax"
    out.mkdir(parents=True, exist_ok=True)
    log = protocol.RowLog(out / "relaxations.csv", protocol.RELAXATION_COLUMNS, resume=True,
                          retry_failed=args.retry_failed)
    # Largest first, so the long relaxations do not trail the run.
    bench = bench.sort_values("n_atoms", ascending=False)
    todo = [row for row in bench.itertuples(index=False) if (int(row.index), 0) not in log.done]
    logger.info("Relaxing %d of %d benchmark structures (%d already done) with %s",
                len(todo), len(bench), len(log.done), spec["mlip"])
    devices = args.devices or spec["devices"]
    slots = protocol.resolve_devices(None, devices, args.workers_per_device
                                     or spec["workers_per_device"])
    # Set in the parent too: MKL fixes its thread count when it is first loaded.
    for key, value in protocol._SINGLE_THREAD_ENV_VARS.items():
        os.environ.setdefault(key, value)
    budgets: dict = {}

    def initargs(counter, live):
        budgets.clear()
        budgets.update(protocol.resolve_device_budgets(live))
        return (counter, live, spec["mlip"], None, False, dict(budgets))

    def tasks():
        for row in todo:
            atoms = ase_read(StringIO(row.cif), format="cif")
            trial_dir = out / "trials" / str(int(row.index))
            yield ((int(row.index), 0), (
                int(row.index), 0, atoms, str(trial_dir), spec["fmax"], True, True,
                spec["timeout_s"], 0.1, None, "cryspr", spec["steps"]))

    started = time.time()
    try:
        report = run_supervised(
            tasks(), protocol._relax_one, slots=slots,
            initializer=protocol._init_relax_worker, initargs=initargs,
            on_result=lambda key, row: log.write(row),
            on_failure=lambda key, status, error: log.write(
                protocol._failed_row(key, status, error)),
            task_timeout=spec["timeout_s"],
            progress=protocol._progress_logger("Relaxed", 100, unit="structure"),
        )
    finally:
        log.close()
    rows = protocol.latest_rows(protocol.read_rows(out / "relaxations.csv",
                                                   protocol.RELAXATION_COLUMNS))
    (out / "manifest.json").write_text(json.dumps({
        "mlip": spec["mlip"], "fmax": spec["fmax"], "steps": spec["steps"],
        "schedule": "cryspr", "slots": slots, "seconds": round(time.time() - started),
        "structures": len(bench), "ok": int((rows["status"] == "ok").sum()),
        "ok_fixed": int((rows["status_fixed"] == "ok").sum()),
        **report.manifest("relax"),
    }, indent=2))
    print(f"{int((rows['status'] == 'ok').sum())}/{len(bench)} relaxed -> {out}")


def relaxed_structures(root: Path) -> pd.DataFrame:
    """Benchmark rows with a free ORB relaxation: ``relaxed_cif``, ``orb_energy_per_atom``."""
    from wyckoff_transformer.cli import protocol

    rows = protocol.latest_rows(protocol.read_rows(root / "bench" / "relax" / "relaxations.csv",
                                                   protocol.RELAXATION_COLUMNS))
    ok = rows[rows["status"] == "ok"][["index", "cif", "energy_per_atom"]].rename(
        columns={"cif": "relaxed_cif", "energy_per_atom": "orb_energy_per_atom"})
    ok["index"] = ok["index"].astype(int)
    bench = pd.read_parquet(root / "bench" / "structures.parquet").drop(columns="cif")
    return bench.merge(ok, on="index")


# --------------------------------------------------------------------------- #
# Resumable worker pools
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a line a kill left half-written
    return rows


def _cpu_init() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    warnings.filterwarnings("ignore")
    logging.getLogger("pymatgen").setLevel(logging.ERROR)


def run_pool(tasks: list[tuple], fn, out_path: Path, workers: int, initializer=_cpu_init,
             initargs: tuple = (), label: str = "structures", every: int = 200) -> None:
    """Run ``fn(task)`` for every task not yet in *out_path*, one JSON line per result."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # A row with an error (an OOM on a shared card, a crash) is retried on the next run.
    done = {row["index"] for row in read_jsonl(out_path) if "error" not in row}
    todo = [task for task in tasks if task[0] not in done]
    logger.info("%s: %d to do, %d already done", out_path.name, len(todo), len(done))
    if not todo:
        return
    started = time.time()
    context = multiprocessing.get_context("spawn")
    with open(out_path, "a", encoding="utf-8") as handle, ProcessPoolExecutor(
            workers, mp_context=context, initializer=initializer, initargs=initargs) as pool:
        futures = {pool.submit(fn, task): task[0] for task in todo}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - one structure must not stop the stage
                row = {"index": futures[future], "error": f"{type(exc).__name__}: {exc}"}
            handle.write(json.dumps(row, default=float) + "\n")
            handle.flush()
            if count % every == 0 or count == len(todo):
                rate = (time.time() - started) / count
                logger.info("%s: %d/%d %s (%.2f s each, %.0f min left)", out_path.name, count,
                            len(todo), label, rate, rate * (len(todo) - count) / 60)


# --------------------------------------------------------------------------- #
# bench-frameworks
# --------------------------------------------------------------------------- #
def _framework_one(task: tuple) -> dict:
    """Framework of one relaxed structure, and the OBELiX frameworks it matches."""
    from pymatgen.core import Structure

    from wyckoff_transformer.transport.frameworks import (
        default_matcher,
        framework,
        framework_formula,
        structure_anions,
    )

    index, path, references, tolerances = task
    structure = Structure.from_file(path)
    anions = structure_anions(structure)
    frame = framework(structure, anions=anions)
    row = {"index": index, "anions": "".join(sorted(anions))}
    if frame is None:
        return row | {"framework_formula": None, "obelix_matches": []}
    formula = framework_formula(frame)
    matcher = default_matcher(**tolerances)
    matches = [key for key, cif in references.get(formula, [])
               if matcher.fit(Structure.from_str(cif, fmt="cif"), frame)]
    return row | {"framework_formula": formula, "obelix_matches": matches,
                  "framework_cif": frame.to(fmt="cif")}


def obelix_frameworks(obelix: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """OBELiX frameworks from the CIFs, grouped; per-entry group and per-group labels."""
    from pymatgen.core import Structure

    from wyckoff_transformer.transport.frameworks import framework, group_frameworks

    frames = {}
    for row in obelix[obelix["cif"].notna()].itertuples(index=False):
        frame = framework(Structure.from_str(row.cif, fmt="cif"))
        if frame is not None:
            frames[row.ID] = frame
    groups = group_frameworks(frames)
    entries = obelix[obelix["ID"].isin(frames)].copy()
    entries["framework_group"] = entries["ID"].map(groups)
    entries["framework_formula"] = entries["ID"].map(
        lambda key: frames[key].composition.reduced_formula)
    entries["framework_cif"] = entries["ID"].map(lambda key: frames[key].to(fmt="cif"))
    return entries, frames


def label_obelix_groups(entries: pd.DataFrame, fast: float, poor: float) -> pd.DataFrame:
    """Per framework group: fast if any measured member >= *fast*, poor if every <= *poor*.

    A censored value ``<x`` counts as at most x; a group with no fast member and one
    member above *poor* is left unlabelled.
    """
    def family(values):
        named = [v for v in values if v and v != "nan"]
        return pd.Series(named).mode().iloc[0] if named else None

    rows = []
    for group, members in entries.groupby("framework_group"):
        measured = members[members["censoring"] == ""]["sigma"]
        is_fast = bool((measured >= fast).any())
        upper = members["sigma"]       # value or upper bound
        is_poor = bool((upper <= poor).all()) and not (members["censoring"] == ">").any()
        rows.append({"framework_group": group, "n_members": len(members),
                     "label": "fast" if is_fast else "poor" if is_poor else None,
                     "sigma_max": float(members["sigma"].max()),
                     "family": family(members["family"]),
                     "fast_family": family(members.loc[members["sigma"] >= fast, "family"]),
                     "example": members.sort_values("sigma", ascending=False)["formula"].iloc[0]})
    return pd.DataFrame(rows)


def run_bench_frameworks(args, config: dict) -> None:
    """Frameworks of every relaxed structure and their OBELiX labels."""
    root = campaign_dir(config, args.out)
    spec = config["benchmark"]["labels"]
    obelix = pd.read_parquet(root / "data" / "obelix.parquet")
    entries, _ = obelix_frameworks(obelix)
    groups = label_obelix_groups(entries, spec["obelix_fast_s_cm"], spec["obelix_poor_s_cm"])
    write_parquet(entries.drop(columns="cif"), root / "bench" / "obelix_frameworks.parquet")
    write_parquet(groups, root / "bench" / "obelix_groups.parquet")
    logger.info("OBELiX: %d framework groups (%s)", len(groups),
                groups["label"].value_counts(dropna=False).to_dict())
    references: dict[str, list[tuple[str, str]]] = {}
    for row in entries.itertuples(index=False):
        references.setdefault(row.framework_formula, []).append((row.ID, row.framework_cif))
    tolerances = config["benchmark"].get("framework_matcher", {})
    relaxed = relaxed_structures(root)
    tasks = [(int(row.index), row.relaxed_cif, references, tolerances)
             for row in relaxed.itertuples(index=False)]
    run_pool(tasks, _framework_one, root / "bench" / "frameworks.jsonl", args.workers)
    frames = pd.DataFrame(read_jsonl(root / "bench" / "frameworks.jsonl"))
    group_of = dict(zip(entries["ID"], entries["framework_group"]))
    label_of = dict(zip(groups["framework_group"], groups["label"]))
    family_of = dict(zip(groups["framework_group"], groups["fast_family"].fillna(groups["family"])))

    def summarise(matches):
        matched = sorted({group_of[key] for key in matches or []})
        labels = {label_of[g] for g in matched}
        label = "fast" if "fast" in labels else "poor" if labels == {"poor"} else None
        families = sorted({family_of[g] for g in matched if label_of[g] == label and family_of[g]})
        return matched, label, families[0] if families else None

    summary = frames["obelix_matches"].map(summarise)
    frames["obelix_groups"] = [s[0] for s in summary]
    frames["obelix_label"] = [s[1] for s in summary]
    frames["obelix_family"] = [s[2] for s in summary]
    write_parquet(frames, root / "bench" / "frameworks.parquet")
    lemat = frames.merge(relaxed[["index", "source"]], on="index")
    lemat = lemat[lemat["source"] == "lemat"]
    print(f"{len(frames)} frameworks; palette compounds labelled by OBELiX: "
          f"{lemat['obelix_label'].value_counts().to_dict()}")


# --------------------------------------------------------------------------- #
# bench-describe
# --------------------------------------------------------------------------- #
def _describe_one(task: tuple) -> dict:
    from pymatgen.core import Structure

    from wyckoff_transformer.transport.descriptors import (
        bvse_barriers,
        coarse_descriptors,
    )

    index, path = task
    structure = Structure.from_file(path)
    started = time.time()
    row = {"index": index, **coarse_descriptors(structure)}
    row["seconds_coarse"] = time.time() - started
    started = time.time()
    row.update(bvse_barriers(structure))
    row["seconds_bvse"] = time.time() - started
    return row


def run_bench_describe(args, config: dict) -> None:
    """Coarse descriptors and BVSE barriers of every relaxed structure (CPU)."""
    root = campaign_dir(config, args.out)
    relaxed = relaxed_structures(root)
    # Largest first: the BVSE map grows with the cell.
    relaxed = relaxed.sort_values("n_atoms", ascending=False)
    tasks = [(int(row.index), row.relaxed_cif) for row in relaxed.itertuples(index=False)]
    run_pool(tasks, _describe_one, root / "bench" / "describe.jsonl", args.workers)
    frame = pd.DataFrame(read_jsonl(root / "bench" / "describe.jsonl"))
    write_parquet(frame, root / "bench" / "describe.parquet")
    print(f"{len(frame)} described -> {root / 'bench' / 'describe.parquet'}")


# --------------------------------------------------------------------------- #
# bench-pes
# --------------------------------------------------------------------------- #
_PES_MODEL = None


def _pes_init(slots, counter, max_atoms_per_batch: int = 8000) -> None:
    global _PES_MODEL
    _cpu_init()
    with counter.get_lock():
        device = slots[counter.value % len(slots)]
        counter.value += 1
    from wyckoff_transformer.transport.pes import OrbProbeEnergy

    _PES_MODEL = OrbProbeEnergy(device=device, max_atoms_per_batch=max_atoms_per_batch)


def _pes_one(task: tuple) -> dict:
    from pymatgen.core import Structure

    from wyckoff_transformer.transport.pes import (
        empty_framework_scan,
        frozen_framework_scan,
    )

    index, path, kinds, spec = task
    structure = Structure.from_file(path)
    row = {"index": index, "device": str(_PES_MODEL.device)}
    if "frozen" in kinds:
        result = frozen_framework_scan(
            structure, _PES_MODEL, step=spec["grid_step"], r_min=spec["r_min"],
            threshold=spec["fv_threshold_ev"], max_orbits=spec["max_orbits"], seed=spec["seed"],
            xi_kwargs={"connected_threshold": spec["connected_log10_threshold"],
                       "disconnected_threshold": spec["disconnected_log10_threshold"],
                       "slope": spec["sigmoid_slope"]})
        row.update({"xi": result.xi, "fv_connected": result.fv_connected,
                    "fv_disconnected": result.fv_disconnected, "mpe": result.mpe,
                    "frozen_evaluations": result.n_evaluations,
                    "n_orbits": result.n_orbits, "n_scanned": result.n_scanned,
                    "frozen_seconds": result.seconds, "grid": list(result.grid)})
    if "empty" in kinds:
        result = empty_framework_scan(structure, _PES_MODEL, step=spec["grid_step"],
                                      r_min=spec["r_min"])
        row.update({"empty_E_1D": result.E_1D, "empty_E_2D": result.E_2D,
                    "empty_E_3D": result.E_3D, "empty_evaluations": result.n_evaluations,
                    "empty_seconds": result.seconds})
    return row


def pes_selection(root: Path, config: dict) -> pd.DataFrame:
    """Where the frozen-framework scan runs: labelled, best BVSE, and a random control.

    The control is drawn from all 12,383 palette compounds, labelled and best-BVSE ones
    included, so that it estimates the whole population's Xi distribution: that is what
    a fast member's percentile among the 12,383 is read from.
    """
    spec = config["benchmark"]["xi"]
    relaxed = relaxed_structures(root)
    frames = pd.read_parquet(root / "bench" / "frameworks.parquet")[
        ["index", "obelix_label", "obelix_groups"]]
    describe = pd.read_parquet(root / "bench" / "describe.parquet")[["index", "bvse_E_1D"]]
    table = relaxed.merge(frames, on="index", how="left").merge(describe, on="index", how="left")
    # Labelled palette compounds, at most max_labelled_per_group per OBELiX framework
    # group: the evaluation weighs groups equally, and the layered group alone labels
    # 550 compounds.
    table["obelix_group"] = table["obelix_groups"].map(
        lambda g: int(g[0]) if isinstance(g, (list, np.ndarray)) and len(g) else -1)
    matched = table[(table["source"] == "lemat") & table["obelix_label"].notna()]
    kept = matched.groupby("obelix_group", group_keys=False).apply(
        lambda part: part.sample(n=min(len(part), spec["max_labelled_per_group"]),
                                 random_state=spec["seed"]))
    labelled = (table["source"] != "lemat") | table["index"].isin(kept["index"])
    unlabelled = table[(table["source"] == "lemat") & table["obelix_label"].isna()]
    best = unlabelled.sort_values("bvse_E_1D").head(spec["n_top_bvse"])
    palette = table[table["source"] == "lemat"]
    control = palette.sample(n=min(spec["n_random_control"], len(palette)),
                             random_state=spec["seed"])
    table["pes_reason"] = None
    table.loc[table["index"].isin(best["index"]), "pes_reason"] = "top_bvse"
    table.loc[table["index"].isin(control["index"]), "pes_reason"] = "control"
    table.loc[labelled, "pes_reason"] = "labelled"
    table["in_control"] = table["index"].isin(control["index"])
    return table[table["pes_reason"].notna()]


def run_bench_pes(args, config: dict) -> None:
    """PES scans with ORB on the GPUs, frozen or empty framework, on the selection.

    ``--pes-all`` scans every relaxed structure instead; at about 30k atom-evaluations
    per second per worker the empty framework of all 12.6k costs some 6e9 of them.
    """
    import multiprocessing

    root = campaign_dir(config, args.out)
    spec = config["benchmark"]["xi"]
    kind = args.pes_kinds
    if kind not in ("frozen", "empty"):
        raise ValueError("--pes-kinds is frozen or empty")
    if args.pes_all:
        selection = relaxed_structures(root)
    else:
        selection = pes_selection(root, config)
        write_parquet(selection[["index", "pes_reason", "in_control"]],
                      root / "bench" / "pes_selection.parquet")
    if args.limit is not None:
        selection = selection.sample(n=min(args.limit, len(selection)), random_state=0)
    selection = selection.sort_values("n_atoms")
    tasks = [(int(row.index), row.relaxed_cif, (kind,), spec)
             for row in selection.itertuples(index=False)]
    devices = (args.devices or config["benchmark"]["relax"]["devices"]).split(",")
    slots = [d for d in devices for _ in range(args.workers_per_device or 2)]
    counter = multiprocessing.get_context("spawn").Value("i", 0)
    out = root / "bench" / f"pes_{kind}.jsonl"
    run_pool(tasks, _pes_one, out, len(slots), initializer=_pes_init,
             initargs=(slots, counter, spec["max_atoms_per_batch"]), every=50)
    frame = pd.DataFrame(read_jsonl(out))
    if "error" in frame:
        frame = frame[frame["error"].isna()].drop(columns="error")
    frame = frame.drop_duplicates("index", keep="last")
    write_parquet(frame, out.with_suffix(".parquet"))
    print(f"{len(frame)} {kind}-framework scans -> {out.with_suffix('.parquet')}")


# --------------------------------------------------------------------------- #
# matcher-calibrate (gate G1b)
# --------------------------------------------------------------------------- #
PROTOCOL = Path(sys.executable).parent / "wyformer-protocol"
PROTOCOL_STAGES = ("screen", "generate", "prescreen", "relax")


def protocol_command(run_dir: Path, stage: str, config: dict, devices: str | None = None,
                     workers_per_device: int | None = None) -> list[str]:
    """The families campaign's protocol arm: wide NEP89 prescreen, then ORB."""
    spec = config["protocol"]
    command = [str(PROTOCOL), str(run_dir / "wyckoff_genes.json.gz"),
               "--output-dir", str(run_dir), "--stage", stage, "--mlip", spec["mlip"],
               "--trial-multiplier", str(spec["trial_multiplier"]),
               "--pyxtal-cores", str(spec["pyxtal_cores"]),
               "--pyxtal-timeout", str(spec["pyxtal_timeout"]), "--relax-from", "prescreen"]
    if stage == "relax":
        command += ["--devices", devices or spec["devices"], "--workers-per-device",
                    str(workers_per_device or spec["workers_per_device"])]
    else:
        command += ["--cores", str(spec["prescreen_cores"])]
    return command


def run_protocol(run_dir: Path, config: dict, devices: str | None = None,
                 workers_per_device: int | None = None) -> None:
    """Every protocol stage of *run_dir* not yet marked done, in order."""
    import subprocess

    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    for stage in PROTOCOL_STAGES:
        marker = run_dir / "logs" / f"{stage}.done"
        if marker.exists():
            continue
        command = protocol_command(run_dir, stage, config, devices, workers_per_device)
        logger.info("[%s] %s", run_dir.name, stage)
        started = time.time()
        with open(run_dir / "logs" / f"{stage}.log", "a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(command)}\n")
            log.flush()
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        if code != 0:
            raise SystemExit(f"[{run_dir.name}] {stage} failed ({code}); see {log.name}")
        marker.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {time.time() - started:.0f} s\n")


def calibration_controls(root: Path, config: dict) -> pd.DataFrame:
    """One OBELiX-labelled palette compound per framework group, fast groups first.

    The lowest-e_hull member of each group, at most ``max_atoms`` in its conventional
    cell; groups are taken round-robin over OBELiX families so that no family fills the
    list.
    """
    spec = config["benchmark"]["matcher_calibration"]
    frames = pd.read_parquet(root / "bench" / "frameworks.parquet")
    bench = pd.read_parquet(root / "bench" / "structures.parquet")
    table = bench.merge(frames[["index", "obelix_groups", "obelix_label", "obelix_family"]],
                        on="index")
    table = table[(table["source"] == "lemat") & table["obelix_label"].notna()
                  & (table["atoms_conventional"] <= spec["max_atoms"])
                  & (table["spacegroup_number"] >= spec.get("min_spacegroup", 1))
                  & (table["free_coordinates"] <= spec.get("max_free_coordinates", 10**6))
                  ].copy()
    table["obelix_group"] = table["obelix_groups"].map(lambda g: int(g[0]))
    best = table.sort_values("energy_above_hull").drop_duplicates("obelix_group")
    best["fast"] = best["obelix_label"] == "fast"
    best = best.sort_values(["fast", "energy_above_hull"], ascending=[False, True])
    chosen, seen_families = [], {}
    pools = [part for _, part in best.groupby("fast", sort=False)]
    for pool in sorted(pools, key=lambda p: not p["fast"].iloc[0]):
        while len(chosen) < spec["n_controls"] and len(pool):
            counts = pool["obelix_family"].map(lambda f: seen_families.get(f, 0))
            pick = pool.loc[counts.idxmin()]
            chosen.append(pick)
            seen_families[pick["obelix_family"]] = seen_families.get(pick["obelix_family"], 0) + 1
            pool = pool.drop(pick.name)
    return pd.DataFrame(chosen).reset_index(drop=True)


def _g1b_one(task: tuple) -> dict:
    """One control track: rebuilt? and the OBELiX groups its framework matches per tolerance.

    Each framework fit is limited to two minutes (StructureMatcher can enumerate
    supercells for much longer); a fit that runs out counts as no match and is logged.
    """
    import signal

    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    from wyckoff_transformer.transport.frameworks import default_matcher, framework

    index, control, group, track, path, bench_path, references, tolerances = task

    def alarm(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, alarm)
    structure = Structure.from_file(path)
    result = {}
    try:
        signal.alarm(120)
        result[f"{track}_rebuilt"] = StructureMatcher().fit(structure,
                                                            Structure.from_file(bench_path))
    except TimeoutError:
        result[f"{track}_rebuilt"] = None
    finally:
        signal.alarm(0)
    frame = framework(structure)
    formula = frame.composition.reduced_formula
    candidates = [(g, Structure.from_str(cif, fmt="cif")) for g, cif in references]
    candidates = [(g, ref) for g, ref in candidates if ref.composition.reduced_formula == formula]
    timeouts = 0
    for ltol, stol, angle in tolerances:
        matcher = default_matcher(ltol=ltol, stol=stol, angle_tol=angle)
        groups = set()
        for g, ref in candidates:
            if g in groups:
                continue
            try:
                signal.alarm(120)
                if matcher.fit(ref, frame):
                    groups.add(g)
            except TimeoutError:
                timeouts += 1
            finally:
                signal.alarm(0)
        key = f"{track}_{ltol}_{stol}_{angle}"
        result[f"{key}_matched"] = group in groups
        result[f"{key}_other_groups"] = len(groups - {group})
    result[f"{track}_fit_timeouts"] = timeouts
    return {"index": index, "control": control, "group": group, "result": result}


def run_matcher_calibrate(args, config: dict) -> None:
    """Gate G1b: relax known-SE genes from their genes; does the matcher find their family?

    Every control is judged at each tolerance of ``matcher_calibration.tolerances``, on
    both protocol tracks.  Also recorded: whether the protocol rebuilt the compound's
    own structure (a full-structure match with its benchmark relaxation), since no
    framework tolerance can rescue a control relaxed into another structure, and which
    other OBELiX groups the framework matches, the price of a looser tolerance.
    """

    from wyckoff_transformer.evaluation.DiffCSP_to_sites import record_to_pyxtal

    root = campaign_dir(config, args.out)
    spec = config["benchmark"]["matcher_calibration"]
    base = root / spec.get("run", "g1b")
    run_dir = base / "protocol"
    run_dir.mkdir(parents=True, exist_ok=True)
    controls_path = base / "controls.parquet"
    if not controls_path.exists():
        controls = calibration_controls(root, config)
        ids = set(controls["immutable_id"])
        columns = ["elements", "wyckoff_letters", "multiplicity", "spacegroup_number"]
        genes = {}
        for split in SPLITS:
            frame = load_split(dataset_cache_dir(config["data"]["lemat"]["cache"]), split,
                               columns=columns)
            for key in ids & set(frame.index):
                genes[key] = record_to_pyxtal(frame.loc[key].to_dict())
        controls["gene"] = controls["immutable_id"].map(lambda key: json.dumps(genes[key]))
        with gzip.open(run_dir / "wyckoff_genes.json.gz", "wt", encoding="utf-8") as handle:
            json.dump([json.loads(g) for g in controls["gene"]], handle)
        write_parquet(controls.drop(columns=["obelix_groups"]), controls_path)
    controls = pd.read_parquet(controls_path)
    print(controls[["immutable_id", "reduced_formula", "obelix_group", "obelix_label",
                    "obelix_family", "spacegroup_number", "free_coordinates",
                    "atoms_conventional"]].to_string())
    run_protocol(run_dir, config, args.devices, args.workers_per_device)

    entries = pd.read_parquet(root / "bench" / "obelix_frameworks.parquet")
    references = list(zip(entries["framework_group"].astype(int), entries["framework_cif"]))
    relaxed = relaxed_structures(root).set_index("immutable_id")
    tasks = []
    for i, control in controls.iterrows():
        for track, directory in (("free", "cifs"), ("fixed", "cifs_fixed_symmetry")):
            path = run_dir / directory / f"{i}.cif"
            if path.exists():
                tasks.append((len(tasks), i, int(control["obelix_group"]), track, str(path),
                              relaxed.loc[control["immutable_id"], "relaxed_cif"],
                              references, spec["tolerances"]))
    out = base / "matches.jsonl"
    run_pool(tasks, _g1b_one, out, args.workers, label="control tracks", every=5)
    rows = {}
    for record in read_jsonl(out):
        if "error" in record:
            continue
        row = rows.setdefault(record["control"], {
            "protocol_index": record["control"],
            "formula": controls.loc[record["control"], "reduced_formula"],
            "group": record["group"]})
        row.update(record["result"])
    rows = [rows[key] for key in sorted(rows)]
    result = pd.DataFrame(rows)
    result.to_csv(base / "matches.csv", index=False)
    need = config["gates"]["g1b_min_matched"]
    for track in ("free", "fixed"):
        rebuilt = int(result[f"{track}_rebuilt"].fillna(False).astype(bool).sum())
        print(f"\n{track} track: the protocol rebuilt {rebuilt}/{len(result)} controls' "
              "own structures")
        for ltol, stol, angle in spec["tolerances"]:
            key = f"{track}_{ltol}_{stol}_{angle}"
            matched = int(result[f"{key}_matched"].fillna(False).astype(bool).sum())
            other = int((result[f"{key}_other_groups"].fillna(0) > 0).sum())
            print(f"  ltol {ltol} stol {stol} angle {angle}: {matched}/{len(result)} matched "
                  f"to their family ({'PASS' if matched >= need else 'FAIL'}, needs {need}); "
                  f"{other} also match another group")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(result.to_string())


# --------------------------------------------------------------------------- #
# md-calibrate (gate G1c)
# --------------------------------------------------------------------------- #
#: The plan's MD references: formula -> (label, space group of the phase meant).  With
#: no space group, or none in LeMat, the lowest-e_hull polymorph stands in.
MD_REFERENCES = {
    "Li6PS5Cl": ("fast", 216), "Li10GeP2S12": ("fast", 137), "Li7P3S11": ("fast", 2),
    "Li3YCl6": ("fast", 164), "Li3InCl6": ("fast", 12), "Li3PS4": ("fast", 62),
    "LiTaOCl4": ("fast", None), "Li3N": ("fast", 191), "LiBH4": ("fast", 186),
    "LiF": ("poor", 225), "LiCl": ("poor", 225), "Li2O": ("poor", 225), "Li3PO4": ("poor", 62),
    "Li4SiO4": ("poor", 11), "LiAlO2": ("poor", 92), "Li2SO4": ("poor", 14),
    "LiAlCl4": ("poor", 14), "Li7La3Zr2O12": ("poor", 142), "LiTi2(PO4)3": ("poor", 167),
}
_MD_CALCULATOR = None


def md_references(root: Path) -> pd.DataFrame:
    from pymatgen.core import Composition

    relaxed = relaxed_structures(root)
    lemat = relaxed[relaxed["source"] == "lemat"]
    rows = []
    for formula, (label, spacegroup) in MD_REFERENCES.items():
        reduced = Composition(formula).reduced_formula
        found = lemat[lemat["reduced_formula"] == reduced].sort_values("energy_above_hull")
        if found.empty:
            logger.warning("MD reference %s is not among the relaxed palette compounds", formula)
            continue
        exact = found[found["spacegroup_number"] == spacegroup] if spacegroup else found.iloc[:0]
        pick = (exact if len(exact) else found).iloc[0]
        rows.append({"reference": formula, "label": label, "index": int(pick["index"]),
                     "immutable_id": pick["immutable_id"],
                     "spacegroup_number": int(pick["spacegroup_number"]),
                     "spacegroup_wanted": spacegroup, "e_hull": pick["energy_above_hull"],
                     "relaxed_cif": pick["relaxed_cif"]})
    return pd.DataFrame(rows)


def _md_init(slots, counter) -> None:
    global _MD_CALCULATOR
    _cpu_init()
    with counter.get_lock():
        device = slots[counter.value % len(slots)]
        counter.value += 1
    from wyckoff_transformer.evaluation.hull_mlips import build_hull_calculator

    _MD_CALCULATOR = build_hull_calculator("orb_conserv_inf", device=device)


def _md_one(task: tuple) -> dict:
    from ase.io import read as ase_read

    from wyckoff_transformer.transport.md import analyse, md_supercell, run_nvt

    key, path, temperature, spec, out_dir = task
    atoms = ase_read(path)
    supercell, repeats = md_supercell(atoms, spec["min_atoms"], spec["min_width"])
    trajectory = run_nvt(supercell, _MD_CALCULATOR, temperature,
                         thermalise_ps=spec["thermalise_ps"],
                         production_ps=spec["production_ps"], sample_fs=spec["sample_fs"],
                         seed=spec["seed"])
    np.savez_compressed(Path(out_dir) / f"{key}.npz", positions=trajectory.positions.astype(
        np.float32), symbols=np.array(trajectory.symbols), dt_ps=trajectory.dt_ps,
        volume=trajectory.volume, temperature=temperature)
    return {"index": key, "supercell_matrix": np.asarray(repeats).tolist(),
            **analyse(trajectory)}


def md_runs(out_dir: Path) -> pd.DataFrame:
    """Every finished run of ``out_dir/md.jsonl``, re-analysed from its saved trajectory.

    The analysis can then change without rerunning the MD; the table is also written
    to ``md_analysis.csv``.
    """
    from wyckoff_transformer.transport.md import Trajectory, analyse

    rows = []
    for row in read_jsonl(out_dir / "md.jsonl"):
        path = out_dir / "trajectories" / f"{row['index']}.npz"
        if "error" in row or not path.exists():
            continue
        saved = np.load(path)
        trajectory = Trajectory(symbols=saved["symbols"].tolist(),
                                positions=saved["positions"].astype(np.float64),
                                dt_ps=float(saved["dt_ps"]),
                                temperature=float(saved["temperature"]),
                                volume=float(saved["volume"]), seconds=row["md_seconds"],
                                mean_temperature=row["mean_temperature"])
        rows.append({"index": row["index"], **analyse(trajectory)})
    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "md_analysis.csv", index=False)
    return runs


def md_summary(runs: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """One row per run key (``<key>_<temperature>``) with Phase 7's filters and tiers.

    The framework must not melt at the hot temperature, Li must move at the warm one,
    and sigma_300K - 1 s.e.m., extrapolated from the warm run with a pessimistic Ea, is
    tiered: >= 1e-4 S/cm electrolyte, >= 1e-6 coating.
    """
    from wyckoff_transformer.transport.md import extrapolate

    runs = runs.assign(key=runs["index"].str.rsplit("_", n=1).str[0])
    warm_k, hot_k = min(spec["temperatures"]), max(spec["temperatures"])
    hot = runs[runs["temperature"] == hot_k].set_index("key")
    warm = runs[runs["temperature"] == warm_k].set_index("key")
    table = warm[["sigma", "sigma_sem", "li_msd_max"]].add_suffix(f"_{warm_k}K").join(
        hot[["sigma", "framework_D_slowest"]].add_suffix(f"_{hot_k}K"), how="outer")

    def to_300(sigma):
        return extrapolate(sigma, warm_k, 300.0, spec["activation_ev"]) \
            if np.isfinite(sigma) else np.nan

    table["sigma_300K"] = table[f"sigma_{warm_k}K"].map(to_300)
    table["sigma_300K_low"] = (table[f"sigma_{warm_k}K"]
                               - table[f"sigma_sem_{warm_k}K"].fillna(0)).map(to_300)
    table["framework_stable"] = table[f"framework_D_slowest_{hot_k}K"] * 6 * \
        spec["production_ps"] <= spec["framework_msd_max"]
    table["li_mobile"] = table[f"li_msd_max_{warm_k}K"] > spec["li_msd_min"]
    passes = table["framework_stable"] & table["li_mobile"]
    table["tier"] = np.select([passes & (table["sigma_300K_low"] >= 1e-4),
                               passes & (table["sigma_300K_low"] >= 1e-6)],
                              ["electrolyte", "coating"], default="")
    return table


def run_md_calibrate(args, config: dict) -> None:
    """Gate G1c: does ORB MD separate fast from poor references (AUROC >= 0.8)?"""
    import multiprocessing

    from sklearn.metrics import roc_auc_score

    root = campaign_dir(config, args.out)
    spec = config["md"]
    references = md_references(root)
    out_dir = root / "g1c"
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    write_parquet(references, out_dir / "references.parquet")
    print(references[["reference", "label", "immutable_id", "spacegroup_number",
                      "spacegroup_wanted", "e_hull"]].to_string())
    tasks = [(f"{row.reference}_{temperature}", row.relaxed_cif, temperature, spec,
              str(out_dir / "trajectories"))
             for temperature in spec["temperatures"] for row in references.itertuples()]
    devices = (args.devices or config["benchmark"]["relax"]["devices"]).split(",")
    slots = [d for d in devices for _ in range(args.workers_per_device or 3)]
    counter = multiprocessing.get_context("spawn").Value("i", 0)
    out = out_dir / "md.jsonl"
    run_pool(tasks, _md_one, out, len(slots), initializer=_md_init, initargs=(slots, counter),
             label="MD runs", every=1)
    table = references.set_index("reference")[["label"]].join(
        md_summary(md_runs(out_dir), spec))
    table.to_csv(out_dir / "references.csv")
    y = (table["label"] == "fast").astype(int)
    auroc = roc_auc_score(y, table["sigma_500K"].fillna(0))
    print(table.round(5).to_string())
    print(f"\nG1c: AUROC of sigma(500 K) fast vs poor = {auroc:.3f} (pass needs "
          f"{config['gates']['g1c_min_auroc']}) -> "
          f"{'PASS' if auroc >= config['gates']['g1c_min_auroc'] else 'FAIL'}")


# --------------------------------------------------------------------------- #
# xi-check
# --------------------------------------------------------------------------- #
#: Xi of Materials Project structures as printed in Maevskiy et al. 2025 (Table I, Table
#: II and Fig. 2), computed there with M3GNet on the MP structures as they are.  The
#: constructor.app page with all 5,997 values is a JavaScript application this script
#: cannot read.
MAEVSKIY_XI = {
    "mp-1211100": 0.9924, "mp-1211296": 0.9898, "mp-570935": 0.9812, "mp-721239": 0.9746,
    "mp-721252": 0.9729, "mp-2530": 0.9712, "mp-721253": 0.9712, "mp-705516": 0.9661,
    "mp-644223": 0.9647, "mp-696127": 0.9645, "mp-641703": 0.9606, "mp-696128": 0.9379,
    "mp-985583": 0.958, "mp-1001069": 0.956, "mp-1097034": 0.951, "mp-1040451": 0.949,
    "mp-1185263": 0.942, "mp-1097036": 0.931, "mp-1222398": 0.896, "mp-755463": 0.887,
    "mp-753720": 0.858, "mp-1222392": 0.82, "mp-34038": 0.814, "mp-680395": 0.783,
    "mp-775806": 0.763, "mp-28336": 0.721, "mp-1222582": 0.717, "mp-753429": 0.716,
    "mp-985582": 0.703, "mp-1177520": 0.658, "mp-1211362": 0.647, "mp-1211446": 0.606,
    "mp-1222482": 0.539, "mp-1211324": 0.538, "mp-1195718": 0.525, "mp-950995": 0.521,
    "mp-1211176": 0.483, "mp-1191476": 0.35, "mp-768738": 0.00,
}


def _xi_check_one(task: tuple) -> dict:
    from pymatgen.core import Structure

    from wyckoff_transformer.transport.pes import frozen_framework_scan

    index, key, cif, spec = task
    structure = Structure.from_str(cif, fmt="cif")
    result = frozen_framework_scan(structure, _PES_MODEL, step=spec["grid_step"],
                                   r_min=spec["r_min"], threshold=spec["fv_threshold_ev"],
                                   max_orbits=spec["max_orbits"], seed=spec["seed"])
    return {"index": index, "key": key, "formula": structure.composition.reduced_formula,
            "n_atoms": len(structure), "xi": result.xi, "fv_connected": result.fv_connected,
            "fv_disconnected": result.fv_disconnected, "mpe": result.mpe,
            "n_orbits": result.n_orbits, "n_scanned": result.n_scanned,
            "seconds": result.seconds}


def run_xi_check(args, config: dict) -> None:
    """ORB Xi on the MP structures of Maevskiy et al.'s tables, as MP has them.

    LeMat-Bulk's CSV holds MP's geometry under the MP id.  The structures are not
    relaxed: the paper ran M3GNet on MP's structures as they are.
    """
    import multiprocessing

    root = campaign_dir(config, args.out)
    spec = config["benchmark"]["xi"]
    cifs = lemat_cifs(data_path(config["data"]["lemat"]["structures"]), set(MAEVSKIY_XI))
    from pymatgen.core import Structure

    sizes = {key: len(Structure.from_str(cif, fmt="cif")) for key, (_, cif) in cifs.items()}
    # Smallest first: several of the highest published values are cells of a few atoms.
    tasks = [(i, key, cifs[key][1], spec)
             for i, key in enumerate(sorted(cifs, key=lambda k: (sizes[k], k)))]
    missing = sorted(set(MAEVSKIY_XI) - set(cifs))
    logger.info("xi-check: %d MP structures (missing from LeMat: %s)", len(cifs), missing)
    devices = (args.devices or config["benchmark"]["relax"]["devices"]).split(",")
    slots = [d for d in devices for _ in range(args.workers_per_device or 2)]
    counter = multiprocessing.get_context("spawn").Value("i", 0)
    out = root / "checks" / "xi_check.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    run_pool(tasks, _xi_check_one, out, len(slots), initializer=_pes_init,
             initargs=(slots, counter, spec["max_atoms_per_batch"]), every=20)
    frame = pd.DataFrame(read_jsonl(out))
    frame["xi_published"] = frame["key"].map(MAEVSKIY_XI)
    write_parquet(frame, out.with_suffix(".parquet"))
    mp = frame[frame["xi_published"].notna()].sort_values("xi_published", ascending=False)
    with pd.option_context("display.width", 200):
        print(mp[["key", "formula", "n_atoms", "xi_published", "xi", "fv_connected",
                  "fv_disconnected", "mpe"]].round(4).to_string(index=False))


# --------------------------------------------------------------------------- #
# bvse-check
# --------------------------------------------------------------------------- #
def _bvse_check_one(task: tuple) -> dict:
    from pymatgen.core import Structure

    from wyckoff_transformer.transport.descriptors import bvse_barriers

    index, key, cif = task
    return {"index": index, "key": key, **bvse_barriers(Structure.from_str(cif, fmt="cif"))}


def run_bvse_check(args, config: dict) -> None:
    """Our bvlain barriers on MP's geometry against LiTraj's published BVEL13k values."""
    import zipfile

    from scipy.stats import spearmanr

    root = campaign_dir(config, args.out)
    with zipfile.ZipFile(data_path(config["data"]["bvel13k"]["zip"])) as archive, \
            archive.open("BVEL13k/BVEL13k_index.csv") as handle:
        published = pd.read_csv(handle)
    bench = pd.read_parquet(root / "bench" / "structures.parquet")
    overlap = bench.merge(published, left_on="immutable_id", right_on="material_id")
    overlap = overlap.sample(n=min(args.limit or 500, len(overlap)), random_state=0)
    logger.info("bvse-check: %d of BVEL13k's MP ids are in the benchmark set; checking %d",
                len(bench.merge(published, left_on="immutable_id", right_on="material_id")),
                len(overlap))
    tasks = [(int(row.index), row.immutable_id, row.cif) for row in overlap.itertuples()]
    out = root / "checks" / "bvse_check.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    run_pool(tasks, _bvse_check_one, out, args.workers)
    ours = pd.DataFrame(read_jsonl(out)).merge(published, left_on="key", right_on="material_id")
    ours.to_csv(out.with_suffix(".csv"), index=False)
    for dim in ("1D", "2D", "3D"):
        both = ours[[f"bvse_E_{dim}", f"E_{dim}"]].replace(np.inf, np.nan).dropna()
        rho = spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic
        mae = (both.iloc[:, 0] - both.iloc[:, 1]).abs().median()
        print(f"E_{dim}: n={len(both)}, Spearman {rho:.3f}, median |ours - BVEL13k| "
              f"{mae:.3f} eV")


# --------------------------------------------------------------------------- #
# mine-gap (Phase 2)
# --------------------------------------------------------------------------- #
def run_mine_gap(args, config: dict) -> None:
    """Band gap of every palette compound: MP's GGA gap where it has one, else 3t16iys0.

    MP's 2026.09.28 GGA/GGA+U gaps (`mp_20260928_gga_gap`) still use the old ids LeMat
    carries for 2,518 of the palette's 2,624 MP entries.  Every compound also gets the
    gene-level prediction, so the two can be compared where both exist.
    """
    import torch

    from wyckoff_transformer.cli.gene_screen import predict_gene_scalars
    from wyckoff_transformer.evaluation.DiffCSP_to_sites import record_to_pyxtal

    sys.path.insert(0, str(REPO / "scripts"))
    from solid_electrolyte_campaign import load_band_gap_regressor

    root = campaign_dir(config, args.out)
    spec = config["mining"]["band_gap"]
    bench = pd.read_parquet(root / "bench" / "structures.parquet")
    palette = bench[bench["source"] == "lemat"][["index", "immutable_id"]].copy()
    ids = set(palette["immutable_id"])
    mp = []
    for split in SPLITS:
        for chunk in pd.read_csv(data_path(spec["mp_dataset"]) / f"{split}.csv.gz",
                                 usecols=["material_id", "band_gap"], chunksize=50_000):
            mp.append(chunk[chunk["material_id"].isin(ids)])
    mp = pd.concat(mp).drop_duplicates("material_id").rename(
        columns={"material_id": "immutable_id", "band_gap": "mp_gga_gap"})
    palette = palette.merge(mp, on="immutable_id", how="left")
    genes = {}
    columns = ["elements", "wyckoff_letters", "multiplicity", "spacegroup_number"]
    for split in SPLITS:
        frame = load_split(dataset_cache_dir(config["data"]["lemat"]["cache"]), split,
                           columns=columns)
        for key in ids & set(frame.index):
            genes[key] = record_to_pyxtal(frame.loc[key].to_dict())
    regressor = load_band_gap_regressor(spec["wandb_run"], torch.device(args.device))
    gene_list = [genes[key] for key in palette["immutable_id"]]
    predictions = []
    for start in range(0, len(gene_list), 5000):
        with torch.no_grad():
            predictions.append(predict_gene_scalars(
                gene_list[start:start + 5000], regressor,
                augmentation_samples=spec["augmentation_samples"])["prediction"].to_numpy())
    palette["band_gap_pred"] = np.concatenate(predictions)
    palette["band_gap"] = palette["mp_gga_gap"].fillna(palette["band_gap_pred"])
    palette["band_gap_source"] = np.where(palette["mp_gga_gap"].notna(), "mp_gga",
                                          spec["wandb_run"])
    write_parquet(palette, root / "mine" / "gaps.parquet")
    both = palette.dropna(subset=["mp_gga_gap", "band_gap_pred"])
    mae = (both["mp_gga_gap"] - both["band_gap_pred"]).abs().mean()
    print(f"{len(palette)} gaps: {palette['mp_gga_gap'].notna().sum()} from MP, MAE of the "
          f"prediction against them {mae:.3f} eV; {(palette['band_gap'] >= spec['min_ev']).sum()}"
          f" at >= {spec['min_ev']} eV")


# --------------------------------------------------------------------------- #
# coordination-reference, mine-screen, mine-pes, mine-rank (Phase 2)
# --------------------------------------------------------------------------- #
MOTIF_SCRIPT = REPO / "scripts" / "screen_coordination_motifs.py"


def run_coordination_reference(args, config: dict) -> None:
    """LeMat's cation-environment frequency table, for the coordination check.

    ``scripts/screen_coordination_motifs.py`` measured it on aspire2a for
    lemat_bulk_fmax1_stress, and its outputs are not on hera. This writes that script's
    shards from the campaign's cache (e_hull within the cut, an anion the analysis
    knows) and lemat_pbe.csv.gz's CIFs, then runs its ``run`` and ``report`` stages.
    """
    import subprocess

    from wyckoff_transformer.evaluation.coordination_motifs import ANIONS

    spec = config["mining"]["coordination"]
    out = campaign_dir(config, args.out) / "mine" / "coordination"
    shards = out / "shards"
    if not (shards / "_DONE").exists():
        shards.mkdir(parents=True, exist_ok=True)
        keep = []
        for split in SPLITS:
            frame = load_split(dataset_cache_dir(config["data"]["lemat"]["cache"]), split,
                               columns=["composition", "energy_above_hull"])
            frame = frame[(frame["energy_above_hull"] <= spec["reference_max_e_hull"])
                          & frame["composition"].map(lambda c: bool(element_set(c) & ANIONS))]
            keep.append(frame[["energy_above_hull"]].assign(split=split))
        keep = pd.concat(keep)
        keep = keep[~keep.index.duplicated()]
        logger.info("coordination reference: %d structures", len(keep))
        n_shard, buffer = 0, []
        with gzip.open(data_path(config["data"]["lemat"]["structures"]), "rt",
                       encoding="utf-8") as handle:
            for chunk in pd.read_csv(handle, usecols=["immutable_id", "cif"], chunksize=100_000):
                buffer.append(chunk[chunk["immutable_id"].isin(keep.index)])
                while sum(map(len, buffer)) >= 20_000:
                    frame = pd.concat(buffer, ignore_index=True)
                    frame.iloc[:20_000].join(keep, on="immutable_id").to_parquet(
                        shards / f"{n_shard:05d}.parquet")
                    n_shard += 1
                    buffer = [frame.iloc[20_000:]]
        frame = pd.concat(buffer, ignore_index=True)
        if len(frame):
            frame.join(keep, on="immutable_id").to_parquet(shards / f"{n_shard:05d}.parquet")
            n_shard += 1
        (shards / "_DONE").write_text(f"{len(keep)} structures in {n_shard} shards\n")
    subprocess.run([sys.executable, str(MOTIF_SCRIPT), "run", "--out", str(out),
                    "--workers", str(args.workers)], check=True)
    subprocess.run([sys.executable, str(MOTIF_SCRIPT), "report", "--out", str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    print(f"environment table -> {out / 'environment_frequency.csv'}")


def _mine_one(task: tuple) -> dict:
    """Bond-valence roles and clean cation environments of one relaxed structure."""
    import signal

    from pymatgen.core import Structure

    from wyckoff_transformer.evaluation.coordination_motifs import analyse
    from wyckoff_transformer.transport.screens import (
        bond_valence_roles,
        environment_keys,
    )

    def alarm(signum, frame):
        raise TimeoutError

    index, immutable_id, path = task
    cif = Path(path).read_text(encoding="utf-8")
    states, reason = bond_valence_roles(Structure.from_str(cif, fmt="cif"))
    # screen_coordination_motifs.py's guard: a few compositions blow up the
    # oxidation-state search.
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(60)
    try:
        summary, sites, _pairs = analyse(immutable_id, cif)
    except TimeoutError:
        summary, sites = {"status": "timeout"}, []
    finally:
        signal.alarm(0)
    return {"index": index, "bv_states": states, "bv_reason": reason,
            "motif_status": summary["status"],
            "environments": ["|".join(map(str, key)) for key in environment_keys(sites)]}


def run_mine_screen(args, config: dict) -> None:
    """Phase 2 screens of the palette compounds outside the families campaign's systems.

    Every compound gets every flag, so the funnel can be read in any order: the gap,
    the generic anion roles (formula, then bond valence on the ORB structure), the
    coordination check against LeMat's environment table, and known-SE frameworks
    (a match to a fast OBELiX framework group).
    """
    from wyckoff_transformer.transport.screens import generic_oxidation_states

    root = campaign_dir(config, args.out)
    spec = config["mining"]
    relaxed = relaxed_structures(root)
    table = relaxed[(relaxed["source"] == "lemat")
                    & ~relaxed["in_families_campaign"].astype(bool)].copy()
    gaps = pd.read_parquet(root / "mine" / "gaps.parquet")[["index", "band_gap",
                                                             "band_gap_source"]]
    frames = pd.read_parquet(root / "bench" / "frameworks.parquet")[
        ["index", "framework_formula", "obelix_label", "obelix_family"]]
    table = table.merge(gaps, on="index", how="left").merge(frames, on="index", how="left")
    table["gap_ok"] = table["band_gap"] >= spec["band_gap"]["min_ev"]
    table["oxidation_states"] = table["reduced_formula"].map(generic_oxidation_states)
    table["known_se"] = table["obelix_label"] == "fast"
    out = root / "mine" / "screen.jsonl"
    run_pool([(int(row.index), row.immutable_id, row.relaxed_cif)
              for row in table.itertuples(index=False)], _mine_one, out, args.workers)
    rows = pd.DataFrame(read_jsonl(out)).drop_duplicates("index", keep="last")
    table = table.merge(rows.drop(columns=["error"], errors="ignore"), on="index", how="left")
    table["roles_ok"] = table["oxidation_states"].notna() & table["bv_states"].notna()

    # The rarity cutoff: the largest count of LeMat structures an environment must reach
    # that removes at most max_removed_fraction of the compounds the check applies to.
    reference_path = root / "mine" / "coordination" / "environment_frequency.csv"
    if reference_path.is_file():
        reference = pd.read_csv(reference_path)
        counts = {f"{r.species}|{r.cn}|{r.geometry}": int(r.structures)
                  for r in reference.itertuples(index=False)}
    else:
        logger.warning("no %s yet (stage coordination-reference): the coordination check "
                       "is not applied", reference_path)
        counts = None
    table["coordination_applied"] = counts is not None
    counts = counts or {}
    table["rarest_environment_count"] = table["environments"].map(
        lambda keys: min((counts.get(k, 0) for k in keys), default=np.inf)
        if isinstance(keys, (list, np.ndarray)) else np.inf)
    applies = table["gap_ok"] & table["roles_ok"]
    rarest = np.sort(table.loc[applies, "rarest_environment_count"].to_numpy())
    allowed = int(np.floor(spec["coordination"]["max_removed_fraction"] * len(rarest)))
    # Removing every compound whose rarest count is below the cutoff; the cutoff is the
    # smallest count present at rank `allowed`, so ties at it are kept.
    cutoff = float(rarest[allowed]) if len(rarest) > allowed and counts else 0.0
    table["coordination_ok"] = table["rarest_environment_count"] >= cutoff
    table["phase2_survivor"] = (table["gap_ok"] & table["roles_ok"] & table["coordination_ok"]
                                & ~table["known_se"])
    write_parquet(table.drop(columns=["relaxed_cif"]), root / "mine" / "screen.parquet")
    funnel = [("outside the families campaign", np.ones(len(table), dtype=bool)),
              ("gap", table["gap_ok"]), ("anion roles, formula", table["oxidation_states"].notna()),
              ("anion roles, bond valence", table["bv_states"].notna()),
              ("coordination", table["coordination_ok"]), ("not a known-SE framework",
                                                           ~table["known_se"])]
    keep = np.ones(len(table), dtype=bool)
    for name, mask in funnel:
        keep &= np.asarray(mask, dtype=bool)
        print(f"{name:32s} {int(keep.sum()):6d}")
    print(f"coordination cutoff: an environment in >= {cutoff:g} LeMat structures "
          f"({int((applies & ~table['coordination_ok']).sum())} of {int(applies.sum())} removed)")
    print(table.loc[applies, "bv_reason"].value_counts().to_string())


def run_mine_pes(args, config: dict) -> None:
    """Empty-framework scans of the Phase 2 survivors the benchmark did not scan.

    They go to ``mine/pes_empty.jsonl``, not the benchmark's file: poor-labelled
    compounds beyond the benchmark's 40-per-group cap would otherwise enter its pAUC.
    """
    import multiprocessing

    root = campaign_dir(config, args.out)
    spec = config["benchmark"]["xi"]
    screen = pd.read_parquet(root / "mine" / "screen.parquet")
    scanned = set(pd.read_parquet(root / "bench" / "pes_empty.parquet")["index"])
    relaxed = relaxed_structures(root).set_index("index")
    # Before the coordination cutoff exists the gap and role screens are enough: the
    # cutoff removes at most 1%.
    wanted = screen[screen["gap_ok"] & screen["roles_ok"] & ~screen["known_se"]
                    & ~screen["index"].isin(scanned)].sort_values("n_atoms")
    tasks = [(int(i), relaxed.loc[i, "relaxed_cif"], ("empty",), spec) for i in wanted["index"]]
    devices = (args.devices or config["benchmark"]["relax"]["devices"]).split(",")
    slots = [d for d in devices for _ in range(args.workers_per_device or 1)]
    counter = multiprocessing.get_context("spawn").Value("i", 0)
    out = root / "mine" / "pes_empty.jsonl"
    run_pool(tasks, _pes_one, out, len(slots), initializer=_pes_init,
             initargs=(slots, counter, spec["max_atoms_per_batch"]), every=100)
    frame = pd.DataFrame(read_jsonl(out))
    if "error" in frame:
        frame = frame[frame["error"].isna()].drop(columns="error")
    write_parquet(frame.drop_duplicates("index", keep="last"), out.with_suffix(".parquet"))
    print(f"{len(frame)} mining scans -> {out.with_suffix('.parquet')}")


def descriptor_values(root: Path) -> pd.DataFrame:
    """Every descriptor measured on any structure, benchmark and mining scans together."""
    values = pd.read_parquet(root / "bench" / "describe.parquet").set_index("index")
    for path in (root / "bench" / "pes_frozen.parquet", root / "bench" / "pes_empty.parquet",
                 root / "mine" / "pes_empty.parquet"):
        if path.is_file():
            pes = pd.read_parquet(path).drop(columns=["device", "grid"], errors="ignore")
            values = values.combine_first(pes.set_index("index"))
    return values


def run_mine_rank(args, config: dict) -> None:
    """Phase 2 ranking: the survivors within a G1 descriptor's line, clustered by framework.

    A descriptor's line is the random control's top ``rank.top_fraction``, the line a
    family's median fast member had to reach to count as rediscovered at G1. With
    several G1 descriptors a survivor is a hit if it is within any line (Phase 6's OR).
    Clusters are ranked by their best member's percentile.
    """
    from pymatgen.core import Structure

    from wyckoff_transformer.transport.frameworks import (
        ANION_LABEL,
        default_matcher,
        group_frameworks,
    )

    root = campaign_dir(config, args.out)
    top = config["mining"]["rank"]["top_fraction"]
    report = pd.read_csv(root / "bench" / "evaluation" / "descriptors.csv")
    passing = dict(zip(report.loc[report["g1"], "descriptor"],
                       report.loc[report["g1"], "orientation"]))
    if not passing:
        raise SystemExit("No descriptor passed G1; the plan stops at Phase 1.")
    screen = pd.read_parquet(root / "mine" / "screen.parquet")
    if not screen["coordination_applied"].all():
        raise SystemExit("mine-screen ran without the coordination reference; rerun it "
                         "after coordination-reference.")
    values = descriptor_values(root)
    control = pd.read_parquet(root / "bench" / "pes_selection.parquet")
    control = control.loc[control["in_control"], "index"]
    table = screen[screen["phase2_survivor"]].copy()
    for name, sign in passing.items():
        oriented = sign * values[name].replace([np.inf, -np.inf], [1e6, -1e6])
        reference = oriented.reindex(control).dropna().to_numpy()
        score = oriented.reindex(table["index"]).to_numpy()
        filled = np.where(np.isfinite(score), score, -np.inf)
        table[name] = values[name].reindex(table["index"]).to_numpy()
        table[f"{name}_percentile"] = np.where(
            np.isfinite(score), (reference[None, :] > filled[:, None]).mean(axis=1)
            + (reference[None, :] == filled[:, None]).mean(axis=1) / 2, np.nan)
    percentiles = table[[f"{name}_percentile" for name in passing]]
    table["best_percentile"] = percentiles.min(axis=1)
    table["hit"] = table["best_percentile"] <= top
    missing = int(percentiles.isna().all(axis=1).sum())
    hits = table[table["hit"]].copy()

    frames = pd.read_parquet(root / "bench" / "frameworks.parquet").set_index("index")
    structures = {int(i): Structure.from_str(frames.loc[i, "framework_cif"], fmt="cif")
                  for i in hits["index"] if isinstance(frames.loc[i, "framework_cif"], str)}
    groups = group_frameworks(structures, default_matcher(**config["benchmark"]["framework_matcher"]))
    next_group = max(groups.values(), default=-1) + 1
    hits["cluster"] = [groups.get(int(i), next_group + k) for k, i in enumerate(hits["index"])]
    hits = hits.sort_values(["best_percentile", "energy_above_hull"])
    clusters = hits.groupby("cluster", sort=False).agg(
        members=("index", "size"), systems=("system", "nunique"),
        framework=("framework_formula", "first"),
        best_percentile=("best_percentile", "min"),
        best_index=("index", "first"), best_formula=("reduced_formula", "first"),
        best_immutable_id=("immutable_id", "first"),
        best_e_hull=("energy_above_hull", "first"), best_gap=("band_gap", "first"),
        min_e_hull=("energy_above_hull", "min"), icsd_members=("icsd_backed", "sum"),
        formulas=("reduced_formula", lambda f: ", ".join(dict.fromkeys(f).keys())[:200]))
    for name, sign in passing.items():
        by_cluster = hits.groupby("cluster", sort=False)[name]
        clusters[f"best_{name}"] = by_cluster.min() if sign < 0 else by_cluster.max()
    # A bare anion lattice (antifluorite, antiperovskite) has a low empty-framework
    # barrier by construction; see the G1 section of docs/se_new_families_campaign.md.
    clusters["anion_only"] = clusters["framework"].fillna("").str.fullmatch(
        rf"{ANION_LABEL}\d*")
    clusters = clusters.reset_index()
    clusters.insert(0, "rank", np.arange(1, len(clusters) + 1))
    write_parquet(hits.drop(columns=["environments"], errors="ignore"),
                  root / "mine" / "hits.parquet")
    clusters.to_csv(root / "mine" / "clusters.csv", index=False)
    print(f"G1 descriptors: {passing}; {len(table)} survivors, {missing} without a value; "
          f"{len(hits)} within the control's top {top:.0%}; {len(clusters)} framework clusters")
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(clusters.head(args.limit or 40).drop(columns=["formulas"]).round(3)
              .to_string(index=False))


def run_mine_md(args, config: dict) -> None:
    """Phase 7's MD on the best member of each of the top Phase 2 clusters.

    Clusters with an anion-only framework are skipped: the empty-framework barrier is
    low there by construction, they are not new framework families, and G1c's Li2O,
    LiF and LiCl already show what MD makes of them.
    """
    import multiprocessing

    root = campaign_dir(config, args.out)
    spec = config["md"]
    clusters = pd.read_csv(root / "mine" / "clusters.csv")
    clusters = clusters[~clusters["anion_only"]].head(args.limit or 20)
    relaxed = relaxed_structures(root).set_index("index")
    out_dir = root / "mine" / "md"
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    keys = {f"c{row.rank:03d}-{row.best_immutable_id}": row for row in clusters.itertuples()}
    tasks = [(f"{key}_{temperature}", relaxed.loc[row.best_index, "relaxed_cif"], temperature,
              spec, str(out_dir / "trajectories"))
             for temperature in spec["temperatures"] for key, row in keys.items()]
    devices = (args.devices or config["benchmark"]["relax"]["devices"]).split(",")
    slots = [d for d in devices for _ in range(args.workers_per_device or 3)]
    counter = multiprocessing.get_context("spawn").Value("i", 0)
    run_pool(tasks, _md_one, out_dir / "md.jsonl", len(slots), initializer=_md_init,
             initargs=(slots, counter), label="MD runs", every=1)
    table = md_summary(md_runs(out_dir), spec)
    info = pd.DataFrame({key: {"rank": row.rank, "formula": row.best_formula,
                               "framework": row.framework, "members": row.members,
                               "e_hull": row.best_e_hull} for key, row in keys.items()}).T
    table = info.join(table).sort_values("rank")
    table.to_csv(out_dir / "candidates.csv")
    with pd.option_context("display.width", 250):
        print(table.round(5).to_string())


def tiered_members(root: Path) -> pd.DataFrame:
    """Every hit in a cluster whose ``mine-md`` run reached a tier, with its cluster rank."""
    candidates = pd.read_csv(root / "mine" / "md" / "candidates.csv", index_col=0)
    tiered = candidates[candidates["tier"].fillna("") != ""]
    clusters = pd.read_csv(root / "mine" / "clusters.csv")
    clusters = clusters[clusters["rank"].isin(tiered["rank"])]
    hits = pd.read_parquet(root / "mine" / "hits.parquet")
    members = hits.merge(clusters[["rank", "cluster", "best_index"]], on="cluster")
    members["md_tier"] = members["rank"].map(tiered.set_index("rank")["tier"])
    members["is_best"] = members["index"] == members["best_index"]
    return members.sort_values(["rank", "is_best", "energy_above_hull"],
                               ascending=[True, False, True]).reset_index(drop=True)


def run_mine_md_members(args, config: dict) -> None:
    """Phase 7's MD on further members of the tiered clusters (``phase8/md/``).

    ``mine-md`` ran each cluster's best member only; the plan's scope is up to three
    structures per candidate family. Members are taken by e_hull, and one whose ORB
    structure matches a member already chosen is skipped: MP, Alexandria and OQMD often
    hold the same ordering.
    """
    import multiprocessing

    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    root = campaign_dir(config, args.out)
    spec = config["md"]
    per_cluster = config["phase8"]["md_members_per_cluster"]
    relaxed = relaxed_structures(root).set_index("index")
    members = tiered_members(root)
    matcher = StructureMatcher()
    chosen, duplicates = [], {}
    for _, group in members.groupby("rank", sort=True):
        picked: list[tuple[int, Structure]] = []
        for row in group.itertuples():
            if len(picked) == per_cluster:
                break
            structure = Structure.from_file(relaxed.loc[row.index, "relaxed_cif"])
            same = next((i for i, other in picked if matcher.fit(structure, other)), None)
            if same is not None:
                duplicates[row.immutable_id] = int(same)
                continue
            picked.append((row.index, structure))
            chosen.append(row)
    chosen = pd.DataFrame(chosen).drop(columns="Index")
    out_dir = root / "phase8" / "md"
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    chosen["key"] = [f"c{row.rank:03d}-{row.immutable_id}" for row in chosen.itertuples()]
    chosen["duplicates"] = chosen["index"].map(
        lambda i: ", ".join(k for k, v in duplicates.items() if v == i))
    todo = chosen[~chosen["is_best"]]
    tasks = [(f"{row.key}_{temperature}", relaxed.loc[row.index, "relaxed_cif"], temperature,
              spec, str(out_dir / "trajectories"))
             for temperature in spec["temperatures"] for row in todo.itertuples()]
    logger.info("%d tiered clusters, %d members chosen, %d new MD structures; duplicates: %s",
                chosen["rank"].nunique(), len(chosen), len(todo), duplicates)
    devices = (args.devices or config["benchmark"]["relax"]["devices"]).split(",")
    slots = [d for d in devices for _ in range(args.workers_per_device or 3)]
    counter = multiprocessing.get_context("spawn").Value("i", 0)
    run_pool(tasks, _md_one, out_dir / "md.jsonl", len(slots), initializer=_md_init,
             initargs=(slots, counter), label="MD runs", every=1)
    table = md_summary(md_runs(out_dir), spec)
    info = todo.set_index("key")[["rank", "index", "immutable_id", "reduced_formula",
                                  "energy_above_hull", "icsd_backed", "duplicates"]]
    table = info.join(table).sort_values(["rank", "energy_above_hull"])
    table.to_csv(out_dir / "candidates.csv")
    chosen.drop(columns=["environments"], errors="ignore").to_csv(out_dir / "members.csv",
                                                                  index=False)
    with pd.option_context("display.width", 250):
        print(table.round(5).to_string())


# --------------------------------------------------------------------------- #
# Phase 8: windows, interface reactions, joins and structure checks
# --------------------------------------------------------------------------- #
def electrode_panel(config: dict) -> dict[str, tuple[str, float]]:
    """``name -> (formula, potential vs Li/Li+ in V)``."""
    return {name: (spec["formula"], float(spec["potential"]))
            for name, spec in config["phase8"]["electrodes"].items()}


def icsd_flags(config: dict) -> pd.Series:
    """``icsd_backed`` by immutable_id over every split of the ICSD-provenance cache."""
    spec = config["data"]["lemat"]
    return pd.concat([load_split(dataset_cache_dir(spec["icsd_cache"]), split,
                                 columns=["icsd_backed"])["icsd_backed"]
                      for split in SPLITS]).astype(bool)


def _mp2020_one(task: tuple) -> dict:
    from pymatgen.core import Structure

    from wyckoff_transformer.transport.electrochem import (
        mp2020_energy,
        mp_run_parameters,
    )

    entry_id, cif, energy = task
    structure = Structure.from_str(cif, fmt="cif")
    return {"index": entry_id, "energy_mp2020": mp2020_energy(structure, energy, entry_id),
            "run_type": mp_run_parameters(structure.composition)["run_type"]}


def run_phase8_entries(args, config: dict) -> None:
    """LeMat-Bulk-fixed entries of every system Phase 8 needs, with MP2020 energies.

    The systems are each tiered member's, alone and joined with each electrode's
    elements. Every LeMat row whose chemical system lies inside one of them is kept with
    its CIF, which MP2020 needs to tell a sulfate from a sulfide and a peroxide from an
    oxide. A row MP2020 refuses keeps a missing ``energy_mp2020``.

    Whether LeMat's U elements in oxides really are GGA+U on every source is checked
    from the data: for each such composition held by MP (whose rows the fix put on MP's
    run type) and by Alexandria or OQMD, the lowest energies per atom are compared. A
    plain-GGA row would sit about 1 eV per U atom below a GGA+U one.
    """
    from itertools import combinations

    from pymatgen.core import Composition

    root = campaign_dir(config, args.out)
    spec = config["phase8"]
    out_dir = root / "phase8"
    out_dir.mkdir(parents=True, exist_ok=True)
    members = tiered_members(root)
    electrodes = [element_set(Composition(formula)) for formula, _ in electrode_panel(config).values()]
    systems = {frozenset(system.split("-")) | electrode
               for system in members["system"] for electrode in electrodes}
    wanted = {"-".join(sorted(subset)) for system in systems
              for size in range(1, len(system) + 1) for subset in combinations(system, size)}
    path = pinned(spec, "lemat_hull", data_path(spec["lemat_hull"]))
    frames = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for chunk in pd.read_csv(handle, usecols=["immutable_id", "full_formula", "chemsys",
                                                  "energy_corrected", "cif", "e_hull"],
                                 chunksize=200_000):
            frames.append(chunk[chunk["chemsys"].isin(wanted)])
    entries = pd.concat(frames, ignore_index=True)
    logger.info("%d systems, %d LeMat-Bulk-fixed rows in them", len(systems), len(entries))
    tasks = [(row.immutable_id, row.cif, row.energy_corrected) for row in entries.itertuples()]
    run_pool(tasks, _mp2020_one, out_dir / "mp2020.jsonl", args.workers, label="entries",
             every=5000)
    corrected = pd.DataFrame(read_jsonl(out_dir / "mp2020.jsonl")).drop_duplicates(
        "index", keep="last").rename(columns={"index": "immutable_id"})
    entries = entries.merge(corrected.drop(columns="error", errors="ignore"), on="immutable_id",
                            how="left")
    entries["icsd_backed"] = entries["immutable_id"].map(icsd_flags(config)).astype("boolean")
    entries["source"] = entries["immutable_id"].str.extract(r"^(mp|agm|oqmd)", expand=False)
    entries["reduced_formula"] = entries["full_formula"].map(
        lambda f: Composition(f).reduced_formula)
    write_parquet(entries, out_dir / "entries.parquet")

    u_rows = entries[entries["run_type"] == "GGA+U"].copy()
    u_rows["energy_per_atom"] = u_rows["energy_corrected"] / u_rows["full_formula"].map(
        lambda f: Composition(f).num_atoms)
    best = u_rows.sort_values("energy_per_atom").groupby(["reduced_formula", "source"]).head(1)
    lowest = best.pivot(index="reduced_formula", columns="source", values="energy_per_atom")
    hull = best.pivot(index="reduced_formula", columns="source", values="e_hull")
    check = pd.DataFrame({f"{other}_minus_mp": lowest[other] - lowest["mp"]
                          for other in ("agm", "oqmd") if other in lowest})
    # Far above the hull the two lowest rows are often different structures; near it a
    # plain-GGA row would stand out by about 1 eV per U atom.
    near = pd.DataFrame({f"{other}_minus_mp_near_hull": check[f"{other}_minus_mp"].where(
        (hull[other] <= 0.1) & (hull["mp"] <= 0.1)) for other in ("agm", "oqmd") if other in lowest})
    check = check.join(near)
    check.to_csv(out_dir / "u_scheme_check.csv")
    refused = entries["energy_mp2020"].isna()
    print(f"{len(entries)} rows; MP2020 refused {int(refused.sum())}; "
          f"GGA+U rows {len(u_rows)} by source: {u_rows['source'].value_counts().to_dict()}")
    print("Lowest GGA+U energy per atom at one composition, other source minus MP (eV/atom):")
    print(check.describe().round(3).to_string())


_PHASE8_ENTRIES = None


def _phase8_init(path: str) -> None:
    global _PHASE8_ENTRIES
    _cpu_init()
    table = pd.read_parquet(path, columns=["immutable_id", "full_formula", "chemsys",
                                           "energy_corrected", "energy_mp2020"])
    table["elements"] = table["chemsys"].map(lambda s: frozenset(s.split("-")))
    _PHASE8_ENTRIES = table


def _phase8_diagram(elements: frozenset, energies: str):
    """The phase diagram of LeMat's rows inside *elements*, raw or MP2020."""
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition

    column = {"raw": "energy_corrected", "mp2020": "energy_mp2020"}[energies]
    rows = _PHASE8_ENTRIES[_PHASE8_ENTRIES["elements"].map(lambda e: e <= elements)
                           & _PHASE8_ENTRIES[column].notna()]
    return PhaseDiagram([PDEntry(Composition(row.full_formula), getattr(row, column),
                                 attribute=row.immutable_id)
                         for row in rows.itertuples()])


def _window_one(task: tuple) -> dict:
    """The window of one composition on the hull, and its members' e_hull."""
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition

    from wyckoff_transformer.transport.electrochem import (
        decomposition_at,
        phases,
        placed_on_hull,
        stability_window,
    )

    key, formula, energies, member_ids, margin, offset, mobile = task
    composition = Composition(formula)
    base = _phase8_diagram(element_set(composition), energies)
    placed = placed_on_hull(base, composition, "candidate", margin)
    diagram = PhaseDiagram(base.all_entries + [placed])
    low, high = stability_window(diagram, "candidate", mobile)
    by_id = {entry.attribute: entry for entry in base.all_entries}
    e_hull = {i: base.get_e_above_hull(by_id[i]) for i in member_ids if i in by_id}
    return {"index": key, "formula": formula, "energies": energies, "mu_low": low,
            "mu_high": high, "reduction_V": -high, "oxidation_V": -low,
            "assemblage": phases(base.get_decomposition(composition)),
            "reduction_products": decomposition_at(diagram, composition, high + offset, mobile)
            if high < -1e-6 else "",
            "oxidation_products": decomposition_at(diagram, composition, low - offset, mobile),
            "e_hull": e_hull}


def _interface_one(task: tuple) -> dict:
    """Richards' eqs. 2, 4 and 5 for one composition against one electrode."""
    from pymatgen.core import Composition

    from wyckoff_transformer.transport.electrochem import (
        chemical_reaction,
        electrochemical_reaction,
    )

    key, formula, electrode, electrode_formula, potential, energies, mobile = task
    composition, counterpart = Composition(formula), Composition(electrode_formula)
    diagram = _phase8_diagram(element_set(composition) | element_set(counterpart), energies)
    chemical = chemical_reaction(diagram, composition, counterpart)
    electrochemical = electrochemical_reaction(diagram, composition, counterpart, -potential,
                                               mobile)
    return {"index": key, "formula": formula, "electrode": electrode, "potential_V": potential,
            "energies": energies, "chemical_eV_atom": chemical["energy"],
            "chemical_x": chemical["x"], "chemical_reaction": chemical["reaction"],
            "no_mixing_eV_atom": electrochemical["no_mixing"],
            "electrochemical_eV_atom": electrochemical["energy"],
            "electrochemical_x": electrochemical["x"],
            "electrochemical_products": electrochemical["products"]}


def join_table(members: pd.DataFrame, entries: pd.DataFrame, mobile: str) -> pd.DataFrame:
    """The plan's join flag for each member composition.

    For each anion that can balance all of Li's charge, the join runs from the Li binary
    to the rest of the formula. LeMat compositions strictly between the two, other than
    the member's own, are counted: all of them, those within 0.1 eV/atom, and the
    ICSD-backed ones (the member's own composition included). The plan flags a
    composition when LeMat holds no other compound on its join. Oxidation states are the
    structure's bond-valence ones where it has them: the formula stage's generic states
    make Li2SnN2 Sn(+2) N(-2).
    """
    from pymatgen.core import Composition

    from wyckoff_transformer.transport.electrochem import (
        position_on_join,
        pseudo_binary_joins,
    )

    by_formula = entries.groupby("reduced_formula").agg(
        e_hull=("e_hull", "min"), icsd=("icsd_backed", "max"), system=("chemsys", "first"))
    rows = []
    members = members.assign(states=members["bv_states"].fillna(members["oxidation_states"]))
    for formula, states in members.drop_duplicates("reduced_formula")[
            ["reduced_formula", "states"]].itertuples(index=False):
        composition = Composition(formula)
        oxidation = {token.rstrip("+-0123456789."): float(token[len(token.rstrip(
            "+-0123456789.")):]) for token in states.split()}
        elements = {str(e) for e in composition}
        local = by_formula[[set(s.split("-")) <= elements for s in by_formula["system"]]]
        for anion, binary, rest in pseudo_binary_joins(composition, oxidation, mobile):
            positions = {f: position_on_join(Composition(f), binary, rest) for f in local.index}
            on = local[[p is not None and 1e-6 < p < 1 - 1e-6 and f != formula
                        for f, p in positions.items()]]
            rows.append({"formula": formula, "anion": anion,
                         "join": f"{binary.reduced_formula}-{rest.reduced_formula}",
                         "position": positions.get(formula),
                         "others_on_join": len(on), "others_within_0.1": int((on["e_hull"] <= 0.1).sum()),
                         "icsd_on_join": int(local.loc[[f for f, p in positions.items()
                                                        if p is not None and 1e-6 < p < 1 - 1e-6],
                                                       "icsd"].fillna(False).sum()),
                         "compounds": ", ".join(on.sort_values("e_hull").index[:8]),
                         "join_flag": len(on) == 0})
    return pd.DataFrame(rows)


def run_phase8_window(args, config: dict) -> None:
    """Windows, interface reactions and the join flag of the tiered members (Richards 2016).

    Every member composition gets its window, placed on the hull, with MP2020 and with
    raw energies; then Richards' eq. 2 (closed), eq. 4 (open to Li at the electrode's
    potential) and eq. 5 (the reservoir alone) against every electrode of the panel.
    """
    root = campaign_dir(config, args.out)
    spec = config["phase8"]
    mobile = config["mobile_ion"]
    out_dir = root / "phase8"
    members = tiered_members(root)
    ids = members.groupby("reduced_formula")["immutable_id"].agg(list)
    window = spec["window"]
    tasks = [(f"window|{formula}|{energies}", formula, energies, member_ids,
              window["hull_margin"], window["product_offset"], mobile)
             for formula, member_ids in ids.items() for energies in window["energies"]]
    tasks += [(f"interface|{formula}|{name}|{energies}", formula, name, electrode, potential,
               energies, mobile)
              for formula in ids.index for name, (electrode, potential) in
              electrode_panel(config).items() for energies in window["energies"]]
    entries_path = out_dir / "entries.parquet"
    run_pool(tasks, _phase8_one, out_dir / "window.jsonl", args.workers,
             initializer=_phase8_init, initargs=(str(entries_path),), label="tasks", every=20)
    rows = [row for row in read_jsonl(out_dir / "window.jsonl") if "error" not in row]
    windows = pd.DataFrame([r for r in rows if r["index"].startswith("window|")])
    interfaces = pd.DataFrame([r for r in rows if r["index"].startswith("interface|")])
    e_hull = pd.DataFrame([{"immutable_id": i, "energies": r["energies"], "e_hull": e}
                           for r in rows if r["index"].startswith("window|")
                           for i, e in r["e_hull"].items()]).pivot(
        index="immutable_id", columns="energies", values="e_hull").add_prefix("e_hull_")
    windows.drop(columns=["e_hull"]).to_csv(out_dir / "windows.csv", index=False)
    interfaces.to_csv(out_dir / "interfaces.csv", index=False)
    entries = pd.read_parquet(entries_path, columns=["immutable_id", "reduced_formula",
                                                     "chemsys", "e_hull", "icsd_backed"])
    lemat = entries.set_index("immutable_id")["e_hull"]
    e_hull["e_hull_lemat"] = lemat.reindex(e_hull.index)
    e_hull.to_csv(out_dir / "e_hull.csv")
    joins = join_table(members, entries, mobile)
    joins.to_csv(out_dir / "joins.csv", index=False)

    worst = (e_hull["e_hull_raw"] - e_hull["e_hull_lemat"]).abs().max()
    print(f"raw e_hull recomputed from the cached rows vs LeMat-Bulk-fixed's: max |diff| "
          f"{worst:.2e} eV/atom over {len(e_hull)} members")
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(windows.pivot(index="formula", columns="energies",
                            values=["reduction_V", "oxidation_V"]).round(2).to_string())
        wide = interfaces.pivot_table(index="formula", columns=["energies", "electrode"],
                                      values="electrochemical_eV_atom")
        print(wide["mp2020"][list(electrode_panel(config))].round(3).to_string())
        print(joins.drop(columns="compounds").to_string(index=False))


def _phase8_one(task: tuple) -> dict:
    return (_window_one if task[0].startswith("window|") else _interface_one)(task)


def run_phase8_check(args, config: dict) -> None:
    """Each tiered member's ORB structure against its LeMat-Bulk-fixed DFT structure.

    MD ran on the ORB structure, the e_hull is the DFT one's, so they must be the same
    crystal. Also the formula's lowest LeMat polymorph and whether it is ICSD-backed.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    root = campaign_dir(config, args.out)
    out_dir = root / "phase8"
    members = tiered_members(root)
    relaxed = relaxed_structures(root).set_index("index")
    entries = pd.read_parquet(out_dir / "entries.parquet",
                              columns=["immutable_id", "reduced_formula", "cif", "e_hull",
                                       "icsd_backed", "source"]).set_index("immutable_id")
    matcher = StructureMatcher()
    rows = []
    for row in members.itertuples():
        dft = Structure.from_str(entries.loc[row.immutable_id, "cif"], fmt="cif")
        orb = Structure.from_file(relaxed.loc[row.index, "relaxed_cif"])
        rms = matcher.get_rms_dist(orb, dft)
        polymorphs = entries[entries["reduced_formula"] == row.reduced_formula].sort_values(
            "e_hull")
        lowest = polymorphs.iloc[0]
        rows.append({
            "rank": row.rank, "immutable_id": row.immutable_id, "formula": row.reduced_formula,
            "md_tier": row.md_tier, "is_best": row.is_best,
            "spacegroup_dft": SpacegroupAnalyzer(dft, symprec=0.1).get_space_group_number(),
            "spacegroup_orb": SpacegroupAnalyzer(orb, symprec=0.1).get_space_group_number(),
            "matches": rms is not None, "rms": None if rms is None else rms[0],
            "volume_ratio": (orb.volume / len(orb)) / (dft.volume / len(dft)),
            "e_hull": entries.loc[row.immutable_id, "e_hull"],
            "icsd_backed": bool(entries.loc[row.immutable_id, "icsd_backed"] or False),
            "polymorphs": len(polymorphs), "lowest_polymorph": polymorphs.index[0],
            "lowest_e_hull": lowest["e_hull"],
            "lowest_icsd": bool(lowest["icsd_backed"] or False),
            "icsd_polymorphs": ", ".join(polymorphs.index[polymorphs["icsd_backed"].fillna(
                False).astype(bool)]),
            "above_lowest": entries.loc[row.immutable_id, "e_hull"] - lowest["e_hull"]})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "structures.csv", index=False)
    with pd.option_context("display.width", 250):
        print(table.round(4).to_string(index=False))


def run_phase8_prototypes(args, config: dict) -> None:
    """Each tiered member's structure type among LeMat's ICSD-backed structures.

    A member whose structure matches an ICSD-backed one with every species relabelled
    (``StructureMatcher.fit_anonymous``, pymatgen's defaults) is a known structure type
    in new chemistry. The framework match of the screens anonymised only cation and
    anion roles, and only against known SEs.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Composition, Structure

    root = campaign_dir(config, args.out)
    out_dir = root / "phase8"
    members = tiered_members(root)
    entries = pd.read_parquet(out_dir / "entries.parquet", columns=["immutable_id", "cif"])
    cifs = entries.set_index("immutable_id")["cif"]
    wanted = {Composition(f).anonymized_formula for f in members["reduced_formula"]}
    icsd = icsd_flags(config)
    icsd = set(icsd.index[icsd])
    path = pinned(config["phase8"], "lemat_hull", data_path(config["phase8"]["lemat_hull"]))
    found = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for chunk in pd.read_csv(handle, usecols=["immutable_id", "full_formula", "cif"],
                                 chunksize=200_000):
            chunk = chunk[chunk["immutable_id"].isin(icsd)]
            anonymous = chunk["full_formula"].map(lambda f: Composition(f).anonymized_formula)
            found.append(chunk[anonymous.isin(wanted)].assign(anonymous=anonymous))
    references = pd.concat(found)
    references["structure"] = [Structure.from_str(c, fmt="cif") for c in references["cif"]]
    matcher = StructureMatcher()
    rows = []
    for row in members.itertuples():
        structure = Structure.from_str(cifs[row.immutable_id], fmt="cif")
        same = references[references["anonymous"] == structure.composition.anonymized_formula]
        hits = [(r.immutable_id, Composition(r.full_formula).reduced_formula)
                for r in same.itertuples() if matcher.fit_anonymous(structure, r.structure)]
        rows.append({"rank": row.rank, "immutable_id": row.immutable_id,
                     "formula": row.reduced_formula, "icsd_same_anonymous_formula": len(same),
                     "icsd_prototype_matches": len(hits),
                     "icsd_prototypes": ", ".join(f"{f} {i}" for i, f in hits[:8])})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "prototypes.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print(table.to_string(index=False))


def md_tables(root: Path) -> pd.DataFrame:
    """Every Phase 2 and Phase 8 MD structure's numbers, by run key."""
    tables = [pd.read_csv(out_dir / "candidates.csv", index_col=0)
              for out_dir in (root / "mine" / "md", root / "phase8" / "md")
              if (out_dir / "candidates.csv").is_file()]
    md = pd.concat(tables)
    md["immutable_id"] = md.index.str.split("-", n=1).str[1]
    return md


def run_phase8_report(args, config: dict) -> None:
    """The plan's Phase 8 fields, one row per tiered member (``phase8/report.csv``).

    Framework and cluster; e_hull (LeMat's PBE, and MP2020 against the same rows); the
    G1 descriptor and the MD numbers where MD ran; the MP2020 and raw windows and the
    electrodes inside them; the join flag; the coordination flag.
    """
    root = campaign_dir(config, args.out)
    out_dir = root / "phase8"
    members = tiered_members(root)
    windows = pd.read_csv(out_dir / "windows.csv").pivot(
        index="formula", columns="energies", values=["reduction_V", "oxidation_V"])
    windows.columns = [f"{name}_{energies}" for name, energies in windows.columns]
    e_hull = pd.read_csv(out_dir / "e_hull.csv", index_col=0)
    joins = pd.read_csv(out_dir / "joins.csv").groupby("formula").agg(
        joins=("join", ", ".join), others_on_join=("others_on_join", "max"),
        icsd_on_join=("icsd_on_join", "max"), join_flag=("join_flag", "all"))
    structures = pd.read_csv(out_dir / "structures.csv").set_index("immutable_id")
    md = md_tables(root).set_index("immutable_id")
    panel = electrode_panel(config)
    report = members[["rank", "cluster", "framework_formula", "md_tier", "index", "immutable_id",
                      "reduced_formula", "source", "icsd_backed", "band_gap", "empty_E_2D",
                      "empty_E_2D_percentile", "rarest_environment_count",
                      "coordination_ok"]].copy()
    report = report.join(e_hull[["e_hull_lemat", "e_hull_mp2020"]], on="immutable_id")
    report = report.join(structures[["matches", "rms", "volume_ratio", "lowest_polymorph",
                                     "above_lowest", "icsd_polymorphs"]], on="immutable_id")
    report = report.join(md[["tier", "sigma_500K", "sigma_300K_low"]].rename(
        columns={"tier": "own_md_tier"}), on="immutable_id")
    report = report.join(windows, on="reduced_formula").join(joins, on="reduced_formula")
    # No clean cation environment (or no ionic states for the motif analysis): the
    # coordination check passed without testing anything.
    report["coordination_tested"] = np.isfinite(report["rarest_environment_count"])
    for energies in config["phase8"]["window"]["energies"]:
        low, high = report[f"reduction_V_{energies}"], report[f"oxidation_V_{energies}"]
        report[f"electrodes_in_window_{energies}"] = [
            ", ".join(name for name, (_, v) in panel.items() if lo - 1e-6 <= v <= hi + 1e-6)
            for lo, hi in zip(low, high)]
    report.to_csv(out_dir / "report.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_colwidth", 40):
        print(report.drop(columns=["index", "cluster", "icsd_polymorphs", "joins"]).round(3)
              .to_string(index=False))


def _dimension_one(task: tuple) -> dict:
    """Dimension of one relaxed structure's framework, the mobile ion removed."""
    import signal

    from pymatgen.analysis.dimensionality import get_dimensionality_larsen
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.core import Structure

    def alarm(signum, frame):
        raise TimeoutError

    index, path, mobile_ion = task
    structure = Structure.from_file(path)
    structure.remove_species([mobile_ion])
    if not len(structure):
        return {"index": index, "framework_dimension": None}
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(120)
    try:
        dimension = get_dimensionality_larsen(CrystalNN().get_bonded_structure(structure))
    finally:
        signal.alarm(0)
    return {"index": index, "framework_dimension": int(dimension)}


def run_mine_dimension(args, config: dict) -> None:
    """Framework dimension of the control and the survivors: what E_2D's line selects.

    A diagnostic run after ``mine-rank``. Anion-only frameworks are a class of their own;
    the rest are 0D (isolated units), 1D, 2D (layers) or 3D (CrystalNN bonds, Larsen's
    algorithm).
    """
    from wyckoff_transformer.transport.frameworks import ANION_LABEL

    root = campaign_dir(config, args.out)
    relaxed = relaxed_structures(root).set_index("index")
    selection = pd.read_parquet(root / "bench" / "pes_selection.parquet")
    control = set(selection.loc[selection["in_control"], "index"])
    screen = pd.read_parquet(root / "mine" / "screen.parquet")
    survivors = set(screen.loc[screen["phase2_survivor"], "index"])
    tasks = [(int(i), relaxed.loc[i, "relaxed_cif"], config["mobile_ion"])
             for i in sorted(control | survivors)]
    out = root / "mine" / "framework_dimension.jsonl"
    run_pool(tasks, _dimension_one, out, args.workers)
    table = pd.DataFrame(read_jsonl(out)).drop_duplicates("index", keep="last")
    table = table.drop(columns="error", errors="ignore").set_index("index")
    frames = pd.read_parquet(root / "bench" / "frameworks.parquet").set_index("index")
    anion_only = frames["framework_formula"].reindex(table.index).fillna("").str.fullmatch(
        rf"{ANION_LABEL}\d*")
    table["framework_class"] = np.where(
        anion_only, "anion only",
        table["framework_dimension"].map(lambda d: "?" if pd.isna(d) else f"{int(d)}D"))
    table["empty_E_2D"] = descriptor_values(root)["empty_E_2D"].reindex(table.index)
    table["in_control"] = table.index.isin(control)
    table["survivor"] = table.index.isin(survivors)
    table["hit"] = table.index.isin(pd.read_parquet(root / "mine" / "hits.parquet")["index"])
    write_parquet(table.reset_index(), out.with_suffix(".parquet"))
    in_control = table[table["in_control"]]
    line = in_control["empty_E_2D"].quantile(config["mining"]["rank"]["top_fraction"])
    summary = pd.concat({
        "control": in_control.groupby("framework_class")["empty_E_2D"].agg(
            ["size", "median", lambda e: (e <= line).mean()]).set_axis(
            ["n", "median_E_2D", "within_line"], axis=1),
        "survivors": table[table["survivor"]].groupby("framework_class")["hit"].agg(
            ["size", "sum", "mean"]).set_axis(["n", "hits", "hit_rate"], axis=1)}, axis=1)
    print(f"line: E_2D <= {line:.3f} eV")
    print(summary.round(3).to_string())


# --------------------------------------------------------------------------- #
# bench-evaluate
# --------------------------------------------------------------------------- #
#: Every descriptor the benchmark scores: (column, orientation, class).  The orientation
#: is fixed from the literature before any score is seen -- +1 when a larger value should
#: mean a better conductor -- and is never chosen from the data.
DESCRIPTORS = (
    # gene level: what the gene alone knows
    ("li_fraction", +1, "gene"),
    ("anion_electronegativity", -1, "gene"),         # a softer anion conducts better
    ("n_li_orbits", +1, "gene"),
    ("li_free_site_fraction", +1, "gene"),
    ("composition_lofo", +1, "gene"),                # fitted, leave-one-family-out
    # coarse structure
    ("corner_sharing_or_isolated", +1, "coarse"),    # Jun et al. 2022
    ("anion_lattice_bcc", +1, "coarse"),             # Wang et al. 2015
    ("volume_per_anion", +1, "coarse"),
    ("li_distortion_mean", +1, "coarse"),            # distorted Li sites, Jun et al.
    ("isolated_anion_fraction", +1, "coarse"),
    ("li_li_shortest_mean", -1, "coarse"),           # Sendek's LLSD has a negative weight
    ("li_anion_shortest_mean", +1, "coarse"),        # Sendek's LASD has a positive weight
    # energy model
    ("bvse_E_1D", -1, "energy"), ("bvse_E_2D", -1, "energy"), ("bvse_E_3D", -1, "energy"),
    ("xi", +1, "energy"), ("fv_connected", +1, "energy"), ("fv_disconnected", +1, "energy"),
    ("mpe", -1, "energy"),
    ("empty_E_1D", -1, "energy"), ("empty_E_2D", -1, "energy"), ("empty_E_3D", -1, "energy"),
    # Xi refitted on ORB: Maevskiy's thresholds were tuned on M3GNet, whose softer PES
    # gives larger free volumes (xi-check); a leave-one-family-out logistic fit on
    # log10 FVc, log10 FVd and MPE is the ORB analogue
    ("xi_lofo", +1, "combination"),
    # combination of the structure descriptors, fitted leave-one-family-out
    ("structure_lofo", +1, "combination"),
)
STRUCTURE_FEATURES = [name for name, _, kind in DESCRIPTORS if kind in ("coarse", "energy")
                      and name not in ("xi", "fv_connected", "fv_disconnected", "mpe")]


def benchmark_table(root: Path, config: dict) -> pd.DataFrame:
    """One row per relaxed structure with every descriptor and both kinds of label."""
    from wyckoff_transformer.transport.sources import gaussian_label_weights

    table = relaxed_structures(root)
    frames = pd.read_parquet(root / "bench" / "frameworks.parquet")
    table = table.merge(frames[["index", "framework_formula", "obelix_groups", "obelix_label",
                                "obelix_family"]], on="index", how="left")
    describe = pd.read_parquet(root / "bench" / "describe.parquet")
    table = table.merge(describe.drop(columns=[c for c in ("error",) if c in describe]),
                        on="index", how="left")
    table["anion_lattice_bcc"] = (table["anion_lattice"] == "bcc").astype(float)
    for kind in ("frozen", "empty"):
        path = root / "bench" / f"pes_{kind}.parquet"
        if path.is_file():
            pes = pd.read_parquet(path).drop(columns=["device", "error"], errors="ignore")
            table = table.merge(pes, on="index", how="left")
    selection = root / "bench" / "pes_selection.parquet"
    if selection.is_file():
        table = table.merge(pd.read_parquet(selection), on="index", how="left")
    # Kahle labels at the configured temperature, Gaussian-smoothed as Maevskiy et al. do.
    spec = config["benchmark"]["labels"]
    kahle = pd.read_parquet(root / "data" / "kahle2020.parquet")
    kahle = kahle[kahle["temperature"] == spec["kahle_temperature_k"]]
    kahle = kahle.groupby("structure_uuid").agg(
        kahle_sigma=("sigma", "mean"), kahle_sigma_sem=("sigma_sem", "max")).reset_index()
    w_pos, w_neg = gaussian_label_weights(kahle["kahle_sigma"].to_numpy(),
                                          kahle["kahle_sigma_sem"].fillna(0).to_numpy(),
                                          spec["kahle_fast_s_cm"], spec["kahle_poor_s_cm"])
    kahle["kahle_w_pos"], kahle["kahle_w_neg"] = w_pos, w_neg
    table = table.merge(kahle.rename(columns={"structure_uuid": "structure_id"}),
                        on="structure_id", how="left")
    table["obelix_group"] = table["obelix_groups"].map(
        lambda groups: groups[0] if isinstance(groups, (list, np.ndarray)) and len(groups) else None)
    return table


def _composition_features(table: pd.DataFrame) -> np.ndarray:
    from pymatgen.core import Composition

    compositions = [Composition(formula).fractional_composition
                    for formula in table["reduced_formula"]]
    elements = sorted({str(e) for c in compositions for e in c.elements})
    return np.array([[c[e] for e in elements] for c in compositions])


def _lofo_scores(features: np.ndarray, table: pd.DataFrame, exp_mask: np.ndarray,
                 families: np.ndarray, y: np.ndarray) -> dict:
    """Leave-one-family-out logistic regression on the experimental labels.

    Returns ``{family: scores of every row from the model fitted without that family}``
    and ``"all"``: the model fitted on every labelled row, used for the Kahle set.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fit_predict(train):
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        model.fit(features[train], y[train])
        return model.decision_function(features)

    # A barrier with no percolation below the cutoff is infinite; it is the worst value,
    # not a missing one.
    features = np.where(np.isposinf(features), 10.0, np.where(np.isneginf(features), -10.0,
                                                              features))
    out = {"all": fit_predict(exp_mask)}
    for family in sorted({f for f in families[exp_mask] if f}):
        train = exp_mask & (families != family)
        if len(set(y[train])) == 2:
            out[family] = fit_predict(train)
    return out


def _percentiles(values: np.ndarray, population: np.ndarray, reference: np.ndarray | None
                 ) -> np.ndarray:
    """Percentile rank of every row among the palette population, 0 = best.

    With *reference* (a random sample of the population, for a descriptor computed on a
    subset only), a row's percentile is the share of the reference that beats it.
    """
    from wyckoff_transformer.transport.scores import percentile_ranks

    if reference is None:
        out = np.full(len(values), np.nan)
        out[population] = percentile_ranks(values[population])
        return out
    ref = values[reference]
    ref = ref[np.isfinite(ref)]
    filled = np.where(np.isfinite(values), values, -np.inf)
    better = (ref[None, :] > filled[:, None]).mean(axis=1)
    ties = (ref[None, :] == filled[:, None]).mean(axis=1)
    return better + ties / 2


def run_bench_evaluate(args, config: dict) -> None:
    """Score every descriptor on both label sets and read gate G1."""
    from wyckoff_transformer.transport.scores import (
        bootstrap_partial_auc,
        random_partial_auc,
        rediscovery,
    )

    root = campaign_dir(config, args.out)
    metrics = config["benchmark"]["metrics"]
    gates = config["gates"]
    table = benchmark_table(root, config)
    palette = (table["source"] == "lemat").to_numpy()
    exp_mask = palette & table["obelix_label"].isin(["fast", "poor"]).to_numpy()
    y_exp = (table["obelix_label"] == "fast").to_numpy().astype(int)
    # A labelled compound whose group has no family name is its own held-out "family",
    # so that the fitted models never score a compound they were trained on.
    held_out = np.array([(f if isinstance(f, str) and f else f"group-{g}") if m else None
                         for f, g, m in zip(table["obelix_family"], table["obelix_group"],
                                            exp_mask)], dtype=object)
    families = np.array([f if m and isinstance(f, str) and f else None
                         for f, m in zip(table["obelix_family"], exp_mask)], dtype=object)
    groups = table["obelix_group"].to_numpy(dtype=object)
    kahle_mask = table["kahle_w_pos"].notna().to_numpy() & (table["source"] == "kahle2020").to_numpy()
    control = table["in_control"].fillna(False).to_numpy(dtype=bool) \
        if "in_control" in table else None
    fast_members = exp_mask & (y_exp == 1)
    rng_seed = config["benchmark"]["xi"]["seed"]

    fitted = {
        "composition_lofo": _lofo_scores(_composition_features(table), table, exp_mask,
                                         held_out, y_exp),
        "xi_lofo": _lofo_scores(
            np.column_stack([np.log10(np.clip(table.get(c, pd.Series(np.nan, index=table.index))
                                              .to_numpy(dtype=float), 1e-6, None))
                             for c in ("fv_connected", "fv_disconnected")]
                            + [table.get("mpe", pd.Series(np.nan, index=table.index))
                               .clip(upper=10).to_numpy(dtype=float)]),
            table, exp_mask & table.get("fv_connected", pd.Series(np.nan, index=table.index))
            .notna().to_numpy(), held_out, y_exp),
        "structure_lofo": _lofo_scores(
            table[[c for c in STRUCTURE_FEATURES if c in table]].to_numpy(dtype=float),
            table, exp_mask, held_out, y_exp),
    }

    # Descriptors computed only on the PES selection: a fitted one is defined only where
    # its inputs were measured, and every one of them is ranked against the control.
    subset_names = {"xi", "fv_connected", "fv_disconnected", "mpe", "empty_E_1D",
                    "empty_E_2D", "empty_E_3D", "xi_lofo"}
    scanned = table.get("fv_connected", pd.Series(np.nan, index=table.index)).notna().to_numpy()
    for scores_by_model in fitted["xi_lofo"].values():
        scores_by_model[~scanned] = np.nan

    def cluster_bootstrap(score, mask, by_group: bool):
        """pAUC with a bootstrap over OBELiX framework groups, not over compounds.

        With *by_group* every framework group weighs the same, its weight shared among
        the palette compounds it labels: the label is a framework's, and one OBELiX
        entry must not count 550 times because 550 compounds share its framework.
        """
        from wyckoff_transformer.transport.scores import partial_auc

        keys = np.array([g if g is not None else f"row{i}" for i, g in enumerate(groups)],
                        dtype=object)[mask]
        s, yy = score[mask], y_exp[mask]
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        weight = 1.0 / counts[inverse] if by_group else np.ones(len(keys))

        def pauc(pick):
            return partial_auc(s[pick], yy[pick] * weight[pick], (1 - yy[pick]) * weight[pick],
                               metrics["pauc_max_fpr"])

        point = pauc(np.arange(len(keys)))
        unique = np.unique(keys)
        rng = np.random.default_rng(rng_seed)
        index_of = {k: np.flatnonzero(keys == k) for k in unique}
        draws = []
        for _ in range(metrics["bootstrap"]):
            pick = np.concatenate([index_of[k] for k in rng.choice(unique, len(unique))])
            if 0 < yy[pick].sum() < len(pick):
                draws.append(pauc(pick))
        low, high = np.quantile(draws, [0.025, 0.975])
        return point, float(low), float(high)

    rows = []
    for name, sign, kind in DESCRIPTORS:
        if name in fitted:
            lofo = fitted[name]
            score = np.full(len(table), np.nan)
            for i in np.flatnonzero(exp_mask):
                score[i] = lofo[held_out[i]][i] if held_out[i] in lofo else np.nan
            kahle_score = lofo["all"]
        elif name in table:
            values = table[name].to_numpy(dtype=float)
            values = np.where(np.isposinf(values), 1e6, np.where(np.isneginf(values), -1e6,
                                                                 values))
            score = sign * values
            kahle_score = score
        else:
            continue
        row = {"descriptor": name, "class": kind, "orientation": sign}
        # Experimental set.
        m = exp_mask & np.isfinite(score)
        row["exp_n"], row["exp_fast"] = int(m.sum()), int(y_exp[m].sum())
        row["exp_fast_groups"] = len({groups[i] for i in np.flatnonzero(m & (y_exp == 1))})
        row["exp_poor_groups"] = len({groups[i] for i in np.flatnonzero(m & (y_exp == 0))})
        if m.any() and 0 < y_exp[m].sum() < m.sum():
            row["exp_pauc"], row["exp_low"], row["exp_high"] = cluster_bootstrap(score, m, True)
            (row["exp_pauc_compound"], row["exp_low_compound"],
             row["exp_high_compound"]) = cluster_bootstrap(score, m, False)
        # Kahle set.
        k = kahle_mask & np.isfinite(kahle_score)
        row["kahle_n"] = int(k.sum())
        if k.sum() > 5:
            row["kahle_pauc"], row["kahle_low"], row["kahle_high"] = bootstrap_partial_auc(
                kahle_score[k], table["kahle_w_pos"].to_numpy()[k],
                table["kahle_w_neg"].to_numpy()[k], metrics["pauc_max_fpr"],
                n=metrics["bootstrap"], seed=rng_seed)
        # Rediscovery among the palette compounds, each family scored by the model that
        # never saw it when the descriptor is fitted.
        subset_only = name in subset_names
        reference = control if subset_only else None
        if name in fitted:
            per_family = {}
            for family in {f for f in families[fast_members] if f}:
                model_scores = fitted[name].get(family, fitted[name]["all"])
                percent = _percentiles(model_scores, palette, reference)
                if subset_only:
                    percent[~np.isfinite(model_scores)] = np.nan
                per_family[family] = percent
            percent = np.full(len(table), np.nan)
            for i in np.flatnonzero(fast_members):
                if families[i] in per_family:
                    percent[i] = per_family[families[i]][i]
        else:
            percent = _percentiles(score, palette, reference)
            if subset_only:
                # Members never scanned have no rank; they are not ranked last.
                percent[~np.isfinite(score)] = np.nan
        result = rediscovery(percent, families, fast_members & np.isfinite(percent),
                             metrics["rediscovery_min_members"], metrics["rediscovery_top_fraction"])
        row["rediscovered"] = result["rediscovered"]
        row["eligible_families"] = result["eligible"]
        row["rediscovery_fraction"] = result["fraction"]
        row["family_medians"] = json.dumps({f: round(v["median_percentile"], 3)
                                            for f, v in result["families"].items()})
        random = random_partial_auc(metrics["pauc_max_fpr"])
        row["g1"] = bool(row.get("exp_low", -1) > random and row.get("kahle_low", -1) > random
                         and result["eligible"] > 0
                         and result["fraction"] >= gates["g1_min_family_fraction"])
        rows.append(row)
    report = pd.DataFrame(rows)
    out = root / "bench" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / "descriptors.csv", index=False)
    write_parquet(table.drop(columns=["obelix_groups"]), out / "table.parquet")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(report.drop(columns=["family_medians"]).round(3).to_string(index=False))
    print(f"\nG1 passes: {report.loc[report['g1'], 'descriptor'].tolist()}")


# --------------------------------------------------------------------------- #
# upload
# --------------------------------------------------------------------------- #
#: Run-directory files each W&B artifact carries, relative to the campaign directory.
UPLOADS = {
    "benchmark": ["bench/structures.parquet", "bench/relaxed.parquet",
                  "bench/relax/relaxations.csv", "bench/frameworks.parquet",
                  "bench/obelix_frameworks.parquet", "bench/obelix_groups.parquet",
                  "bench/describe.parquet", "bench/pes_selection.parquet",
                  "bench/pes_frozen.parquet", "bench/pes_empty.parquet", "bench/evaluation",
                  "data"],
    # Not mine/coordination/shards: LeMat's own CIFs, rebuilt by coordination-reference.
    "mining": ["mine/gaps.parquet", "mine/screen.jsonl", "mine/screen.parquet",
               "mine/pes_empty.jsonl", "mine/pes_empty.parquet", "mine/hits.parquet",
               "mine/clusters.csv", "mine/framework_dimension.jsonl",
               "mine/framework_dimension.parquet", "mine/coordination/environment_frequency.csv",
               "mine/coordination/sharing_frequency.csv", "mine/coordination/report.md",
               "mine/coordination/results", "mine/md"],
    "gates": ["g1b/controls.parquet", "g1b/matches.csv", "g1b_v2/controls.parquet",
              "g1b_v2/matches.csv", "g1c/references.parquet", "g1c/references.csv",
              "g1c/md.jsonl", "g1c/md_analysis.csv", "g1c/trajectories", "checks"],
    # Not phase8/entries.parquet: LeMat-Bulk-fixed's own rows, rebuilt by phase8-entries.
    "phase8": ["phase8/mp2020.jsonl", "phase8/u_scheme_check.csv", "phase8/window.jsonl",
               "phase8/windows.csv", "phase8/interfaces.csv", "phase8/e_hull.csv",
               "phase8/joins.csv", "phase8/structures.csv", "phase8/prototypes.csv",
               "phase8/report.csv", "phase8/md"],
}


def run_upload(args, config: dict) -> None:
    """Log the benchmark, the gate calibrations and the mining to W&B, one artifact each.

    The relaxed CIFs are packed into ``bench/relaxed.parquet`` first, so that the
    artifact does not depend on the run directory's trial layout.
    """
    import subprocess

    import wandb

    from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
    from wyckoff_transformer.paths import wandb_dir

    root = campaign_dir(config, args.out)
    relaxed = relaxed_structures(root)[["index", "immutable_id", "relaxed_cif",
                                        "orb_energy_per_atom"]]
    relaxed["relaxed_cif"] = [Path(path).read_text(encoding="utf-8")
                              for path in relaxed["relaxed_cif"]]
    write_parquet(relaxed, root / "bench" / "relaxed.parquet")

    def git(*command):
        return subprocess.run(["git", *command], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()

    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, id=config["campaign_id"],
                     name=config["campaign_id"], job_type="benchmark", resume="allow",
                     dir=wandb_dir(), config={**config, "code_commit": git("rev-parse", "HEAD"),
                                              "code_dirty": bool(git("status", "--porcelain"))})
    descriptors = pd.read_csv(root / "bench" / "evaluation" / "descriptors.csv")
    references = pd.read_csv(root / "g1c" / "references.csv")
    run.log({"descriptors": wandb.Table(dataframe=descriptors),
             "md_references": wandb.Table(dataframe=references)})
    run.summary["g1_passing_descriptors"] = descriptors.loc[descriptors["g1"],
                                                            "descriptor"].tolist()
    candidates_path = root / "mine" / "md" / "candidates.csv"
    if candidates_path.is_file():
        candidates = pd.read_csv(candidates_path).rename(columns={"Unnamed: 0": "key"})
        run.log({"phase2_md_candidates": wandb.Table(dataframe=candidates)})
        for tier in ("electrolyte", "coating"):
            run.summary[f"phase2_{tier}_tier"] = candidates.loc[candidates["tier"] == tier,
                                                                "formula"].tolist()
    for table_name, relative in (("phase8_report", "phase8/report.csv"),
                                 ("phase8_interfaces", "phase8/interfaces.csv")):
        if (root / relative).is_file():
            run.log({table_name: wandb.Table(dataframe=pd.read_csv(root / relative))})
    for name, paths in UPLOADS.items():
        artifact = wandb.Artifact(f"{config['campaign_id']}-{name}", type=f"se_{name}")
        for relative in paths:
            path = root / relative
            if path.is_dir():
                artifact.add_dir(str(path), name=relative)
            elif path.is_file():
                artifact.add_file(str(path), name=relative)
            else:
                logger.warning("upload: %s is missing", path)
        run.log_artifact(artifact)
    run.finish()
    print(f"W&B run {WANDB_ENTITY}/{WANDB_PROJECT}/{config['campaign_id']}")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("data", "bench-set", "bench-relax", "bench-frameworks",
                                          "bench-describe", "bench-pes", "bench-evaluate", "xi-check",
                                          "matcher-calibrate", "md-calibrate", "bvse-check",
                                          "mine-gap", "coordination-reference", "mine-screen",
                                          "mine-pes", "mine-rank", "mine-md", "mine-dimension",
                                          "mine-md-members", "phase8-entries", "phase8-window",
                                          "phase8-check", "phase8-prototypes",
                                          "phase8-report", "upload"))
    parser.add_argument("--workers", type=int, default=16,
                        help="CPU stages: worker processes (hera has 24 physical cores).")
    parser.add_argument("--pes-kinds", type=str, default="frozen",
                        help="bench-pes: frozen or empty framework.")
    parser.add_argument("--pes-all", action="store_true",
                        help="bench-pes: every relaxed structure, not the selection.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="mine-gap: the torch device of the band-gap regressor.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Campaign directory; default <runs store>/<campaign_id>.")
    parser.add_argument("--limit", type=int, default=None,
                        help="bench-relax, bench-pes: only N structures (a pilot); "
                             "mine-rank: clusters printed; mine-md: clusters run (20).")
    parser.add_argument("--devices", type=str, default=None,
                        help="bench-relax: override the config's devices.")
    parser.add_argument("--workers-per-device", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true",
                        help="bench-relax: re-run structures whose relaxation failed.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    {"data": run_data, "bench-set": run_bench_set, "bench-relax": run_bench_relax,
     "bench-frameworks": run_bench_frameworks, "bench-describe": run_bench_describe,
     "bench-pes": run_bench_pes, "bench-evaluate": run_bench_evaluate,
     "xi-check": run_xi_check, "matcher-calibrate": run_matcher_calibrate,
     "md-calibrate": run_md_calibrate, "bvse-check": run_bvse_check,
     "mine-gap": run_mine_gap, "coordination-reference": run_coordination_reference,
     "mine-screen": run_mine_screen, "mine-pes": run_mine_pes, "mine-rank": run_mine_rank,
     "mine-md": run_mine_md, "mine-dimension": run_mine_dimension,
     "mine-md-members": run_mine_md_members, "phase8-entries": run_phase8_entries,
     "phase8-window": run_phase8_window, "phase8-check": run_phase8_check,
     "phase8-prototypes": run_phase8_prototypes, "phase8-report": run_phase8_report,
     "upload": run_upload}[args.stage](args, config)


if __name__ == "__main__":
    main()
