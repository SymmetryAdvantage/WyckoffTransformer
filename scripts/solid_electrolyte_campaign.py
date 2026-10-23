"""Generative search for new Li solid electrolytes in the known SE families.

The pipeline of docs/archive/101_Generative_Pipeline_for_Di.pdf up to the hull, with the
families, models and thresholds in one campaign yaml
(yamls/campaigns/solid_electrolyte_families_hera_20261022.yaml); the write-up is
docs/solid_electrolyte_campaign.md.

    PY=.venv/bin/python S=scripts/solid_electrolyte_campaign.py
    $PY $S systems                      # the expanded system list, nothing run
    $PY $S generate --device cuda:0     # draws, dedup, gene novelty, OBELiX, energy score
    $PY $S bandgap --device cuda:0      # predicted band gap of every candidate gene
    $PY $S select                       # filters, energy-ranked top-K, one gene file per run
    $PY $S relax                        # wide NEP89 -> ORB protocol; ORB also on iapetus
    $PY $S report --wandb               # candidates, tiers, funnel; W&B artifacts

Every stage resumes: ``generate`` and ``bandgap`` skip systems already written, and
``relax`` skips protocol stages that already finished (the protocol itself resumes a
stage at trial granularity).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import queue
import shutil
import subprocess
import sys
import threading
import time
from functools import cache
from itertools import product
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from kappagen_gene_campaign import dedup_pool, generate_pool, system_seed

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
from wyckoff_transformer.cli.csp import load_trainer
from wyckoff_transformer.cli.gene_screen import (
    check_regressor_field,
    predict_gene_scalars,
    score_genes,
)
from wyckoff_transformer.evaluation.gene_hash import GeneKeyTable
from wyckoff_transformer.evaluation.protocol import GeneFingerprinter
from wyckoff_transformer.formula_energy import hf_reference
from wyckoff_transformer.formula_energy.screen import HullLookup
from wyckoff_transformer.paths import cache_root, data_path, runs_root
from wyckoff_transformer.trainer import WyckoffTrainer

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "yamls/campaigns/solid_electrolyte_families_hera_20261022.yaml"
PROTOCOL = Path(sys.executable).parent / "wyformer-protocol"
CPU_STAGES = ("screen", "generate", "prescreen")
GPU_STAGES = ("relax", "score")
#: Protocol outputs worth keeping; the extxyz draws and per-trial logs are not.
PROTOCOL_UPLOADS = (
    "wyckoff_genes.json.gz", "screen.json", "pyxtal.csv", "prescreen.csv",
    "prescreen_selection.csv", "relaxations.csv", "structures.csv",
    "structures_fixed_symmetry.csv", "funnel.json", "manifest.json",
    "cifs", "cifs_fixed_symmetry",
)


# --------------------------------------------------------------------------- #
# Campaign definition
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def canonical_system(elements) -> str:
    return "-".join(sorted(set(elements)))


def expand_systems(config: dict) -> pd.DataFrame:
    """One row per (family, system), with the protocol run the system goes to.

    Families keep their yaml order, which is also the order the runs are relaxed in.
    """
    rows = []
    for family in config["families"]:
        split_by = family.get("split_by") or []
        for template in family["templates"]:
            slots = [slot if isinstance(slot, list) else [slot] for slot in template]
            for elements in product(*slots):
                if len(set(elements)) != len(elements):
                    raise ValueError(f"{family['name']}: repeated element in {elements}")
                system = canonical_system(elements)
                split = next((element for element in split_by if element in elements), None)
                run = f"{family['name']}_{split}" if split else family["name"]
                rows.append({"family": family["name"], "system": system, "run": run})
    frame = pd.DataFrame(rows)
    duplicated = frame["system"].duplicated()
    if duplicated.any():
        raise ValueError(f"Systems listed twice: {sorted(frame.loc[duplicated, 'system'])}")
    return frame


def campaign_dir(config: dict, out: Path | None) -> Path:
    return out if out is not None else runs_root() / config["campaign_id"]


def system_file(root: Path, system: str) -> Path:
    return root / "systems" / f"{system}.parquet"


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.rename(path)


# --------------------------------------------------------------------------- #
# Known solid electrolytes
# --------------------------------------------------------------------------- #
def reduced_formula(formula: str) -> str | None:
    from pymatgen.core import Composition

    try:
        return Composition(formula).reduced_formula
    except (ValueError, KeyError, TypeError):
        return None


def load_obelix(config: dict) -> tuple[dict[str, list[str]], set[str]]:
    """OBELiX's known SEs: reduced formula -> entry ids, and their chemical systems.

    Fractional (doped) compositions stay fractional, so they only ever match a
    generated composition at exactly that stoichiometry -- the parent compounds
    listed with integer formulas are what this catches.
    """
    from pymatgen.core import Composition

    spec = config["screens"]["obelix"]
    path = data_path(spec["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        raise ValueError(f"{path} has sha256 {digest}, the campaign pins {spec['sha256']}")
    table = pd.read_csv(path)
    formulas: dict[str, list[str]] = {}
    systems: set[str] = set()
    for entry_id, composition in zip(table["ID"], table["Reduced Composition"]):
        try:
            parsed = Composition(composition)
        except (ValueError, KeyError, TypeError):
            logger.warning("OBELiX %s: cannot parse %r", entry_id, composition)
            continue
        formulas.setdefault(parsed.reduced_formula, []).append(str(entry_id))
        systems.add(canonical_system(element.symbol for element in parsed.elements))
    logger.info("OBELiX: %d entries, %d distinct reduced formulas, %d systems",
                len(table), len(formulas), len(systems))
    return formulas, systems


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def load_hf(repo: str, revision: str, device: torch.device) -> WyckoffTrainer:
    trainer = WyckoffTrainer.from_huggingface(repo, device=device, revision=revision)
    if hasattr(trainer.model, "_orig_mod"):
        trainer.model = trainer.model._orig_mod
    trainer.model.eval()
    return trainer


def run_generate(args, config: dict) -> None:
    root = campaign_dir(config, args.out)
    systems = expand_systems(config)
    if args.only:
        systems = systems[systems["system"].isin(args.only.split(","))]
    todo = [s for s in systems["system"] if not system_file(root, s).exists()]
    logger.info("%d systems, %d left to generate", len(systems), len(todo))
    if not todo:
        return

    generation = config["generation"]
    energy = config["screens"]["energy"]
    generator = load_hf(generation["hf_model"], generation["hf_revision"], args.device)
    regressor = load_hf(energy["hf_model"], energy["hf_revision"], args.device)
    repo, revision = hf_reference.parse_spec(energy["hull_reference"])
    check_regressor_field(
        regressor, hf_reference.reference_field(repo, revision),
        f"hull reference {repo}@{revision}", energy["allow_incompatible_energy"])
    lookup = HullLookup(hf_reference.load_hf_reference(repo, revision))
    screens = config["screens"]
    key_table = GeneKeyTable.load(
        cache_root() / screens["novelty_reference_cache"] / screens["novelty_key_table"])
    logger.info("Novelty reference: %d gene keys, %s", len(key_table), key_table.meta)
    obelix_formulas, obelix_systems = load_obelix(config)
    fingerprinter = GeneFingerprinter()
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for system in todo:
        t0 = time.time()
        genes = generate_pool(generator, system, generation["n_draws"], generation["batch"],
                              system_seed(system, generation["seed"]))
        with gzip.open(raw_dir / f"{system}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(genes, handle)
        pool = dedup_pool(genes, fingerprinter)
        pool["chemsys"] = pool["gene"].map(lambda g: canonical_system(g["species"]))
        pool["exact_system"] = pool["chemsys"] == system
        keys = torch.tensor(pool[["key_low", "key_high"]].to_numpy(), dtype=torch.int64)
        pool["gene_novel"] = (~key_table.contains(keys)).numpy() if len(pool) else []
        t1 = time.time()

        # Only genes of exactly the requested system can be selected; scoring the rest
        # would cost hull lookups for systems nobody asked for.
        exact = pool[pool["exact_system"]]
        scored = []
        for start in range(0, len(exact), args.score_batch):
            part = exact.iloc[start:start + args.score_batch]
            result = score_genes(list(part["gene"]), regressor, None,
                                 augmentation_samples=energy["augmentation_samples"],
                                 hull_lookup=lookup).sort_index()
            result.index = part.index
            scored.append(result)
        columns = ["formula", "predicted_formation_energy", "hull_energy", "score", "reason"]
        scored = pd.concat(scored) if scored else pd.DataFrame(columns=columns)
        pool = pool.join(scored[columns].rename(columns={
            "score": "predicted_e_hull", "reason": "energy_reason"}))
        pool["reduced_formula"] = pool["formula"].map(
            lambda f: reduced_formula(f) if isinstance(f, str) else None)
        pool["obelix_ids"] = pool["reduced_formula"].map(
            lambda f: ",".join(obelix_formulas.get(f, [])) if f else "")
        pool["obelix_known"] = pool["obelix_ids"] != ""
        pool["obelix_same_chemsys"] = pool["chemsys"].isin(obelix_systems)
        pool["requested_system"] = system
        pool["spacegroup_number"] = pool["gene"].map(lambda g: int(g["group"]))
        pool["n_atoms"] = pool["gene"].map(lambda g: int(sum(g["numIons"])))
        pool["gene"] = pool["gene"].map(json.dumps)
        pool.attrs = {}
        write_parquet(pool, system_file(root, system))
        logger.info(
            "%s: %d valid of %d draws, %d unique, %d exact system, %d novel of those, "
            "%d OBELiX-known, %d energy-scored (generate+novelty %.0f s, score %.0f s)",
            system, len(genes), generation["n_draws"], len(pool), int(pool["exact_system"].sum()),
            int((pool["exact_system"] & pool["gene_novel"]).sum()),
            int(pool["obelix_known"].sum()), int(pool["predicted_e_hull"].notna().sum()),
            t1 - t0, time.time() - t1)
        with open(root / "systems" / f"{system}.json", "w", encoding="utf-8") as handle:
            json.dump({"system": system, "n_draws": generation["n_draws"],
                       "n_valid_draws": len(genes)}, handle)


# --------------------------------------------------------------------------- #
# bandgap
# --------------------------------------------------------------------------- #
def load_band_gap_regressor(wandb_run: str, device: torch.device):
    regressor = load_trainer(device=device, wandb_run=wandb_run,
                             wandb_entity=WANDB_ENTITY, wandb_project=WANDB_PROJECT)
    if regressor.target_name != "band_gap":
        raise ValueError(f"{wandb_run} predicts {regressor.target_name}, not band_gap")
    if hasattr(regressor.model, "_orig_mod"):
        regressor.model = regressor.model._orig_mod
    regressor.model.eval()
    return regressor


def run_bandgap(args, config: dict) -> None:
    """Predicted band gaps of every exact-system gene, one column set per model.

    ``screens.band_gap.wandb_run`` writes ``band_gap_pred``, the one the screen reads;
    each ``also_predict`` entry writes ``band_gap_pred_<name>``, recorded and reported
    but not screened on.
    """
    root = campaign_dir(config, args.out)
    spec = config["screens"]["band_gap"]
    models = [("", spec["wandb_run"])] + [
        (f"_{extra['name']}", extra["wandb_run"]) for extra in spec.get("also_predict", [])]
    systems = list(expand_systems(config)["system"])
    for path in (system_file(root, system) for system in systems):
        if not path.exists():
            raise SystemExit(f"{path.stem} has not been generated yet")
    for suffix, wandb_run in models:
        column = f"band_gap_pred{suffix}"
        todo = [system for system in systems
                if column not in pq.read_schema(system_file(root, system)).names]
        logger.info("%s (%s): %d systems left", column, wandb_run, len(todo))
        if not todo:
            continue
        regressor = load_band_gap_regressor(wandb_run, args.device)
        for system in todo:
            path = system_file(root, system)
            pool = pd.read_parquet(path)
            pool[column] = float("nan")
            pool[f"band_gap_std{suffix}"] = float("nan")
            pool[f"band_gap_reason{suffix}"] = None
            exact = pool[pool["exact_system"]]
            for start in range(0, len(exact), args.score_batch):
                part = exact.iloc[start:start + args.score_batch]
                with torch.no_grad():
                    predicted = predict_gene_scalars(
                        [json.loads(gene) for gene in part["gene"]], regressor,
                        augmentation_samples=spec["augmentation_samples"])
                predicted.index = part.index
                pool.loc[part.index, column] = predicted["prediction"]
                pool.loc[part.index, f"band_gap_std{suffix}"] = predicted["prediction_std"]
                pool.loc[part.index, f"band_gap_reason{suffix}"] = predicted["reason"]
            write_parquet(pool, path)
            gapped = (pool[column] >= spec["min_ev"]).sum()
            logger.info("%s %s: %d of %d exact-system genes predicted >= %.2f eV", column,
                        system, gapped, len(exact), spec["min_ev"])


# --------------------------------------------------------------------------- #
# select
# --------------------------------------------------------------------------- #
@cache
def _oxidation_states(formula: str) -> str | None:
    from pymatgen.core import Composition

    composition = Composition(formula).reduced_composition
    elements = list(composition.elements)
    for states in product(*(element.common_oxidation_states for element in elements)):
        if abs(sum(composition[e] * s for e, s in zip(elements, states))) > 1e-6:
            continue
        anions = [e.X for e, s in zip(elements, states) if s < 0]
        cations = [e.X for e, s in zip(elements, states) if s > 0]
        if anions and cations and min(anions) < max(cations):
            continue
        return " ".join(f"{e}{s:+d}" for e, s in sorted(zip(elements, states),
                                                         key=lambda pair: pair[0].symbol))
    return None


def oxidation_states(formula) -> str | None:
    """One common oxidation state per element summing to zero, e.g. ``Cl-1 Li+1 Y+3``.

    Every anion must be more electronegative than every cation. This is the charge
    balance of an ionic insulator, which every family's prototypes satisfy except the
    mixed-valence-hydrogen closo-borates and amide-borohydrides (Li2B12H12, Li4BN3H10).
    The energy regressor ranks impossible stoichiometries such as Li46P4S7Cl4 below the
    hull, so without this screen they fill the top-K; structure validity would reject
    them anyway, after their relaxation was paid for. The ICSD-prior search of
    ``coordination_motifs.guess_oxidation_states`` explodes combinatorially on
    formulas like Li13P6S39Cl8 (27 s) and admits Zintl-type states (Y+1, S-1); one state
    per element from ``common_oxidation_states`` is a few hundred products at most.
    """
    return _oxidation_states(formula) if isinstance(formula, str) else None


def select_system(pool: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """The system's relaxation candidates, best predicted hull distance first."""
    min_gap = config["screens"]["band_gap"]["min_ev"]
    selection = config["selection"]
    pool = pool.assign(oxidation_states=pool["reduced_formula"].map(oxidation_states))
    masks = {
        "unique": pd.Series(True, index=pool.index),
        "exact_system": pool["exact_system"],
        "gene_novel": pool["gene_novel"],
        "not_obelix": ~pool["obelix_known"],
        "charge_balanced": pool["oxidation_states"].notna(),
        "band_gap": pool["band_gap_pred"] >= min_gap,
        "energy_scored": pool["predicted_e_hull"].notna(),
        "max_atoms": pool["n_atoms"] <= selection.get("max_atoms_per_gene", float("inf")),
    }
    counts, keep = {}, pd.Series(True, index=pool.index)
    for name, mask in masks.items():
        keep &= mask.fillna(False).astype(bool)
        counts[name] = int(keep.sum())
        counts[f"{name}_draws"] = int(pool.loc[keep, "count"].sum())
    passed = pool[keep].sort_values("predicted_e_hull", kind="stable")
    passed = passed[passed.groupby("reduced_formula").cumcount() < selection["max_per_formula"]]
    passed = passed.head(selection["top_k_per_system"])
    # The wide arm costs about the same per atom whatever the chemistry, so a system
    # whose best genes are 300-atom cells gets fewer of them, not 400.
    budget = selection.get("atom_budget_per_system")
    if budget is not None:
        passed = passed[passed["n_atoms"].cumsum() <= budget]
    counts["selected"] = len(passed)
    counts["selected_atoms"] = int(passed["n_atoms"].sum())
    return passed, counts


def run_select(args, config: dict) -> None:
    root = campaign_dir(config, args.out)
    systems = expand_systems(config)
    funnel, chosen_all = [], []
    for row in systems.itertuples():
        pool = pd.read_parquet(system_file(root, row.system))
        if "band_gap_pred" not in pool.columns:
            raise SystemExit(f"{row.system}: run the bandgap stage first")
        with open(root / "systems" / f"{row.system}.json", encoding="utf-8") as handle:
            meta = json.load(handle)
        chosen, counts = select_system(pool, config)
        funnel.append({"family": row.family, "system": row.system, "run": row.run,
                       "draws": meta["n_draws"], "valid_draws": meta["n_valid_draws"],
                       **counts})
        chosen = chosen.assign(family=row.family, run=row.run)
        chosen_all.append(chosen)
    funnel = pd.DataFrame(funnel)
    selected_dir = root / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    funnel.to_csv(selected_dir / "funnel_by_system.csv", index=False)
    by_family = funnel.drop(columns=["system", "run"]).groupby("family", sort=False).sum()
    by_family.insert(0, "systems", funnel.groupby("family", sort=False).size())
    by_family.to_csv(selected_dir / "funnel_by_family.csv")
    selected = pd.concat(chosen_all, ignore_index=True)
    # The protocol indexes genes by their position in the run's gene file.
    selected["protocol_index"] = selected.groupby("run", sort=False).cumcount()
    write_parquet(selected, selected_dir / "selected.parquet")
    for run, part in selected.groupby("run", sort=False):
        run_dir = root / "protocol" / run
        run_dir.mkdir(parents=True, exist_ok=True)
        genes = [json.loads(gene) for gene in part.sort_values("protocol_index")["gene"]]
        target = run_dir / "wyckoff_genes.json.gz"
        if target.exists():
            with gzip.open(target, "rt", encoding="utf-8") as handle:
                if json.load(handle) != genes:
                    raise SystemExit(f"{target} exists with other genes; the protocol run in "
                                     "it would no longer match its inputs")
            continue
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            json.dump(genes, handle)
    print(by_family.to_string())
    print(f"{len(selected)} genes selected over {selected['run'].nunique()} protocol runs")


# --------------------------------------------------------------------------- #
# relax
# --------------------------------------------------------------------------- #
def run_order(config: dict) -> list[str]:
    return list(dict.fromkeys(expand_systems(config)["run"]))


def protocol_command(run_dir: Path, stage: str, config: dict, limit: int | None) -> list[str]:
    spec = config["protocol"]
    command = [
        str(PROTOCOL), str(run_dir / "wyckoff_genes.json.gz"),
        "--output-dir", str(run_dir), "--stage", stage,
        "--mlip", spec["mlip"],
        "--trial-multiplier", str(spec["trial_multiplier"]),
        "--pyxtal-cores", str(spec["pyxtal_cores"]),
        "--pyxtal-timeout", str(spec["pyxtal_timeout"]),
        "--relax-from", "prescreen",
    ]
    if stage in GPU_STAGES:
        command += ["--devices", spec["devices"],
                    "--workers-per-device", str(spec["workers_per_device"])]
    else:
        command += ["--cores", str(spec["prescreen_cores"])]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


def call_logged(command: list[str], run_dir: Path, stage: str) -> int:
    with open(run_dir / "logs" / f"{stage}.log", "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        return subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)


def mark_done(run_dir: Path, stage: str, started: float, where: str = "hera") -> None:
    (run_dir / "logs" / f"{stage}.done").write_text(
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {time.time() - started:.0f} s on {where}\n")
    logger.info("[%s] %s done on %s in %.0f s", run_dir.name, stage, where,
                time.time() - started)


def run_stages(run_dir: Path, stages, config: dict, limit: int | None) -> bool:
    """Run *stages* of one protocol run here, in order; False at the first failure."""
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    for stage in stages:
        if (run_dir / "logs" / f"{stage}.done").exists():
            continue
        command = protocol_command(run_dir, stage, config, limit)
        logger.info("[%s] %s", run_dir.name, " ".join(command[2:]))
        started = time.time()
        code = call_logged(command, run_dir, stage)
        if code != 0:
            logger.error("[%s] %s failed with exit code %d after %.0f s; see %s",
                         run_dir.name, stage, code, time.time() - started,
                         run_dir / "logs" / f"{stage}.log")
            return False
        mark_done(run_dir, stage, started)
    return True


def remote_relax(run_dir: Path, config: dict, remote: dict, limit: int | None) -> bool:
    """The relax stage of one run on another host, its outputs synced back.

    The run directory goes over whole, relaxes there through the host's own launcher
    and comes back; ``score`` then runs here. Score reads only the aggregated
    ``structures*.csv`` and ``cifs*/`` relax wrote, never the remote paths
    ``relaxations.csv`` records, so the round trip is transparent to it.
    """
    if (run_dir / "logs" / "relax.done").exists():
        return True
    host = remote["host"]
    # Named after the local campaign directory, so a pilot (--out) has its own.
    remote_dir = f"{remote['runs_root']}/{run_dir.parent.parent.name}/protocol/{run_dir.name}"
    spec = config["protocol"]
    started = time.time()
    steps = [
        ["ssh", host, f"mkdir -p {remote_dir}"],
        ["rsync", "-a", "--exclude", "logs/", f"{run_dir}/", f"{host}:{remote_dir}/"],
        ["ssh", host, " ".join([
            f"cd {remote['checkout']} &&", remote["launcher"], "wyformer-protocol",
            f"{remote_dir}/wyckoff_genes.json.gz", "--output-dir", remote_dir,
            "--stage", "relax", "--relax-from", "prescreen", "--mlip", spec["mlip"],
            "--trial-multiplier", str(spec["trial_multiplier"]),
            "--devices", remote["devices"],
            "--workers-per-device", str(remote["workers_per_device"]),
        ] + (["--limit", str(limit)] if limit is not None else []))],
        ["rsync", "-a", "--exclude", "logs/", f"{host}:{remote_dir}/", f"{run_dir}/"],
    ]
    for command in steps:
        logger.info("[%s] %s", run_dir.name, " ".join(command))
        code = call_logged(command, run_dir, "relax")
        if code != 0:
            logger.error("[%s] relax on %s failed at `%s` (exit %d); see %s", run_dir.name,
                         host, " ".join(command[:2]), code, run_dir / "logs" / "relax.log")
            return False
    mark_done(run_dir, "relax", started, where=host)
    return True


def fits(size: tuple[int, int], remote: dict) -> bool:
    """Whether a run of *size* (genes, atoms) is within a remote host's caps."""
    n_genes, n_atoms = size
    return (n_genes <= remote.get("max_genes", n_genes)
            and n_atoms <= remote.get("max_atoms", n_atoms))


class RelaxQueue:
    """Runs whose CPU stages are done, for whichever relax lane is free first.

    A remote lane takes only runs that fit its caps and gives back a run it could
    not relax (the host down, the SSH agent gone with the session that forwarded
    it), which then goes to the local lane only. The local lane therefore stays
    until no remote run is in flight, since one may yet come back. It takes the
    runs no remote may have first, and never one an idle remote lane is waiting for.
    """

    def __init__(self, remotes: list[dict] = ()):
        self._runs: list[tuple[str, tuple[int, int], bool]] = []
        self._remotes = list(remotes)
        self._idle_remotes: list[dict] = []
        self._closed = False
        self._remote_in_flight = 0
        self._cond = threading.Condition()

    def put(self, run: str, size: tuple[int, int], remote_ok: bool = True) -> None:
        with self._cond:
            self._runs.append((run, size, remote_ok))
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def take(self, remote: dict | None = None) -> str | None:
        """The first run this lane may take, waiting for one; None when there will be none.

        *remote* is the remote host this lane relaxes on, or None for the local lane.
        """
        def wanted_by(remotes):
            return [remote_ok and any(fits(size, r) for r in remotes)
                    for _, size, remote_ok in self._runs]

        with self._cond:
            while True:
                if remote is not None:
                    allowed = wanted_by([remote])
                else:
                    allowed = [not wanted for wanted in wanted_by(self._remotes)]
                    if not any(allowed):
                        allowed = [not wanted for wanted in wanted_by(self._idle_remotes)]
                if any(allowed):
                    run = self._runs.pop(allowed.index(True))[0]
                    self._remote_in_flight += remote is not None
                    self._cond.notify_all()
                    return run
                if self._closed and (remote is not None
                                     or not (self._runs or self._remote_in_flight)):
                    return None
                if remote is not None:
                    self._idle_remotes.append(remote)
                self._cond.wait()
                if remote is not None:
                    self._idle_remotes.remove(remote)

    def remote_done(self, run: str, size: tuple[int, int], succeeded: bool) -> None:
        """A remote lane's verdict on *run*; a failed one goes to the local lane."""
        with self._cond:
            self._remote_in_flight -= 1
            if not succeeded:
                self._runs.append((run, size, False))
            self._cond.notify_all()


class CpuQueue:
    """Protocol runs waiting for their CPU stages, shared by the CPU lanes.

    The first lane takes them in campaign order. Any other lane takes the smallest
    run a remote host may relax first, so that host has work from the start rather
    than once the first lane reaches the small runs.
    """

    def __init__(self, runs: list[tuple[str, tuple[int, int]]], remotes: list[dict] = ()):
        self._runs = list(runs)
        self._remotes = list(remotes)
        self._lock = threading.Lock()

    def take(self, lane: int) -> tuple[str, tuple[int, int]] | None:
        with self._lock:
            if not self._runs:
                return None
            position = 0
            if lane > 0:
                for_remote = [(size[1], i) for i, (_, size) in enumerate(self._runs)
                              if any(fits(size, remote) for remote in self._remotes)]
                if for_remote:
                    position = min(for_remote)[1]
            return self._runs.pop(position)


def run_size(run_dir: Path, limit: int | None) -> tuple[int, int]:
    """Genes and conventional-cell atoms of one protocol run."""
    with gzip.open(run_dir / "wyckoff_genes.json.gz", "rt", encoding="utf-8") as handle:
        genes = json.load(handle)[:limit]
    return len(genes), sum(sum(gene["numIons"]) for gene in genes)


def run_relax(args, config: dict) -> None:
    """Lanes: CPU stages here; relax here and on the remote hosts; score here.

    The CPU stages of run i+1 overlap the relaxations of run i, and scoring -- CPU
    and memory, no GPU -- has a lane of its own so it never holds a GPU idle. A
    remote host takes only runs within its ``max_genes`` and ``max_atoms``, so that a
    slow one is not left finishing a large run alone at the end.

    ``--cpu-lanes``, ``--cpu-cores`` and ``--remote-max-atoms`` change where and how
    fast the stages run, not what they compute: every trial's seed is fixed by the
    trial itself.
    """
    root = campaign_dir(config, args.out)
    runs = run_order(config)
    if args.runs:
        wanted = args.runs.split(",")
        runs = [run for run in runs if run in wanted]
    if args.cpu_cores is not None:
        config["protocol"]["pyxtal_cores"] = config["protocol"]["prescreen_cores"] = args.cpu_cores
    remotes = [] if args.no_remote else config["protocol"].get("remote_relax", [])
    if args.remote_max_atoms is not None:
        remotes = [{**remote, "max_atoms": args.remote_max_atoms} for remote in remotes]
    failed = []
    ready = []
    for run in runs:
        if (root / "protocol" / run / "wyckoff_genes.json.gz").exists():
            ready.append((run, run_size(root / "protocol" / run, args.limit)))
        else:
            logger.error("[%s] no gene file; run the select stage first", run)
            failed.append(run)
    cpu_queue = CpuQueue(ready, remotes)
    relax_queue = RelaxQueue(remotes)
    score_queue: queue.Queue = queue.Queue()

    def cpu_lane(lane: int):
        while (item := cpu_queue.take(lane)) is not None:
            run, size = item
            if run_stages(root / "protocol" / run, CPU_STAGES, config, args.limit):
                relax_queue.put(run, size)
            else:
                failed.append(run)

    def local_relax_lane():
        while (run := relax_queue.take()) is not None:
            if run_stages(root / "protocol" / run, ("relax",), config, args.limit):
                score_queue.put(run)
            else:
                failed.append(run)

    def remote_relax_lane(remote: dict):
        while (run := relax_queue.take(remote)) is not None:
            run_dir = root / "protocol" / run
            succeeded = remote_relax(run_dir, config, remote, args.limit)
            if succeeded:
                score_queue.put(run)
            else:
                logger.warning("[%s] handing the run to the local relax lane", run)
            relax_queue.remote_done(run, run_size(run_dir, args.limit), succeeded)

    def score_lane():
        while (run := score_queue.get()) is not None:
            if not run_stages(root / "protocol" / run, ("score",), config, args.limit):
                failed.append(run)

    relax_lanes = [threading.Thread(target=local_relax_lane)] + [
        threading.Thread(target=remote_relax_lane, args=(remote,)) for remote in remotes]
    cpu_lanes = [threading.Thread(target=cpu_lane, args=(lane,))
                 for lane in range(args.cpu_lanes)]
    scorer = threading.Thread(target=score_lane)
    for lane in [*cpu_lanes, scorer, *relax_lanes]:
        lane.start()
    for lane in cpu_lanes:
        lane.join()
    relax_queue.close()
    for lane in relax_lanes:
        lane.join()
    score_queue.put(None)
    scorer.join()
    if failed:
        raise SystemExit(f"Protocol runs with a failed stage: {failed}")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def relaxed_symmetry(cif: Path) -> tuple[str | None, int | None, str | None]:
    """The relaxed structure's reduced formula and space group, from its CIF."""
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        structure = Structure.from_file(cif)
        analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
        return (structure.composition.reduced_formula, analyzer.get_space_group_number(),
                analyzer.get_space_group_symbol())
    except Exception as error:  # noqa: BLE001 -- a bad CIF is reported, not fatal
        logger.warning("%s: %s", cif, error)
        return None, None, None


def run_report(args, config: dict) -> None:
    root = campaign_dir(config, args.out)
    selected = pd.read_parquet(root / "selected" / "selected.parquet")
    obelix_formulas, _ = load_obelix(config)
    tiers = config["report"]["e_hull_tiers"]
    headline = config["report"]["headline_e_hull"]
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    scored, missing = [], []
    for run in run_order(config):
        run_dir = root / "protocol" / run
        if not (run_dir / "logs" / "score.done").exists():
            missing.append(run)
            continue
        genes = selected[selected["run"] == run].set_index("protocol_index")
        for track, name, cifs in (("free", "structures.csv", "cifs"),
                                  ("fixed_symmetry", "structures_fixed_symmetry.csv",
                                   "cifs_fixed_symmetry")):
            frame = pd.read_csv(run_dir / name, index_col="index")
            gaps = [column for column in genes.columns if column.startswith("band_gap_pred")]
            frame = frame.join(genes[["family", "requested_system", "reduced_formula",
                                      "spacegroup_number", "n_atoms", *gaps,
                                      "predicted_e_hull", "oxidation_states", "count",
                                      "gene"]],
                               rsuffix="_gene")
            frame["run"], frame["track"] = run, track
            frame["cif_path"] = [str(run_dir / cifs / f"{index}.cif") for index in frame.index]
            scored.append(frame.reset_index())
    if missing:
        logger.warning("Runs not scored yet, left out: %s", missing)
    if not scored:
        raise SystemExit("No protocol run has been scored yet")
    scored = pd.concat(scored, ignore_index=True)
    scored.to_parquet(report_dir / "scored.parquet", index=False)

    good = (scored["has_structure"].fillna(False).astype(bool)
            & scored["valid_structure"].fillna(False).astype(bool)
            & scored["unique_structure"].fillna(False).astype(bool)
            & scored["novel_structure"].fillna(False).astype(bool))
    candidates = scored[good & (scored["e_above_hull"] <= max(tiers))].copy()
    symmetry = candidates["cif_path"].map(lambda path: relaxed_symmetry(Path(path)))
    candidates["relaxed_formula"] = symmetry.map(lambda s: s[0])
    candidates["relaxed_spacegroup_number"] = symmetry.map(lambda s: s[1])
    candidates["relaxed_spacegroup"] = symmetry.map(lambda s: s[2])
    candidates["obelix_ids_relaxed"] = candidates["relaxed_formula"].map(
        lambda f: ",".join(obelix_formulas.get(f, [])) if f else "")
    candidates = candidates[candidates["obelix_ids_relaxed"] == ""]
    candidates = candidates.sort_values(["track", "e_above_hull"])
    candidates.to_csv(report_dir / "candidates.csv", index=False)
    cif_dir = report_dir / "cifs"
    if cif_dir.exists():
        shutil.rmtree(cif_dir)
    for row in candidates[candidates["track"] == "free"].itertuples():
        target = (cif_dir / row.family / row.requested_system /
                  f"{row.relaxed_formula}_sg{row.relaxed_spacegroup_number}_{row.run}_{row.index}.cif")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row.cif_path, target)

    summary = []
    for (track, family), part in scored[good].groupby(["track", "family"], sort=False):
        known = candidates[(candidates["track"] == track) & (candidates["family"] == family)]
        row = {"track": track, "family": family, "relaxed": int(scored[
            (scored["track"] == track) & (scored["family"] == family)
            & scored["has_structure"].fillna(False).astype(bool)].shape[0]),
            "valid_unique_novel": len(part)}
        for tier in tiers:
            row[f"e_hull<={tier}"] = int((known["e_above_hull"] <= tier).sum())
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(report_dir / "tiers_by_family.csv", index=False)
    print(summary.to_string(index=False))
    free = candidates[(candidates["track"] == "free") & (candidates["e_above_hull"] <= headline)]
    print(f"{len(free)} novel, not-known-SE structures at e_above_hull <= {headline} (free track)")
    if args.wandb:
        upload(root, config, summary, len(free))


def upload(root: Path, config: dict, summary: pd.DataFrame, n_headline: int) -> None:
    import wandb

    from wyckoff_transformer.paths import wandb_dir

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                            text=True, check=True).stdout.strip()
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, id=config["campaign_id"],
                     name=config["campaign_id"], job_type="campaign", resume="allow",
                     dir=wandb_dir(), config={**config, "code_commit": commit})
    by_family = pd.read_csv(root / "selected" / "funnel_by_family.csv")
    run.log({"funnel_by_family": wandb.Table(dataframe=by_family),
             "tiers_by_family": wandb.Table(dataframe=summary)})
    run.summary["candidates_headline"] = n_headline
    screens = wandb.Artifact(f"{config['campaign_id']}-screens", type="campaign_screens")
    screens.add_dir(str(root / "systems"), name="systems")
    screens.add_dir(str(root / "selected"), name="selected")
    run.log_artifact(screens)
    protocol = wandb.Artifact(f"{config['campaign_id']}-protocol", type="protocol_eval")
    for run_dir in sorted((root / "protocol").iterdir()):
        for name in PROTOCOL_UPLOADS:
            path = run_dir / name
            if path.is_dir():
                protocol.add_dir(str(path), name=f"{run_dir.name}/{name}")
            elif path.is_file():
                protocol.add_file(str(path), name=f"{run_dir.name}/{name}")
    run.log_artifact(protocol)
    report = wandb.Artifact(f"{config['campaign_id']}-candidates", type="candidates")
    report.add_dir(str(root / "report"))
    run.log_artifact(report)
    run.finish()


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("systems", "generate", "bandgap", "select",
                                          "relax", "report"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None,
                        help="Campaign directory; the runs store's <campaign_id> by default.")
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    parser.add_argument("--only", type=str, default=None,
                        help="generate: comma-separated systems, e.g. Li-Y-Cl,Cl-Li-P-S.")
    parser.add_argument("--score-batch", type=int, default=5000)
    parser.add_argument("--runs", type=str, default=None,
                        help="relax: comma-separated protocol runs, in campaign order.")
    parser.add_argument("--limit", type=int, default=None,
                        help="relax: pass --limit to the protocol, for a smoke test.")
    parser.add_argument("--no-remote", action="store_true",
                        help="relax: ignore protocol.remote_relax and relax on this host only.")
    parser.add_argument("--cpu-lanes", type=int, default=1,
                        help="relax: protocol runs whose CPU stages run at once.")
    parser.add_argument("--cpu-cores", type=int, default=None,
                        help="relax: cores per CPU lane, in place of the config's "
                             "pyxtal_cores and prescreen_cores.")
    parser.add_argument("--remote-max-atoms", type=int, default=None,
                        help="relax: the largest run, in atoms, a remote host may take.")
    parser.add_argument("--wandb", action="store_true", help="report: log to W&B.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = load_config(args.config)

    if args.stage == "systems":
        systems = expand_systems(config)
        print(systems.groupby(["family", "run"], sort=False).size().to_string())
        print(f"{len(systems)} systems in {systems['run'].nunique()} protocol runs")
        return
    {"generate": run_generate, "bandgap": run_bandgap, "select": run_select,
     "relax": run_relax, "report": run_report}[args.stage](args, config)


if __name__ == "__main__":
    main()
