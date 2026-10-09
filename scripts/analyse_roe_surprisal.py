"""Replay the rules of engagement on one fully relaxed pool: lookup against surprisal.

`scripts/platforms/aspire2a/roe_surprisal_in_pbs.sh` draws a pool, scores every
gene's predicted e_hull and surprisal, and relaxes every unique gene. Each mode
is then a *selection* from that pool, and every arm is scored against the same
relaxation outcomes, so the comparison carries no relaxation noise between arms:

==================  ==========================================================
broadside           the first B draws, duplicates included (a duplicate slot
                    is a miss, charged at its representative's cost)
fire-discipline     the first B genes that pass the screen
fire-control        the B lowest predicted e_hull among the genes that pass
                    the screen, over the whole pool
==================  ==========================================================

with the screen filled two ways:

``lookup``     uniqueness and novelty by fingerprint against the reference,
               the protocol's own screen (``protocol/screen.json``);
``surprisal``  uniqueness against the pool itself, which needs no reference,
               and a band of the generator's own surprisal (pool quantiles).

Ablations: ``unique`` (self-uniqueness alone), ``energy`` (fire-control with
self-uniqueness and no novelty lever at all) and ``lookup+plausible`` (the
lookup, then the least surprising 30% of the novel genes, then energy -- the
best arm of the e9ywwsie report).

MetaSUN and SUN are computed by the protocol's own
:func:`~wyckoff_transformer.evaluation.protocol.funnel_structure_metrics`, so
they mean exactly what they mean in docs/de_novo_ranking_protocol.md. That
definition groups structures by their sampled gene before matching, so two
genes that relax to the same structure both count; ``strict`` columns collapse
such pairs with StructureMatcher, as a check.

    python scripts/analyse_roe_surprisal.py $WYFORMER_RUNS/roe_surprisal/cfg
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from wyckoff_transformer.evaluation.protocol import (
    METASTABLE_THRESHOLD,
    STABLE_THRESHOLD,
    GeneFingerprinter,
    GeneScreen,
    funnel_structure_metrics,
    load_genes,
    read_screen,
)
from wyckoff_transformer.formula_energy.prefilter import wilson_interval

logger = logging.getLogger(__name__)

BUDGETS = (250, 500, 1000, 2000)
#: Fixed before this pool was drawn: the best band of "band, then rank by
#: energy" in docs/archive/e9ywwsie_generative_novelty_report.md.
BAND = (0.40, 0.80)
#: The lookup-plus-likelihood arm of the same report.
LEAST_SURPRISING = 0.30
#: The band surface is read on this grid, as in that report.
BAND_LOWER = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
BAND_UPPER = (0.6, 0.7, 0.8, 0.9, 1.0)
SPLITS = 40
STRUCTURE_COLUMNS = (
    "has_structure", "valid_structure", "unique_structure", "novel_structure",
    "e_above_hull", "n_atoms", "n_trials", "relax_seconds", "pyxtal_seconds", "formula")


# --------------------------------------------------------------------------- #
# The pool
# --------------------------------------------------------------------------- #
def _as_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() == "true"
                      if not isinstance(value, (bool, np.bool_)) else bool(value))


#: The protocol's two readouts: the structure kept after the symmetry release and
#: the rattle, and the one relaxed with the gene's symmetry held, which is still
#: the gene's Wyckoff representation at the end.
TRACKS = {
    "free": ("structures.csv", "cifs"),
    "fixed_symmetry": ("structures_fixed_symmetry.csv", "cifs_fixed_symmetry"),
}


def load_pool(root: Path, variants: Sequence[str], track: str = "free",
              ) -> tuple[pd.DataFrame, pd.DataFrame, GeneScreen]:
    """One row per sampled gene, and one per relaxed representative.

    Each gene is mapped to its representative -- the first gene of its
    fingerprint class, which is what the protocol relaxed -- with the protocol's
    own fingerprinter, and the result is checked against ``screen.json``.
    """
    genes = load_genes(root / "pool" / "wyckoff_genes.json.gz")
    screen = read_screen(root / "protocol" / "screen.json")
    if screen.n_sampled != len(genes):
        raise ValueError(f"screen.json covers {screen.n_sampled} genes, the pool {len(genes)}")

    fingerprinter = GeneFingerprinter()
    first: dict = {}
    representative = np.full(len(genes), -1, dtype=np.int64)
    for index, gene in enumerate(genes):
        try:
            fingerprint = fingerprinter.fingerprint(gene)
        except Exception:  # noqa: BLE001 - mirrors screen_genes
            continue
        representative[index] = first.setdefault(fingerprint, index)
    counts = pd.Series(representative[representative >= 0]).value_counts().to_dict()
    if counts != screen.counts:
        raise ValueError("The fingerprint classes disagree with the protocol's screen.json")

    frame = pd.DataFrame({"representative": representative},
                         index=pd.RangeIndex(len(genes), name="index"))
    frame["is_representative"] = frame["representative"] == frame.index
    novel = set(screen.novel)
    frame["gene_novel"] = frame["representative"].isin(novel)

    energy = pd.read_csv(root / "scores" / "gene_screen.csv", index_col=0).sort_index()
    frame["predicted_e_hull"] = energy["score"].reindex(frame.index).astype(float)
    for variant in variants:
        scored = pd.read_csv(root / "scores" / f"gene_novelty_{variant}.csv", index_col=0)
        frame[f"surprisal_{variant}"] = scored["surprisal"].reindex(frame.index).astype(float)

    structures = pd.read_csv(root / "protocol" / TRACKS[track][0], index_col=0)
    for column in ("has_structure", "valid_structure", "unique_structure", "novel_structure",
                   "relaxed_fingerprint_changed"):
        structures[column] = _as_bool(structures[column])
    energies = structures["e_above_hull"].astype(float)
    alive = (structures["has_structure"] & structures["valid_structure"]
             & structures["unique_structure"])
    structures["metastable"] = alive & (energies <= METASTABLE_THRESHOLD)
    structures["metasun"] = structures["metastable"] & structures["novel_structure"]
    structures["sun"] = (alive & (energies <= STABLE_THRESHOLD)
                         & structures["novel_structure"])

    # Outcomes belong to the representative and are read through it; a gene with
    # no representative (-1) reads an all-missing row.
    outcome = structures.reindex(frame["representative"].to_numpy())
    outcome.index = frame.index
    for column in ("metastable", "metasun", "sun", "novel_structure", "e_above_hull",
                   "n_atoms", "n_trials", "relax_seconds", "relaxed_fingerprint_changed"):
        frame[column] = outcome[column].to_numpy()
    for column in ("metastable", "metasun", "sun", "novel_structure",
                   "relaxed_fingerprint_changed"):
        frame[column] = frame[column].fillna(False).astype(bool)
    frame.loc[frame["representative"] < 0, ["metastable", "metasun", "sun"]] = False
    return frame, structures, screen


# --------------------------------------------------------------------------- #
# Strict uniqueness
# --------------------------------------------------------------------------- #
def _match_group(paths: list) -> list:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    loaded = [(index, Structure.from_file(path)) for index, path in paths]
    matcher = StructureMatcher()
    pairs = []
    for a in range(len(loaded)):
        for b in range(a + 1, len(loaded)):
            if matcher.fit(loaded[a][1], loaded[b][1]):
                pairs.append((loaded[a][0], loaded[b][0]))
    return pairs


def strict_classes(root: Path, structures: pd.DataFrame, workers: int,
                   track: str = "free") -> pd.Series:
    """A class id per MetaSUN structure: those StructureMatcher calls the same.

    Only structures with the same reduced formula can match, so only those are
    compared, and only MetaSUN ones, since those are the hits the classes are
    for. The classes are the connected components of the match graph.
    """
    from pymatgen.core import Composition

    hits = structures.index[structures["metasun"]]
    groups: dict = defaultdict(list)
    for index in hits:
        path = root / "protocol" / TRACKS[track][1] / f"{index}.cif"
        if path.is_file():
            formula = Composition(structures.at[index, "formula"]).reduced_formula
            groups[formula].append((int(index), str(path)))
    jobs = [members for members in groups.values() if len(members) > 1]
    logger.info("Strict uniqueness: %d MetaSUN structures, %d formulas with >1 member",
                len(hits), len(jobs))
    parent = {int(index): int(index) for index in hits}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    with Pool(workers) as pool:
        for pairs in pool.imap_unordered(_match_group, jobs, chunksize=1):
            for a, b in pairs:
                root_a, root_b = find(a), find(b)
                if root_a != root_b:
                    parent[max(root_a, root_b)] = min(root_a, root_b)
    return pd.Series({index: find(index) for index in parent}, name="strict_class")


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def band_mask(values: pd.Series, eligible: pd.Series, lower: float, upper: float) -> pd.Series:
    """Genes in the (lower, upper] quantile band of *values*, quantiles over *eligible*."""
    scored = values[eligible & values.notna()]
    quantile = scored.rank(pct=True)
    keep = (quantile > lower) & (quantile <= upper)
    return keep.reindex(values.index, fill_value=False)


def first(mask: pd.Series, budget: int) -> tuple[list, int]:
    """Discipline: the first *budget* genes the mask passes, and the draws that took."""
    passed = mask.index[mask.to_numpy()]
    slots = list(passed[:budget])
    consumed = int(slots[-1]) + 1 if len(slots) == budget else int(len(mask))
    return slots, consumed


def ranked(mask: pd.Series, score: pd.Series, budget: int) -> tuple[list, int]:
    """Control: the *budget* lowest scores among what the mask passes, whole pool drawn.

    A gene the regressor could not score ranks last, as ``PredictedHullFilter``
    ranks an undecided gene.
    """
    candidates = score[mask].fillna(np.inf).sort_values(kind="stable")
    return list(candidates.index[:budget]), int(len(mask))


def build_arms(frame: pd.DataFrame, budget: int, variants: Sequence[str],
               band=BAND) -> dict:
    """Arm name -> (slots, draws consumed, mode, lever)."""
    unique = frame["is_representative"]
    lookup = unique & frame["gene_novel"]
    energy = frame["predicted_e_hull"]
    arms = {
        "broadside": (list(frame.index[:budget]), budget, "broadside", "none"),
        "fire-discipline/unique": (*first(unique, budget), "fire-discipline", "unique"),
        "fire-discipline/lookup": (*first(lookup, budget), "fire-discipline", "lookup"),
        "fire-control/energy": (*ranked(unique, energy, budget), "fire-control", "unique"),
        "fire-control/lookup": (*ranked(lookup, energy, budget), "fire-control", "lookup"),
    }
    for variant in variants:
        surprisal = frame[f"surprisal_{variant}"]
        in_band = unique & band_mask(surprisal, unique, *band)
        plausible = lookup & band_mask(surprisal, lookup, 0.0, LEAST_SURPRISING)
        arms[f"fire-discipline/surprisal-{variant}"] = (
            *first(in_band, budget), "fire-discipline", f"surprisal-{variant}")
        arms[f"fire-control/surprisal-{variant}"] = (
            *ranked(in_band, energy, budget), "fire-control", f"surprisal-{variant}")
        arms[f"fire-control/lookup+plausible-{variant}"] = (
            *ranked(plausible, energy, budget), "fire-control", f"lookup+plausible-{variant}")
    return arms


def evaluate(frame: pd.DataFrame, structures: pd.DataFrame, slots: list, consumed: int,
             budget: int, strict: Optional[pd.Series]) -> dict:
    """One arm, per reconstruction slot, per draw and per relaxation hour.

    A slot holding a gene whose representative an earlier slot already holds is
    a miss: the protocol relaxed the representative once, and a campaign at
    these rules would have spent the slot on it again.
    """
    representatives = frame.loc[slots, "representative"]
    distinct = sorted({int(r) for r in representatives if r >= 0})
    screen = GeneScreen(n_sampled=budget, valid=distinct, counts={r: 1 for r in distinct})
    metrics = funnel_structure_metrics(screen, structures.loc[structures.index.intersection(distinct)])
    metasun = int(metrics["metastable_among_novel"] or 0)
    sun = int(metrics["stable_among_novel"] or 0)
    low, high = wilson_interval(metasun, budget)
    sun_low, sun_high = wilson_interval(sun, budget)

    chosen = frame.loc[slots]
    seconds = float(chosen["relax_seconds"].fillna(0).sum())
    trials = int(chosen["n_trials"].fillna(0).sum())
    novel_structures = int(metrics["novel_structure"] or 0)
    result = {
        "budget": budget,
        "filled": len(slots),
        "duplicate_slots": int(len(slots) - len(distinct)),
        "draws": consumed,
        "gene_novel_slots": int(chosen["gene_novel"].sum()),
        "valid_structure": metrics["valid_structure"],
        "novel_structure": novel_structures,
        "metastable": metrics["metastable"],
        "metasun": metasun,
        "metasun_rate": metasun / budget,
        "metasun_ci": [low, high],
        "sun": sun,
        "sun_rate": sun / budget,
        "sun_ci": [sun_low, sun_high],
        "metastable_given_novel": (metasun / novel_structures) if novel_structures else None,
        "metasun_per_draw": metasun / consumed if consumed else None,
        "trials": trials,
        "relax_hours": seconds / 3600,
        "metasun_per_relax_hour": metasun / (seconds / 3600) if seconds else None,
        "trials_per_metasun": trials / metasun if metasun else None,
        "mean_atoms": float(chosen["n_atoms"].mean()),
        "median_predicted_e_hull": float(chosen["predicted_e_hull"].median()),
    }
    # Whether a hit is still the gene's Wyckoff representation once relaxed: a
    # relaxed fingerprint that differs from the gene's means relaxation moved the
    # structure off the orbit set the screen judged.
    hits = structures.reindex(distinct)
    hit_mask = hits["metasun"].fillna(False).astype(bool)
    changed = hits["relaxed_fingerprint_changed"].fillna(False).astype(bool)
    result["metasun_fingerprint_kept"] = int((hit_mask & ~changed).sum())
    result["metasun_fingerprint_changed"] = int((hit_mask & changed).sum())
    result["fingerprint_changed_slots"] = int(changed.sum())
    if strict is not None:
        hits = [r for r in distinct if bool(structures.at[r, "metasun"])] \
            if distinct else []
        classes = {strict.get(r, r) for r in hits}
        result["metasun_strict"] = len(classes)
        result["metasun_strict_rate"] = len(classes) / budget
    return result


def fisher(a: dict, b: dict, key: str = "metasun") -> float:
    from scipy.stats import fisher_exact

    table = [[a[key], a["budget"] - a[key]], [b[key], b["budget"] - b[key]]]
    return float(fisher_exact(table)[1])


# --------------------------------------------------------------------------- #
# The band, and what the surprisal knows
# --------------------------------------------------------------------------- #
def _auc(score: pd.Series, label: pd.Series) -> Optional[float]:
    usable = score.notna() & label.notna()
    values = score[usable].to_numpy(dtype=float)
    positive = label[usable].to_numpy(dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    ranks = pd.Series(values).rank().to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def estimator_quality(frame: pd.DataFrame, variants: Sequence[str]) -> dict:
    from scipy.stats import spearmanr

    reps = frame[frame["is_representative"]]
    relaxed = reps[reps["e_above_hull"].notna()]
    report = {"predicted_e_hull": {
        "auc_metastable": _auc(-relaxed["predicted_e_hull"], relaxed["metastable"]),
        "auc_metasun": _auc(-relaxed["predicted_e_hull"], relaxed["metasun"]),
        "spearman_e_above_hull": float(spearmanr(
            relaxed["predicted_e_hull"], relaxed["e_above_hull"], nan_policy="omit")[0]),
    }}
    for variant in variants:
        column = f"surprisal_{variant}"
        novel = relaxed[relaxed["gene_novel"]]
        bins = relaxed[relaxed[column].notna()].copy()
        bins["decile"] = pd.qcut(bins[column], 10, labels=False, duplicates="drop")
        report[column] = {
            "auc_gene_novel": _auc(reps[column], reps["gene_novel"]),
            "auc_novel_structure": _auc(relaxed[column], relaxed["novel_structure"]),
            "auc_metasun": _auc(relaxed[column], relaxed["metasun"]),
            "auc_low_surprisal_metasun_given_gene_novel": _auc(-novel[column], novel["metasun"]),
            "auc_low_surprisal_metastable_given_gene_novel": _auc(
                -novel[column], novel["metastable"]),
            "spearman_e_above_hull": float(spearmanr(
                relaxed[column], relaxed["e_above_hull"], nan_policy="omit")[0]),
            "spearman_predicted_e_hull": float(spearmanr(
                relaxed[column], relaxed["predicted_e_hull"], nan_policy="omit")[0]),
            "unscored_representatives": int(reps[column].isna().sum()),
            "deciles": [
                {"decile": int(d), "n": int(len(g)),
                 "median_surprisal": float(g[column].median()),
                 "median_e_above_hull": float(g["e_above_hull"].median()),
                 "gene_novel": float(g["gene_novel"].mean()),
                 "novel_structure": float(g["novel_structure"].mean()),
                 "metastable": float(g["metastable"].mean()),
                 "metasun": float(g["metasun"].mean()),
                 "sun": float(g["sun"].mean())}
                for d, g in bins.groupby("decile")],
        }
    if len(variants) > 1:
        a, b = (f"surprisal_{v}" for v in variants[:2])
        report["spearman_between_variants"] = float(spearmanr(
            reps[a], reps[b], nan_policy="omit")[0])
    return report


def band_surface(frame: pd.DataFrame, variant: str, budget: int) -> dict:
    """MetaSUN per slot over the band grid, for both modes, in sample."""
    unique = frame["is_representative"]
    surprisal = frame[f"surprisal_{variant}"]
    surface = {"fire-discipline": {}, "fire-control": {}}
    for lower in BAND_LOWER:
        for upper in BAND_UPPER:
            if upper <= lower:
                continue
            in_band = unique & band_mask(surprisal, unique, lower, upper)
            key = f"{lower:.1f}-{upper:.1f}"
            for mode, (slots, _) in (
                    ("fire-discipline", first(in_band, budget)),
                    ("fire-control", ranked(in_band, frame["predicted_e_hull"], budget))):
                hits = int(frame.loc[slots, "metasun"][frame.loc[slots, "is_representative"]].sum())
                surface[mode][key] = hits / budget
    return surface


def split_half(frame: pd.DataFrame, variant: str, budget: int, seed: int = 0) -> dict:
    """The band chosen on one half of the pool, spent on the other.

    The quantiles are recomputed inside each half, so neither half sees the
    other's surprisal distribution.
    """
    rng = np.random.default_rng(seed)
    reps = frame[frame["is_representative"]]
    half_budget = budget // 2
    held = {"fire-discipline": [], "fire-control": []}
    preregistered = {"fire-discipline": [], "fire-control": []}
    chosen = {"fire-discipline": [], "fire-control": []}

    def rate(part: pd.DataFrame, lower: float, upper: float, mode: str) -> float:
        surprisal = part[f"surprisal_{variant}"]
        everyone = pd.Series(True, index=part.index)
        in_band = band_mask(surprisal, everyone, lower, upper)
        if mode == "fire-discipline":
            slots, _ = first(in_band, half_budget)
        else:
            slots, _ = ranked(in_band, part["predicted_e_hull"], half_budget)
        return float(part.loc[slots, "metasun"].sum()) / half_budget

    grid = [(lo, up) for lo in BAND_LOWER for up in BAND_UPPER if up > lo]
    for _ in range(SPLITS):
        mask = rng.random(len(reps)) < 0.5
        fit, test = reps[mask], reps[~mask]
        for mode in held:
            best = max(grid, key=lambda band: rate(fit, *band, mode))
            chosen[mode].append(f"{best[0]:.1f}-{best[1]:.1f}")
            held[mode].append(rate(test, *best, mode))
            preregistered[mode].append(rate(test, *BAND, mode))
    return {
        mode: {
            "half_budget": half_budget,
            "held_out_metasun_rate": float(np.mean(held[mode])),
            "preregistered_band_metasun_rate": float(np.mean(preregistered[mode])),
            "chosen_bands": pd.Series(chosen[mode]).value_counts().to_dict(),
        }
        for mode in held
    }


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def markdown(report: dict, main_budget: int) -> str:
    lines = []
    arms = report["arms"][str(main_budget)]
    broadside = arms["broadside"]
    lines.append(f"### Arms at B = {main_budget}\n")
    lines.append("| arm | MetaSUN | 95% CI | SUN | gene-novel slots | novel structure | "
                 "P(meta \\| novel) | draws | MetaSUN per draw | relax h | MetaSUN per relax h "
                 "| atoms | strict MetaSUN | MetaSUN, fingerprint kept | p vs broadside |")
    lines.append("|" + "---|" * 15)
    for name, arm in arms.items():
        lines.append(
            f"| {name} | {_fmt(arm['metasun_rate'])} | "
            f"[{_fmt(arm['metasun_ci'][0])}, {_fmt(arm['metasun_ci'][1])}] | "
            f"{_fmt(arm['sun_rate'], 4)} | {arm['gene_novel_slots']} | {arm['novel_structure']} | "
            f"{_fmt(arm['metastable_given_novel'])} | {arm['draws']} | "
            f"{_fmt(arm['metasun_per_draw'])} | {_fmt(arm['relax_hours'], 1)} | "
            f"{_fmt(arm['metasun_per_relax_hour'], 1)} | {_fmt(arm['mean_atoms'], 1)} | "
            f"{_fmt(arm.get('metasun_strict_rate'))} | "
            f"{_fmt(arm['metasun_fingerprint_kept'] / arm['budget'])} | "
            f"{'--' if name == 'broadside' else '%.1e' % fisher(arm, broadside)} |")
    lines.append("\n### MetaSUN per slot against the budget\n")
    budgets = list(report["arms"])
    lines.append("| arm | " + " | ".join(f"B = {b}" for b in budgets) + " |")
    lines.append("|---|" + "---|" * len(budgets))
    for name in arms:
        cells = []
        for b in budgets:
            arm = report["arms"][b][name]
            short = "*" if arm["filled"] < arm["budget"] else ""
            cells.append(f"{_fmt(arm['metasun_rate'])}{short}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("\n`*` the arm ran out of candidates before filling the budget; empty slots are misses.")
    return "\n".join(lines)


def upload(root: Path, out_dir: Path, report: dict, name: str, main_budget: int) -> None:
    """One W&B run per pool: the pool, its scores, the protocol outputs and the report.

    Nothing under the runs store is the only copy of a result.
    """
    import wandb

    from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
    from wyckoff_transformer.cli.protocol_wandb import add_protocol_outputs
    from wyckoff_transformer.paths import wandb_dir

    manifest_path = root / "pool" / "pool_manifest.json"
    config = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    config.update({"root": str(root), "band": list(BAND), "variants": report["variants"]})
    run = wandb.init(dir=wandb_dir(), entity=WANDB_ENTITY, project=WANDB_PROJECT,
                     name=name, id=name, resume="allow", job_type="roe_surprisal",
                     config=config)
    try:
        artifact = wandb.Artifact(name, type="roe", metadata=config)
        for sub in ("pool", "scores"):
            for path in sorted((root / sub).glob("*")):
                if path.is_file() and ".tmp" not in path.name:
                    artifact.add_file(str(path), name=f"{sub}/{path.name}")
        add_protocol_outputs(artifact, root / "protocol", prefix="protocol/")
        for path in sorted(out_dir.glob("*")):
            if path.is_file():
                artifact.add_file(str(path), name=f"analysis/{path.name}")
        summary = {"pool/" + key: value for key, value in report["pool"].items()
                   if isinstance(value, (int, float))}
        for arm, values in report["arms"][str(main_budget)].items():
            for key in ("metasun_rate", "sun_rate", "metasun_per_draw",
                        "metasun_per_relax_hour", "metasun_strict_rate"):
                if values.get(key) is not None:
                    summary[f"B{main_budget}/{arm}/{key}"] = values[key]
        run.summary.update(summary)
        run.log_artifact(artifact)
    finally:
        run.finish()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="The pool's root: pool/, scores/, protocol/.")
    parser.add_argument("--variants", type=str, default=None,
                        help="Surprisal variants, comma-separated (default: every "
                             "scores/gene_novelty_*.csv).")
    parser.add_argument("--budgets", type=str, default=",".join(map(str, BUDGETS)))
    parser.add_argument("--main-budget", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--track", choices=sorted(TRACKS), default="free",
                        help="The protocol readout to score: 'free' (after the symmetry "
                             "release and the rattle) or 'fixed_symmetry' (relaxed with the "
                             "gene's symmetry held, so still its Wyckoff representation).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: ROOT/analysis, or ROOT/analysis_<track> off the "
                             "free track.")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Log the pool, scores, protocol outputs and report to this "
                             "W&B run (also its id and the artifact name).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = args.root
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variants = sorted(path.stem.removeprefix("gene_novelty_")
                          for path in (root / "scores").glob("gene_novelty_*.csv"))
    budgets = [int(b) for b in args.budgets.split(",")]
    out_dir = args.out_dir or root / (
        "analysis" if args.track == "free" else f"analysis_{args.track}")
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, structures, screen = load_pool(root, variants, args.track)
    strict = None if args.no_strict else strict_classes(
        root, structures, args.workers, args.track)

    report = {
        "root": str(root),
        "track": args.track,
        "variants": variants,
        "band": list(BAND),
        "pool": {
            "sampled": len(frame),
            "unique_genes": int(frame["is_representative"].sum()),
            "gene_novel_unique": int((frame["is_representative"] & frame["gene_novel"]).sum()),
            "relaxed_with_structure": int(structures["has_structure"].sum()),
            "metasun_unique": int(structures["metasun"].sum()),
            "sun_unique": int(structures["sun"].sum()),
            "metasun_strict_classes": None if strict is None else int(strict.nunique()),
        },
        "arms": {},
    }
    for budget in budgets:
        arms = build_arms(frame, budget, variants)
        report["arms"][str(budget)] = {
            name: {"mode": mode, "lever": lever,
                   **evaluate(frame, structures, slots, consumed, budget, strict)}
            for name, (slots, consumed, mode, lever) in arms.items()}

    main_arms = report["arms"][str(args.main_budget)]
    contrasts = {}
    for variant in variants:
        for mode in ("fire-discipline", "fire-control"):
            a, b = main_arms[f"{mode}/lookup"], main_arms[f"{mode}/surprisal-{variant}"]
            contrasts[f"{mode}: lookup - surprisal-{variant}"] = {
                "difference": a["metasun_rate"] - b["metasun_rate"], "p": fisher(a, b)}
    report["contrasts"] = contrasts
    report["estimators"] = estimator_quality(frame, variants)
    report["band_surface"] = {
        variant: {str(b): band_surface(frame, variant, b) for b in (250, args.main_budget)}
        for variant in variants}
    report["split_half"] = {
        variant: split_half(frame, variant, args.main_budget) for variant in variants}

    frame.to_csv(out_dir / "genes.csv.gz")
    with open(out_dir / "report.json", "wt", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, default=float)
    (out_dir / "tables.md").write_text(markdown(report, args.main_budget), encoding="utf-8")
    print(markdown(report, args.main_budget))
    print(f"\nReport: {out_dir / 'report.json'}")
    if args.wandb_name:
        upload(root, out_dir, report, args.wandb_name, args.main_budget)


if __name__ == "__main__":
    main()
