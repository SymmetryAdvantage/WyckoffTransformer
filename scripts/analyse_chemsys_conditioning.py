"""Would conditioning generation on a chemical system and a space group raise (M)SUN?

The claim under test: naive unconditional generation is the wrong instrument for
(M)SUN, because a model trained to imitate an archive (a) reproduces structures
the archive already holds and (b) spends its budget in well-explored chemical
systems whose hull is already occupied. Naming the chemical system -- and
possibly the space group -- should therefore lift the yield.

Nothing new is generated here. Every ``protocol_eval`` artifact in the W&B
project is pooled into one per-gene table, each gene labelled with the chemical
system and space group it was drawn in and with how much LeMat-Bulk already holds
there, and the pool is asked what a different sampling distribution over those
cells would have been worth. Three things are measured:

*What the models actually do.*  Where the budget lands in the archive's density,
and how each stage of the cascade -- novelty, metastability, MetaSUN, SUN --
moves with it, with run fixed effects and cluster-bootstrap intervals over
chemical systems, before and after controlling for arity, positional degrees of
freedom and cell size.

*What a reallocation would pay.*  A conditioning target is a cell a caller can
name in advance, and conditioning lets the whole budget be spent inside one, so
a cell's price is its own rate. Cells are therefore ranked on one random half of
the pool and read off the other, repeatedly, so that the number reported is a
prediction and not the maximum of a noisy table. The finer-grained version fits
an *a priori* score -- archive statistics of the system, of its elements and of
the space group, and nothing about the drawn gene or its relaxation -- with
5-fold cross-fitting whose folds are split by chemical system, and reads the
yield off the top of the out-of-fold score.

*What conditioning already costs.*  Three of the runs draw their (system, space
group) from the training-matched prior of ``wyckoff_transformer.system_prior``
rather than sampling the space group alone, so they are conditioned generation
with the archive's own prior. They are compared against the rest both overall and
within matched cells.

    python scripts/analyse_chemsys_conditioning.py --work-dir <dir> --fetch
    python scripts/analyse_chemsys_conditioning.py --work-dir <dir>

``--fetch`` downloads the newest version of every ``protocol_eval`` collection
into ``<work-dir>/artifacts`` and is needed once; afterwards the pool is rebuilt
from what is on disk. ``--work-dir`` defaults to ``<runs root>/chemsys_study``,
which is a scratch location: the numbers that matter are written to
``<work-dir>/results/*.json`` and the reading of them is in
``docs/chemical_system_conditioning.md``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
from wyckoff_transformer.paths import data_path, runs_root

logger = logging.getLogger(__name__)

ARTIFACT_TYPE = "protocol_eval"
#: LeMat-Bulk's per-entry labels: ``immutable_id, chemsys, formula, e_hull, e_form``
#: for all 5,335,299 rows, which is where "how explored is this system" comes from.
LABELS = ("lemat-bulk", "labels.parquet")
#: The readout the protocol develops against: post-rattle, symmetry released.
P = "free"
#: A cell needs this many observed genes before its rate is allowed to be ranked.
MIN_CELL = 100
#: Halves drawn for the cross-fitted cell selection.
N_SPLITS = 20
#: Cluster-bootstrap resamples, clustered on chemical system.
N_BOOT = 400

CRYSTAL_SYSTEM_BOUNDS = ((1, 2, "triclinic"), (3, 15, "monoclinic"),
                         (16, 74, "orthorhombic"), (75, 142, "tetragonal"),
                         (143, 167, "trigonal"), (168, 194, "hexagonal"),
                         (195, 230, "cubic"))
#: Bins over ``log10(1 + LeMat-Bulk entries in the system)``.
CROWD_EDGES = (-0.01, 0.001, 1.5, 2.5, 10)
CROWD_LABELS = ("unseen", "1-31", "32-316", ">=317")

#: Features of a *cell*, all computable before anything is generated.
SYS_FEATURES = ["log_n_entries", "log_n_formulas", "log_n_hull", "log_n_near_hull",
                "min_e_hull", "min_e_form", "arity", "system_unseen",
                "el_log_rows_mean", "el_log_rows_min", "el_log_rows_max",
                "el_log_systems_mean", "el_log_systems_min"]
SG_FEATURES = ["spacegroup", "log_sg"]


def crystal_system(sg: int) -> str:
    for lo, hi, name in CRYSTAL_SYSTEM_BOUNDS:
        if lo <= sg <= hi:
            return name
    return "?"


def wrate(df: pd.DataFrame, col: str) -> float:
    """A per-*sampled*-gene rate.

    ``structures.csv`` holds one row per unique gene; ``screen.json`` holds the
    multiplicity each represents. Weighting by it is what makes these numbers
    reproduce ``funnel.json`` rather than a deduplicated readout in which
    uniqueness is 1.0 by construction.
    """
    w = df["count"]
    return float((df[col].astype(bool) * w).sum() / w.sum()) if w.sum() else np.nan


# --------------------------------------------------------------------- inputs

def fetch_artifacts(dest: Path, entity: str, project: str) -> None:
    """Download the newest version of every ``protocol_eval`` collection."""
    import wandb  # noqa: PLC0415

    dest.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=300)
    for coll in api.artifact_type(ARTIFACT_TYPE, project=f"{entity}/{project}").collections():
        target = dest / coll.name
        if (target / "manifest.json").is_file():
            logger.info("have %s", coll.name)
            continue
        artifact = next(iter(coll.artifacts()))
        logger.info("downloading %s %s", coll.name, artifact.version)
        artifact.download(root=str(target))


def build_chemsys_stats(out: Path) -> pd.DataFrame:
    """Per-chemical-system statistics of what LeMat-Bulk already holds."""
    if out.is_file():
        return pd.read_parquet(out)
    df = pd.read_parquet(data_path(*LABELS),
                         columns=["immutable_id", "chemsys", "formula", "e_hull", "e_form"])
    logger.info("labels: %d rows, %d systems", len(df), df.chemsys.nunique())
    df["is_hull"] = df.e_hull <= 1e-6
    df["is_near_hull"] = df.e_hull <= 0.1
    stats = df.groupby("chemsys", sort=False).agg(
        n_entries=("immutable_id", "size"),
        n_formulas=("formula", "nunique"),
        n_hull=("is_hull", "sum"),
        n_near_hull=("is_near_hull", "sum"),
        min_e_hull=("e_hull", "min"),
        min_e_form=("e_form", "min"))
    stats["arity"] = stats.index.str.count("-") + 1
    stats.to_parquet(out)

    rows: dict[str, int] = {}
    systems: dict[str, int] = {}
    for name, n in zip(stats.index, stats.n_entries.to_numpy()):
        for element in name.split("-"):
            rows[element] = rows.get(element, 0) + int(n)
            systems[element] = systems.get(element, 0) + 1
    json.dump({"element_rows": rows, "element_systems": systems},
              open(out.with_name("element_stats.json"), "w"), indent=1)
    return stats


def space_group_counts(out: Path, run_dir: Path | None = None) -> pd.Series:
    """Training rows per space group, from any run's ``system_prior.npz``.

    The prior is built from the same tensor cache the models train on, so its
    ``global_sg_counts`` is exactly the marginal generation draws its start
    token from.
    """
    if out.is_file():
        return pd.read_parquet(out)["n"]
    candidates = sorted((run_dir or runs_root()).glob("*/system_prior.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No system_prior.npz under {run_dir or runs_root()}; build one with "
            "`wyformer-system-prior build <dataset>` or fetch a chemsys run's.")
    prior = np.load(candidates[0], allow_pickle=True)
    counts = pd.DataFrame({"n": prior["global_sg_counts"]},
                          index=pd.Index(prior["space_groups"], name="spacegroup"))
    counts.to_parquet(out)
    return counts["n"]


def load_run(run_dir: Path) -> pd.DataFrame:
    """One row per unique gene of one protocol artifact."""
    manifest = json.load(open(run_dir / "manifest.json"))
    screen = json.load(open(run_dir / "screen.json"))
    genes = json.load(gzip.open(run_dir / "wyckoff_genes.json.gz"))
    counts = {int(k): v for k, v in screen["counts"].items()}
    novel_genes = set(screen["novel"])

    frames = {}
    for readout, name in (("free", "structures.csv"),
                          ("fixed", "structures_fixed_symmetry.csv")):
        path = run_dir / name
        if path.is_file():
            df = pd.read_csv(path).set_index("index")
            frames[readout] = df[~df.index.duplicated(keep="first")]

    base = frames["free"]
    out = pd.DataFrame(index=base.index)
    out["run"] = run_dir.name.removeprefix("protocol_")
    out["count"] = [counts.get(i, 0) for i in out.index]
    out["gene_novel"] = [i in novel_genes for i in out.index]
    out["spacegroup"] = [genes[i]["group"] for i in out.index]
    out["crystal_system"] = out.spacegroup.map(crystal_system)
    out["chemsys"] = ["-".join(sorted(set(genes[i]["species"]))) for i in out.index]
    out["arity"] = out.chemsys.str.count("-") + 1
    out["n_atoms_gene"] = [sum(genes[i]["numIons"]) for i in out.index]
    out["dof_positional"] = base["dof_positional"]
    out["formula"] = base["formula"]

    for readout, df in frames.items():
        has = df["has_structure"].fillna(False).astype(bool)
        valid = has & df["valid_structure"].fillna(False).astype(bool)
        unique = valid & df["unique_structure"].fillna(False).astype(bool)
        novel = unique & df["novel_structure"].fillna(False).astype(bool)
        energy = df["e_above_hull"]
        out[f"{readout}_has_structure"] = has
        out[f"{readout}_valid"] = valid
        out[f"{readout}_unique"] = unique
        out[f"{readout}_novel"] = novel
        out[f"{readout}_e_hull"] = energy
        out[f"{readout}_metastable"] = unique & energy.notna() & (energy <= 0.1)
        out[f"{readout}_stable"] = unique & energy.notna() & (energy <= 0.0)
        out[f"{readout}_msun"] = novel & energy.notna() & (energy <= 0.1)
        out[f"{readout}_sun"] = novel & energy.notna() & (energy <= 0.0)
    out["reference_cache"] = manifest.get("reference_cache")
    out["temperature"] = manifest.get("sampling_temperature")
    return out.reset_index()


def assemble(work: Path) -> pd.DataFrame:
    """Pool every downloaded artifact and attach the archive's view of each cell."""
    runs = sorted(p for p in (work / "artifacts").iterdir()
                  if (p / "manifest.json").is_file())
    if not runs:
        raise FileNotFoundError(f"No artifacts under {work / 'artifacts'}; pass --fetch.")
    references = {json.load(open(p / "manifest.json")).get("reference_cache") for p in runs}
    if len(references) > 1:
        raise ValueError(
            f"The artifacts were scored against different novelty references ({references}); "
            "gene novelty, novel_structure, MetaSUN and SUN are not comparable across them. "
            "Re-score the odd ones with `wyformer-protocol-wandb --from-artifact "
            "--stages screen,score` first.")
    logger.info("pooling %d runs, all scored against %s", len(runs), references.pop())

    pool = pd.concat([load_run(p) for p in runs], ignore_index=True)
    stats = build_chemsys_stats(work / "chemsys_stats.parquet")
    pool = pool.merge(stats, how="left", left_on="chemsys", right_index=True,
                      suffixes=("", "_ref"))
    for col in ("n_entries", "n_formulas", "n_hull", "n_near_hull"):
        pool[col] = pool[col].fillna(0).astype(int)
    pool["system_known"] = pool.n_entries > 0
    pool["log_n"] = np.log10(pool.n_entries + 1)
    pool["crowd"] = pd.cut(pool.log_n, list(CROWD_EDGES), labels=list(CROWD_LABELS))
    pool["symmetry"] = np.where(pool.spacegroup < 75, "low (sg<75)", "high (sg>=75)")
    pool["arity_g"] = pool.arity.clip(upper=5)
    sg = space_group_counts(work / "sg_counts.parquet")
    pool["log_sg"] = np.log10(pool.spacegroup.map(sg).fillna(0) + 1)
    pool["family"] = np.where(pool.run.str.startswith("chemsys_"),
                              "system+sg prior", "unconditional")
    pool.to_parquet(work / "pool.parquet")
    return pool


# ------------------------------------------------------------------ measures

def describe(pool: pd.DataFrame, work: Path) -> dict:
    """Per run, pooled, and by every cell coordinate the hypothesis names."""
    report: dict = {}
    total = pool["count"].sum()

    per_run = pool.groupby("run").apply(lambda d: pd.Series({
        "sampled": int(d["count"].sum()),
        "gene_novel": wrate(d, "gene_novel"),
        "valid": wrate(d, f"{P}_valid"),
        "novel": wrate(d, f"{P}_novel"),
        "metastable": wrate(d, f"{P}_metastable"),
        "msun": wrate(d, f"{P}_msun"),
        "sun": wrate(d, f"{P}_sun")}), include_groups=False)
    print("=" * 100, "\nPER RUN (free readout, per sampled gene)\n", per_run.round(4), sep="")
    report["per_run"] = per_run.round(5).reset_index().to_dict("records")

    unique = pool[pool[f"{P}_unique"]]
    novel = pool[pool[f"{P}_novel"]]
    known = unique[~unique[f"{P}_novel"]]
    funnel = {
        "sampled": int(total),
        "novel_structures": int((pool[f"{P}_novel"] * pool["count"]).sum()),
        "msun_structures": int((pool[f"{P}_msun"] * pool["count"]).sum()),
        "sun_structures": int((pool[f"{P}_sun"] * pool["count"]).sum()),
        "gene_novel": wrate(pool, "gene_novel"),
        "valid": wrate(pool, f"{P}_valid"),
        "novel": wrate(pool, f"{P}_novel"),
        "metastable": wrate(pool, f"{P}_metastable"),
        "stable": wrate(pool, f"{P}_stable"),
        "msun": wrate(pool, f"{P}_msun"),
        "sun": wrate(pool, f"{P}_sun"),
        "P(metastable|novel)": wrate(novel, f"{P}_metastable"),
        "P(stable|novel)": wrate(novel, f"{P}_stable"),
        "P(metastable|known)": wrate(known, f"{P}_metastable"),
        "P(stable|known)": wrate(known, f"{P}_stable"),
    }
    print("=" * 100, "\nPOOLED FUNNEL\n", json.dumps(funnel, indent=1), sep="")
    report["pooled_funnel"] = funnel

    for label, keys in (("crowding", ["crowd"]), ("arity", ["arity_g"]),
                        ("crystal system", ["crystal_system"]),
                        ("crowding x arity", ["crowd", "arity_g"]),
                        ("crowding x symmetry", ["crowd", "symmetry"])):
        table = pool.groupby(keys, observed=True).apply(lambda d: pd.Series({
            "n": int(d["count"].sum()),
            "share": float(d["count"].sum() / total),
            "gene_novel": wrate(d, "gene_novel"),
            "novel": wrate(d, f"{P}_novel"),
            "metastable": wrate(d, f"{P}_metastable"),
            "msun": wrate(d, f"{P}_msun"),
            "sun": wrate(d, f"{P}_sun")}), include_groups=False)
        table = table[table.n >= 30]
        print("=" * 100, f"\nBY {label.upper()}\n", table.round(4).to_string(), sep="")
        report[label] = table.round(5).reset_index().to_dict("records")

    top = pool.groupby("spacegroup")["count"].sum().sort_values(ascending=False).head(15)
    sg_table = pool[pool.spacegroup.isin(top.index)].groupby("spacegroup").apply(
        lambda d: pd.Series({
            "n": int(d["count"].sum()),
            "share": float(d["count"].sum() / total),
            "novel": wrate(d, f"{P}_novel"),
            "metastable": wrate(d, f"{P}_metastable"),
            "msun": wrate(d, f"{P}_msun"),
            "sun": wrate(d, f"{P}_sun")}), include_groups=False).sort_values("n", ascending=False)
    print("=" * 100, "\nTOP 15 SPACE GROUPS BY BUDGET\n", sg_table.round(4).to_string(), sep="")
    report["space_group"] = sg_table.round(5).reset_index().to_dict("records")

    # The element-level reading of "well-explored": the exact system may be
    # obscure while the elements in it are worked to death, so the same question
    # is asked of the rarest element each gene uses.
    elements = pool.chemsys.str.split("-")
    log_rows = {k: np.log10(v + 1) for k, v in
                json.load(open(work / "element_stats.json"))["element_rows"].items()}
    pool = pool.assign(el_rows_min=elements.map(lambda e: min(log_rows.get(x, 0) for x in e)))
    pool["el_quintile"] = pd.qcut(pool.el_rows_min, 5,
                                  labels=["Q1 rarest", "Q2", "Q3", "Q4", "Q5 commonest"])
    el_table = pool.groupby("el_quintile", observed=True).apply(lambda d: pd.Series({
        "n": int(d["count"].sum()),
        "median_rows_of_rarest_element": float(10 ** d.el_rows_min.median()),
        "novel": wrate(d, f"{P}_novel"),
        "metastable": wrate(d, f"{P}_metastable"),
        "msun": wrate(d, f"{P}_msun"),
        "sun": wrate(d, f"{P}_sun")}), include_groups=False)
    print("=" * 100, "\nBY THE ARCHIVE'S DEPTH IN THE GENE'S RAREST ELEMENT\n",
          el_table.round(4).to_string(), sep="")
    report["element_depth"] = el_table.round(5).reset_index().to_dict("records")

    # Where the space group sits in the training marginal, which is what
    # generation draws its start token from. Sorted by frequency rather than by
    # symmetry, because the two are not the same variable and only one of them
    # moves SUN.
    pool["sg_quintile"] = pd.qcut(pool.log_sg, 5,
                                  labels=["Q1 rarest", "Q2", "Q3", "Q4", "Q5 commonest"])
    sg_freq = pool.groupby("sg_quintile", observed=True).apply(lambda d: pd.Series({
        "n": int(d["count"].sum()),
        "median_training_rows": float(10 ** d.log_sg.median()),
        "novel": wrate(d, f"{P}_novel"),
        "metastable": wrate(d, f"{P}_metastable"),
        "msun": wrate(d, f"{P}_msun"),
        "sun": wrate(d, f"{P}_sun")}), include_groups=False)
    print("=" * 100, "\nBY SPACE-GROUP FREQUENCY IN TRAINING\n",
          sg_freq.round(4).to_string(), sep="")
    report["space_group_frequency"] = sg_freq.round(5).reset_index().to_dict("records")

    symmetry = pool.groupby("symmetry").apply(lambda d: pd.Series({
        "n": int(d["count"].sum()),
        "share": float(d["count"].sum() / total),
        "novel": wrate(d, f"{P}_novel"),
        "metastable": wrate(d, f"{P}_metastable"),
        "msun": wrate(d, f"{P}_msun"),
        "sun": wrate(d, f"{P}_sun")}), include_groups=False)
    print("\nBY SYMMETRY (the other reading of the same axis)\n",
          symmetry.round(4).to_string(), sep="")
    report["symmetry"] = symmetry.round(5).reset_index().to_dict("records")

    # Crowding within geometry, since a crowded system also holds smaller cells:
    # if the gradient is geometry it should flatten inside these bins.
    pool["dof_bin"] = pd.cut(pool.dof_positional, [-0.1, 0.5, 3.5, 9.5, 1e4],
                             labels=["0", "1-3", "4-9", "10+"])
    pool["size_bin"] = pd.cut(pool.n_atoms_gene, [0, 8, 16, 32, 1e5],
                              labels=["<=8", "9-16", "17-32", ">32"])
    for name in ("dof_bin", "size_bin"):
        table = pool.groupby([name, "crowd"], observed=True).apply(lambda d: pd.Series({
            "n": int(d["count"].sum()),
            "novel": wrate(d, f"{P}_novel"),
            "metastable": wrate(d, f"{P}_metastable"),
            "msun": wrate(d, f"{P}_msun")}), include_groups=False)
        table = table[table.n >= 40]
        print("=" * 100, f"\nCROWDING WITHIN {name}\n", table.round(4).to_string(), sep="")
        report[f"crowding_within_{name}"] = table.round(5).reset_index().to_dict("records")

    # The hypothesis's own mechanism, stated as a comparison: are novel structures
    # further from the hull where the archive already reaches it?
    e_by_crowd = novel.groupby("crowd", observed=True).apply(lambda d: pd.Series({
        "n_novel": len(d),
        "median_e_hull": float(d[f"{P}_e_hull"].median()),
        "p25_e_hull": float(d[f"{P}_e_hull"].quantile(0.25)),
        "p75_e_hull": float(d[f"{P}_e_hull"].quantile(0.75)),
        "P(metastable)": float(d[f"{P}_metastable"].mean()),
        "P(stable)": float(d[f"{P}_stable"].mean())}), include_groups=False)
    print("=" * 100, "\ne_above_hull OF NOVEL STRUCTURES, BY CROWDING OF THEIR SYSTEM\n",
          e_by_crowd.round(4).to_string(), sep="")
    report["novel_e_hull_by_crowd"] = e_by_crowd.round(5).reset_index().to_dict("records")

    by_hull = novel.assign(has_hull_entry=novel.n_hull > 0).groupby("has_hull_entry").apply(
        lambda d: pd.Series({
            "n_novel": len(d),
            "median_e_hull": float(d[f"{P}_e_hull"].median()),
            "P(metastable)": float(d[f"{P}_metastable"].mean()),
            "P(stable)": float(d[f"{P}_stable"].mean()),
            "median_n_entries": float(d.n_entries.median())}), include_groups=False)
    print("=" * 100, "\nNOVEL STRUCTURES, SPLIT ON WHETHER THEIR SYSTEM ALREADY HOLDS A HULL ENTRY\n",
          by_hull.round(4).to_string(), sep="")
    report["novel_by_hull_entry"] = by_hull.round(5).reset_index().to_dict("records")

    # What the pool can resolve at all on the readout that matters. 67 SUN
    # events is what makes every SUN statement in this study directional.
    from scipy import stats  # noqa: PLC0415

    p0 = funnel["sun"]
    report["sun_power"] = {}
    for lift in (1.5, 2.0, 3.0):
        p1 = p0 * lift
        pbar = (p0 + p1) / 2
        z_alpha, z_beta = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
        n = ((z_alpha * np.sqrt(2 * pbar * (1 - pbar))
              + z_beta * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2 / (p1 - p0) ** 2)
        report["sun_power"][f"{lift:g}x"] = int(np.ceil(n))
        print(f"SUN: an {lift:g}x lift over {p0:.4f} needs {int(np.ceil(n))} genes per arm "
              "at 80% power")
    return report


def _design(pool: pd.DataFrame, regressor: str, controls: list[str]) -> np.ndarray:
    """``[regressor, *controls, run dummies]`` as one dense array.

    Built once and indexed by row for every bootstrap resample: rebuilding it
    per resample is what made the interval cost minutes rather than seconds.
    """
    dummies = pd.get_dummies(pool["run"], drop_first=True).to_numpy(float)
    columns = [pool[regressor].to_numpy(float)]
    columns += [pool[c].to_numpy(float) for c in controls]
    return np.column_stack(columns + [dummies])


def _odds_ratio(X, y, w, rows=None):
    """Odds ratio of the first column of *X*, on *rows* if given."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    if rows is not None:
        X, y, w = X[rows], y[rows], w[rows]
    if y.sum() < 5 or (~y).sum() < 5:
        return np.nan
    model = LogisticRegression(max_iter=1000, C=1e6)
    model.fit(X, y, sample_weight=w)
    return float(np.exp(model.coef_[0][0]))


def effects(pool: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Odds ratios per decade, with run fixed effects and cluster-bootstrap CIs.

    The cluster is the chemical system: genes drawn in one system share whatever
    makes that system easy or hard, and the systems are what the hypothesis is
    about, so treating genes as independent would understate every interval.
    """
    index = pool.groupby("chemsys").indices
    systems = np.array(list(index))
    rng = np.random.default_rng(seed)
    resamples = [np.concatenate([index[s] for s in rng.choice(systems, len(systems), True)])
                 for _ in range(n_boot)]
    weight = pool["count"].to_numpy(float)

    report: dict = {}
    targets = ((f"{P}_novel", "novel"), (f"{P}_metastable", "metastable"),
               (f"{P}_msun", "msun"), (f"{P}_sun", "sun"))
    geometry = ["arity", "dof_positional", "n_atoms_gene"]
    for regressor, controls, label in (
            ("log_n", [], "system crowding, unadjusted"),
            ("log_n", geometry + ["log_sg"], "system crowding, adjusted"),
            ("log_sg", [], "space-group frequency, unadjusted"),
            ("log_sg", geometry + ["log_n"], "space-group frequency, adjusted")):
        report[label] = {}
        X = _design(pool, regressor, controls)
        for target, name in targets:
            y = pool[target].to_numpy(bool)
            point = _odds_ratio(X, y, weight)
            values = np.array([_odds_ratio(X, y, weight, r) for r in resamples])
            values = values[np.isfinite(values)]
            lo, hi = np.percentile(values, [2.5, 97.5])
            report[label][name] = {"or_per_decade": point, "ci95": [float(lo), float(hi)]}
            print(f"{label:34s} {name:11s} OR/decade = {point:.3f}  [{lo:.3f}, {hi:.3f}]",
                  flush=True)
    return report


def _cell_rates(df, keys, target):
    return df.groupby(keys, observed=True).apply(lambda d: pd.Series({
        "n": int(d["count"].sum()), "rate": wrate(d, target)}),
        include_groups=False).dropna()


def crossfit_cells(pool: pd.DataFrame, seed: int = 0) -> dict:
    """Rank cells on half the pool, read the winner's rate off the other half."""
    report: dict = {}
    for label, keys in (("crowding", ["crowd"]), ("symmetry", ["symmetry"]),
                        ("space group", ["spacegroup"]),
                        ("crowding x symmetry", ["crowd", "symmetry"]),
                        ("crowding x arity", ["crowd", "arity_g"]),
                        ("crowding x arity x symmetry", ["crowd", "arity_g", "symmetry"])):
        for target, name in ((f"{P}_msun", "msun"), (f"{P}_sun", "sun")):
            rng = np.random.default_rng(seed)
            rates, bases, picked = [], [], []
            for _ in range(N_SPLITS):
                mask = rng.random(len(pool)) < 0.5
                a, b = pool[mask], pool[~mask]
                fit = _cell_rates(a, keys, target)
                fit = fit[fit.n >= MIN_CELL / 2]
                if fit.empty:
                    continue
                best = fit.rate.idxmax()
                held = _cell_rates(b, keys, target)
                if best not in held.index or held.loc[best, "n"] < 20:
                    continue
                picked.append(str(best))
                rates.append(held.loc[best, "rate"])
                bases.append(wrate(b, target))
            if not rates:
                print(f"  {label:30s} {name}: no cell survived")
                continue
            counts = pd.Series(picked).value_counts()
            entry = {"held_out_rate": float(np.mean(rates)),
                     "held_out_base": float(np.mean(bases)),
                     "enrichment": float(np.mean(rates) / np.mean(bases)),
                     "rate_p10_p90": [float(np.percentile(rates, 10)),
                                      float(np.percentile(rates, 90))],
                     "splits_used": len(rates),
                     "most_picked": counts.index[0],
                     "most_picked_share": float(counts.iloc[0] / len(picked))}
            report[f"{label}|{name}"] = entry
            print(f"  {label:30s} {name}: held-out {entry['held_out_rate']:.3f} vs base "
                  f"{entry['held_out_base']:.3f} = {entry['enrichment']:.2f}x  "
                  f"(p10-p90 {entry['rate_p10_p90'][0]:.3f}-{entry['rate_p10_p90'][1]:.3f}, "
                  f"picked {entry['most_picked']} {entry['most_picked_share']:.0%})")
    return report


def _cell_features(pool: pd.DataFrame, work: Path) -> pd.DataFrame:
    element = json.load(open(work / "element_stats.json"))
    log_rows = {k: np.log10(v + 1) for k, v in element["element_rows"].items()}
    log_systems = {k: np.log10(v + 1) for k, v in element["element_systems"].items()}
    elements = pool.chemsys.str.split("-")
    return pd.DataFrame({
        "log_n_entries": pool.log_n,
        "log_n_formulas": np.log10(pool.n_formulas + 1),
        "log_n_hull": np.log10(pool.n_hull + 1),
        "log_n_near_hull": np.log10(pool.n_near_hull + 1),
        # A system the archive never saw has no minimum. 0.5 eV/atom says "nothing
        # here is known to be anywhere near the hull", which is what absence means;
        # system_unseen lets the model treat the fill as its own category.
        "min_e_hull": pool.min_e_hull.fillna(0.5),
        "min_e_form": pool.min_e_form.fillna(0.0),
        "arity": pool.arity,
        "system_unseen": (~pool.system_known).astype(int),
        "el_log_rows_mean": elements.map(lambda e: np.mean([log_rows.get(x, 0) for x in e])),
        "el_log_rows_min": elements.map(lambda e: min(log_rows.get(x, 0) for x in e)),
        "el_log_rows_max": elements.map(lambda e: max(log_rows.get(x, 0) for x in e)),
        "el_log_systems_mean": elements.map(lambda e: np.mean([log_systems.get(x, 0) for x in e])),
        "el_log_systems_min": elements.map(lambda e: min(log_systems.get(x, 0) for x in e)),
        "spacegroup": pool.spacegroup,
        "log_sg": pool.log_sg,
    }, index=pool.index)


def crossfit_score(pool: pd.DataFrame, work: Path, seed: int = 0) -> dict:
    """Fit an a-priori cell score and read the yield off the top of it.

    The folds are split **by chemical system**: a system that appeared in the fit
    must not appear in the readout, or the score would be memorising which
    systems happened to work rather than predicting which ones will.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    from sklearn.model_selection import GroupKFold  # noqa: PLC0415

    features = _cell_features(pool, work)
    weight = pool["count"].to_numpy(float)
    groups = pool.chemsys.to_numpy()
    blocks = {"chemical system only": SYS_FEATURES,
              "space group only": SG_FEATURES,
              "system + space group": SYS_FEATURES + SG_FEATURES}
    report: dict = {}
    for target, name in ((f"{P}_msun", "msun"), (f"{P}_sun", "sun"),
                         (f"{P}_novel", "novel"), (f"{P}_metastable", "metastable")):
        y = pool[target].to_numpy(bool)
        report[name] = {}
        for block, columns in blocks.items():
            oof = np.zeros(len(y))
            for train, test in GroupKFold(n_splits=5).split(features, y, groups):
                model = HistGradientBoostingClassifier(
                    max_depth=4, max_iter=200, learning_rate=0.06,
                    min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
                model.fit(features[columns].iloc[train], y[train], sample_weight=weight[train])
                oof[test] = model.predict_proba(features[columns].iloc[test])[:, 1]
            order = np.argsort(-oof)
            base = (y * weight).sum() / weight.sum()
            cumulative = np.cumsum(weight[order])
            rows = []
            for q in (0.1, 0.2, 0.3, 0.5):
                k = int(np.searchsorted(cumulative, q * weight.sum())) + 1
                sel = order[:k]
                rate = (y[sel] * weight[sel]).sum() / weight[sel].sum()
                rows.append({"top_q": q, "n_sampled": int(weight[sel].sum()),
                             "rate": float(rate), "enrichment": float(rate / base)})
            report[name][block] = {"base_rate": float(base), "quantiles": rows}
            print(f"[{name}] {block}: base {base:.4f} -> top 10% "
                  f"{rows[0]['rate']:.4f} ({rows[0]['enrichment']:.2f}x), top 50% "
                  f"{rows[-1]['rate']:.4f} ({rows[-1]['enrichment']:.2f}x)", flush=True)
    return report


def space_group_reallocation(pool: pd.DataFrame, seed: int = 0) -> dict:
    """What re-weighting the start-token table alone would pay.

    The space group is not something the model chooses: it is the start token,
    drawn from a fixed table before the sequence begins, so the joint is
    ``p_table(G) * p_theta(gene | G)``. Replacing ``p_table`` changes only which
    cells are visited, and the pool's per-group rates estimate ``p_theta(.|G)``'s
    yield without bias -- so unlike everything else in this study, this
    reallocation is identified rather than merely selected.

    Restricted to the runs that draw their start token that way: the
    chemical-system runs take theirs from the prior's ``p(G|S)`` instead.

    Two rules are priced. A smooth one -- keep the rarest quintiles of the
    group's training frequency -- and a per-group one, ranked on half the pool
    and spent on the other, which is where 228 groups and 61 SUN events show
    what they are worth.
    """
    unconditional = pool[pool.family == "unconditional"]
    weight = unconditional["count"].to_numpy(float)
    quintile = pd.qcut(unconditional.log_sg, 5, labels=False).to_numpy()
    report: dict = {"n_genes": int(weight.sum()), "smooth_rule": {}, "per_group_rule": {}}

    def rate_ci(mask, target):
        n = weight[mask].sum()
        k = (unconditional[target].to_numpy(bool)[mask] * weight[mask]).sum()
        p_hat, z = k / n, 1.959964
        d = 1 + z * z / n
        centre = (p_hat + z * z / (2 * n)) / d
        half = z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / d
        return {"rate": float(p_hat), "ci95": [float(max(0.0, centre - half)),
                                               float(min(1.0, centre + half))],
                "k": int(k), "n": int(n)}

    print("=" * 100, "\nRE-WEIGHTING THE START-TOKEN TABLE (unconditional runs only)", sep="")
    for label, mask in (("as drawn", np.ones(len(unconditional), bool)),
                        ("rarest two quintiles", quintile <= 1),
                        ("commonest two quintiles", quintile >= 3)):
        report["smooth_rule"][label] = {}
        line = []
        for target, name in ((f"{P}_msun", "MetaSUN"), (f"{P}_sun", "SUN")):
            entry = rate_ci(mask, target)
            report["smooth_rule"][label][name] = entry
            line.append(f"{name} {entry['rate']:.4f} [{entry['ci95'][0]:.4f}, "
                        f"{entry['ci95'][1]:.4f}] (k={entry['k']}, n={entry['n']})")
        print(f"  {label:26s} " + "   ".join(line))

    for target, name in ((f"{P}_msun", "msun"), (f"{P}_sun", "sun")):
        rng = np.random.default_rng(seed)
        gains = []
        for _ in range(N_SPLITS):
            mask = rng.random(len(unconditional)) < 0.5
            a, b = unconditional[mask], unconditional[~mask]
            rates = _cell_rates(a, ["spacegroup"], target)
            rates = rates[rates.n >= 20]
            if rates.empty:
                continue
            chosen = set(rates.index[rates.rate > wrate(a, target)])
            picked = b[b.spacegroup.isin(chosen)]
            if picked["count"].sum() < 50:
                continue
            gains.append(wrate(picked, target) / wrate(b, target))
        report["per_group_rule"][name] = {
            "enrichment": float(np.mean(gains)),
            "p10_p90": [float(np.percentile(gains, 10)), float(np.percentile(gains, 90))],
            "splits_used": len(gains)}
        print(f"  per-group selection, {name}: {np.mean(gains):.2f}x "
              f"[{np.percentile(gains, 10):.2f}, {np.percentile(gains, 90):.2f}]")
    return report


def prior_runs(pool: pd.DataFrame) -> dict:
    """The runs that already draw (system, space group) from the training prior.

    Two questions: does that prior put them anywhere different, and are they
    worse at building once they are there. The second is confounded -- they are
    different checkpoints, not the same model with the sampler swapped -- but a
    within-cell gap is the only direct evidence the pool holds on what forcing a
    cell costs.
    """
    report: dict = {}
    overall = pool.groupby("family").apply(lambda d: pd.Series({
        "n": int(d["count"].sum()),
        "unseen_system_share": float((d["count"] * ~d.system_known).sum() / d["count"].sum()),
        "median_n_entries": float(d.n_entries.median()),
        "distinct_systems_per_1000": float(d.groupby("run").chemsys.nunique().mean()),
        "low_symmetry_share": float((d["count"] * (d.spacegroup < 75)).sum() / d["count"].sum()),
        "novel": wrate(d, f"{P}_novel"),
        "metastable": wrate(d, f"{P}_metastable"),
        "msun": wrate(d, f"{P}_msun"),
        "sun": wrate(d, f"{P}_sun")}), include_groups=False)
    print("=" * 100, "\nSYSTEM-PRIOR-SAMPLED RUNS vs THE REST\n",
          overall.round(4).to_string(), sep="")
    report["overall"] = overall.round(5).reset_index().to_dict("records")

    # Generation draws the space group from the training marginal unless the
    # prior replaces it, so the KL against that marginal says whether the prior
    # moved the budget anywhere at all.
    share = pool.groupby(["family", "spacegroup"])["count"].sum().unstack(0).fillna(0)
    share = share.div(share.sum(), axis=1)
    training = pool.drop_duplicates("spacegroup").set_index("spacegroup")["log_sg"]
    training = (10 ** training - 1).reindex(share.index).fillna(0)
    training = training / training.sum()
    report["sg_marginal_kl_vs_training"] = {}
    for family in share.columns:
        q = share[family].to_numpy()
        mask = q > 0
        kl = float((q[mask] * np.log(q[mask] / training.to_numpy()[mask])).sum())
        report["sg_marginal_kl_vs_training"][family] = kl
        print(f"KL(space-group marginal of {family!r} || training) = {kl:.4f} nats")

    cells = pool.groupby(["crowd", "arity_g", "symmetry", "family"], observed=True).apply(
        lambda d: pd.Series({"n": int(d["count"].sum()),
                             "novel": wrate(d, f"{P}_novel"),
                             "meta": wrate(d, f"{P}_metastable"),
                             "msun": wrate(d, f"{P}_msun")}), include_groups=False)
    wide = cells.unstack("family")
    wide = wide[(wide[("n", "system+sg prior")] >= 80) & (wide[("n", "unconditional")] >= 80)]
    print("\nWITHIN-CELL COMPARISON (cells with >=80 genes in each family)")
    print(wide.round(4).to_string())
    n = wide[("n", "system+sg prior")]
    gaps = {key: float(((wide[(key, "system+sg prior")] - wide[(key, "unconditional")]) * n).sum() / n.sum())
            for key in ("novel", "meta", "msun")}
    print(f"budget-weighted within-cell gaps (prior - unconditional): {gaps}")
    # The unstack leaves tuple column labels, which are not JSON keys.
    flat = wide.round(5).copy()
    flat.columns = [f"{metric}|{family}" for metric, family in flat.columns]
    report["within_cell"] = flat.reset_index().to_dict("records")
    report["within_cell_gaps"] = gaps
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="scratch directory for artifacts, the pool and results "
                             "(default: <runs root>/chemsys_study)")
    parser.add_argument("--fetch", action="store_true",
                        help="download every protocol_eval artifact first")
    parser.add_argument("--rebuild", action="store_true",
                        help="re-pool the artifacts even if pool.parquet exists")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help="cluster-bootstrap resamples for the odds ratios")
    parser.add_argument("--wandb-entity", default=WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    work = args.work_dir or (runs_root() / "chemsys_study")
    work.mkdir(parents=True, exist_ok=True)
    results = work / "results"
    results.mkdir(exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)

    if args.fetch:
        fetch_artifacts(work / "artifacts", args.wandb_entity, args.wandb_project)
    pool_path = work / "pool.parquet"
    pool = (assemble(work) if args.rebuild or not pool_path.is_file()
            else pd.read_parquet(pool_path))
    pool = pool.reset_index(drop=True)

    report = {"descriptive": describe(pool, work)}
    print("=" * 100, "\nODDS RATIOS PER DECADE (run fixed effects, cluster bootstrap)", sep="")
    report["effects"] = effects(pool, n_boot=args.n_boot)
    print("=" * 100, "\nCROSS-FITTED BEST CELL", sep="")
    report["crossfit_cells"] = crossfit_cells(pool)
    print("=" * 100, "\nCROSS-FITTED A-PRIORI CELL SCORE", sep="")
    report["crossfit_score"] = crossfit_score(pool, work)
    report["space_group_reallocation"] = space_group_reallocation(pool)
    report["prior_runs"] = prior_runs(pool)

    out = results / "chemsys_conditioning.json"
    json.dump(report, open(out, "w"), indent=1, default=float)
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
