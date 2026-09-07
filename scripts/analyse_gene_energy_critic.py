"""Measure how far a Wyckoff gene determines energy, and what ranking genes by a
predicted energy could buy.

Three sub-commands, each printing the tables quoted in
``docs/gene_energy_critic_study.md``:

``uniqueness``
    How far the pair (reduced formula, energy) determines the Wyckoff gene: the
    share of rows in formulas holding several distinct genes, the energy gap
    between a formula's lowest-energy gene and the next distinct one, how much
    data a "one ground state per formula" target discards, and the exact-formula
    lookup baseline from train to test.

``signal``
    Training-free lower bounds on gene to ``energy_above_hull`` predictability:
    the mean absolute error of predicting a held-out entry's energy by the train
    mean over rows sharing its reduced formula, its formula and space group, or
    its exact gene. Stratified by positional degrees of freedom.

``enrichment``
    End-to-end test on a real generated set: rank the sampled genes by a
    training-free proxy critic and report the MetaSUN rate of the top slices
    against the measured energies, alongside the ceiling a perfect energy
    ranking would reach.

Run with ``uv run python scripts/analyse_gene_energy_critic.py <sub-command>``.
No sub-command trains anything; ``signal`` and ``enrichment`` are the evidence
the study cites for whether a learned critic is worth building.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

LEMAT_CACHE = Path("cache/lemat_bulk_ehull/data.pkl.gz")
MP20_CACHE = Path("cache/mp_20/data.pkl.gz")
DEFAULT_GENES = Path("generated/upi73i4k/wyckoff_genes_ehull0_n2500.json.gz")
DEFAULT_STRUCTURES = Path("generated/upi73i4k/protocol/structures.csv")

ENERGY_COLUMNS = ("energy_above_hull", "e_above_hull", "formation_energy_per_atom")
SITE_RE = re.compile(r"^(\d+)([a-zA-Z])$")


def pick_energy_column(df: pd.DataFrame) -> str:
    for name in ENERGY_COLUMNS:
        if name in df.columns:
            return name
    raise KeyError(f"No energy column among {ENERGY_COLUMNS} in {list(df.columns)}")


def reduced_formula(elements, multiplicity) -> str:
    """``Au1 Sc3 Zn2``: element counts divided by their greatest common divisor."""
    counts: dict[str, int] = {}
    for element, mult in zip(elements, multiplicity):
        counts[str(element)] = counts.get(str(element), 0) + int(mult)
    divisor = 0
    for value in counts.values():
        divisor = math.gcd(divisor, value)
    return " ".join(f"{k}{v // divisor}" for k, v in sorted(counts.items()))


def gene_key(spacegroup, elements, letters, multiplicity) -> str:
    """Space group plus the sorted multiset of (element, Wyckoff letter, multiplicity)."""
    sites = sorted(zip(map(str, elements), map(str, letters), (int(m) for m in multiplicity)))
    return f"{int(spacegroup)}|" + ",".join(f"{e}{l}{m}" for e, l, m in sites)


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    formulas, genes, dofs = [], [], []
    for elements, multiplicity, spacegroup, letters, dof in zip(
        df["elements"].values,
        df["multiplicity"].values,
        df["spacegroup_number"].values,
        df["wyckoff_letters"].values,
        df["dof"].values,
    ):
        formulas.append(reduced_formula(elements, multiplicity))
        genes.append(gene_key(spacegroup, elements, letters, multiplicity))
        dofs.append(int(sum(int(x) for x in dof)))
    df = df.assign(fkey=formulas, gkey=genes, sdof=dofs)
    return df.assign(fsg=df["fkey"] + "|" + df["spacegroup_number"].astype(str))


def load_cache(path: Path) -> dict[str, pd.DataFrame]:
    started = time.time()
    data = pd.read_pickle(path)
    if not isinstance(data, dict):
        data = {"all": data}
    sizes = ", ".join(f"{k} {len(v)}" for k, v in data.items())
    print(f"loaded {path} in {time.time() - started:.0f}s: {sizes}", flush=True)
    return data


# --------------------------------------------------------------------------- #
# uniqueness
# --------------------------------------------------------------------------- #
def report_uniqueness(df: pd.DataFrame, name: str, energy: str) -> pd.DataFrame:
    n = len(df)
    values = df[energy]
    # "metastable" only means something for an above-hull column, not for formation energy.
    metastable = f", share <= 0.1 {(values <= 0.1).mean():.1%}" if "hull" in energy else ""
    print(
        f"\n[{name}] rows {n}; {energy}: median {values.median():.3f}, "
        f"p90 {values.quantile(0.9):.3f}{metastable}"
    )
    df = df.sort_values(["fkey", energy], kind="stable")
    grouped = df.groupby("fkey", sort=False)
    entries, genes = grouped.size(), grouped["gkey"].nunique()
    print(f"  reduced formulas {grouped.ngroups}; ground-state-only keeps {grouped.ngroups / n:.1%} of rows")
    print(
        f"  formulas with >=2 distinct genes: {(genes >= 2).mean():.1%} of formulas, "
        f"holding {entries[genes >= 2].sum() / n:.1%} of rows"
    )
    print(f"  rows sharing formula AND gene with another row: {(entries - genes).sum() / n:.1%}")

    ground_state = grouped.first()
    differs = df[df["gkey"].values != df["fkey"].map(ground_state["gkey"]).values]
    nearest = differs.groupby("fkey", sort=False)[energy].first()
    gaps = (nearest - ground_state[energy].reindex(nearest.index)).values
    if len(gaps):
        quantiles = "/".join(f"{x:.3f}" for x in np.quantile(gaps, [0.1, 0.25, 0.5, 0.75, 0.9]))
        print(f"  gap to nearest different-gene polymorph, n={len(gaps)}: q10/25/50/75/90 = {quantiles}")
        for threshold in (0.005, 0.01, 0.025, 0.05, 0.1):
            print(f"    share of gaps <= {threshold:.3f}: {(gaps <= threshold).mean():.1%}")
    return ground_state


def cmd_uniqueness(args: argparse.Namespace) -> None:
    data = load_cache(Path(args.cache))
    energy = pick_energy_column(next(iter(data.values())))
    print(f"energy column: {energy}", flush=True)
    data = {split: add_keys(df) for split, df in data.items()}
    ground_states = {split: report_uniqueness(df, split, energy) for split, df in data.items()}

    if "train" in data and "test" in data:
        train, test = data["train"], data["test"]
        gs_gene = ground_states["train"]["gkey"]
        all_genes = train.groupby("fkey")["gkey"].agg(set)
        known = test["fkey"].isin(gs_gene.index)
        print(f"\n[test vs train] test formula present in train: {known.mean():.1%}")
        subset = test[known]
        hit_gs = np.array([gs_gene[f] == g for f, g in zip(subset["fkey"], subset["gkey"])])
        hit_any = np.array([g in all_genes[f] for f, g in zip(subset["fkey"], subset["gkey"])])
        print(
            f"  exact-formula lookup baseline over all test rows: "
            f"{hit_gs.sum() / len(test):.1%} (ground-state gene) / {hit_any.sum() / len(test):.1%} (any polymorph)"
        )
        both = pd.concat([train.assign(split="train"), test.assign(split="test")])
        lowest = both.sort_values(energy, kind="stable").drop_duplicates("fkey", keep="first")
        print(
            f"  test rows that are their formula's lowest-energy entry in train+test: "
            f"{(lowest['split'] == 'test').sum() / len(test):.1%}"
        )


# --------------------------------------------------------------------------- #
# signal
# --------------------------------------------------------------------------- #
def cmd_signal(args: argparse.Namespace) -> None:
    data = load_cache(Path(args.cache))
    energy = pick_energy_column(data["train"])
    train, test = add_keys(data["train"]), add_keys(data["test"])
    truth = test[energy].values
    print(
        f"\ntest {energy}: mean {truth.mean():.3f}, sd {truth.std():.3f}, "
        f"median {np.median(truth):.3f}; metastable (<= 0.1) {(truth <= 0.1).mean():.1%}"
    )
    within = train.groupby("fkey")[energy].std().dropna()
    print(f"within-formula sd of {energy} (train): mean {within.mean():.3f}, median {within.median():.3f}")

    fallback = train[energy].mean()
    predictions = {"global mean": pd.Series(np.full(len(test), fallback), index=test.index)}
    for label, column in (("formula", "fkey"), ("formula+SG", "fsg"), ("exact gene", "gkey")):
        predictions[label] = test[column].map(train.groupby(column)[energy].mean())

    header = f"{'baseline':<14} {'coverage':>9} {'MAE|cov':>8} {'MAE(all)':>9} {'top-10% metastable':>19}"
    print(f"\n{header}")
    for label, prediction in predictions.items():
        covered = prediction.notna().values
        filled = prediction.fillna(fallback).values
        mae_covered = np.abs(filled[covered] - truth[covered]).mean() if covered.any() else float("nan")
        order = np.argsort(filled, kind="stable")[: len(truth) // 10]
        print(
            f"{label:<14} {covered.mean():>8.1%} {mae_covered:>8.3f} "
            f"{np.abs(filled - truth).mean():>9.3f} {(truth[order] <= 0.1).mean():>18.1%}"
        )

    best = (
        predictions["exact gene"]
        .fillna(predictions["formula+SG"])
        .fillna(predictions["formula"])
        .fillna(fallback)
        .values
    )
    print("\nby positional dof (exact gene where covered, else formula+SG, else formula):")
    print(f"{'dof':<8} {'n':>8} {'sd(E_hull)':>11} {'MAE':>7} {'gene cov':>9} {'top-10% meta':>13}")
    for low, high in ((0, 0), (1, 2), (3, 5), (6, 10), (11, 10**9)):
        selected = ((test["sdof"] >= low) & (test["sdof"] <= high)).values
        ys, ps = truth[selected], best[selected]
        if not len(ys):
            continue
        order = np.argsort(ps, kind="stable")[: max(1, len(ys) // 10)]
        label = str(low) if low == high else (f">{low - 1}" if high > 10**8 else f"{low}-{high}")
        coverage = predictions["exact gene"][selected].notna().mean()
        print(
            f"{label:<8} {len(ys):>8} {ys.std():>11.3f} {np.abs(ps - ys).mean():>7.3f} "
            f"{coverage:>8.1%} {(ys[order] <= 0.1).mean():>12.1%}"
        )


# --------------------------------------------------------------------------- #
# enrichment
# --------------------------------------------------------------------------- #
def parse_generated_gene(gene: dict) -> tuple[str, str, int]:
    """``{"group": 123, "sites": [["1a", "2g"]], "species": ["Sc"], ...}`` to keys."""
    counts: dict[str, int] = {}
    sites: list[tuple[str, str, int]] = []
    for species, letters in zip(gene["species"], gene["sites"]):
        for site in letters:
            matched = SITE_RE.match(site)
            if matched is None:
                raise ValueError(f"Unparsable Wyckoff site {site!r}")
            multiplicity, letter = int(matched.group(1)), matched.group(2)
            counts[species] = counts.get(species, 0) + multiplicity
            sites.append((species, letter, multiplicity))
    divisor = 0
    for value in counts.values():
        divisor = math.gcd(divisor, value)
    formula = " ".join(f"{k}{v // divisor}" for k, v in sorted(counts.items()))
    key = f"{int(gene['group'])}|" + ",".join(f"{e}{l}{m}" for e, l, m in sorted(sites))
    return formula, key, len(sites)


def print_slices(generated: pd.DataFrame, column: str, label: str, sizes: tuple[int, ...]) -> None:
    frame = generated.copy()
    scores = frame[column]
    frame["_rank"] = scores.fillna(scores.max() + 1e6)
    frame = frame.sort_values("_rank", kind="stable")
    base = generated["metasun"].mean()
    print(f"\n{label}")
    print(f"{'kept':>6} {'MetaSUN':>9} {'lift':>6} {'SUN':>8} {'novel':>7} {'valid':>7} {'median e_hull':>14}")
    for size in sizes:
        head = frame.head(size)
        print(
            f"{size:>6} {head['metasun'].mean():>9.3f} {head['metasun'].mean() / base:>5.2f}x "
            f"{head['sun'].mean():>8.4f} {head['novel_structure'].mean():>7.3f} "
            f"{head['valid_structure'].mean():>7.3f} {head['e_above_hull'].median():>14.3f}"
        )


def cmd_enrichment(args: argparse.Namespace) -> None:
    with gzip.open(args.genes, "rt") as handle:
        genes = json.load(handle)
    generated = pd.DataFrame(
        [parse_generated_gene(g) for g in genes], columns=["fkey", "gkey", "n_sites"]
    ).assign(index=range(len(genes)))

    scored = pd.read_csv(args.structures)
    generated = generated.merge(
        scored[["index", "e_above_hull", "valid_structure", "novel_structure", "unique_structure"]],
        on="index",
        how="left",
    )
    for flag in ("valid_structure", "novel_structure", "unique_structure"):
        # .eq(True) maps a missing flag to False without pandas' object-dtype downcast warning.
        generated[flag] = generated[flag].eq(True)
    stable = generated["e_above_hull"].notna()
    selected = generated["valid_structure"] & generated["novel_structure"] & generated["unique_structure"]
    generated["metasun"] = stable & (generated["e_above_hull"] <= 0.1) & selected
    generated["sun"] = stable & (generated["e_above_hull"] <= 0.0) & selected
    print(
        f"generated genes {len(generated)}; MetaSUN base {generated['metasun'].mean():.4f} "
        f"({generated['metasun'].sum()}), SUN base {generated['sun'].mean():.4f} ({generated['sun'].sum()})",
        flush=True,
    )

    train = add_keys(load_cache(Path(args.cache))["train"])
    energy = pick_energy_column(train)
    generated["critic_formula"] = generated["fkey"].map(train.groupby("fkey")[energy].mean())
    generated["critic_formula_min"] = generated["fkey"].map(train.groupby("fkey")[energy].min())
    generated["critic_gene"] = generated["gkey"].map(train.groupby("gkey")[energy].mean())
    print(
        f"\ncritic coverage: formula {generated['critic_formula'].notna().mean():.1%}, "
        f"exact gene {generated['critic_gene'].notna().mean():.1%}"
    )

    measured = generated.dropna(subset=["e_above_hull"])
    for column in ("critic_formula", "critic_formula_min"):
        subset = measured.dropna(subset=[column])
        pearson = subset[column].corr(subset["e_above_hull"])
        spearman = subset[column].corr(subset["e_above_hull"], method="spearman")
        print(f"{column}: n={len(subset)}, Pearson {pearson:.3f}, Spearman {spearman:.3f} vs measured e_above_hull")

    sizes = tuple(args.sizes)
    print_slices(generated, "critic_formula", "formula-mean critic (training-free)", sizes)
    print_slices(generated, "critic_formula_min", "formula-min critic (training-free)", sizes)
    generated["_oracle"] = generated["e_above_hull"].fillna(9.0)
    print_slices(generated, "_oracle", "ORACLE: rank by the true measured energy (ceiling)", sizes)
    achievable = min(1.0, generated["metasun"].sum() / sizes[len(sizes) // 2])
    print(
        f"\nFor reference, perfectly identifying every MetaSUN gene would give "
        f"{achievable:.3f} at {sizes[len(sizes) // 2]} kept."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    uniqueness = subparsers.add_parser("uniqueness", help="does (formula, energy) determine the gene?")
    uniqueness.add_argument("--cache", default=str(LEMAT_CACHE), help=f"dataset cache (default {LEMAT_CACHE})")
    uniqueness.set_defaults(func=cmd_uniqueness)

    signal = subparsers.add_parser("signal", help="training-free gene to energy predictability")
    signal.add_argument("--cache", default=str(LEMAT_CACHE), help=f"dataset cache (default {LEMAT_CACHE})")
    signal.set_defaults(func=cmd_signal)

    enrichment = subparsers.add_parser("enrichment", help="proxy critic on a real generated set")
    enrichment.add_argument("--cache", default=str(LEMAT_CACHE))
    enrichment.add_argument("--genes", default=str(DEFAULT_GENES))
    enrichment.add_argument("--structures", default=str(DEFAULT_STRUCTURES))
    enrichment.add_argument("--sizes", type=int, nargs="+", default=[100, 250, 500, 1000, 2500])
    enrichment.set_defaults(func=cmd_enrichment)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
