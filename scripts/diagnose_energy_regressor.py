"""Separate known-gene regression errors from generated-structure energy gaps.

Uses the immutable precision-study artifact and the model's original dataset
cache. No inference, MLIP calculation, relaxation, or DFT calculation is run.
All energies are formation energies in eV/atom on the saved common reference.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gc
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from wyckoff_transformer.dataset_cache import build_info, dataset_cache_dir, load_split
from wyckoff_transformer.evaluation.gene_hash import GENE_HASH_VERSION, gene_key
from wyckoff_transformer.paths import runs_root, wandb_dir


SOURCE = "symmetry-advantage/WyckoffTransformer/energy_regressor_precision_rycc5sdi:v1"
KEY_COLUMNS = [
    "spacegroup_number", "elements", "site_symmetries_augmented",
    "sites_enumeration_augmented",
]
ENERGY = "formation_energy_per_atom"
MINIMUM = "gene_min_formation_energy_per_atom"
COMPONENTS = ["regressor_error", "structure_gap", "structure_error"]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def scan_reference(cache, wanted):
    """Keep every matching archive row, including its split and stored target."""
    matched = []
    sizes = {}
    for split in ("train", "val", "test"):
        frame = load_split(cache, split, columns=KEY_COLUMNS + [ENERGY, MINIMUM])
        sizes[split] = len(frame)
        columns = [frame[c].values for c in KEY_COLUMNS]
        for i, (identifier, values, energy, minimum) in enumerate(zip(
            frame.index, zip(*columns), frame[ENERGY].values, frame[MINIMUM].values,
        )):
            key = ":".join(map(str, gene_key(dict(zip(KEY_COLUMNS, values)))))
            if key in wanted:
                matched.append(dict(
                    key=key, split=split, immutable_id=str(identifier),
                    formation_energy_per_atom=float(energy),
                    gene_min_formation_energy_per_atom=float(minimum),
                ))
            if (i + 1) % 500_000 == 0:
                print(f"Scanned {split}: {i + 1:,}/{len(frame):,}", flush=True)
        print(f"Finished {split}: {len(frame):,} rows, {len(matched):,} matches so far", flush=True)
        del frame, columns
        gc.collect()
    return pd.DataFrame(matched), sizes


def reference_minima(matched):
    """Recompute minima; refuse inconsistent or incomplete stored gene targets."""
    if not np.isfinite(matched[[ENERGY, MINIMUM]].to_numpy()).all():
        raise ValueError("Non-finite reference energies")
    grouped = matched.groupby("key", sort=True)
    minima = grouped.agg(**{
        MINIMUM: (ENERGY, "min"),
        "stored_target_min": (MINIMUM, "min"),
        "stored_target_max": (MINIMUM, "max"),
        "n_reference_rows": (ENERGY, "size"),
    })
    for column in ["stored_target_min", "stored_target_max"]:
        if not np.allclose(minima[MINIMUM], minima[column], rtol=0, atol=1e-10):
            raise ValueError("Recomputed archive minimum disagrees with stored training target")
    counts = pd.crosstab(matched.key, matched.split)
    for split in ["train", "val", "test"]:
        minima[f"n_{split}"] = counts.get(split, pd.Series(dtype=int)).reindex(minima.index, fill_value=0)
    minima["gene_in_train"] = minima.n_train > 0
    train = matched[matched.split == "train"].groupby("key")[ENERGY].min()
    minima["train_min_formation_energy_per_atom"] = train
    best = matched.loc[matched.groupby("key")[ENERGY].idxmin()].set_index("key")
    minima["minimum_immutable_id"] = best.immutable_id
    minima["minimum_split"] = best.split
    return minima.drop(columns=["stored_target_min", "stored_target_max"])


def decompose(frame, minima):
    """Use final genes, common finite pairs, and explicit unknown-gene exclusions."""
    out = frame.copy()
    out["reference_known"] = out.final_key.isin(minima.index)
    labelled = out.final_gene_novel_fixed.notna()
    recorded_known = out.loc[labelled, "final_gene_novel_fixed"].astype(str) == "False"
    if not np.array_equal(recorded_known, out.loc[labelled, "reference_known"]):
        raise ValueError("Reference membership differs from the pinned precision study")
    if MINIMUM in out:
        available = out[MINIMUM].notna()
        rebuilt = out.loc[available, "final_key"].map(minima[MINIMUM])
        if not np.allclose(out.loc[available, MINIMUM], rebuilt, rtol=0, atol=1e-10):
            raise ValueError("Archive sample targets changed since the precision study")
    for column in minima.columns:
        out[column] = out.final_key.map(minima[column])
    # Archive truth in the parent report is ORB. Use its saved DFT row energy
    # here, so the archive control compares DFT with DFT. B remains cross-scale.
    if ENERGY in out and "dataset" in out:
        out[ENERGY] = out[ENERGY].where(out.dataset == "a", out.truth)
    else:
        out[ENERGY] = out.truth
    out["regressor_error"] = out.final_prediction - out[MINIMUM]
    out["structure_gap"] = out[ENERGY] - out[MINIMUM]
    out["structure_error"] = out.final_prediction - out[ENERGY]
    out["decomposition_available"] = np.isfinite(out[COMPONENTS].to_numpy()).all(axis=1)
    out["decomposition_exclusion"] = np.select(
        [~out.reference_known, ~np.isfinite(out.final_prediction), ~np.isfinite(out[ENERGY])],
        ["unknown final gene", "missing prediction", "missing structure energy"],
        default="",
    )
    valid = out[out.decomposition_available]
    if not np.allclose(
        valid.structure_error, valid.regressor_error - valid.structure_gap,
        rtol=0, atol=1e-10,
    ):
        raise ValueError("Residual decomposition does not close")
    return out


def summarize(frame, **labels):
    """Component MAEs share one denominator; squared errors include covariance."""
    f = frame[frame.decomposition_available]
    result = dict(
        **labels, n_rows=len(frame), n=len(f), excluded=len(frame) - len(f),
        n_genes=f.final_key.nunique(),
    )
    available = frame[np.isfinite(frame.structure_error)]
    result["n_total_pairs"] = len(available)
    result["n_genes_total"] = available.final_key.nunique()
    result["total_mae_all_finite"] = float(available.structure_error.abs().mean()) if len(available) else None
    if f.empty:
        return result
    for column in COMPONENTS:
        error = f[column]
        result.update({f"{column}_{stat}": float(value) for stat, value in {
            "mae": error.abs().mean(), "rmse": np.sqrt((error**2).mean()),
            "bias": error.mean(), "median_absolute": error.abs().median(),
            "p90_absolute": error.abs().quantile(.9),
        }.items()})
    # MAEs do not add. This identity does, including cancellation or reinforcement.
    result["regressor_mse"] = float((f.regressor_error**2).mean())
    result["gap_mse"] = float((f.structure_gap**2).mean())
    result["cross_term"] = float(-2 * (f.regressor_error * f.structure_gap).mean())
    result["structure_mse"] = float((f.structure_error**2).mean())
    result["fraction_below_archive_50meV"] = float((f.structure_gap < -.05).mean())
    result["fraction_above_archive_50meV"] = float((f.structure_gap > .05).mean())
    result["fraction_within_archive_50meV"] = float((f.structure_gap.abs() <= .05).mean())
    if not np.isclose(
        result["structure_mse"], result["regressor_mse"] + result["gap_mse"] + result["cross_term"],
        rtol=1e-10, atol=1e-12,
    ):
        raise ValueError("Squared-error decomposition does not close")
    return result


def summaries(frame):
    results = []

    def add(f, description, **labels):
        for weighting in ["source_rows", "unique_geometry", "equal_gene"]:
            selected = f
            if weighting == "unique_geometry":
                # Archive rows have distinct IDs, not saved geometry digests.
                keys = ["final_key", "geometry"] if (f.dataset != "a").all() else ["index"]
                selected = f.drop_duplicates(keys)
            result = summarize(selected, description=description, weighting=weighting, **labels)
            if weighting == "equal_gene":
                valid = selected[selected.decomposition_available]
                available = selected[np.isfinite(selected.structure_error)]
                result["total_mae_all_finite"] = (
                    float(available.assign(a=available.structure_error.abs()).groupby("final_key").a.mean().mean())
                    if len(available) else None
                )
                # Average absolute/squared/signed error within each gene first.
                for column in COMPONENTS:
                    by_gene = valid.groupby("final_key")[column]
                    if len(valid):
                        result[f"{column}_mae"] = float(valid.assign(a=valid[column].abs()).groupby("final_key").a.mean().mean())
                        result[f"{column}_bias"] = float(by_gene.mean().mean())
                        result[f"{column}_rmse"] = float(np.sqrt(valid.assign(a=valid[column]**2).groupby("final_key").a.mean().mean()))
                # Other distribution statistics above describe source rows only.
                result = {k: v for k, v in result.items() if k not in {
                    "regressor_mse", "gap_mse", "structure_mse", "cross_term",
                } and not any(s in k for s in ["median_absolute", "p90_absolute", "fraction_"])}
            results.append(result)

    for (dataset, cohort), f in frame[frame.dataset == "a"].groupby(["dataset", "cohort"]):
        add(f, "archive split", dataset=dataset, cohort=cohort)
        for known, ff in f.groupby("gene_in_train", dropna=False):
            add(ff, "archive training-gene membership", dataset=dataset, cohort=cohort, gene_in_train=str(known))
    c = frame[frame.dataset == "c"]
    add(c, "DFT all", dataset="c")
    for (known, novel), f in c.groupby(["reference_known", "novel_structure_fixed"], dropna=False):
        add(f, "DFT gene and structure novelty", dataset="c", reference_known=str(known), novel_structure_fixed=str(novel))
    for known, f in c.groupby("gene_in_train", dropna=False):
        add(f, "DFT training-gene membership", dataset="c", gene_in_train=str(known))
        for novel, ff in f.groupby("novel_structure_fixed", dropna=False):
            add(ff, "DFT training-gene membership and structure novelty", dataset="c", gene_in_train=str(known), novel_structure_fixed=str(novel))
    for (cohort, novel), f in c.groupby(["cohort", "novel_structure_fixed"], dropna=False):
        add(f, "DFT per cohort and structure novelty", dataset="c", cohort=cohort, novel_structure_fixed=str(novel))
    for (track, valid), f in frame[frame.dataset == "b"].groupby(["track", "valid_structure"], dropna=False):
        add(f, "ORB final gene (cross-scale)", dataset="b", track=track, valid_structure=str(valid))
        for (same, novel), ff in f.groupby(["same_gene", "novel_structure"], dropna=False):
            add(ff, "ORB gene preservation and structure novelty (cross-scale)", dataset="b", track=track, valid_structure=str(valid), same_gene=str(same), novel_structure=str(novel))
    return pd.DataFrame(results)


def plot(frame, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = frame[(frame.dataset == "c") & frame.decomposition_available].drop_duplicates(["geometry", "final_key"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for novel, f in c.groupby("novel_structure_fixed"):
        label = "Novel structure" if novel else "Known structure"
        for ax, column in zip(axes[:2], ["regressor_error", "structure_gap"]):
            values = np.sort(f[column].abs())
            ax.plot(values, np.arange(1, len(f) + 1) / len(f), label=f"{label} (n={len(f):,})")
    for ax, title in zip(axes[:2], ["Prediction versus archive minimum", "DFT structure versus archive minimum"]):
        ax.set(title=title, xlabel="Absolute difference (eV/atom)", ylabel="Fraction within difference", xlim=(0, .5), ylim=(0, 1))
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=.25)
    f = c[c.novel_structure_fixed == True]
    axes[2].scatter(f.regressor_error, f.structure_gap, s=7, alpha=.4)
    axes[2].set(xlabel="Prediction − archive minimum (eV/atom)", ylabel="DFT structure − archive minimum (eV/atom)", title="Novel structures with known genes")
    axes[2].set_xscale("symlog", linthresh=.025)
    axes[2].set_yscale("symlog", linthresh=.025)
    axes[2].axhline(0, color="grey", linewidth=.6)
    axes[2].axvline(0, color="grey", linewidth=.6)
    for suffix in ["png", "pdf"]:
        fig.savefig(root / f"decomposition.{suffix}", dpi=180)
    plt.close(fig)


def write_report(root, summary, provenance, url):
    s = summary[(summary.weighting == "source_rows") & (summary.description == "DFT gene and structure novelty")]
    lines = [
        "# Energy regressor residual decomposition", "",
        f"Measured {provenance['date']} at git `{provenance['git_commit']}` "
        f"(dirty={provenance['git_dirty']}; exact analysis source is in the artifact). [W&B run]({url}).",
        "",
        f"Source: `{provenance['source_artifact']['name']}`. Model revision: `{provenance['model']['revision']}`.",
        "Training run: `min_energy_adamw_wsd_h7x_xl-20261015-185731`.", "",
        "The final gene is used throughout. Recomputed minima include train, validation "
        "and test, matching the training target. Every matched archive row and split is saved. "
        "All energies are eV/atom on the pinned model's elemental reference; no fitted correction is applied.",
        "",
        "`regressor_error = prediction − archive minimum`; "
        "`structure_gap = DFT structure − archive minimum`; "
        "`structure_error = regressor_error − structure_gap`. "
        "Archive controls use their saved DFT row energies; B uses ORB and C uses DFT. "
        "The second term includes differences in geometry and in the underlying calculations. "
        "It is not a direct measurement of relaxation failure. MAEs are not additive.",
        "",
        "| Known gene | Novel structure | Total N | Decomposed N | Genes | Regressor MAE | Structure-gap MAE | Total MAE | Gap bias |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s.to_dict("records"):
        values = [r.get(k) for k in ["regressor_error_mae", "structure_gap_mae", "total_mae_all_finite", "structure_gap_bias"]]
        fmt = [f"{1000 * x:.1f}" if pd.notna(x) else "—" for x in values]
        lines.append(f"| {r['reference_known']} | {r['novel_structure_fixed']} | {r['n_total_pairs']:.0f} | {r['n']:.0f} | {r['n_genes_total']:.0f} | " + " | ".join(fmt) + " |")
    novel = s[(s.reference_known == "True") & (s.novel_structure_fixed == "True")].iloc[0]
    lines += [
        "", f"For novel structures with known genes, an exact archive-minimum predictor "
        f"would have {1000 * novel.structure_gap_mae:.1f} meV/atom MAE, compared with "
        f"{1000 * novel.structure_error_mae:.1f} for the regressor. "
        "The regressor remains accurate against its actual target in this subgroup. "
        "Its structure-energy error is dominated by the difference between the archive target and the generated structure.",
        "", f"{100 * novel.fraction_below_archive_50meV:.1f}% of these structures lie more than "
        f"50 meV/atom below the archive minimum, and {100 * novel.fraction_above_archive_50meV:.1f}% "
        "lie more than 50 meV/atom above it. A one-sided failure to reach an existing minimum "
        "cannot explain both directions. The archive is incomplete, and differences in geometry "
        "and calculation settings remain mixed in this term. This diagnostic does not separate them.",
    ]
    lines += [
        "", "Table errors and biases are meV/atom. Unknown genes have no archive minimum "
        "and remain explicit exclusions from the decomposition; their total errors remain in rows.csv.gz.",
        "", "summary.csv also includes archive controls by training-gene membership, "
        "cohort splits, unique-geometry summaries, and equal-gene MAE/RMSE/bias. "
        "ORB results are explicitly cross-scale; their structure gaps include the OMat24/MP discrepancy. "
        "Finite-pair denominators are identical across components. "
        "The squared-error identity, including its cross term, is verified for every summary.",
        "", "No new model predictions, relaxations, MLIP calculations or DFT calculations were run.", "",
    ]
    (root / "diagnostic.md").write_text("\n".join(lines))


def main():
    import wandb
    from wyckoff_transformer.cli.template_novelty import fetch_entry
    from wyckoff_transformer.dataset_manifest import load_manifest
    from wyckoff_transformer.energy_fields import check_compatible
    from wyckoff_transformer.field_provenance import check_against_dataset, target_field

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", default=SOURCE)
    parser.add_argument("--source-local", type=Path, default=runs_root() / "regressor_precision_20261028")
    parser.add_argument("--output", type=Path, default=runs_root() / "energy_regressor_diagnostic_20261028")
    parser.add_argument("--resume-run", help="Resume this diagnostic's own W&B run after reusing its reference scan")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    with wandb.init(
        entity="symmetry-advantage", project="WyckoffTransformer",
        job_type="energy_regressor_diagnostic", mode="online", dir=wandb_dir(),
        id=args.resume_run, resume="must" if args.resume_run else None,
        name="Known-gene energy residual decomposition 2026-10-28",
        config=dict(git_commit=commit, git_dirty=dirty, source_artifact=args.source_artifact),
    ) as run:
        source = run.use_artifact(args.source_artifact)
        inputs = root / "source"
        inputs.mkdir(exist_ok=True)
        required = ["rows.csv.gz", "field_provenance.json", "model.json", "provenance.json", "elemental_references.json"]
        for name in required:
            fetch_entry(source.get_entry(name), inputs, args.source_local)
        fields = json.loads((inputs / "field_provenance.json").read_text())
        dataset = fields["dataset"]
        cache = dataset_cache_dir(dataset)
        check_against_dataset(fields, dataset, "known-gene decomposition")
        manifest = load_manifest(dataset)
        check_compatible(target_field(fields), manifest.resolve(ENERGY).energy, "archive energies versus model minimum")
        recorded_build = fields["cache_build"]
        current_build = build_info(cache)
        if any(b != recorded_build for b in current_build.values()):
            raise ValueError("Dataset cache differs from the model's recorded training cache")
        frame = pd.read_csv(inputs / "rows.csv.gz", low_memory=False)
        wanted = set(frame.final_key.dropna())
        provenance = dict(
            date=datetime.now(UTC).isoformat(), git_commit=commit, git_dirty=dirty,
            source_artifact=dict(name=source.qualified_name, id=source.id, digest=source.digest),
            source_files={name: source.manifest.entries[name].digest for name in required},
            model=json.loads((inputs / "model.json").read_text()),
            dataset=dataset, cache_build=current_build, gene_hash_version=GENE_HASH_VERSION,
            units="eV/atom", no_new_calculations=True,
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            requested_keys_sha256=hashlib.sha256("\n".join(sorted(wanted)).encode()).hexdigest(),
        )
        # A rerun can reuse only a lookup built for the same inputs and cache.
        lookup_record = root / "lookup_provenance.json"
        signature = {k: provenance[k] for k in ["cache_build", "gene_hash_version", "requested_keys_sha256"]}
        signature["lookup_version"] = 1
        matched_path = root / "reference_rows.csv.gz"
        if matched_path.exists() and lookup_record.exists() and json.loads(lookup_record.read_text()) == signature:
            matched = pd.read_csv(matched_path)
            sizes = {s: len(load_split(cache, s, columns=[])) for s in current_build}
            print("Reusing verified reference scan", flush=True)
        else:
            matched, sizes = scan_reference(cache, wanted)
            matched.to_csv(matched_path, index=False)
            dump(lookup_record, signature)
        minima = reference_minima(matched)
        minima.to_csv(root / "gene_minima.csv.gz")
        rows = decompose(frame, minima)
        rows.to_csv(root / "rows.csv.gz", index=False)
        summary = summaries(rows)
        summary.to_csv(root / "summary.csv", index=False)
        finite = rows[rows.decomposition_available]
        c = finite[finite.dataset == "c"]
        per_gene = c.groupby(["final_key", "novel_structure_fixed"]).agg(
            n=("structure_error", "size"),
            regressor_mae=("regressor_error", lambda x: x.abs().mean()),
            structure_gap_mae=("structure_gap", lambda x: x.abs().mean()),
            total_mae=("structure_error", lambda x: x.abs().mean()),
        )
        per_gene.to_csv(root / "dft_per_gene.csv.gz")
        outlier_ids = set(c.structure_gap.abs().nlargest(25).index) | set(c.regressor_error.abs().nlargest(25).index)
        c.loc[sorted(outlier_ids)].to_csv(root / "dft_outliers.csv", index=False)
        provenance["scan_split_sizes"] = sizes
        dump(root / "provenance.json", provenance)
        dump(root / "verification.json", dict(
            matched_reference_rows=len(matched), matched_genes=len(minima),
            decompose_rows=len(finite),
            maximum_closure_error=float((finite.structure_error - finite.regressor_error + finite.structure_gap).abs().max()),
            recomputed_minima_equal_stored_targets=True,
            original_membership_and_archive_sample_targets_unchanged=True,
            exclusion_counts=rows[~rows.decomposition_available].groupby(["dataset", "decomposition_exclusion"]).size().rename("n").reset_index().to_dict("records"),
        ))
        plot(rows, root)
        write_report(root, summary, provenance, run.url)
        artifact = wandb.Artifact(f"energy_regressor_diagnostic_{run.id}", type="energy_regressor_evaluation", metadata=provenance)
        for path in root.iterdir():
            if path.is_file() and path.name != "wandb_result.json":
                artifact.add_file(str(path))
        for name in required[1:]:
            artifact.add_file(str(inputs / name), name=f"source/{name}")
        artifact.add_file(__file__, name="code/diagnose_energy_regressor.py")
        artifact.add_file("src/wyckoff_transformer/tests/test_energy_regressor_diagnostic.py", name="code/test_energy_regressor_diagnostic.py")
        artifact.add_file("docs/energy_regressor_precision.md", name="docs/energy_regressor_precision.md")
        run.log_artifact(artifact).wait()
        run.summary.update(dict(matched_genes=len(minima), matched_reference_rows=len(matched), decomposition_rows=len(finite)))
        dump(root / "wandb_result.json", dict(url=run.url, id=run.id, artifact=artifact.qualified_name))
        print(summary[(summary.description == "DFT gene and structure novelty") & (summary.weighting == "source_rows")].to_string(index=False), flush=True)
        print("W&B", run.url, flush=True)


if __name__ == "__main__":
    main()
