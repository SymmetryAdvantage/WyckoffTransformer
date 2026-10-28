"""Repeat the saved precision study with one observed minimum per final gene.

Run with the platform's documented interpreter. Inputs and outputs are W&B
artifacts; no new prediction, relaxation, MLIP calculation or DFT is run.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from wyckoff_transformer.evaluation.gene_minimum_evaluation import (
    add_archive_candidates, error_statistics, select_minima,
)
from wyckoff_transformer.paths import runs_root, wandb_dir


SOURCE = "symmetry-advantage/WyckoffTransformer/energy_regressor_diagnostic_53xvvvq7:v2"
CODE = [
    "scripts/evaluate_energy_regressor_gene_minima.py",
    "src/wyckoff_transformer/evaluation/gene_minimum_evaluation.py",
    "src/wyckoff_transformer/tests/test_gene_minimum_evaluation.py",
]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def make_pools(frame, reference):
    frame = frame.copy()
    frame["candidate_id"] = "source:" + frame.index.astype(str)
    frame["origin"] = np.where(frame.dataset == "a", "archive_sample", "generated")
    pools, definitions = [], {}

    def add(name, f, energy_source, group="final_key", primary=True):
        f = f.copy()
        f["candidate_id"] = name + "|" + f.candidate_id
        f["pool"] = name
        f["group_key"] = f[group]
        definitions[name] = dict(energy_source=energy_source, group_gene=group, primary=primary)
        pools.append(f)

    a, b, c = [frame[frame.dataset == part] for part in ["a", "b", "c"]]
    add("a_orb", a, "ORB")
    add("a_dft", a.assign(truth=a.formation_energy_per_atom), "DFT", primary=False)
    for track, f in b.groupby("track"):
        add(f"b_{track}", f, "ORB")
        add(f"b_initial_{track}", f, "ORB", group="initial_key", primary=False)
    add("c_generated_dft", c, "DFT")
    add("c_archive_union_dft", add_archive_candidates(c, reference), "DFT")
    for cohort, f in c.groupby("cohort"):
        add(f"c_cohort:{cohort}", f, "DFT", primary=False)
    return pd.concat(pools, ignore_index=True), definitions


def summarize(candidates, selected):
    results = []

    def add(f, pool, description, selection="minimum", predictor="final_prediction", **labels):
        results.append(dict(
            pool=pool, description=description, selection=selection,
            predictor=predictor, **labels, **error_statistics(f, prediction=predictor),
        ))

    for pool, original in candidates.groupby("pool", sort=True):
        f = selected[selected.pool == pool]
        add(original, pool, "overall", selection="all_rows")
        add(f, pool, "overall")
        if pool.startswith("a_"):
            for split, ff in f.groupby("cohort"):
                add(ff, pool, "archive split", cohort=split)
                if pool == "a_orb":
                    add(ff.assign(truth=ff.gene_min_formation_energy_per_atom), pool, "archive DFT gene minimum", cohort=split)
                    add(ff.assign(truth=ff.formation_energy_per_atom), pool, "archive DFT structure", cohort=split)
                    add(ff.assign(final_prediction=ff.formation_energy_per_atom), pool, "archive DFT versus ORB", cohort=split)
        if pool.startswith("b_"):
            add(f, pool, "overall", predictor="initial_prediction")
            axes = ["initial_gene_novel_reference", "final_gene_novel_reference", "novel_structure", "same_gene", "valid_structure", "reference"]
        elif pool.startswith("c_"):
            axes = ["final_gene_novel_fixed", "novel_structure_fixed", "novel_structure_mp20", "gene_in_train", "origin"]
        else:
            axes = ["gene_in_train"]
        for axis in axes:
            for value, ff in f.groupby(axis, dropna=False):
                add(ff, pool, "marginal split", axis=axis, value=str(value))
        if pool in ["c_generated_dft", "c_archive_union_dft"]:
            for (novel_gene, novel_structure), ff in f.groupby(["final_gene_novel_fixed", "novel_structure_fixed"], dropna=False):
                labels = dict(novel_gene=str(novel_gene), novel_structure=str(novel_structure))
                add(ff, pool, "gene and structure novelty", **labels)
                known = ff[
                    np.isfinite(ff.gene_min_formation_energy_per_atom)
                    & np.isfinite(ff.final_prediction) & np.isfinite(ff.truth)
                ]
                add(known.assign(truth=known.gene_min_formation_energy_per_atom), pool, "prediction versus archive minimum", **labels)
                add(known.assign(final_prediction=known.truth, truth=known.gene_min_formation_energy_per_atom), pool, "structure versus archive minimum", **labels)
    return pd.DataFrame(results)


def lookup(summary, pool, description="overall", **labels):
    mask = (summary.pool == pool) & (summary.description == description) & (summary.selection == "minimum") & (summary.predictor == "final_prediction")
    for key, value in labels.items():
        mask &= summary[key].astype(str) == str(value)
    found = summary[mask]
    if len(found) != 1:
        raise ValueError(f"Expected one summary for {pool}/{description}/{labels}, got {len(found)}")
    return found.iloc[0]


def metric(value):
    return f"{1000 * value:.1f}" if pd.notna(value) else "—"


def report(summary, selected, provenance, url):
    def table(headers, rows):
        return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows]

    body = [
        "## E Repeat: one lowest-energy observed structure per gene", "",
        f"Measured {provenance['date'][:10]} at git `{provenance['git_commit']}` with working-tree "
        f"source preserved in the [W&B analysis run]({url}). The model and training run "
        "at the top of this report are unchanged. Inputs are pinned to "
        f"`{provenance['source_artifact']['name']}`; its lineage points to the original precision study.", "",
        "The minimum is chosen by observed **energy per atom**, before filtering predictions "
        "or splitting novelty categories. Gene identity is the augmentation-invariant final "
        "gene recovered from the saved structure. Ties use the geometry identifier, then "
        "the pinned source-row ID. Every pool contains exactly one retained structure per "
        "resolved gene with a finite energy. Missing predictions remain exclusions. "
        "Rows without recoverable genes or energies are retained in the selection audit. "
        "Novelty labels belong to the selected structure; a gene is never selected separately "
        "inside known and novel structure categories.", "",
        "ORB and DFT minima are computed separately. Protocol minima are pooled across "
        "all available cohorts within each relaxation track. Archive ORB minima are chosen "
        "across the original 15,000 sampled rows, then reported by the selected row's split; "
        "they are not minima over the full ORB archive. DFT minima are chosen across all "
        "Figshare cohorts. These are observed minima at the available search budgets, not "
        "estimates of global physical minima. Energy normalization, exclusions and the "
        "cross-scale ORB comparison follow the original study. The inherited pooling of "
        "raw DFT energies assumes the same nominal MP PBE/PBE+U scale; differences between "
        "calculations remain unresolved. Separate minima within each DFT cohort are also saved.", "",
        "**Errors below are meV/atom.** Bias is prediction minus the retained observation. "
        "No new inference, relaxation, MLIP calculation or DFT calculation was run.", "",
        "### Archive ORB minima", "",
    ]
    rows = []
    for split in ["train", "val", "test"]:
        r = lookup(summary, "a_orb", "archive split", cohort=split)
        values = [lookup(summary, "a_orb", desc, cohort=split).mae for desc in ["archive DFT gene minimum", "archive DFT structure", "archive DFT versus ORB"]]
        rows.append([split, int(r.n), metric(r.mae), metric(r.rmse), metric(r.bias), *map(metric, values)])
    body += table(["Split", "N", "ORB MAE", "ORB RMSE", "ORB bias", "DFT gene-min MAE", "DFT structure MAE", "DFT–ORB MAE"], rows)
    body += ["", "### Protocol ORB minima", ""]
    rows = []
    for track in ["fixed_symmetry", "free"]:
        pool = f"b_{track}"
        for predictor in ["initial_prediction", "final_prediction"]:
            r = summary[(summary.pool == pool) & (summary.description == "overall") & (summary.selection == "minimum") & (summary.predictor == predictor)].iloc[0]
            rows.append([track, predictor.removesuffix("_prediction"), int(r.n), int(r.excluded), metric(r.mae), metric(r.rmse), metric(r.bias)])
    body += table(["Track", "Predictor gene", "N", "Excluded", "MAE", "RMSE", "Bias"], rows)
    body += ["", "Both predictor columns score the same structures selected by final gene. "
             "Initial-gene scores therefore include any gene change during relaxation. "
             "A companion selection by initial gene is in summary.csv.", "", "### Generated DFT minima", ""]
    rows = []
    for axis, reference in [("novel_structure_fixed", "LeMat-fixed"), ("novel_structure_mp20", "MP-20")]:
        for novel in [False, True]:
            r = lookup(summary, "c_generated_dft", "marginal split", axis=axis, value=str(novel))
            rows.append([reference, novel, int(r.n), int(r.excluded), metric(r.mae), metric(r.rmse), metric(r.bias)])
    body += table(["Novelty reference", "Novel structure", "N", "Excluded", "MAE", "RMSE", "Bias"], rows)
    body += ["", "### Including the archive minimum as an eligible structure", "",
        "The generated-only pool retains the best generated observation for each gene. "
        "The second pool also offers that gene's lowest DFT structure from all "
        "LeMat-Bulk-fixed training-dataset splits. It contains the same eligible genes; "
        "archive-only genes are not added. No ORB energy enters this union. Archive IDs "
        "and splits are saved, and all such archive structures are known to LeMat-fixed. "
        "MP-20 novelty is left unavailable for archive candidates rather than inferred.", ""]
    rows = []
    for pool, label in [("c_generated_dft", "Generated only"), ("c_archive_union_dft", "Generated + archive")]:
        for novel in [False, True]:
            r = lookup(summary, pool, "marginal split", axis="final_gene_novel_fixed", value=str(novel))
            rows.append([label, novel, int(r.n), int(r.excluded), metric(r.mae), metric(r.rmse), metric(r.bias)])
    body += table(["Eligible observations", "Novel gene", "N", "Excluded", "MAE", "RMSE", "Bias"], rows)
    body += ["", "### Known-gene diagnostic after minimum selection", ""]
    rows = []
    for pool, label in [("c_generated_dft", "Generated only"), ("c_archive_union_dft", "Generated + archive")]:
        for novel in [False, True]:
            labels = dict(novel_gene="False", novel_structure=str(novel))
            r = lookup(summary, pool, "gene and structure novelty", **labels)
            reg = lookup(summary, pool, "prediction versus archive minimum", **labels)
            gap = lookup(summary, pool, "structure versus archive minimum", **labels)
            rows.append([label, novel, int(r.n), metric(reg.mae), metric(gap.mae), metric(r.mae)])
    body += table(["Eligible observations", "Novel selected structure", "N", "Prediction vs archive minimum MAE", "Structure vs archive minimum MAE", "Total MAE"], rows)
    main = lookup(summary, "c_generated_dft")
    unknown = lookup(summary, "c_generated_dft", "marginal split", axis="final_gene_novel_fixed", value="True")
    joint = lookup(summary, "c_generated_dft", "gene and structure novelty", novel_gene="False", novel_structure="True")
    body += ["",
        f"Generated-only selection retains {int(main.n_rows):,} genes "
        f"({int(main.n):,} finite prediction–target pairs) and has {metric(main.mae)} meV/atom "
        f"overall MAE. Novel genes have {metric(unknown.mae)} meV/atom MAE. "
        f"Novel selected structures with known genes have {metric(joint.mae)} meV/atom MAE. "
        "Selecting one observed minimum per gene removes repeated weighting, but does not "
        "guarantee that the generated minimum coincides with the archive minimum. "
        "The gap between them still mixes geometry and calculation differences.", "",
        "The full artifact contains the selected rows and their source IDs, every candidate's "
        "selection status, all pooled and marginal summaries, per-cohort DFT minima, the "
        "archive-union candidates, exact input digests, verification records, figure and source. "
        "Reproduce with `scripts/evaluate_energy_regressor_gene_minima.py` using the platform's "
        "documented interpreter. `--update-doc` replaces this section in the repository report.", "",
    ]
    # Preserve the per-cohort MP-20 comparison without selecting minima by novelty.
    cohorts = ["", "## Per-cohort generated DFT minima against MP-20", "",
               "Each cohort independently retains one minimum per final gene, then splits "
               "the retained structures by novelty. These are companion pools, not subsets "
               "of the globally selected DFT pool.", ""]
    rows = []
    for pool in sorted(summary.pool.unique()):
        if not pool.startswith("c_cohort:"):
            continue
        parts = []
        for novel in [False, True]:
            r = summary[(summary.pool == pool) & (summary.description == "marginal split") & (summary.axis == "novel_structure_mp20") & (summary.value == str(novel))]
            parts += [int(r.iloc[0].n), metric(r.iloc[0].mae)] if len(r) else [0, "—"]
        rows.append([pool.removeprefix("c_cohort:").removesuffix("/data.csv.gz"), *parts])
    cohorts += table(["Cohort", "Known N", "Known MAE", "Novel N", "Novel MAE"], rows)
    return "\n".join(body), "\n".join(body + cohorts)


def plot(candidates, selected, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")

    def curve(ax, f, label, style="-"):
        r = (f.final_prediction - f.truth).abs().to_numpy()
        r = np.sort(r[np.isfinite(r)])
        if len(r):
            ax.plot(r, np.arange(1, len(r) + 1) / len(r), style, label=f"{label} (n={len(r):,})")

    for split in ["train", "val", "test"]:
        curve(axes[0], selected[(selected.pool == "a_orb") & (selected.cohort == split)], split)
    for track in ["fixed_symmetry", "free"]:
        curve(axes[1], candidates[candidates.pool == f"b_{track}"], track + " before", "--")
        curve(axes[1], selected[selected.pool == f"b_{track}"], track + " minimum")
    for novel in [False, True]:
        label = "Novel" if novel else "Known"
        for f, suffix, style in [(candidates, "before", "--"), (selected, "minimum", "-")]:
            curve(axes[2], f[(f.pool == "c_generated_dft") & (f.novel_structure_fixed == novel)], label + " " + suffix, style)
    for ax, title in zip(axes, ["Archive ORB minima", "Protocol ORB final-gene minima", "Generated DFT minima"]):
        ax.set(title=title, xlabel="Absolute error (eV/atom)", ylabel="Fraction within error", xlim=(0, .5), ylim=(0, 1))
        ax.grid(alpha=.25)
        ax.legend(fontsize=7, loc="lower right")
    for extension in ["png", "pdf"]:
        fig.savefig(root / f"gene_minima_ecdf.{extension}", dpi=180)
    plt.close(fig)


def main():
    import wandb
    from wyckoff_transformer.cli.template_novelty import fetch_entry
    from wyckoff_transformer.energy_fields import EnergySource, check_compatible
    from wyckoff_transformer.field_provenance import target_field

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", default=SOURCE)
    parser.add_argument("--source-local", type=Path, default=runs_root() / "energy_regressor_diagnostic_20261028")
    parser.add_argument("--output", type=Path, default=runs_root() / "energy_regressor_gene_minima_20261028")
    parser.add_argument("--update-doc", action="store_true")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    with wandb.init(
        entity="symmetry-advantage", project="WyckoffTransformer", mode="online",
        job_type="energy_regressor_gene_minima", dir=wandb_dir(),
        name="Energy precision: one observed minimum per gene 2026-10-28",
        config=dict(git_commit=commit, source_artifact=args.source_artifact),
    ) as run:
        source = run.use_artifact(args.source_artifact)
        inputs = root / "source"
        inputs.mkdir(exist_ok=True)
        required = ["rows.csv.gz", "reference_rows.csv.gz", "source/field_provenance.json", "source/model.json", "provenance.json"]
        for name in required:
            fetch_entry(source.get_entry(name), inputs, args.source_local)
        fields = json.loads((inputs / "source/field_provenance.json").read_text())
        expected = target_field(fields)
        dft = replace(expected, aggregate="none")
        orb = replace(dft, source=EnergySource(method="mlip", dft_settings="PBE_OMat24", correction="none", mlip_model="orb-v3-conservative-inf-omat-20250404"))
        check_compatible(expected, dft, "DFT observations and archive candidates")
        differences = check_compatible(expected, orb, "explicit cross-scale ORB repeat", allow=True)
        frame = pd.read_csv(inputs / "rows.csv.gz", low_memory=False)
        reference = pd.read_csv(inputs / "reference_rows.csv.gz")
        rebuilt = reference.groupby("key").formation_energy_per_atom.min()
        if not np.allclose(reference.key.map(rebuilt), reference.gene_min_formation_energy_per_atom, rtol=0, atol=1e-10):
            raise ValueError("Pinned reference rows disagree with recorded archive minima")
        candidates, definitions = make_pools(frame, reference)
        selected, audit = select_minima(candidates)
        summary = summarize(candidates, selected)
        selected.to_csv(root / "rows.csv.gz", index=False)
        audit.to_csv(root / "selection_audit.csv.gz", index=False)
        summary.to_csv(root / "summary.csv", index=False)
        selected[selected.origin == "archive"].to_csv(root / "selected_archive_rows.csv.gz", index=False)
        # Every final gene must have exactly one observation in each primary pool.
        if selected.duplicated(["pool", "group_key"]).any():
            raise ValueError("More than one retained structure per pool/gene")
        generated_keys = set(selected[selected.pool == "c_generated_dft"].final_key)
        union_keys = set(selected[selected.pool == "c_archive_union_dft"].final_key)
        if generated_keys != union_keys:
            raise ValueError("The archive union changed the evaluated genes")
        counts = audit.groupby(["pool", "selection_status"]).size().rename("n").reset_index()
        counts.to_csv(root / "selection_counts.csv", index=False)
        provenance = dict(
            date=datetime.now(UTC).isoformat(), git_commit=commit,
            git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
            source_artifact=dict(name=source.qualified_name, id=source.id, digest=source.digest),
            source_files={name: source.manifest.entries[name].digest for name in required},
            model=json.loads((inputs / "source/model.json").read_text()),
            code_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in CODE},
            pools=definitions, selection="observed per-atom energy before prediction/novelty filtering",
            units="eV/atom", energy_fields=dict(DFT=dft.to_dict(), ORB=orb.to_dict()),
            cross_scale_differences=differences, no_new_calculations=True,
        )
        dump(root / "provenance.json", provenance)
        dump(root / "verification.json", dict(
            minima_equal_poolwise_grouped_observed_minima=True,
            one_retained_structure_per_pool_gene=True,
            archive_union_evaluates_same_genes=True,
            reconstructed_archive_minima_equal_stored_targets=True,
            selection_uses_neither_prediction_nor_novelty=True,
            selected_rows=len(selected), primary_DFT_genes=len(generated_keys),
        ))
        section, full_report = report(summary, selected, provenance, run.url)
        (root / "report.md").write_text(full_report)
        if args.update_doc:
            doc = Path("docs/energy_regressor_precision.md")
            text = doc.read_text()
            if "## E Repeat:" in text:
                start = text.index("## E Repeat:")
                end = text.index("## Reproduction and saved outputs", start)
                text = text[:start] + text[end:]
            text = text.replace("## Reproduction and saved outputs", section + "\n## Reproduction and saved outputs", 1)
            note = "The repeat retaining one lowest-energy observed structure per gene is in §E.\n\n"
            if note not in text:
                text = text.replace("## Energy normalization and coverage", note + "## Energy normalization and coverage", 1)
            doc.write_text(text)
        plot(candidates, selected, root)
        artifact = wandb.Artifact(f"energy_regressor_gene_minima_{run.id}", type="energy_regressor_evaluation", metadata=provenance)
        for path in root.iterdir():
            if path.is_file() and path.name != "wandb_result.json":
                artifact.add_file(str(path))
        for name in CODE:
            artifact.add_file(name, name="code/" + name)
        if args.update_doc:
            artifact.add_file("docs/energy_regressor_precision.md", name="docs/energy_regressor_precision.md")
        for name in required[2:]:
            artifact.add_file(str(inputs / name), name="source/" + name)
        run.log_artifact(artifact).wait()
        for pool in ["a_orb", "b_fixed_symmetry", "b_free", "c_generated_dft", "c_archive_union_dft"]:
            r = lookup(summary, pool)
            run.summary[pool] = {k: r[k].item() if hasattr(r[k], "item") else r[k] for k in ["n", "excluded", "mae", "rmse", "bias"]}
        dump(root / "wandb_result.json", dict(url=run.url, id=run.id, artifact=artifact.qualified_name))
        print(section, flush=True)
        print("W&B", run.url, flush=True)


if __name__ == "__main__":
    main()
