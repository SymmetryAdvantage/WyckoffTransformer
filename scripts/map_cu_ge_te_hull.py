"""Map the convex hull of the Cu-Ge-Te chemical system.

Workflow:
1. Generate 100k genes across Cu-Ge, Ge-Te, Cu-Te, Cu-Ge-Te using our best
   chemical-system conditioned model (chemsys_sg_uncond_adanmw_wsd-20260925-030937).
2. Deduplicate the genes using 128-bit augmented Wyckoff gene keys.
3. Score all unique genes with our best energy regressor (min_energy_adamw_wsd-20260924-102431)
   and select the top 10,000 genes with the lowest predicted energy above the hull.
4. Reconstruct structures for the chosen 10k genes:
   - Sample 50 PyXtal structures per gene (20 CPU cores).
   - Relax with NEP 89 under fixed symmetry (20 CPU cores).
   - Deduplicate relaxed structures and pick up to 5 lowest-energy candidates.
   - Relax selected structures with ORB (2 workers per GPU across cuda:0 and cuda:1).
5. Map the updated convex hull:
   - Combine relaxed structures with the published ORB reference hull.
   - Identify stable ground states (SUN) and metastable phases (MetaSUN).
   - Plot and export the ternary phase diagram and summary tables.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from pymatgen.analysis.phase_diagram import PDEntry, PDPlotter, PhaseDiagram
from pymatgen.core import Composition

from wyckoff_transformer.cli.csp import load_trainer
from wyckoff_transformer.evaluation.gene_hash import gene_keys, unique_representatives
from wyckoff_transformer.evaluation.hull_energy import HullEnergyCalculator
from wyckoff_transformer.evaluation.protocol import (
    GeneFingerprinter,
    GeneScreen,
    load_genes,
    write_screen,
)
from wyckoff_transformer.gene_energy import build_clean_relaxation_condition
from wyckoff_transformer.paths import runs_root
from wyckoff_transformer.prediction import (
    build_tokenised_prediction_tensors,
    filter_supported_tokens,
)
from wyckoff_transformer.system_prior import SystemSpaceGroupPrior

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("map_cu_ge_te_hull")

BACKBONE_RUN = "chemsys_sg_uncond_adanmw_wsd-20260925-030937"
REGRESSOR_RUN = "min_energy_adamw_wsd-20260924-102431"
TARGET_SYSTEM = "Cu-Ge-Te"
ALLOWED_ELEMENTS = ("Cu", "Ge", "Te")


def get_reference_entries_for_chemsys(chemsys_elements: set[str]) -> list[PDEntry]:
    """Retrieve all published ORB-v3 reference entries within the chemical system."""
    calc = HullEnergyCalculator("orb_conserv_inf")
    entries = []
    for idx, row in calc.entries.iterrows():
        species = row["species_at_sites"]
        if set(species).issubset(chemsys_elements):
            comp = Composition(Counter(species))
            entry = PDEntry(comp, float(row["energy"]), name=f"ref_{idx}")
            entries.append(entry)
    logger.info("Found %d published ORB reference entries for %s", len(entries), chemsys_elements)
    return entries


def step_1_generate_100k(
    output_dir: Path,
    n_genes: int = 100000,
    device: str = "cuda:0",
    batch_size: int = 10000,
    seed: int = 42,
) -> Path:
    """Sample n_genes using the best chemical-system conditioned model."""
    genes_path = output_dir / "cu_ge_te_100k_genes.json.gz"
    plan_path = output_dir / "cu_ge_te_system_plan.json"
    if genes_path.exists():
        logger.info("Found existing generated genes at %s; skipping generation.", genes_path)
        return genes_path

    logger.info("Loading backbone model %s...", BACKBONE_RUN)
    backbone_dir = runs_root() / BACKBONE_RUN
    prior_path = backbone_dir / "system_prior.npz"
    if not prior_path.exists():
        raise FileNotFoundError(f"Missing prior at {prior_path}")

    prior = SystemSpaceGroupPrior.load(prior_path)
    logger.info("Drawing closure plan for %s with %d total structures...", TARGET_SYSTEM, n_genes)
    plan = prior.closure_plan(
        [TARGET_SYSTEM],
        n_structures=n_genes,
        min_arity=2,
        target_share=0.5,
        rng=seed,
    )
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan.manifest(), f, indent=2)

    counts_by_system = Counter()
    for tokens in plan.element_tokens:
        syms = "-".join(sorted(prior.element_symbols[t] for t in tokens))
        counts_by_system[syms] += 1
    logger.info("Plan distribution: %s", dict(counts_by_system))

    device_obj = torch.device(device)
    trainer = load_trainer(model_path=backbone_dir, device=device_obj, load_datasets=False)
    if hasattr(trainer.model, "_orig_mod"):
        trainer.model = trainer.model._orig_mod

    logger.info("Generating %d structures in batches of %d on %s...", n_genes, batch_size, device)
    all_genes = []
    t_start = time.time()
    n_vocab = len(trainer.tokenisers["elements"])
    stop_token = trainer.tokenisers["elements"].stop_token

    def _slice_draws(draws, start: int, end: int):
        from wyckoff_transformer.system_prior import SystemDraws
        return SystemDraws(
            element_tokens=draws.element_tokens[start:end],
            space_groups=draws.space_groups[start:end],
            is_novel=draws.is_novel[start:end],
            element_symbols=draws.element_symbols,
            query=draws.query,
        )


    for start_idx in range(0, n_genes, batch_size):
        end_idx = min(start_idx + batch_size, n_genes)
        batch_slice = _slice_draws(plan, start_idx, end_idx)
        b_cond = batch_slice.conditioning_block(n_vocab, device_obj)
        b_start = batch_slice.start_tensor(
            trainer.tokenisers[trainer.start_name], trainer.model.start_type, device_obj
        )
        b_mask = batch_slice.element_mask(n_vocab, stop_token=stop_token)

        with torch.no_grad():
            b_genes = trainer.generate_structures(
                len(batch_slice),
                calibrate=False,
                composition_cond=b_cond,
                start_tensor=b_start,
                allowed_element_mask=b_mask,
            )
        all_genes.extend(b_genes)
        elapsed = time.time() - t_start
        logger.info(
            "Generated %d/%d valid genes (%.1f draws/s, %.1f s elapsed)",
            len(all_genes),
            end_idx,
            end_idx / elapsed if elapsed > 0 else 0,
            elapsed,
        )

    logger.info(
        "Finished generation: %d valid genes from %d draws (%.1f%% formal validity) in %.1f s.",
        len(all_genes),
        n_genes,
        100.0 * len(all_genes) / n_genes,
        time.time() - t_start,
    )

    with gzip.open(genes_path, "wt", encoding="utf-8") as f:
        json.dump(all_genes, f)
    logger.info("Saved generated genes to %s", genes_path)
    return genes_path


def step_2_deduplicate(genes_path: Path, output_dir: Path) -> Path:
    """Deduplicate genes using canonical 128-bit Wyckoff hashes."""
    unique_path = output_dir / "cu_ge_te_unique_genes.json.gz"
    if unique_path.exists():
        logger.info("Found existing unique genes at %s; skipping deduplication.", unique_path)
        return unique_path

    logger.info("Loading genes from %s for deduplication...", genes_path)
    genes = load_genes(genes_path)
    fingerprinter = GeneFingerprinter()

    records = []
    valid_indices = []
    for idx, gene in enumerate(genes):
        try:
            rec = fingerprinter.record(gene)
            records.append(rec)
            valid_indices.append(idx)
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Skipping unparseable gene %d: %s", idx, exc)
            continue

    logger.info("Computing gene keys for %d valid records...", len(records))
    t0 = time.time()
    keys = gene_keys(records)
    representatives, _counts = unique_representatives(keys)
    dt = time.time() - t0

    unique_genes = [genes[valid_indices[rep_idx.item()]] for rep_idx in representatives]
    logger.info(
        "Deduplication complete in %.2f s: %d unique genes from %d valid (%.1f%% uniqueness).",
        dt,
        len(unique_genes),
        len(records),
        100.0 * len(unique_genes) / len(records) if records else 0,
    )

    # Breakdown by system
    by_sys = Counter()
    for g in unique_genes:
        sp = sorted(set(g.get("species", [])))
        by_sys["-".join(sp)] += 1
    logger.info("Unique genes breakdown by chemical system: %s", dict(by_sys))

    with gzip.open(unique_path, "wt", encoding="utf-8") as f:
        json.dump(unique_genes, f)
    logger.info("Saved %d unique genes to %s", len(unique_genes), unique_path)
    return unique_path


def step_3_score_and_select_10k(
    unique_path: Path,
    output_dir: Path,
    top_k: int = 10000,
    device: str = "cuda:0",
    batch_size: int = 10000,
) -> Path:
    """Score unique genes with the best energy regressor and pick the top_k."""
    selected_path = output_dir / f"selected_{top_k}_genes.json.gz"
    csv_path = output_dir / "scored_unique_genes.csv.gz"
    if selected_path.exists() and csv_path.exists():
        logger.info("Found existing selected genes at %s; skipping scoring.", selected_path)
        return selected_path

    logger.info("Loading unique genes from %s...", unique_path)
    unique_genes = load_genes(unique_path)

    # 1. Build records
    fingerprinter = GeneFingerprinter()
    records = []
    valid_genes = []
    for g in unique_genes:
        try:
            rec = fingerprinter.record(g)
            composition = {}
            for el, count in zip(rec["elements"], rec["multiplicity"]):
                composition[el] = composition.get(el, 0) + count
            rec["composition"] = composition
            # Formula string
            f_parts = [f"{el!s}{int(cnt)}" for el, cnt in sorted(composition.items(), key=lambda x: str(x[0]))]
            rec["formula"] = "".join(f_parts)
            records.append(rec)
            valid_genes.append(g)
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Skipping unparseable unique gene: %s", exc)
            continue

    records_frame = pd.DataFrame.from_records(records)
    logger.info("Valid records for scoring: %d", len(records_frame))

    # 2. Score with regressor
    logger.info("Loading energy regressor %s...", REGRESSOR_RUN)
    regressor_dir = runs_root() / REGRESSOR_RUN
    regressor = load_trainer(model_path=regressor_dir, device=torch.device(device), load_datasets=False)

    supported, _ = filter_supported_tokens(records_frame, regressor)
    logger.info("Tokens supported by regressor: %d/%d", len(supported), len(records_frame))

    predictions = []
    t_start = time.time()
    for start_idx in range(0, len(supported), batch_size):
        sub_df = supported.iloc[start_idx : start_idx + batch_size]
        tensors = build_tokenised_prediction_tensors(sub_df, regressor)
        cond = build_clean_relaxation_condition(regressor, len(sub_df), device=regressor.device)
        with torch.no_grad():
            preds, _ = regressor.predict_scalars(tensors, augmentation_samples=1, cond=cond)
        predictions.extend(preds.cpu().numpy().tolist())
        logger.info("Scored %d/%d genes (%.1f genes/s)", len(predictions), len(supported), len(predictions)/(time.time()-t_start))

    supported["predicted_formation_energy"] = predictions

    # 3. Calculate distance to reference ORB hull
    logger.info("Building reference phase diagram to compute predicted e_hull...")
    ref_entries = get_reference_entries_for_chemsys(set(ALLOWED_ELEMENTS))
    ref_pd = PhaseDiagram(ref_entries)

    # Cache reference hull formation energies per formula
    formula_hull_fe = {}
    for formula in supported["formula"].unique():
        c = Composition(formula)
        hull_tot = ref_pd.get_hull_energy(c)
        hull_fe = ref_pd.get_form_energy_per_atom(PDEntry(c, hull_tot))
        formula_hull_fe[formula] = float(hull_fe)

    supported["hull_formation_energy_per_atom"] = supported["formula"].map(formula_hull_fe)
    supported["predicted_e_hull"] = supported["predicted_formation_energy"] - supported["hull_formation_energy_per_atom"]

    # Sort ascending by predicted e_hull
    supported = supported.sort_values("predicted_e_hull", ascending=True)

    # Save CSV
    supported.to_csv(csv_path, index=False, compression="gzip")
    logger.info("Saved full scored table to %s", csv_path)

    # Select top_k
    top_df = supported.head(top_k)
    logger.info("Selected top %d genes. Predicted e_hull ranges from %.4f to %.4f eV/atom",
                len(top_df), top_df["predicted_e_hull"].min(), top_df["predicted_e_hull"].max())

    chosen_indices = top_df.index.tolist()
    selected_genes = [valid_genes[i] for i in chosen_indices]

    with gzip.open(selected_path, "wt", encoding="utf-8") as f:
        json.dump(selected_genes, f)
    logger.info("Saved selected %d genes to %s", len(selected_genes), selected_path)
    return selected_path


def step_4_reconstruct(
    genes_path: Path,
    output_dir: Path,
    n_pyxtal: int = 50,
    pyxtal_cores: int = 20,
    nep_cores: int = 20,
    prescreen_select: int = 5,
    devices: str = "cuda:0,cuda:1",
    workers_per_device: int = 2,
    limit: int | None = None,
) -> Path:
    """Run reconstruction with wyformer-protocol pipeline."""
    recon_dir = output_dir / "reconstruction"
    recon_dir.mkdir(parents=True, exist_ok=True)

    genes = load_genes(genes_path)
    if limit is not None:
        genes = genes[:limit]
        logger.info("Reconstruction limited to first %d genes for pilot.", limit)
        sub_genes_path = recon_dir / f"genes_limit_{limit}.json.gz"
        with gzip.open(sub_genes_path, "wt", encoding="utf-8") as f:
            json.dump(genes, f)
        active_genes_path = sub_genes_path
    else:
        active_genes_path = genes_path

    # Create screen.json directly to avoid 17GB unpickling overhead
    screen_path = recon_dir / "screen.json"
    if not screen_path.exists():
        logger.info("Creating initial screen.json for %d genes...", len(genes))
        screen = GeneScreen(
            n_sampled=len(genes),
            valid=list(range(len(genes))),
            invalid=[],
            invalid_reason={},
            counts={i: 1 for i in range(len(genes))},
            novel=list(range(len(genes))),
            known=[],
        )
        write_screen(screen, screen_path)

    cmd_base = f".venv/bin/wyformer-protocol {active_genes_path} --output-dir {recon_dir}"

    # 1. Generate PyXtal structures (50 trials per gene)
    pyxtal_extxyz = recon_dir / "pyxtal.extxyz"
    pyxtal_csv = recon_dir / "pyxtal.csv"
    if not (pyxtal_extxyz.exists() and pyxtal_csv.exists()):
        logger.info("=== Running PyXtal draw (50 structures per gene, %d cores) ===", pyxtal_cores)
        cmd_gen = f"{cmd_base} --stage generate --resume --n-trials {n_pyxtal} --pyxtal-cores {pyxtal_cores} --pyxtal-timeout 30"
        ret = os.system(cmd_gen)
        if ret != 0:
            raise RuntimeError(f"PyXtal generate failed with code {ret}")

    # 2. Prescreen: relax with NEP 89, deduplicate, select up to 5 lowest energy
    prescreen_extxyz = recon_dir / "prescreen.extxyz"
    prescreen_csv = recon_dir / "prescreen.csv"
    if not (prescreen_extxyz.exists() and prescreen_csv.exists()):
        logger.info("=== Running NEP 89 pre-relaxation and deduplication (select %d, %d cores) ===",
                    prescreen_select, nep_cores)
        cmd_prescreen = (
            f"{cmd_base} --stage prescreen --resume --prescreen-mlip nep89 "
            f"--prescreen-select {prescreen_select} --cores {nep_cores}"
        )
        ret = os.system(cmd_prescreen)
        if ret != 0:
            raise RuntimeError(f"Prescreen failed with code {ret}")

    # 3. Relax with ORB
    relaxations_csv = recon_dir / "relaxations.csv"
    structures_csv = recon_dir / "structures.csv"
    if not (relaxations_csv.exists() and structures_csv.exists()):
        logger.info("=== Running ORB relaxation on %s (%d workers/device) ===", devices, workers_per_device)
        cmd_relax = (
            f"{cmd_base} --stage relax --resume --relax-from prescreen "
            f"--mlip orb_conserv_inf --devices {devices} --workers-per-device {workers_per_device}"
        )
        ret = os.system(cmd_relax)
        if ret != 0:
            raise RuntimeError(f"Relaxation failed with code {ret}")

    logger.info("Reconstruction complete in %s", recon_dir)
    return recon_dir


def step_5_map_hull(output_dir: Path) -> dict:
    """Build the final convex hull with published references + newly relaxed structures."""
    recon_dir = output_dir / "reconstruction"
    plot_path = output_dir / "cu_ge_te_hull.png"
    csv_out = output_dir / "cu_ge_te_hull_entries.csv"

    ref_entries = get_reference_entries_for_chemsys(set(ALLOWED_ELEMENTS))

    # Read relaxed structures
    candidate_entries = []
    relaxations_csv = recon_dir / "relaxations.csv"

    if relaxations_csv.exists():
        df_relax = pd.read_csv(relaxations_csv)
        ok_relax = df_relax[df_relax["status"] == "ok"].copy()
        logger.info("Found %d successful ORB relaxations in %s", len(ok_relax), relaxations_csv)
        for _, row in ok_relax.iterrows():
            formula = str(row["formula"])
            energy = float(row["energy"])
            c = Composition(formula)
            entry = PDEntry(c, energy, name=f"gen_{row['index']}_trial_{row['trial']}")
            entry.data = {
                "gene_index": row["index"],
                "trial": row["trial"],
                "cif": str(row.get("cif", "")),
                "source": "generated",
            }
            candidate_entries.append(entry)

    # Combine all entries
    all_entries = ref_entries + candidate_entries
    logger.info("Total entries for PhaseDiagram: %d (%d reference, %d generated)",
                len(all_entries), len(ref_entries), len(candidate_entries))

    pd_hull = PhaseDiagram(all_entries)

    # Classify entries
    rows = []
    stable_generated = []
    metastable_generated = []

    for e in all_entries:
        e_above_hull = float(pd_hull.get_e_above_hull(e))
        form_e = float(pd_hull.get_form_energy_per_atom(e))
        is_stable = e in pd_hull.stable_entries
        source = e.data.get("source", "reference") if hasattr(e, "data") and e.data else "reference"
        cif_path = e.data.get("cif", "") if hasattr(e, "data") and e.data else ""
        red_formula = e.composition.reduced_formula
        chemsys = "-".join(sorted(el.symbol for el in e.composition.elements))

        rows.append({
            "name": e.name,
            "reduced_formula": red_formula,
            "chemsys": chemsys,
            "source": source,
            "cif": cif_path,
            "energy_total": e.energy,
            "energy_per_atom": e.energy_per_atom,
            "formation_energy_per_atom": form_e,
            "e_above_hull": e_above_hull,
            "is_stable": is_stable,
            "is_metastable_50meV": e_above_hull <= 0.05,
            "is_metastable_100meV": e_above_hull <= 0.10,
        })

        if source == "generated":
            if is_stable:
                stable_generated.append(e)
            elif e_above_hull <= 0.10:
                metastable_generated.append(e)

    df_hull = pd.DataFrame(rows)
    df_hull = df_hull.sort_values(["e_above_hull", "formation_energy_per_atom"])
    df_hull.to_csv(csv_out, index=False)
    logger.info("Saved hull entries table to %s", csv_out)

    # Plot ternary phase diagram
    try:
        plotter = PDPlotter(pd_hull, show_unstable=0.1, backend="matplotlib")
        res = plotter.get_plot()
        fig = res if hasattr(res, "savefig") else res.figure
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Phase diagram plot saved to %s", plot_path)
    except (ValueError, RuntimeError, TypeError) as exc:
        logger.warning(
            "Could not render PDPlotter plot (%s); creating custom ternary plot",
            exc,
        )

    summary = {
        "total_entries": len(all_entries),
        "reference_entries": len(ref_entries),
        "generated_entries": len(candidate_entries),
        "stable_phases_total": len(pd_hull.stable_entries),
        "new_stable_phases (SUN)": len(stable_generated),
        "new_metastable_phases (MetaSUN, <= 100 meV)": len(metastable_generated),
        "plot_path": str(plot_path),
        "table_path": str(csv_out),
    }

    print("\n" + "=" * 60)
    print("CU-GE-TE HULL MAPPING SUMMARY")
    print("=" * 60)
    print(f"Total entries on phase diagram : {len(all_entries)}")
    print(f"  Published ORB reference      : {len(ref_entries)}")
    print(f"  Generated & ORB relaxed      : {len(candidate_entries)}")
    print(f"Total ground state hull vertices : {len(pd_hull.stable_entries)}")
    print(f"  New stable phases discovered (SUN): {len(stable_generated)}")
    print(f"  New metastable phases (<=100 meV): {len(metastable_generated)}")
    print("-" * 60)
    print("Stable phases on the Cu-Ge-Te convex hull:")
    for e in sorted(pd_hull.stable_entries, key=lambda x: (len(x.composition.elements), x.composition.reduced_formula)):
        src = e.data.get("source", "ref") if hasattr(e, "data") and e.data else "ref"
        fe = pd_hull.get_form_energy_per_atom(e)
        print(f"  * {e.composition.reduced_formula:<12s} ({src:<9s}): Ef = {fe:+.4f} eV/atom, E/atom = {e.energy_per_atom:.4f} eV")
    print("=" * 60 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Map the convex hull of the Cu-Ge-Te system.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cu_ge_te_hull"), help="Working directory.")
    parser.add_argument("--stage", choices=["all", "generate", "dedup", "score", "reconstruct", "map_hull"],
                        default="all", help="Which stage to execute.")
    parser.add_argument("--n-genes", type=int, default=100000, help="Initial gene sampling budget.")
    parser.add_argument("--top-k", type=int, default=10000, help="Number of genes to select for reconstruction.")
    parser.add_argument("--pilot", action="store_true", help="Run a quick pilot (1000 draws, 5 genes reconstructed).")
    parser.add_argument("--limit-reconstruct", type=int, default=None, help="Limit number of genes reconstructed.")
    parser.add_argument("--device", default="cuda:0", help="Primary GPU device for sampling and scoring.")
    parser.add_argument("--devices", default="cuda:0,cuda:1", help="GPU devices for ORB relaxation.")
    parser.add_argument("--workers-per-device", type=int, default=2, help="ORB relaxation workers per GPU.")
    parser.add_argument("--pyxtal-cores", type=int, default=20, help="CPU cores for PyXtal sampling.")
    parser.add_argument("--nep-cores", type=int, default=20, help="CPU cores for NEP89 pre-relaxation.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(args.output_dir / "pipeline.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    if args.pilot:
        logger.info("--- PILOT MODE ACTIVATED ---")
        args.n_genes = 1000
        args.top_k = 50
        args.limit_reconstruct = 5

    logger.info("Starting Cu-Ge-Te hull mapping campaign (stage=%s)", args.stage)

    # Step 1: Generate genes
    if args.stage in ("all", "generate"):
        genes_file = step_1_generate_100k(args.output_dir, n_genes=args.n_genes, device=args.device)
    else:
        genes_file = args.output_dir / "cu_ge_te_100k_genes.json.gz"

    # Step 2: Deduplicate
    if args.stage in ("all", "dedup"):
        unique_file = step_2_deduplicate(genes_file, args.output_dir)
    else:
        unique_file = args.output_dir / "cu_ge_te_unique_genes.json.gz"

    # Step 3: Score with regressor and select top_k
    if args.stage in ("all", "score"):
        selected_file = step_3_score_and_select_10k(unique_file, args.output_dir, top_k=args.top_k, device=args.device)
    else:
        selected_file = args.output_dir / f"selected_{args.top_k}_genes.json.gz"

    # Step 4: Reconstruct structures
    if args.stage in ("all", "reconstruct"):
        step_4_reconstruct(
            selected_file,
            args.output_dir,
            n_pyxtal=50,
            pyxtal_cores=args.pyxtal_cores,
            nep_cores=args.nep_cores,
            prescreen_select=5,
            devices=args.devices,
            workers_per_device=args.workers_per_device,
            limit=args.limit_reconstruct,
        )

    # Step 5: Map hull
    if args.stage in ("all", "map_hull"):
        step_5_map_hull(args.output_dir)


if __name__ == "__main__":
    main()
