"""Recompute SUN / MetaSUN from `relaxed_e_above_hull` instead of the as-submitted energy.

LeMat-GenBench relaxes every structure (fmax 0.02, 50 steps) and stores the hull
energy twice -- `e_above_hull_<mlip>` from a single point on the structure as
submitted, and `relaxed_e_above_hull_<mlip>` after the relaxation. Its SUN and
stability metrics read the *unrelaxed* one; `relaxed_e_above_hull` is written and
never read anywhere under metrics/ or benchmarks/.

This reruns the stability preprocessing on the same valid set (pinned by the
original run's `valid_structure_ids`, so no revalidation is needed), then feeds
the relaxed energies to the benchmark's own SUN metric by writing them into the
property it reads. Everything downstream -- thresholds, structure-matcher
uniqueness, novelty against LeMat-Bulk -- is stock.

Stage 1 (`--stage preprocess`) is the expensive half and caches both energies to
CSV; stage 2 (`--stage sun`) reads that cache, so the SUN side can be re-run
cheaply under either energy.
"""
import argparse
import json
import os
import pickle
import sys
import time
from functools import lru_cache
from pathlib import Path

# The stability preprocessor fans out with joblib.  Set these before it imports
# numpy/torch/MACE so a worker always represents one CPU thread.
for _thread_env in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_env] = "1"

GENBENCH = Path("/home/kna/sun-forest/external/lemat-genbench")
sys.path.insert(0, str(GENBENCH / "src"))

import pandas as pd  # noqa: E402
from pymatgen.core import Structure  # noqa: E402


def patch_mace_hull_loader(cache_dir: Path) -> None:
    """Bridge GenBench's retired MACE-hull parquet path to its current layout.

    The installed GenBench revision looks for
    ``threshold_0_001/mace_mp_above_hull_dataset.parquet``.  The public dataset
    now stores the matching entries at ``data/mace_mp-00000-of-00001.parquet``;
    its composition matrix remains at the original threshold path.  Patch only
    the MACE dataframe loader, leaving GenBench's phase-diagram code intact.
    """
    import pandas as pd
    from scipy import sparse
    from huggingface_hub import hf_hub_download
    from lemat_genbench.preprocess import reference_energies

    original_retrieve_df = reference_energies._retrieve_df
    original_retrieve_matrix = reference_energies._retrieve_matrix

    @lru_cache(maxsize=None)
    def retrieve_df(hull_type="dft", threshold=0.001):
        if hull_type != "mace_mp" or threshold != 0.001:
            return original_retrieve_df(hull_type, threshold)
        path = hf_hub_download(
            repo_id="LeMaterial/LeMat-Bulk-MLIP-Hull",
            repo_type="dataset",
            filename="data/mace_mp-00000-of-00001.parquet",
            cache_dir=str(cache_dir),
        )
        dataset = pd.read_parquet(path)
        for column in ("elements", "species_at_sites"):
            if column in dataset.columns:
                dataset[column] = dataset[column].apply(
                    lambda x: x.tolist() if hasattr(x, "tolist") else x
                )
        return dataset

    reference_energies._retrieve_df = retrieve_df

    @lru_cache(maxsize=None)
    def retrieve_matrix(hull_type="dft", threshold=0.001):
        if hull_type != "mace_mp" or threshold != 0.001:
            return original_retrieve_matrix(hull_type, threshold)
        path = hf_hub_download(
            repo_id="LeMaterial/LeMat-Bulk-MLIP-Hull",
            repo_type="dataset",
            filename="threshold_0_001/mace_mp_above_hull_composition_matrix.npz",
            cache_dir=str(cache_dir),
        )
        return sparse.load_npz(path).toarray()

    reference_energies._retrieve_matrix = retrieve_matrix


def load_valid_structures(cif_list: Path, results_json: Path) -> list[Structure]:
    """Rebuild the exact valid set the original run evaluated.

    The runner parses every CIF path in order, silently dropping the ones pymatgen
    rejects, and `valid_structure_ids` indexes into *that* list -- not into the
    path list. So the parse has to be replayed the same way before indexing.
    """
    paths = [line.strip() for line in cif_list.read_text().splitlines() if line.strip()]
    parsed = []
    for path in paths:
        try:
            parsed.append(Structure.from_file(path))
        except Exception:
            continue
    ids = json.loads(results_json.read_text())["validity_filtering"]["valid_structure_ids"]
    print(f"parsed {len(parsed)}/{len(paths)} CIFs; selecting {len(ids)} valid")
    return [parsed[i] for i in ids]


def preprocess(args) -> None:
    import torch
    from lemat_genbench.preprocess.multi_mlip_preprocess import MultiMLIPStabilityPreprocessor

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    patch_mace_hull_loader(args.hull_cache_dir)

    structures = load_valid_structures(args.cif_list, args.results_json)
    preprocessor = MultiMLIPStabilityPreprocessor(
        mlip_names=["mace"],
        mlip_configs={"mace": {"model_type": "mp", "device": "cpu", "hull_type": "mace_mp"}},
        relax_structures=True,
        relaxation_config={"fmax": 0.02, "steps": 50},
        calculate_formation_energy=True,
        calculate_energy_above_hull=True,
        extract_embeddings=False,
        timeout=300,
        n_jobs=args.workers,
    )
    t0 = time.time()
    processed = preprocessor(structures).processed_structures
    print(f"preprocessed {len(processed)} structures in {time.time() - t0:.0f}s")

    rows = []
    for i, s in enumerate(processed):
        p = s.properties
        rows.append({
            "index": i,
            "formula": s.composition.reduced_formula,
            "e_above_hull": p.get("e_above_hull_mace"),
            "relaxed_e_above_hull": p.get("relaxed_e_above_hull_mace"),
            "formation_energy": p.get("formation_energy_mace"),
            "relaxed_formation_energy": p.get("relaxed_formation_energy_mace"),
            "relaxation_rmse": p.get("relaxation_rmse_mace"),
            "relaxation_steps": p.get("relaxation_steps_mace"),
        })
    pd.DataFrame(rows).to_csv(args.energies_csv, index=False)
    with open(args.structures_pkl, "wb") as f:
        pickle.dump(processed, f)
    print(f"wrote {args.energies_csv} and {args.structures_pkl}")


def run_sun(args) -> None:
    from lemat_genbench.benchmarks.sun_benchmark import SUNBenchmark

    with open(args.structures_pkl, "rb") as f:
        structures = pickle.load(f)

    source = "relaxed_e_above_hull_mace" if args.energy == "relaxed" else "e_above_hull_mace"
    missing = 0
    for s in structures:
        value = s.properties.get(source)
        if value is None:
            missing += 1
        # The metric reads `e_above_hull_mean` first and falls back to `e_above_hull`;
        # both are set so the substitution cannot be bypassed by the fallback.
        s.properties["e_above_hull_mean"] = value
        s.properties["e_above_hull"] = value
    print(f"energy source: {source} ({missing} structures without one)")

    benchmark = SUNBenchmark(
        stability_threshold=0.0,
        metastability_threshold=0.1,
        reference_dataset="LeMaterial/LeMat-Bulk",
        reference_config="compatible_pbe",
        fingerprint_method="structure-matcher",
        cache_reference=True,
        max_reference_size=None,
        include_metasun=True,
    )
    t0 = time.time()
    result = benchmark.evaluate(structures)
    print(f"SUN benchmark took {time.time() - t0:.0f}s")
    print(json.dumps(result.final_scores, indent=2, default=str))
    args.sun_json.write_text(json.dumps(
        {"energy_source": source, "final_scores": result.final_scores}, indent=2, default=str))
    print(f"wrote {args.sun_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["preprocess", "sun"], required=True)
    ap.add_argument("--energy", choices=["relaxed", "unrelaxed"], default="relaxed")
    ap.add_argument("--cif-list", dest="cif_list", type=Path)
    ap.add_argument("--results-json", dest="results_json", type=Path)
    ap.add_argument("--energies-csv", dest="energies_csv", type=Path)
    ap.add_argument("--structures-pkl", dest="structures_pkl", type=Path)
    ap.add_argument("--sun-json", dest="sun_json", type=Path)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--hull-cache-dir", type=Path,
                    default=Path(".cache/lemat_hull"))
    args = ap.parse_args()
    preprocess(args) if args.stage == "preprocess" else run_sun(args)


if __name__ == "__main__":
    main()
