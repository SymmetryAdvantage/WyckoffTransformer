if __name__ == "__main__":
    # We want to avoid messing with the environment variables in case we are used as a module.
    # The code is parallelised by structure
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "1"

from typing import Optional
import argparse
import gzip
import pickle
from pathlib import Path

from wyckoff_transformer.data import read_all_MP_csv


cache_folder = Path(__file__).parent.parent / "cache"  # Adjusted path

def get_cache_data_file_name(dataset:str):
    return cache_folder / dataset / "data.pkl.gz"


def get_cache_tensors_file_name(dataset:str):
    return cache_folder / dataset / "tensors.pkl.gz"


def cache_dataset(
    dataset:str, n_jobs:Optional[int] = None, symmetry_precision:float = 0.1, symmetry_a_tol:float = 5,
    max_wp:Optional[int] = None, scalar_columns:Optional[list] = None,
    sort_by_letter:Optional[bool] = None, observed_gene_minimum_target:bool = False):
    """
    Loads a dataset, tokenizes and caches it.
    """
    if observed_gene_minimum_target and max_wp is not None:
        raise ValueError(
            "observed_gene_minimum_target cannot follow --max-wp: truncation changes "
            "the gene while leaving its energy label attached. Drop over-long rows "
            "instead, as cache_a_dataset_reusing.py --max-sites does.")
    datasets_pd = read_all_MP_csv(
        Path(__file__).parent.parent.resolve() / "data" / dataset,
        n_jobs=n_jobs, symmetry_precision=symmetry_precision,
        symmetry_a_tol=symmetry_a_tol, max_wp=max_wp, scalar_columns=scalar_columns,
        sort_by_letter=sort_by_letter)
    if observed_gene_minimum_target:
        from wyckoff_transformer.gene_energy import add_observed_gene_minimum

        add_observed_gene_minimum(datasets_pd)

    cache_data_file_name = get_cache_data_file_name(dataset)
    cache_data_file_name.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_data_file_name, "wb") as f:
        pickle.dump(datasets_pd, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="The dataset to cache.")
    parser.add_argument("--n-jobs", type=int, help="Number of jobs to use.")
    parser.add_argument("--max-wp", type=int, help="Maximum number of Wyckoff positions to consider.")
    parser.add_argument("--symmetry-precision", type=float, default=0.1,
        help="Symmetry precision. Passed to pyxtal.from_seed")
    parser.add_argument("--symmetry-a-tol", type=float, default=5,
        help="Symmetry angular precision. Passed to pyxtal.from_seed")
    parser.add_argument("--sort-by-letter", dest="sort_by_letter", action="store_true",
        default=None,
        help="Order each structure's sites by Wyckoff letter instead of keeping pyxtal's "
             "own order. Implied by --max-wp, which truncates after sorting; needed on its "
             "own to reproduce a cache built that way, such as lemat_bulk_ehull and "
             "lemat_bulk_fmax1.")
    parser.add_argument("--no-sort-by-letter", dest="sort_by_letter", action="store_false",
        help="Force pyxtal's own site order even under --max-wp.")
    parser.add_argument("--scalar-columns", nargs="*", default=None,
        help="Per-structure scalar columns to carry from the source CSVs into the cache, "
             "beyond the ones carried automatically. Conditioning labels go here. A named "
             "column missing from any split is an error.")
    parser.add_argument("--observed-gene-minimum-target", action="store_true",
        help="Add gene_min_formation_energy_per_atom from the lowest "
             "formation_energy_per_atom observed for each augmented Wyckoff gene "
             "across all splits.")
    args = parser.parse_args()
    cache_dataset(args.dataset, args.n_jobs, args.symmetry_precision, args.symmetry_a_tol,
                  max_wp=args.max_wp, scalar_columns=args.scalar_columns,
                  sort_by_letter=args.sort_by_letter,
                  observed_gene_minimum_target=args.observed_gene_minimum_target)


if __name__ == "__main__":
    main()
