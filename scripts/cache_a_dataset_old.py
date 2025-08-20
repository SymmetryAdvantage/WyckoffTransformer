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

from data import read_all_MP_csv


cache_folder = Path(__file__).parent.parent / "cache"  # Adjusted path

def get_cache_data_file_name(dataset:str):
    return cache_folder / dataset / "data.pkl.gz"


def get_cache_tensors_file_name(dataset:str):
    return cache_folder / dataset / "tensors.pkl.gz"


def cache_dataset(
    dataset:str, n_jobs:Optional[int] = None, symmetry_precision:float = 0.1, symmetry_a_tol:float = 5,
    max_wp:Optional[int] = None):
    """
    Loads a dataset, tokenizes and caches it.
    """
    if dataset in ('mp_20', 'perov_5', 'carbon_24'):
        datasets_pd = read_all_MP_csv(
            mp_path=Path(__file__).parent.parent.resolve() / "cdvae" / "data" / dataset,  # Adjusted path
            n_jobs=n_jobs, symmetry_precision=symmetry_precision, symmetry_a_tol=symmetry_a_tol, max_wp=max_wp)
    else:
        datasets_pd = read_all_MP_csv(
            Path(__file__).parent.parent.resolve() / "data" / dataset,  # Adjusted path
            file_format="csv.gz", n_jobs=n_jobs, symmetry_precision=symmetry_precision,
            symmetry_a_tol=symmetry_a_tol, max_wp=max_wp)

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
    args = parser.parse_args()
    cache_dataset(args.dataset, args.n_jobs, args.symmetry_precision, args.symmetry_a_tol, max_wp=args.max_wp)


if __name__ == "__main__":
    main()
