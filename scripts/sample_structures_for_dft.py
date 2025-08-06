if __name__ == "__main__":
    # Ensure the parent directory is in the system path for imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

from typing import Tuple, Optional
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import pickle
from tqdm.auto import tqdm
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure
from evaluation.generated_dataset import GeneratedDataset, load_all_from_config
from evaluation.novelty import NoveltyFilter, filter_by_unique_structure

from collections import Counter
from operator import itemgetter
from itertools import chain

class TopElements():
    def __init__(self, dataset="mp_20"):
        self.dataset = dataset
        self.cache_path = Path(__file__).parent.parent.joinpath("cache", dataset, "top_elements.pkl") # Adjusted path
        if self.cache_path.exists():
            with self.cache_path.open("rb") as f:
                self.data = pickle.load(f)
        else:
            mp_20 = pd.concat(
                [GeneratedDataset.from_cache(('split', part), dataset).data.elements for part in 
                 ('train', 'val', 'test')],
                axis=0, verify_integrity=True)
            element_counts = Counter(chain(*mp_20))
            self.data = frozenset(map(itemgetter(0), element_counts.most_common(30)))
            with self.cache_path.open("wb") as f:
                pickle.dump(self.data, f)

    def check_represented_composition(self, structure: Structure) -> bool:
        for element in structure.composition:
            if element not in self.data:
                return False
        return True


def to_cif(structure: Structure) -> str:
    """
    Convert a pymatgen structure to a CIF string containing the primitive structure.
    """
    return str(CifWriter(structure.to_primitive()))    


def write_novel_structures(
    transformations: Tuple[str, ...],
    dataset: pd.DataFrame,
    save_path_root: Path,
    novel_save_count: int,
    novelty_filter: NoveltyFilter,
    filter_by_elements: Optional[bool] = False,
    random_seed: int = 42,
    truncation_heuristic: int = 15) -> None:

    dataset_head = dataset.sample(
        min(novel_save_count*truncation_heuristic, len(dataset)),
        random_state=random_seed)
    if filter_by_elements:
        representative_element_filter = TopElements()
        dataset_head = dataset_head[
            dataset_head.structure.apply(representative_element_filter.check_represented_composition)]
    unique = filter_by_unique_structure(dataset_head)
    novel = novelty_filter.get_novel(unique)
    if len(novel) < novel_save_count:
        raise ValueError(f"Only {len(novel)} novel structures found, expected {novel_save_count}. "
                          "Try increasing the dataset size or increasing truncation_heuristic in case"
                          "novelty is low.")
    novel_cif = novel.structure.iloc[:novel_save_count].apply(to_cif)
    print(novel.group.value_counts())
    site_counts = novel.structure.iloc[:novel_save_count].apply(lambda s: len(s.to_primitive().sites))
    print(f"Novel_{novel_save_count} primitive site counts: "
          f"{site_counts.mean():.1f} ± {site_counts.std():.1f}; max: {site_counts.max():.1f}")
    novel_cif.name = "cif"
    cif_path = save_path_root.joinpath(*transformations)
    cif_path.mkdir(parents=True, exist_ok=True)
    novel_cif.to_csv(cif_path.joinpath("cif.csv.gz"), index_label="index")    
    print(f"Saved {len(novel_cif)} novel CIFs to {cif_path}")


def main():
    parser = ArgumentParser(description="Sample CIFs for novel generated structurees.")
    parser.add_argument("transformations", nargs="+", type=str,
                        help="Transformations defining the generated dataset to use for generating structures.")
    parser.add_argument("--filter-by-elements", action="store_true",
                        help="Filter structures by top 30 elements in MP-20.")
    parser.add_argument("--novel_save_count", type=int, default=105,
                        help="Number of novel structures to save.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument("--dataset", type=str, default="mp_20",
                        help="Name of the dataset to use.")
    args = parser.parse_args()
    mp_20_transofrmations = [tuple(args.transformations)]
    all_datasets = load_all_from_config(datasets=mp_20_transofrmations, dataset_name=args.dataset)
    # wycryst_transformations = ("WyCryst", "CrySPR", "CHGNet_fix")
    # all_datasets[wycryst_transformations] = GeneratedDataset.from_cache(wycryst_transformations, "mp_20_biternary")
    novelty_reference = pd.concat([
        GeneratedDataset.from_cache(('split', 'train'), args.dataset).data,
        GeneratedDataset.from_cache(('split', 'val'), args.dataset).data], axis=0, verify_integrity=True)
    novelty_filter = NoveltyFilter(novelty_reference)

    novel_save_path = Path(__file__).parent.parent.joinpath("generated", "Dropbox", f"novel_{args.novel_save_count}") # Adjusted path
    novel_save_path.mkdir(parents=True, exist_ok=True)

    for transformations in tqdm(all_datasets.keys()):
        this_data = all_datasets[transformations].data
        write_novel_structures(transformations, this_data, novel_save_path, args.novel_save_count, novelty_filter,
                               filter_by_elements=args.filter_by_elements, random_seed=args.random_seed)


if __name__ == "__main__":
    main()
