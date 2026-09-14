from pathlib import Path
import ast
import argparse
import pandas as pd

from wyckoff_transformer.paths import data_path

def filter_by_n_elements(
    dataset: pd.DataFrame,
    min_elements: int,
    max_elements: int
):
    """
    Filters the dataset by the number of elements in the structure.

    Args:
        dataset (pd.DataFrame): The dataset.
        min_elements (int): The minimum number of elements, inclusive.
        max_elements (int): The maximum number of elements, inclusive.

    Returns:
        pd.DataFrame: The filtered dataset.
    """
    assert dataset['elements'].apply(lambda x: len(x) == len(set(x))).all()
    return dataset.loc[
        dataset["elements"].apply(lambda x: min_elements <= len(x) <= max_elements)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_elements", type=int, default=2)
    parser.add_argument("--max_elements", type=int, default=3)
    parser.add_argument("--output_dir", type=Path,
        default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = data_path("mp_20_biternary")
    datasets = ('train', 'test', 'val')
    mp_20_path = data_path("mp_20")
    for dataset_name in datasets:
        print(f"Reading mp_20/{dataset_name} dataset")
        dataset = pd.read_csv(mp_20_path / f"{dataset_name}.csv", index_col=0, converters={"elements": ast.literal_eval})
        print(f"It has {len(dataset)} structures")
        filtered_dataset = filter_by_n_elements(dataset, args.min_elements, args.max_elements)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        filtered_dataset.to_csv(args.output_dir / f"{dataset_name}.csv.gz")
        print(f"Saved {len(filtered_dataset)} structures to {args.output_dir / f'{dataset_name}.csv.gz'}")


if __name__ == "__main__":
    main()
