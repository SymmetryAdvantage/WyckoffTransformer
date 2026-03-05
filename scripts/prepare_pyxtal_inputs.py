import argparse
from pathlib import Path
from wyckoff_transformer.evaluation.generated_dataset import GeneratedDataset

def remove_all_extensions(file_path):
    path = Path(file_path)
    while path.suffix:
        path = path.with_suffix('')
    return path


def main():
    parser = argparse.ArgumentParser(description='Prepare input files for pyxtal')
    parser.add_argument("--dataset", type=str, default='mp_20', help="The dataset to be used")
    parser.add_argument("transformations", type=str, nargs="+")
    args = parser.parse_args()
    dataset = GeneratedDataset.from_transformations(args.transformations, dataset=args.dataset)
    pyxtal_columns = ["group", "sites", "species", "numIons"]
    output_path = dataset.wyckoffs_file.with_name(
        remove_all_extensions(dataset.wyckoffs_file).name + "_pyxtal.json.gz")
    dataset.data.loc[:, pyxtal_columns].to_json(output_path, orient="records", index=False)
    print(f"Saved {len(dataset.data)} entries to {output_path}")

if __name__ == "__main__":
    main()