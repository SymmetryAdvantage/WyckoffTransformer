from pathlib import Path
import shutil
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent.parent
from wyckoff_transformer.evaluation.generated_dataset import DATA_KEYS  # noqa: E402

def process_dataset(
    original_path, new_path,
    dataset_name, dataset_config, transformations):

    this_dataset_path = (new_path / dataset_name).joinpath(*transformations)
    this_dataset_path.mkdir(parents=True, exist_ok=True)
    new_config = OmegaConf.create()
    for key in dataset_config:
        if key in DATA_KEYS:
            if isinstance(dataset_config[key], str):
                old_data_path = original_path / dataset_config[key]
                new_data_path = this_dataset_path / Path(dataset_config[key]).name
                new_config[key] = str(new_data_path.relative_to(new_path))
            else:
                old_data_path = original_path / dataset_config[key]["path"]
                new_data_path = this_dataset_path / Path(dataset_config[key]["path"]).name
                new_config[key] = dataset_config[key]
                new_config[key]["path"] = str(new_data_path.relative_to(new_path))
            print(f"Copying {old_data_path} to {new_data_path}")
            if old_data_path.is_dir():
                shutil.copytree(old_data_path, new_data_path)
            else:
                shutil.copy(old_data_path, new_data_path)
        else:
            new_config[key] = process_dataset(
                original_path, new_path,
                dataset_name, dataset_config[key], transformations + [key])
    return new_config

def main():
    generated_original = PROJECT_ROOT / "generated"
    generated_new = PROJECT_ROOT / "generated_new"
    datasets_config = OmegaConf.load(generated_original / "datasets.yaml")
    new_config = OmegaConf.create()
    for dataset_name, dataset_config in datasets_config.items():
        print(f"Processing {dataset_name}")
        new_config[dataset_name] = process_dataset(
            generated_original, generated_new,
            dataset_name, dataset_config, [])
    new_config_path = generated_new / "datasets.yaml"
    OmegaConf.save(new_config, new_config_path)
    print(f"Saved new config to {new_config_path}")

if __name__ == "__main__":
    main()