from pathlib import Path
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description='Create a CRP training script for a given model')
    parser.add_argument("config", type=Path, help="The configuration file. "
        "The path will be used verbatim in the script, so it should be relative to the project root")
    parser.add_argument("dataset", type=str, help="Dataset to use")
    parser.add_argument("--tempalte", type=Path, help="Path to the template script",
        default=Path(__file__).parent.resolve() / "templates" / "train.sh.template")
    parser.add_argument("--config-root", type=Path, help="Root directory for the config files",
        default=Path(__file__).parent.parent.resolve() / "yamls" / "models")
    args = parser.parse_args()
    with open(args.tempalte, 'rt', encoding="ascii") as f:
        template = f.read()
    config_hierarchy = args.config.resolve().relative_to(args.config_root)
    crp_script = template.format(config=args.config, dataset=args.dataset)
    script_location = Path(__file__).parent.resolve() / "train" / config_hierarchy.with_suffix(".sh")
    script_location.parent.mkdir(parents=True, exist_ok=True)
    with open(script_location, 'wt', encoding="ascii") as f:
        f.write(crp_script)
    print(f"Created script at {script_location}")

if __name__ == "__main__":
    main()