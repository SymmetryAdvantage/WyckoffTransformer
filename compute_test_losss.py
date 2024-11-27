from pathlib import Path
import argparse
import torch
from omegaconf import OmegaConf
import wandb
from wyckoff_transformer.trainer import WyckoffTrainer

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Compute test loss for a WanDB run')
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"), help="Device to run on")
    parser.add_argument("--save-predictions", type=Path, help="Save predictions to this file")
    parser.add_argument("--augmentation-samples", type=int, default=10,
        help="Number of samples to use for data augmentation")
    parser.add_argument("wandb_run", type=str, help="The W&B run ID.")
    args = parser.parse_args()
    wandb_run = wandb.Api().run(f"WyckoffTransformer/{args.wandb_run}")
    wandb_config = OmegaConf.create(dict(wandb_run.config))
    if base_config_name := (wandb_config.get("base_config", None)):
        base_config = OmegaConf.load(Path(__file__).parent.resolve() / "yamls" / "models" / f"{base_config_name}.yaml")
        final_config = OmegaConf.merge(base_config, wandb_config)
    else:
        final_config = wandb_config
    run_dir = Path("runs", args.wandb_run)
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        wandb_run.file("best_model_params.pt").download(run_dir)
    trainer = WyckoffTrainer.from_config(final_config, args.device, run_dir)
    trainer.model.load_state_dict(torch.load(trainer.run_path / "best_model_params.pt",
        weights_only=False, map_location=args.device))
    print(trainer.evaluate(trainer.test_dataset))
    if args.save_predictions:
        all_predictions = []
        for augmentation_ix in range(args.augmentation_samples):
            start_tokens, data_tokens, target, padding_mask = \
                trainer.test_dataset.get_augmented_data(no_batch=True)
            predictions = trainer.model(start_tokens, data_tokens, padding_mask, None)
            all_predictions.append(predictions.unsqueeze(1))
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.cat(all_predictions, axis=1), args.save_predictions)

if __name__ == '__main__':
    main()