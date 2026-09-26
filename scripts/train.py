from pathlib import Path
import argparse
import os
import logging
from omegaconf import OmegaConf
import torch
import wandb
import torch._dynamo
torch._dynamo.config.cache_size_limit = 128  # default is 64, set to 128 to avoid cache misses

from wyckoff_transformer.tokenization import refuse_if_obsolete, warn_if_obsolete
from wyckoff_transformer.dataset_manifest import refuse_if_obsolete_dataset
from wyckoff_transformer.paths import runs_root, wandb_dir
from wyckoff_transformer.distributed import init_distributed, shutdown_distributed
from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT  # noqa: E402
from wyckoff_transformer.trainer import train_from_config  # noqa: E402
# from wyckoff_transformer.bigtrainer import train_from_config


def main():
    parser = argparse.ArgumentParser(
        description='Train a model. To train on several GPUs of one node, launch it with '
                    'torchrun --standalone --nproc-per-node N; see docs/distributed_training.md')
    parser.add_argument("config", type=Path, help="The configuration file")
    parser.add_argument("dataset", type=str, help="Dataset to use")
    parser.add_argument("device", type=torch.device,
                        help="Device to train on. Under torchrun, `cuda` or `cpu`: each rank "
                             "takes cuda:LOCAL_RANK of the cards CUDA_VISIBLE_DEVICES shows.")
    parser.add_argument("--pilot", action="store_true", help="Run a pilot run by setting epochs to 3")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--run-path", type=Path, default=None,
                        help="Set the path for saving run data (default: the runs store)")
    parser.add_argument("--torch-num-thread", type=int, help="Number of threads for torch")
    parser.add_argument("--production", action="store_true", help="Train on the combined train+val+test dataset")
    parser.add_argument("--no-test", action="store_true", help="Skip loading and evaluating the test dataset")
    parser.add_argument("--compile", dest="compile_model", action="store_true", default=None,
                        help="Force WyckoffTrainer_args.compile_model=true")
    parser.add_argument("--no-compile", dest="compile_model", action="store_false", default=None,
                        help="Force WyckoffTrainer_args.compile_model=false")
    parser.add_argument("--resume", type=str, metavar="RUN_ID", default=None,
                        help="Continue the W&B run with this id from the last checkpoint in its "
                             "run directory, instead of starting a new one. The config given "
                             "must be the one that run started with.")
    parser.add_argument("--reschedule", action="store_true",
                        help="Allow a resumed run to change its learning-rate horizon: the "
                             "epoch count and the scheduler's own settings may differ from the "
                             "ones the run started with, and the run's config.yaml is rewritten "
                             "to the new schedule. Every other config difference is still "
                             "refused. Use it to land a run on a deadline -- bringing the decay "
                             "forward -- not to change what is being trained.")
    parser.add_argument("--allow-obsolete-tokeniser", action="store_true",
        help="Train on a tokeniser configuration marked obsolete. Only for "
             "reproducing a previous run; the reason it is obsolete is logged, "
             "and the resulting model inherits whatever it says.")
    parser.add_argument("--allow-obsolete-dataset", action="store_true",
        help="Start a run on a dataset yamls/datasets/ marks obsolete, or has no manifest "
             "for. Only for reproducing a previous run; the reason is logged.")
    parser.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY,
                        help="W&B entity to log under. Pinned by default so a run's home does not "
                             "depend on the shell's W&B configuration.")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT, help="W&B project")
    parser.add_argument("--dist-backend", choices=("nccl", "gloo"), default=None,
                        help="Collective backend under torchrun (default: nccl on cuda, gloo on cpu)")
    args = parser.parse_args()
    # A no-op outside torchrun. Under it, before anything touches the GPU, so that every
    # rank's CUDA context lands on its own card.
    distributed, args.device = init_distributed(args.device, args.dist_backend)
    if args.run_path is None:
        args.run_path = runs_root()
    
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        logging.basicConfig(level=logging.DEBUG)

    if args.torch_num_thread:
        torch.set_num_threads(args.torch_num_thread)

    if args.device.type == "cuda":
        # UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
        torch.set_float32_matmul_precision('high')
        
    config = OmegaConf.load(args.config)
    if args.pilot:
        print("Pilot run; overwriting epochs to 3")
        config['optimisation']['epochs'] = 3
        config['optimisation']['validation_period'] = 1
        tags = ["pilot"]
    else:
        tags = []
    config['name'] = args.config.stem
    config['dataset'] = args.dataset
    # Before W&B opens a run for it: an obsolete dataset must not leave an empty run behind.
    # A resumed run is existing work, so it only warns (train_from_config does that).
    if not args.resume:
        refuse_if_obsolete_dataset(args.dataset, "training", allow=args.allow_obsolete_dataset)
    if args.compile_model is not None:
        config['model']['WyckoffTrainer_args']['compile_model'] = args.compile_model

    tokeniser_config_path = Path(__file__).parent.parent.resolve() / "yamls" / "tokenisers" / f"{config.tokeniser.name}.yaml"
    tokeniser_config = OmegaConf.load(tokeniser_config_path)
    if args.allow_obsolete_tokeniser:
        warn_if_obsolete(tokeniser_config, "training, --allow-obsolete-tokeniser was given")
    else:
        refuse_if_obsolete(tokeniser_config, "training")
    # Several augmented fields are supported: AugmentedCascadeDataset draws one
    # variant index per example and gathers every augmented field with it, so
    # they stay paired -- which is what `site_symmetries` alongside
    # `sites_enumeration` needs, the two naming one Wyckoff position between them.
    # The dataset asserts they have equal variant counts.
    config['tokeniser'] = tokeniser_config
    config['production_training'] = args.production

    wandb_config = OmegaConf.to_container(config)
    args.run_path.mkdir(parents=True, exist_ok=True)
    if distributed.is_main:
        run = wandb.init(
            dir=wandb_dir(),
            entity=args.wandb_entity,
            project=args.wandb_project,
            job_type="train",
            tags=tags,
            config=wandb_config,
            # Log back into the same run rather than opening a second one, so the loss curve of a
            # resumed run is continuous and its run directory is the one holding the checkpoint.
            # "must" rather than "allow": a typo in the id has to fail, not silently start afresh.
            id=args.resume,
            resume="must" if args.resume else None,
            settings=wandb.Settings(
                    init_timeout=180
                )
            )
    # The other ranks log nothing, but share the run's id: it names the run directory they
    # read the package data and the checkpoint from. A disabled run makes every wandb call
    # in the trainer a no-op there.
    run_id = distributed.broadcast_object(wandb.run.id if distributed.is_main else None)
    if not distributed.is_main:
        run = wandb.init(mode="disabled", id=run_id, entity=args.wandb_entity,
                         project=args.wandb_project, config=wandb_config)
    with run:
        if distributed.enabled:
            # Next to `code`, not in `config`: the number of GPUs is how a run was executed,
            # not what it trains, and a chain may resume on a different count.
            wandb.config.update({"distributed": {
                "world_size": distributed.world_size, "backend": distributed.backend}},
                allow_val_change=True)

        # The commit this process trains with, as a launcher that knows it passes it in:
        # scripts/platforms/aspire2a/train_in_pbs.sh reads it from the checkout, since the
        # container has no git for W&B to ask. In the W&B config only -- not in `config`,
        # which a resume must match exactly -- and overwritten by every link of a chain,
        # each of which prints its own commit in its log.
        code = {key: os.environ[variable] for key, variable in (
            ("commit", "WYFORMER_GIT_COMMIT"), ("branch", "WYFORMER_GIT_BRANCH"))
            if os.environ.get(variable)}
        if code:
            wandb.config.update({"code": code}, allow_val_change=True)

        if not args.resume:
            configuration_artifact = wandb.Artifact(name=f"config_{config.name}_{wandb.run.id}", type="config")
            configuration_artifact.add_file(args.config, name="model.yaml")
            configuration_artifact.add_file(tokeniser_config_path, name="tokeniser.yaml")
            wandb.log_artifact(configuration_artifact)

        if args.debug:
            config["model"]['WyckoffTrainer_args']['compile_model'] = False
            with torch.autograd.detect_anomaly():
                train_from_config(config, args.device, run_path=args.run_path, production_training=args.production, no_test=args.no_test, resume=bool(args.resume), reschedule=args.reschedule, distributed=distributed, allow_obsolete_dataset=args.allow_obsolete_dataset)
        else:
            train_from_config(config, args.device, run_path=args.run_path, production_training=args.production, no_test=args.no_test, resume=bool(args.resume), reschedule=args.reschedule, distributed=distributed, allow_obsolete_dataset=args.allow_obsolete_dataset)
    shutdown_distributed(distributed)


if __name__ == '__main__':
    main()
