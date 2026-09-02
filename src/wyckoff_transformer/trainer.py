from typing import Tuple, Dict, Optional, List, Any, Union, Set
import importlib
import math
import shutil
from random import randint
import logging
from functools import partial
from pathlib import Path
import gzip
import json
import pickle
import numpy as np
import torch
from torch import nn
from torch import Tensor
from omegaconf import OmegaConf, DictConfig
from tqdm import trange
import wandb
from huggingface_hub import snapshot_download


import wyckoff_transformer
from wyckoff_transformer.cascade.dataset import AugmentedCascadeDataset, AugmentedCascadeLoader, TargetClass
from wyckoff_transformer.cascade.model import CascadeTransformer
from wyckoff_transformer.tokenization import (
    load_tensors_and_tokenisers,
    load_wyckoff_mappings, WYCKOFF_MAPPINGS_FILENAME,
    get_wp_index, WyckoffProcessor)
from wyckoff_transformer.generator import WyckoffGenerator
from wyckoff_transformer.evaluation import (
    evaluate_and_log, StatisticalEvaluator, smac_validity_from_counter)

logger = logging.getLogger(__file__)
start_token_distribution_file_name = "spacegroup_distribution.json"

# Transforms applied to the conditioning feature on its way into the model. The feature is stored,
# logged and passed around in its physical units everywhere else, so --condition-value 0.1 always
# means 0.1 eV/atom regardless of the transform in use.
# log1p suits energy above hull: it is near-linear where the interesting structures are
# (0.03 -> 0.0296) and compressive in the tail (37 -> 3.65), which keeps a handful of extreme
# outliers from dominating the scale of the AdaLN modulation input.
CONDITION_TRANSFORMS = {
    "log1p": torch.log1p,
}


def get_condition_transform(name: Optional[str]):
    """Resolve a conditioning transform by name; None means the identity."""
    if name is None:
        return None
    try:
        return CONDITION_TRANSFORMS[name]
    except KeyError:
        raise ValueError(
            f"Unknown condition_transform {name!r}; available: {sorted(CONDITION_TRANSFORMS)}") from None


#: Prefix `torch.compile` adds to every parameter name of the module it wraps.
_COMPILE_PREFIX = "_orig_mod."


def load_model_weights(
        model: nn.Module,
        path: Path,
        device: torch.device | str = "cpu") -> None:
    """Load a checkpoint into `model` regardless of which side was compiled.

    `torch.compile` renames every parameter with a `_orig_mod.` prefix, so a
    checkpoint saved from a compiled model loads only into another compiled model
    and vice versa. Training compiles on GPU while generation has no reason to,
    which otherwise makes a GPU run's checkpoint unreadable on CPU. The prefix is
    therefore normalised to whatever this particular `model` expects.

    Args:
        model: Destination module, compiled or not.
        path: Checkpoint written by `torch.save(model.state_dict(), ...)`.
        device: Device to map the storages onto. Without this a CUDA-trained
            checkpoint refuses to load where no GPU is visible.
    """
    state_dict = torch.load(path, map_location=device, weights_only=True)
    weights = {key.removeprefix(_COMPILE_PREFIX): value for key, value in state_dict.items()}
    if any(key.startswith(_COMPILE_PREFIX) for key in model.state_dict()):
        weights = {_COMPILE_PREFIX + key: value for key, value in weights.items()}
    model.load_state_dict(weights)


class WyckoffTrainer():
    # Fixed seed for the schedule_free_lag evaluations, so x and z are compared on identical
    # batches and successive checkpoints are comparable to each other.
    LAG_EVAL_SEED = 20260831

    #: Whether `scheduler` is step-indexed (WSD and friends, stepped after every optimiser
    #: step) rather than metric-driven (ReduceLROnPlateau, stepped on the validation loss).
    #: A class attribute so that trainers built by tests via __new__, which skip __init__,
    #: still have a sane default rather than an AttributeError in the training loop.
    scheduler_steps_per_batch = False

    #: Optimiser steps the step-indexed schedule was sized for, or None when the schedule
    #: does not depend on a horizon. train() checks this against the run it is about to do.
    scheduler_total_steps = None

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Optional[Dict[str, torch.tensor]],
        val_dataset: Optional[Dict[str, torch.tensor]],
        tokenisers: dict,
        token_engineers: dict,
        cascade_order: Tuple[str],
        cascade_is_target: Dict[str, bool],
        augmented_fields: List[str]|None,
        start_name: str,
        start_dtype: torch.dtype,
        target: TargetClass|str,
        evaluation_samples: int,
        multiclass_next_token_with_order_permutation: bool,
        optimisation_config: dict,
        device: torch.DeviceObjType,
        augmented_storage_device: Optional[torch.DeviceObjType] = None,
        batch_size: Optional[int] = None,
        train_batch_size: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        run_path: Optional[Path] = None,
        target_name = None,
        weights_path: Optional[Path] = None,
        test_dataset: Optional[Dict[str, torch.tensor]] = None,
        compile_model: bool = False,
        max_sequence_length: Optional[int] = None,
        start_token_distribution: Optional[Dict[str, Any]] = None,
        processor: Optional[WyckoffProcessor] = None,
        tokeniser_config: Optional[DictConfig] = None,
        production_training: bool = False,
        condition_feature: Optional[str] = None,
        condition_transform: Optional[str] = None,
    ):
        """
        Initializes the WyckoffTrainer.

        Args:
            model: The CascadeTransformer model to train.
            train_dataset: Dictionary containing training tensor data.
            val_dataset: Dictionary containing validation tensor data.
            tokenisers: Dictionary of tokenisers for each cascade field.
            token_engineers: Dictionary of token engineers for specialized fields.
            cascade_order: Tuple defining the order of fields in the cascade.
            cascade_is_target: Dictionary indicating if a field is a prediction target.
            augmented_fields: List of fields that have augmented variants.
            start_name: Name of the initial field (e.g., 'spacegroup_number').
            start_dtype: Data type for the start tokens.
            target: The prediction target type (NextToken, NumUniqueTokens, or Scalar).
                NumUniqueTokens: predict the number of unique tokens in the cascade&sequence encountered so far.
                    Intended for debugging the ability of the model to count.
            evaluation_samples: Number of samples to use during evaluation.
            multiclass_next_token_with_order_permutation: Train a permutation-invariant model by permuting the sequences,
                If the target is the next token, predict the 0-th cascade field as a multiclass target.
                You might also want to ensure that the model is permutation-invariant; pay attention to the positional encoding.
            optimisation_config: Configuration dictionary for optimizer and scheduler.
            device: Torch device to use for training.
            augmented_storage_device: Device to store augmented data on (if different from training device).
            batch_size: Deprecated. Use train_batch_size instead.
            train_batch_size: Mini-batch size for training.
            val_batch_size: Mini-batch size for validation.
            test_batch_size: Mini-batch size for testing.
            run_path: Path to save run artifacts and checkpoints.
            target_name: Name of the scalar target field (if target is Scalar).
            weights_path: Path to pre-trained model weights to load.
            test_dataset: Dictionary containing test tensor data.
            compile_model: Whether to use torch.compile on the model.
            max_sequence_length: Maximum number of rows in the sequence.
            start_token_distribution: Optional pre-computed distribution of start tokens.
            processor: Optional WyckoffProcessor instance.
            tokeniser_config: Configuration for the tokenisers.
            production_training: If True, merges all dataset splits (train/val/test) for training.
            condition_feature: Name of the feature to condition on via AdaLN.
            condition_transform: Name of a transform from CONDITION_TRANSFORMS applied to the
                conditioning feature on its way into the model. The stored data and the values
                accepted by generate_structures stay in physical units.
        """
        if isinstance(target, str):
            target = TargetClass[target]
        if target != TargetClass.Scalar:
            is_target_in_order = [cascade_is_target[field] for field in cascade_order]
            if not all(is_target_in_order[:-1]):
                raise NotImplementedError("Only one not targret field is supported "
                    "at the moment and it must be the last")
            self.cascade_target_count = sum(is_target_in_order)
        else:
            self.cascade_target_count = 0
        self.token_engineers = token_engineers
        self.processor = processor or WyckoffProcessor(
            config={},
            tokenisers=tokenisers,
            token_engineers=token_engineers,
        )
        self.cascade_is_target = cascade_is_target
        self.tokeniser_config = tokeniser_config
        # Nothing else will work in foreseeable future
        self.dtype = torch.int64
        if batch_size is not None:
            logger.warning("batch_size is deprecated, use train_batch_size, val_batch_size, test_batch_size")
            if train_batch_size is None:
                train_batch_size = batch_size
            elif train_batch_size != batch_size:
                raise ValueError("batch_size and train_batch_size differ")

        self.run_path = run_path
        self.condition_feature = condition_feature
        self.condition_transform = condition_transform
        self._condition_transform_fn = get_condition_transform(condition_transform)
        extra_fields = [condition_feature] if condition_feature is not None else None

        if target == TargetClass.NextToken:
            # Sequences have difference lengths, so we need to make sure that
            # long sequences don't dominate the loss, so we don't average the loss
            self.criterion = nn.CrossEntropyLoss(reduction="sum")
        elif target == TargetClass.NumUniqueTokens:
            if not multiclass_next_token_with_order_permutation:
                raise NotImplementedError("NumUniqueTokens is not implemented without permutations")
            self.criterion = nn.MSELoss(reduction="none")
        elif target == TargetClass.Scalar:
            # Assumes the batch size is the same for all batches
            self.criterion = nn.MSELoss(reduction='mean')
            self.testing_criterion = nn.L1Loss(reduction='mean')
        else:
            raise ValueError(f"Unknown target: {target}")
        
        self.model = model
        if compile_model:
            # Transformer doesn't support fullgraph=True
            self.compiled_model = torch.compile(self.model, fullgraph=False)
            self.model = self.compiled_model
        self.tokenisers = tokenisers
        self.device = device
        self.augmented_fields = augmented_fields

        self.masks_dict = {field: tokenisers[field].mask_token for field in cascade_order}
        self.pad_dict = {field: tokenisers[field].pad_token for field in cascade_order}
        self.stops_dict = {field: tokenisers[field].stop_token for field in cascade_order}
        self.num_classes_dict = {field: len(tokenisers[field]) for field in cascade_order}
        self.start_name = start_name
        self.max_sequence_length = max_sequence_length
        self.production_training = production_training

        if train_dataset is not None:
            self.train_dataset = AugmentedCascadeDataset(
                data=train_dataset,
                cascade_order=cascade_order,
                masks=self.masks_dict,
                pads=self.pad_dict,
                stops=self.stops_dict,
                num_classes=self.num_classes_dict,
                start_field=start_name,
                augmented_fields=augmented_fields,
                batch_size=train_batch_size,
                dtype=self.dtype,
                start_dtype=start_dtype,
                device=self.device,
                augmented_storage_device=augmented_storage_device,
                target_name=target_name,
                extra_fields=extra_fields)
            self.train_loader = AugmentedCascadeLoader.from_dataset(self.train_dataset)
            self.max_sequence_length = self.train_dataset.max_sequence_length
        else:
            self.train_dataset = None
            self.train_loader = None
        
        if "lr_per_sqrt_n_samples" in optimisation_config.optimiser:
            if "config" in optimisation_config.optimiser and "lr" in optimisation_config.optimiser.config:
                raise ValueError("Cannot specify both lr and lr_per_sqrt_n_samples")
            if train_dataset is None:
                # Optimizer is unused in generation-only mode, but we still need a valid config.
                samples_per_step = 1
            elif train_batch_size is None:
                samples_per_step = len(self.train_dataset)
            else:
                samples_per_step = train_batch_size
            optimisation_config.optimiser.update(
                {"config": {"lr": optimisation_config.optimiser.lr_per_sqrt_n_samples * samples_per_step**0.5}})
        optimizer_module_obj = importlib.import_module(optimisation_config.optimiser.get("module", "torch.optim"))
        self.optimizer = getattr(optimizer_module_obj, optimisation_config.optimiser.name)(
            model.parameters(), **optimisation_config.optimiser.config)
        # Generation-only mode has no training set, so a step-indexed schedule has no
        # horizon to size itself from -- and nothing ever steps it. Skipping it here is
        # what lets a run trained under such a schedule be sampled without its dataset.
        if "scheduler" in optimisation_config and self.train_loader is None:
            logger.info("No training set; skipping the LR schedule, which generation never steps.")
            self.scheduler = None
            self.scheduler_steps_per_batch = False
        elif "scheduler" in optimisation_config:
            scheduler_config = dict(optimisation_config.scheduler.get("config", {}))
            scheduler_module = importlib.import_module(
                optimisation_config.scheduler.get("module", "torch.optim.lr_scheduler"))
            scheduler_name = optimisation_config.scheduler.name
            scheduler_factory = getattr(scheduler_module, scheduler_name)
            if scheduler_name == "ReduceLROnPlateau":
                # Metric-driven: it takes the mode positionally and is stepped with the
                # validation loss, once every validation_period epochs.
                self.scheduler = scheduler_factory(self.optimizer, 'min', **scheduler_config)
                self.scheduler_steps_per_batch = False
            else:
                # Step-indexed: it is stepped after every optimiser step and takes no metric.
                if scheduler_name in getattr(scheduler_module, "NEEDS_TOTAL_STEPS", ()):
                    # The horizon is in optimiser steps, which is what the schedule indexes on;
                    # `epochs` counts passes, each of them batches_per_epoch steps.
                    scheduler_config.setdefault(
                        "total_steps",
                        optimisation_config.epochs * self.train_loader.batches_per_epoch)
                    self.scheduler_total_steps = scheduler_config["total_steps"]
                self.scheduler = scheduler_factory(self.optimizer, **scheduler_config)
                self.scheduler_steps_per_batch = True
        else:
            self.scheduler = None
            self.scheduler_steps_per_batch = False

        if val_dataset is not None:
            self.val_dataset = AugmentedCascadeDataset(
                data=val_dataset,
                cascade_order=cascade_order,
                masks=self.masks_dict,
                pads=self.pad_dict,
                stops=self.stops_dict,
                num_classes=self.num_classes_dict,
                start_field=start_name,
                augmented_fields=augmented_fields,
                batch_size=val_batch_size,
                dtype=self.dtype,
                start_dtype=start_dtype,
                device=device,
                augmented_storage_device=augmented_storage_device,
                target_name=target_name,
                extra_fields=extra_fields,
                )
            self.val_loader = AugmentedCascadeLoader.from_dataset(self.val_dataset)
            if self.max_sequence_length is None:
                self.max_sequence_length = self.val_dataset.max_sequence_length
        else:
            self.val_dataset = None
            self.val_loader = None

        if test_dataset is None:
            self.test_dataset = None
            self.test_loader = None
        else:
            self.test_dataset = AugmentedCascadeDataset(
                data=test_dataset,
                cascade_order=cascade_order,
                masks=self.masks_dict,
                pads=self.pad_dict,
                stops=self.stops_dict,
                num_classes=self.num_classes_dict,
                start_field=start_name,
                augmented_fields=augmented_fields,
                batch_size=test_batch_size,
                dtype=self.dtype,
                start_dtype=start_dtype,
                device=device,
                augmented_storage_device=augmented_storage_device,
                target_name=target_name,
                extra_fields=extra_fields,
            )
            self.test_loader = AugmentedCascadeLoader.from_dataset(self.test_dataset)

        if self.train_dataset is not None and self.val_dataset is not None:
            assert self.train_dataset.max_sequence_length == self.val_dataset.max_sequence_length
        if self.max_sequence_length is None:
            raise ValueError("max_sequence_length must be available from datasets or provided explicitly")

        # Pre-cast conditioning feature to (num_examples, condition_dim) float32 on self.device,
        # so the loss path can index without a per-step .to(device).to(float32).
        if self.condition_feature is not None:
            for ds in (self.train_dataset, self.val_dataset, self.test_dataset):
                if ds is None:
                    continue
                cond_tensor = ds.data[self.condition_feature].to(self.device, dtype=torch.float32)
                if cond_tensor.dim() == 1:
                    cond_tensor = cond_tensor.unsqueeze(1)
                ds.data[self.condition_feature] = cond_tensor
                # Stored in physical units; the transform is applied on the way into the model.
                # Validate once here rather than per step, which would force a device sync.
                self._validate_condition_values(cond_tensor)
    
        # Optional: omit, or set to null, to train without a norm constraint. The pre-clip norm
        # is logged either way, so a run can watch the gradient scale without being shaped by it.
        self.clip_grad_norm = optimisation_config.get("clip_grad_norm", None)
        self.cascade_len = len(cascade_order)
        self.cascade_order = cascade_order
        self.epochs = optimisation_config.epochs
        if weights_path is not None:
            load_model_weights(self.model, weights_path, device)

        self.validation_period = optimisation_config.validation_period
        self.early_stopping_patience_epochs = optimisation_config.early_stopping_patience_epochs
        self.target = target
        self.multiclass_next_token_with_order_permutation = multiclass_next_token_with_order_permutation
        self.evaluation_samples = evaluation_samples
        self.start_token_distribution = start_token_distribution


    def _validate_condition_values(self, values: Tensor):
        """Reject conditioning values the transform cannot represent, with a legible message."""
        if self.condition_transform == "log1p" and bool((values < 0).any()):
            raise ValueError(
                f"condition_transform='log1p' requires {self.condition_feature} >= 0, "
                "but negative values were supplied")


    def transform_condition(self, values: Optional[Tensor]) -> Optional[Tensor]:
        """Map a conditioning tensor from physical units to what the model consumes."""
        if values is None or self._condition_transform_fn is None:
            return values
        return self._condition_transform_fn(values)


    @staticmethod
    def get_start_token_distribution_path(run_path: Path) -> Path:
        return run_path / start_token_distribution_file_name


    @staticmethod
    def load_start_token_distribution_file(distribution_path: Path) -> Dict[str, Any]:
        with distribution_path.open("rt", encoding="ascii") as f:
            return json.load(f)


    def _build_start_token_distribution(self) -> Dict[str, Any]:
        if self.train_dataset is None or self.val_dataset is None:
            raise ValueError("Cannot build start-token distribution without train and validation datasets")
        if self.model.start_type == "categorial":
            max_start = self.model.start_embedding.num_embeddings
            start_counts = torch.bincount(self.train_dataset.start_tokens, minlength=max_start)
            if not self.production_training:
                start_counts += torch.bincount(self.val_dataset.start_tokens, minlength=max_start)
            return {
                "start_name": self.start_name,
                "start_type": self.model.start_type,
                "max_sequence_length": int(self.max_sequence_length),
                "counts": start_counts.cpu().tolist(),
            }

        if self.model.start_type == "one_hot":
            if self.production_training:
                all_starts = self.train_dataset.start_tokens
            else:
                all_starts = torch.cat([self.train_dataset.start_tokens, self.val_dataset.start_tokens], dim=0)
            unique_vectors, inverse_indices = torch.unique(all_starts, dim=0, return_inverse=True)
            counts = torch.bincount(inverse_indices)
            return {
                "start_name": self.start_name,
                "start_type": self.model.start_type,
                "max_sequence_length": int(self.max_sequence_length),
                "vectors": unique_vectors.cpu().tolist(),
                "counts": counts.cpu().tolist(),
            }
        raise ValueError(f"Unknown start type: {self.model.start_type}")


    def save_start_token_distribution(self) -> Path:
        if self.run_path is None:
            raise ValueError("run_path must be set to save start-token distribution")
        if self.start_token_distribution is None:
            self.start_token_distribution = self._build_start_token_distribution()
        distribution_path = self.get_start_token_distribution_path(self.run_path)
        with distribution_path.open("wt", encoding="ascii") as f:
            json.dump(self.start_token_distribution, f)
        artifact = wandb.Artifact(name=f"spacegroup_distribution_{wandb.run.id}", type="dataset_stats")
        artifact.add_file(distribution_path)
        wandb.log_artifact(artifact)
        return distribution_path


    def _sample_start_tokens_from_distribution(self, n_structures: int) -> torch.Tensor:
        if self.start_token_distribution is None:
            if self.train_dataset is not None and self.val_dataset is not None:
                self.start_token_distribution = self._build_start_token_distribution()
            else:
                raise ValueError(
                    "No start-token distribution available. Save or provide "
                    f"{start_token_distribution_file_name} before generation.")

        counts = torch.tensor(self.start_token_distribution["counts"], dtype=torch.float32, device=self.device)
        if counts.sum() <= 0:
            raise ValueError("Start-token distribution counts are empty")
        sample_indices = torch.distributions.Categorical(probs=counts / counts.sum()).sample((n_structures,))

        start_type = self.start_token_distribution["start_type"]
        if start_type == "categorial":
            return sample_indices
        if start_type == "one_hot":
            vectors = torch.tensor(self.start_token_distribution["vectors"], dtype=torch.float32, device=self.device)
            return vectors[sample_indices]
        raise ValueError(f"Unknown start type in distribution: {start_type}")


    @classmethod
    def from_config(cls, config_dict: dict|DictConfig,
                    device: torch.device,
                    use_cached_tensors: bool = True,
                    run_path: Optional[Path] = Path("runs"),
                    load_datasets: bool = True,
                    production_training: bool = False,
                    no_test: bool = False):
        config = OmegaConf.create(config_dict)
        if config.model.WyckoffTrainer_args.get("multiclass_next_token_with_order_permutation", False) and \
            not config.model.CascadeTransformer_args.learned_positional_encoding_only_masked:

            raise ValueError("Multiclass target with order permutation requires learned positional encoding only masked, ",
                            "otherwise the Transformer is not permutation invariant.")
        if load_datasets:
            tensors, tokenisers, token_engineers = load_tensors_and_tokenisers(
                config.dataset, config.tokeniser.name, use_cached_tensors=use_cached_tensors,
                tokenizer_path=run_path / "wyckoff_processor.json" if not use_cached_tensors else None)
            processor = WyckoffProcessor(
                config=config.get("tokeniser", {}),
                tokenisers=tokenisers,
                token_engineers=token_engineers,
            )
            train_data = tensors["train"]
            val_data = tensors["val"]
            test_data = None
            if not no_test:
                if "test" in tensors:
                    test_data = tensors["test"]
                else:
                    logger.warning("Test dataset not found in tensors. Setting test_data to None and no_test to True.")
                    test_data = None
                    no_test = True

            if production_training:
                # Merge all datasets into one for training and validation
                merged_data = {}
                relevant_splits = [train_data, val_data]
                if test_data is not None:
                    relevant_splits.append(test_data)
                
                # We assume all shards have the same keys (fields)
                for field in train_data.keys():
                    if isinstance(train_data[field], list):
                        # Augmented fields (list of lists of tensors)
                        # We need to flatten and concat
                        merged_field = []
                        for split in relevant_splits:
                            merged_field.extend(split[field])
                        merged_data[field] = merged_field
                    else:
                        merged_data[field] = torch.cat([split[field] for split in relevant_splits], dim=0)
                
                train_data = merged_data
                val_data = merged_data
                test_data = None

            distribution = None
            max_sequence_length = None
        else:
            processor_path = run_path / "wyckoff_processor.json"
            processor = WyckoffProcessor.from_pretrained(processor_path)
            tokenisers = processor.tokenisers
            token_engineers = processor.token_engineers
            train_data = None
            val_data = None
            test_data = None
            distribution_path = cls.get_start_token_distribution_path(run_path)
            if not distribution_path.exists():
                raise FileNotFoundError(
                    f"Missing {distribution_path}. This file is required for generation without datasets.")
            distribution = cls.load_start_token_distribution_file(distribution_path)
            max_sequence_length = int(distribution["max_sequence_length"])
        model = CascadeTransformer.from_config_and_tokenisers(config, tokenisers, device)
        # model.to(torch.float32)
        # Our hihgly dynamic concat-heavy workflow doesn't benefit much from compilation
        # torch._dynamo.config.cache_size_limit = 128
        # model = torch.compile(model, dynamic=True)
        if config.model.CascadeTransformer_args.start_type == "categorial":
            start_dtype = torch.int64
        # one-hots are encoded by a linear layer
        elif config.model.CascadeTransformer_args.start_type == "one_hot":
            start_dtype = torch.float32
        else:
            raise ValueError(f"Unknown start type: {config.model.CascadeTransformer_args.start_type}")
        return cls(
            model, train_data, val_data, tokenisers, token_engineers, config.model.cascade.order,
            config.model.cascade.get("is_target", None),
            config.model.cascade.get("augmented", None),
            config.model.start_token,
            optimisation_config=config.optimisation, device=device,
            run_path=run_path,
            start_dtype=start_dtype,
            test_dataset=test_data,
            max_sequence_length=max_sequence_length,
            start_token_distribution=distribution,
            processor=processor,
            tokeniser_config=config.tokeniser,
            production_training=production_training,
            **config.model.WyckoffTrainer_args)


    @classmethod
    def from_huggingface(
        cls,
        repo_id: str,
        device: torch.device,
        revision: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        load_datasets: bool = False,
    ) -> "WyckoffTrainer":
        """Load a WyckoffTrainer from a HuggingFace Hub repository.

        Args:
            repo_id: HuggingFace repo ID, e.g. 'username/model-name'.
            device: Torch device to use.
            revision: Optional git revision (branch, tag, or commit SHA).
            cache_dir: Optional local directory to cache downloaded files.
            load_datasets: Whether to load datasets (requires dataset config).
        """
        local_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        model_path = Path(local_dir)
        config = OmegaConf.load(model_path / "config.yaml")
        trainer = cls.from_config(
            config,
            device=device,
            use_cached_tensors=False,
            run_path=model_path,
            load_datasets=load_datasets,
        )
        load_model_weights(trainer.model, model_path / "best_model_params.pt", device)
        return trainer


    def get_loss(
        self,
        dataset: AugmentedCascadeDataset,
        known_seq_len: int,
        known_cascade_len: int|None,
        loader: Optional[AugmentedCascadeLoader] = None,
        no_batch: bool = False,
        testing: bool = False,
        return_n_samples: bool = False,
        rescale_to_viable: bool = True) -> Tensor | tuple[Tensor, int]:
        """
        Computes loss on the dataset.

        Args:
            return_n_samples: also return the number of examples the summed loss was computed
                over. Needed to rescale a mini-batch loss to a whole-split estimate.
            rescale_to_viable: put the summed loss on the scale of a pass over every example
                viable at this known_seq_len. What evaluation wants, since it sums over every
                known_seq_len and so has to weight each one by how much data reaches it.
                Training draws known_seq_len from that same distribution instead (see
                AugmentedCascadeDataset.sample_known_seq_len) and so passes False.
        """
        logging.debug("Known sequence length: %i", known_seq_len)
        logging.debug("Known cascade length: %s", str(known_cascade_len))
        # Step 1: Get the data
        if self.multiclass_next_token_with_order_permutation:
            if loader is not None and not no_batch:
                batch_selection = loader.get_next_viable_batch(known_seq_len)
            else:
                batch_selection = slice(None)

            if self.target == TargetClass.NextToken:
                # Once we have sampled the first cascade field, the prediction target is no longer multiclass
                # However, we still need to permute the sequence so that the autoregression is
                # permutation-invariant.
                start_tokens, masked_data, target, batch_selection = dataset.get_masked_multiclass_cascade_data(
                    known_seq_len, known_cascade_len, multiclass_target=(known_cascade_len == 0),
                    target_type=self.target, batch_target_is_viable=batch_selection, return_chosen_indices=True)
            elif self.target == TargetClass.NumUniqueTokens:
                start_tokens, masked_data, target, batch_selection = dataset.get_masked_multiclass_cascade_data(
                    known_seq_len, known_cascade_len, multiclass_target=False, target_type=self.target,
                    batch_target_is_viable=batch_selection, return_chosen_indices=True)
                logging.debug("Target: %s", target)
                # Counts are integers, as they should be, but MSE needs a float
                target = target.float()
            else:
                raise ValueError(f"Target {self.target} is not supported by "
                                  "multiclass_next_token_with_order_permutation")
        else:
            if loader is not None and not no_batch:
                batch_selection = loader.get_next_batch()
            else:
                batch_selection = slice(None)

            if self.target == TargetClass.Scalar:
                start_tokens, masked_data, target, padding_mask = dataset.get_augmented_data(batch_selection=batch_selection)
            else:
                # get_masked_cascade_data doesn't support batching in original code and still doesn't
                # as it expects to use the whole dataset or a specific set of indices.
                # Actually, the original code had: if self.batch_size is not None: raise NotImplementedError
                # Let's keep that behavior but allow passing indices if we want to in future.
                start_tokens, masked_data, target, batch_selection = dataset.get_masked_cascade_data(
                    known_seq_len, known_cascade_len, return_chosen_indices=True)

        cond = None
        if self.condition_feature is not None:
            cond = self.transform_condition(dataset.data[self.condition_feature][batch_selection])

        # Step 2: Get the prediction
        if self.target == TargetClass.NextToken:
            # No padding, as we have already discarded the padding
            prediction = self.model(start_tokens, masked_data, None, known_cascade_len, cond=cond)
        elif self.target == TargetClass.NumUniqueTokens:
            # No padding, as we have already discarded the padding
            prediction = self.model(start_tokens, masked_data, None, None, cond=cond)
        elif self.target == TargetClass.Scalar:
            logger.debug("Start tokens size: %s", start_tokens.size())
            #logger.debug("Start tokens isnan: %s", start_tokens.isnan().any())
            #logger.debug("Masked data isnan: %s", any((a.isnan().any() for a in masked_data)))
            #logger.debug("Padding mask isnan: %s", padding_mask.isnan().any())
            prediction = self.model(start_tokens, masked_data, padding_mask, None, cond=cond).squeeze()
            #logger.debug("Prediction isnan: %s", prediction.isnan().any())
        else:
            raise ValueError(f"Unknown target: {self.target}")
        # Step 3: Calculate the loss
        # logger.debug("Target isnan: %s", target.isnan().any())
        if logger.isEnabledFor(logging.DEBUG):
            # Logging arguments are evaluated eagerly, so without this guard every step pays for
            # two device reductions and the synchronisation they imply, whatever the log level.
            logger.debug("Target min: %s, max: %s", target.min(), target.max())
            logger.debug("Prediction shape: %s", prediction.shape)
        if testing:
            loss = self.testing_criterion(prediction, target)
        else:
            loss = self.criterion(prediction, target)
        n_samples = start_tokens.size(0)
        if (rescale_to_viable and self.target == TargetClass.NextToken
                and self.multiclass_next_token_with_order_permutation):
            # The batch holds as many examples viable at this known_seq_len as the data allows,
            # so the summed cross-entropy is put back on the scale of a pass over every viable
            # example. Without this, a known_seq_len only 0.1% of the data reaches would
            # contribute to the split's total NLL as if it reached all of it. In batchless mode
            # the batch already is every viable example, so the factor is exactly 1.
            n_viable = dataset.viable_count(known_seq_len)
            if n_viable != n_samples:
                loss = loss * (n_viable / n_samples)
        if return_n_samples:
            return loss, n_samples
        return loss


    def train_epoch(self):
        if self.train_dataset is None:
            raise ValueError("train_dataset is not available")
        self.model.train()
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        for _ in trange(self.train_loader.batches_per_epoch, leave=False):
            self.optimizer.zero_grad(set_to_none=True)
            if self.target in (TargetClass.NextToken, TargetClass.NumUniqueTokens):
                known_cascade_len = (
                    randint(0, self.cascade_target_count - 1)
                    if self.target == TargetClass.NextToken else 0)
                if self.multiclass_next_token_with_order_permutation:
                    # Weight each known_seq_len by how much data reaches it here, in the
                    # sampling, rather than in the loss. clip_grad_norm survives this and does
                    # not survive the alternative; it also never draws a known_seq_len no
                    # example is viable at.
                    known_seq_len = self.train_dataset.sample_known_seq_len()
                else:
                    known_seq_len = randint(0, self.train_dataset.max_sequence_length - 1)
            elif self.target == TargetClass.Scalar:
                # Use full sequences
                known_cascade_len = None
                known_seq_len = self.train_dataset.max_sequence_length - 1
            else:
                raise ValueError(f"Unknown target: {self.target}")
            if self.target == TargetClass.NextToken and self.multiclass_next_token_with_order_permutation:
                # Every step is a draw from the split's own distribution over known_seq_len, so
                # the per-known_seq_len weight lives in the sampling and the loss needs no
                # reweighting -- only a reduction. A per-example mean, rather than the sum the
                # criterion returns, is what makes the learning rate and clip_grad_norm mean
                # something on their own: gradient norms land in [0.05, 5] instead of [1e3, 1e5],
                # so a threshold can be set above the working range and actually catch outliers,
                # and the learning rate stops being a function of train_batch_size.
                # (The non-multiclass branch below keeps its summed loss: the learning rates in
                # the other ~130 configs are tuned against that scale.)
                loss, n_samples = self.get_loss(
                    self.train_dataset, known_seq_len, known_cascade_len, loader=self.train_loader,
                    rescale_to_viable=False, return_n_samples=True)
                loss = loss / n_samples
            else:
                loss = self.get_loss(
                    self.train_dataset, known_seq_len, known_cascade_len, loader=self.train_loader)
            if self.target == TargetClass.NumUniqueTokens:
                # Predictions are [batch_size, cascade_size]
                # Unreduced MSE is [batch_size, cascade_size]
                # We avoid averaging them at the level of self.criterion, so we can log
                # the loss for each cascade field separately.
                loss = loss.mean()
            loss.backward()
            # Measure the norm whether or not it is constrained: an infinite max_norm makes
            # clip_grad_norm_ a no-op that still returns the pre-clip norm. Worth logging even
            # when clipping is on -- a threshold that binds on every step is not catching
            # outliers, it is setting the step size, and only this metric shows the difference.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                math.inf if self.clip_grad_norm is None else self.clip_grad_norm)
            self.optimizer.step()
            if self.scheduler_steps_per_batch:
                self.scheduler.step()
            wandb.log({"loss.batch.train": loss,
                       "grad_norm": grad_norm,
                       "known_seq_len": known_seq_len,
                       "known_cascade_len": known_cascade_len})


    @torch.no_grad()
    def schedule_free_lag(self) -> Dict[str, float]:
        """How far the reported iterate has fallen behind the one the optimiser is moving.

        Schedule-free optimisers report and checkpoint `x`, a weighted mean of the raw iterates
        `z`. With the default r=0 that mean is uniform over the whole run, so `x` only converges
        if `z` settles. A full-batch gradient vanishes at a stationary point and `z` does settle;
        a mini-batch gradient never vanishes, so `z` keeps moving and `x` trails it by a distance
        that grows with the step count. Every mini-batched run in this project has a rising
        training loss; no full-batch one does.

        The signature is unambiguous and appears long before the loss curve bends: `lag` climbing
        without bound, and `loss_x_minus_z` -- normally negative, since averaging is the point --
        rising towards zero. An average that is worse than what it averages has gone stale, and
        the fix is to shorten the window (schedule-free's `r`), restart the averaging, or stop
        `z` drifting at all (`weight_decay`).

        Returns an empty dict for optimisers that keep no such average.
        """
        state = getattr(self.optimizer, "state", None)
        if state is None or self.train_dataset is None:
            return {}
        params = [p for group in self.optimizer.param_groups for p in group["params"]]
        if not any(state.get(p, {}).get("z") is not None for p in params):
            return {}
        # `evaluate` leaves the optimiser in eval mode, so the parameters hold x; make sure of it.
        self.optimizer.eval()
        x = [p.detach().clone() for p in params]
        # A parameter that never received a gradient has no z: the optimiser has never moved it,
        # so x and z coincide there and it contributes nothing to the lag.
        z = [state.get(p, {}).get("z", p).detach() for p in params]
        flat_x = torch.cat([t.flatten() for t in x])
        flat_z = torch.cat([t.flatten() for t in z])
        norm_x = flat_x.norm()
        metrics = {"lag": (flat_x - flat_z).norm().item(),
                   "norm_x": norm_x.item(),
                   "norm_z": flat_z.norm().item()}
        metrics["lag_relative"] = metrics["lag"] / metrics["norm_x"] if norm_x > 0 else float("nan")
        # Evaluate both iterates on the same batches and the same permutations: the quantity of
        # interest is their difference, and an unpaired comparison would bury it in estimator
        # noise. A fixed seed also makes the metric comparable from checkpoint to checkpoint.
        # The training stream is restored afterwards so the monitor cannot alter the run.
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(self.LAG_EVAL_SEED)
            loss_x = self.evaluate(self.train_dataset, self.train_loader).sum().item()
            for p, zi in zip(params, z):
                p.copy_(zi)
            try:
                torch.manual_seed(self.LAG_EVAL_SEED)
                loss_z = self.evaluate(self.train_dataset, self.train_loader).sum().item()
            finally:
                for p, xi in zip(params, x):
                    p.copy_(xi)
        finally:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        metrics["loss_x"] = loss_x
        metrics["loss_z"] = loss_z
        metrics["loss_x_minus_z"] = loss_x - loss_z
        return metrics


    @torch.no_grad()
    def evaluate(self, dataset: AugmentedCascadeDataset, loader: Optional[AugmentedCascadeLoader] = None) -> Tensor:
        """
        Evaluates the model by calculating the average loss on the dataset.
        Args:
            dataset: The dataset to evaluate on.
            loader: The loader to use for batching.
        Returns:
            The average loss on the dataset.
        """
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()
        
        if self.target == TargetClass.Scalar:
            if loader is None:
                raise ValueError("Evaluation of Scalar target requires a loader")
            if not loader.fix_batch_size:
                raise NotImplementedError("Only fixed batch size is supported for Scalar evaluation")
            
            loss = torch.zeros(1, device=self.device)
            # Augmentation
            for _ in range(self.evaluation_samples):
                for _ in range(loader.batches_per_epoch):
                    loss += self.get_loss(dataset, self.max_sequence_length - 1, None, loader=loader, testing=True)
            # Above we check that the batch size is the same for all batches
            return loss / self.evaluation_samples / loader.batches_per_epoch

        loss = torch.zeros(self.cascade_len, device=self.device)          

        # set to batching mode
        # for _ in range(self.evaluation_samples):
        #     for _ in range(dataset.batches_per_epoch):
        #         for known_seq_len in range(dataset.max_sequence_length):

        #             if self.target == TargetClass.NextToken:
        #                 for known_cascade_len in range(self.cascade_target_count):
        #                     loss[known_cascade_len] += self.get_loss(
        #                         dataset,
        #                         known_seq_len=known_seq_len,
        #                         known_cascade_len=known_cascade_len,
        #                         no_batch=False,      # mini-batch mode
        #                     )
        #             else:  # TargetClass.NumUniqueTokens
        #                 loss += self.get_loss(
        #                     dataset,
        #                     known_seq_len=known_seq_len,
        #                     known_cascade_len=0,
        #                     no_batch=False,          # mini-batch mode
        #                 ).sum(dim=0)

        # return loss / self.evaluation_samples / len(dataset)

        # Without a loader every viable example of the split goes through the model in a single
        # forward pass per (known_seq_len, known_cascade_len). That is exact, but the activations
        # scale with the whole split times max_sequence_length, which does not fit on a GPU for
        # datasets the size of LeMat-Bulk. With a loader we evaluate one mini-batch per
        # (known_seq_len, known_cascade_len) instead and rescale each summed loss by
        # n_viable / n_samples, which keeps the returned quantity on the same scale as the
        # exact path (mean total NLL per structure) so runs stay comparable.
        for _ in range(self.evaluation_samples):
            for known_seq_len in range(dataset.max_sequence_length):
                if dataset.viable_count(known_seq_len) == 0:
                    # Nothing in this split reaches that far. The padded width comes from the
                    # tensors and can exceed the longest sequence of an individual split, which
                    # leaves both the sampled and the exhaustive path with an empty batch.
                    continue
                if self.target == TargetClass.NextToken:
                    for known_cascade_len in range(self.cascade_target_count):
                        # get_loss already rescales a sampled batch to the whole viable set.
                        loss[known_cascade_len] += self.get_loss(
                            dataset, known_seq_len, known_cascade_len, loader=loader,
                            no_batch=loader is None)
                else: # NumUniqueTokens
                    if loader is None:
                        loss += self.get_loss(dataset, known_seq_len, 0, no_batch=True).sum(dim=0)
                    else:
                        batch_loss, n_samples = self.get_loss(
                            dataset, known_seq_len, 0, loader=loader, no_batch=False,
                            return_n_samples=True)
                        loss += batch_loss.sum(dim=0) * (
                            dataset.viable_count(known_seq_len) / n_samples)
            # ln(P) = ln p(t_n|t_n-1, ..., t_1) + ... + ln p(t_2|t_1)
            # We are minimising the negative log likelihood of the whole sequences
        return loss / self.evaluation_samples / len(dataset)


    def train(self):
        if self.train_dataset is None:
            raise ValueError("train() requires a train dataset")
        if not self.production_training and self.val_dataset is None:
             raise ValueError("train() requires a validation dataset (or production_training=True)")
        if self.scheduler_total_steps is not None:
            # The schedule places its warmup and decay as fractions of a fixed horizon, so a
            # run that does not consume exactly that many steps gets a schedule meant for a
            # different run: too few and the decay -- where most of the gain is -- never
            # happens, too many and the tail is spent at the floor. The horizon is derived
            # from `epochs` at construction, so this only fires if one of them was changed
            # afterwards.
            planned = self.epochs * self.train_loader.batches_per_epoch
            if planned != self.scheduler_total_steps:
                raise ValueError(
                    f"The learning rate schedule was built for {self.scheduler_total_steps} "
                    f"optimiser steps, but this run is {self.epochs} epochs x "
                    f"{self.train_loader.batches_per_epoch} batches = {planned} steps. "
                    f"Change `epochs` in the config rather than on the trainer.")
            if self.early_stopping_patience_epochs < self.epochs:
                logging.warning(
                    "early_stopping_patience_epochs (%d) is below epochs (%d): a step-indexed "
                    "schedule that stops early never completes its decay, which is where most "
                    "of the improvement appears.",
                    self.early_stopping_patience_epochs, self.epochs)
        best_val_loss = float('inf')
        best_val_epoch = 0
        self.run_path.mkdir(exist_ok=True)
        best_model_params_path = self.run_path / "best_model_params.pt"
        self.save_start_token_distribution()

        wandb.define_metric("loss.epoch.val.total", step_metric="epoch", summary="min")
        wandb.define_metric("loss.epoch.train.total", step_metric="epoch", summary="min")
        wandb.define_metric("loss.epoch.val_best", step_metric="epoch", summary="min")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("known_seq_len", hidden=True)
        wandb.define_metric("known_cascade_len", hidden=True)

        for epoch in (train_tqdm := trange(self.epochs)):
            self.train_epoch()
            if epoch % self.validation_period == 0 or epoch == self.epochs - 1:
                train_loss = self.evaluate(self.train_dataset, self.train_loader)
                raw_losses = {
                    "train": train_loss,
                    "val": train_loss if self.production_training else self.evaluate(self.val_dataset, self.val_loader),
                }
                if self.test_dataset is not None:
                    raw_losses['test'] = self.evaluate(self.test_dataset, self.test_loader)
                loss_dict = {}
                if self.target == TargetClass.Scalar:
                    total_val_loss = raw_losses['val']
                    for name, loss in raw_losses.items():
                        loss_dict[name] = {"mae": loss.item()}
                else:
                    total_val_loss = raw_losses['val'].sum()
                    for name, loss in raw_losses.items():
                        loss_dict[name] = {name: loss[i] for i, name in enumerate(self.cascade_order)}
                        loss_dict[name]["total"] = loss.sum().item()
                logged = {"loss.epoch": loss_dict,
                          "lr": self.optimizer.param_groups[0]['lr'],
                          "epoch": epoch}
                lag_metrics = self.schedule_free_lag()
                if lag_metrics:
                    logged["schedule_free"] = lag_metrics
                wandb.log(logged, commit=False)
                if total_val_loss < best_val_loss:
                    best_val_loss = total_val_loss
                    best_val_epoch = epoch
                    torch.save(self.model.state_dict(), best_model_params_path)
                    best_model_artifact = wandb.Artifact(
                        name=f"best_model_{wandb.run.id}",
                        type="model",
                        metadata={"epoch": epoch})
                    best_model_artifact.add_file(best_model_params_path)
                    wandb.log_artifact(best_model_artifact)
                    train_tqdm.set_description(
                        f"Epoch {epoch}; loss_epoch.val {total_val_loss.item():.4f} "
                        f"saved to {best_model_params_path}")
                    wandb.log({"loss.epoch.val_best": best_val_loss}, commit=False)
                if epoch - best_val_epoch > self.early_stopping_patience_epochs:
                    print(f"Early stopping at epoch {epoch} after more than "
                          f"{self.early_stopping_patience_epochs} epochs without improvement")
                    break
                # Don't step the scheduler on the tail epoch to presereve
                # patience behaviour
                if (self.scheduler and not self.scheduler_steps_per_batch
                        and epoch % self.validation_period == 0):
                    self.scheduler.step(total_val_loss)

        # Make sure we log the last evaluation results
        wandb.log({}, commit=True)


    def generate_structures(
            self,
            n_structures: int,
            calibrate: bool,
            compute_validity_per_known_sequence_length: bool = False,
            start_tensor: Optional[torch.Tensor] = None,
            required_element_set: Optional[Union[str, Set[int]]] = None,
            allowed_element_set: Union[str, Set[int]] = "all",
            temperature: float = 1.0,
            cond: Optional[torch.Tensor] = None,
            ) -> List[dict] | Tuple[List[dict], List, List]:
        """
        Generates structures by autoregressively sampling from the model.

        Args:
            n_structures: The number of structures to generate.
            calibrate: Whether to calibrate the generation probabilities on the validation dataset.
            compute_validity_per_known_sequence_length: Whether to compute the formal validity of
                the generated tensors separately for each known sequence length.
            start_tensor: Optional tensor of start tokens. If None, sampled from the training distribution.
            required_element_set: If set (including an empty set), activates element-constrained
                generation. A set of required element token IDs or a dash-separated string (e.g. "Li-O")
                that MUST appear in every generated structure.
            allowed_element_set: Controls the pool of allowed elements. "all" allows every element in
                the vocab; "fix" restricts to required_element_set; a dash-separated string or Set[int]
                defines a custom pool. Only used when element-constrained generation is active.
            temperature: Softmax temperature for sampling.
            cond: Optional tensor of shape [n_structures, condition_dim] for AdaLN conditioning,
                in physical units (any condition_transform is applied here, not by the caller).
        """
        generator = WyckoffGenerator(
            self.model, self.cascade_order, self.cascade_is_target, self.token_engineers,
            self.masks_dict, self.max_sequence_length)
            
        condition_feature = self.condition_feature

        if calibrate:
            if self.val_dataset is None:
                raise ValueError("Calibration requires a validation dataset")
            generator.calibrate(self.val_dataset, condition_feature=condition_feature,
                                condition_transform=self.transform_condition)
        if start_tensor is None:
            start_tensor = self._sample_start_tokens_from_distribution(n_structures)
        else:
            if start_tensor.size(0) != n_structures:
                raise ValueError("Custom start tensor must have the same number of samples as requested structures.")
            if hasattr(self, 'train_dataset') and self.train_dataset is not None:
                start_tensor = start_tensor.to(self.device).to(self.train_dataset.start_tokens.dtype)
            else:
                start_tensor = start_tensor.to(self.device).to(torch.int64 if self.model.start_type == "categorial" else torch.float32)

        if condition_feature is not None and cond is None:
            if hasattr(self, 'train_dataset') and self.train_dataset is not None:
                random_indices = torch.randint(
                    0, self.train_dataset.num_examples, (n_structures,), device=self.device)
                cond = self.train_dataset.data[condition_feature][random_indices]
            else:
                raise ValueError(
                    f"condition_feature={condition_feature!r} is set but no `cond` was provided "
                    "and no train_dataset is available to sample from."
                )

        if cond is not None:
            # Both branches above, and anything a caller passes in, are in physical units.
            self._validate_condition_values(cond)
            cond = self.transform_condition(cond)

        if required_element_set is not None:
            if 'elements' not in self.tokenisers:
                raise ValueError("Element vocabulary ('elements') not found in self.tokenisers.")
            generated_tensors = generator.generate_tensors(
                start=start_tensor,
                required_element_set=required_element_set,
                allowed_element_set=allowed_element_set,
                temperature=temperature,
                elements_vocab=self.tokenisers['elements'],
                cond=cond
            )
        elif compute_validity_per_known_sequence_length:
            generated_tensors, ss_validitity, enum_validity = generator.generate_tensors(
                start_tensor, compute_validity=True, cond=cond)
        else:
            generated_tensors = generator.generate_tensors(start_tensor, compute_validity=False, cond=cond)

        generated_cascade_order = self.cascade_order
        if self.cascade_order[-1] == "harmonic_site_symmetries":
            del generated_tensors[-1]
            generated_cascade_order = generated_cascade_order[:-1]
        generated_tensors = torch.stack(generated_tensors, dim=-1)

        if 'sites_enumeration' in self.tokenisers:
            letter_from_ss_enum_idx = self.tokenisers['sites_enumeration'].get_letter_from_ss_enum_idx(self.run_path)
        else:
            letter_from_ss_enum_idx = None
        ss_from_letter = load_wyckoff_mappings(self.run_path).ss_from_letter
        to_pyxtal = partial(self.processor.tensor_to_pyxtal,
                            cascade_order=generated_cascade_order,
                            letter_from_ss_enum_idx=letter_from_ss_enum_idx,
                            ss_from_letter=ss_from_letter,
                            wp_index=get_wp_index())
        structures = list(map(to_pyxtal, start_tensor.detach().cpu(), generated_tensors.detach().cpu()))
        logger.info("Generated %d Wyckoffs", len(structures))
        valid_structures = [s for s in structures if s is not None]
        logger.info("From which %d are valid", len(valid_structures))
        if compute_validity_per_known_sequence_length:
            return valid_structures, ss_validitity, enum_validity
        return valid_structures

    def generate_evaluate_and_log_wp(
        self,
        generation_name: str,
        calibrate: bool,
        n_structures: int,
        evaluator: Optional[StatisticalEvaluator]):

        generated_wp, ss_validitity, enum_validity = self.generate_structures(
            n_structures, calibrate, compute_validity_per_known_sequence_length=True)
        validity_data = [[known_seq_len, ss_validitity, enum_validity] for
            known_seq_len, (ss_validitity, enum_validity) in
            enumerate(zip(ss_validitity, enum_validity))]
        validity_table = wandb.Table(data=validity_data, columns=["known_seq_len", "ss_validity", "enumeration_validity"])
        # Important note. The logged values denote the validity specifially *at* given known sequence length,
        # not the validity of the sequences from the start up to the known sequence length.
        wandb.log({
            "ss_validity": wandb.plot.line(validity_table, "known_seq_len", "ss_validity",
                title="Site Symmetry validity"),
            "enumeration_validity": wandb.plot.line(validity_table, "known_seq_len", "enumeration_validity",
                title="Enumeration validity")
        })
        file_name = self.run_path / f"generated_wp_{generation_name}.json.gz"
        saved_wyckoffs = wandb.Artifact(name=f"generated_wp_{generation_name}_{wandb.run.id}", type="generated_data")
        with gzip.open(file_name, "wt") as f:
            json.dump(generated_wp, f)
        saved_wyckoffs.add_file(file_name)
        wandb.log_artifact(saved_wyckoffs)
        if evaluator is not None:
            evaluate_and_log(generated_wp, generation_name, n_structures, evaluator)

    @torch.no_grad()
    def predict_scalars(
        self,
        prediction_data: Dict[str, torch.Tensor | List[torch.Tensor]],
        augmentation_samples: int = 1) -> Tuple[Tensor, Tensor]:
        """
        Predict scalar targets for pre-tokenised data.

        Args:
            prediction_data: Tokenised cascade data in the same format as AugmentedCascadeDataset expects.
                Must include the start field (``self.start_name``), every field from ``self.cascade_order``,
                engineered fields, and augmented variants if applicable.
            augmentation_samples: Number of random augmentation draws to average over. Each draw samples
                a random augmented variant when available.

        Returns:
            A tuple (mean_predictions, all_predictions) where:
                - mean_predictions is a tensor of shape [num_examples] with the average prediction across
                  augmentation samples.
                - all_predictions is a tensor of shape [augmentation_samples, num_examples] with raw
                  predictions per augmentation sample.
        """
        if self.target != TargetClass.Scalar:
            raise ValueError("predict_scalars is only available for Scalar targets.")
        if augmentation_samples < 1:
            raise ValueError("augmentation_samples must be at least 1.")

        prediction_data = prediction_data.copy()
        dummy_target_name = "__scalar_prediction_dummy__"
        num_examples = prediction_data[self.start_name].shape[0]
        prediction_data[dummy_target_name] = torch.zeros(num_examples, dtype=torch.float32)

        prediction_dataset = AugmentedCascadeDataset(
            data=prediction_data,
            cascade_order=self.cascade_order,
            masks=self.masks_dict,
            pads=self.pad_dict,
            stops=self.stops_dict,
            num_classes=self.num_classes_dict,
            start_field=self.start_name,
            augmented_fields=self.augmented_fields,
            batch_size=None,
            dtype=self.dtype,
            start_dtype=self.train_dataset.start_tokens.dtype,
            device=self.device,
            augmented_storage_device=None,
            target_name=dummy_target_name,
        )

        was_training = self.model.training
        self.model.eval()
        try:
            sample_predictions = []
            for _ in range(augmentation_samples):
                start_tokens, cascade_tokens, _, padding_mask = \
                    prediction_dataset.get_augmented_data() # Defaults to all examples
                preds = self.model(start_tokens, cascade_tokens, padding_mask, None).squeeze()
                sample_predictions.append(preds)
            stacked_predictions = torch.stack(sample_predictions, dim=0)
            mean_predictions = stacked_predictions.mean(dim=0)
        finally:
            if was_training:
                self.model.train()
        return mean_predictions, stacked_predictions


def train_from_config(
    config_dict: dict,
    device: torch.device,
    run_path: Path = Path(__file__).resolve().parent.parent / "runs",
    production_training: bool = False,
    no_test: bool = False):

    if wandb.run is None:
        raise ValueError("W&B run must be initialized")
    this_run_path = run_path / wandb.run.id
    this_run_path.mkdir(parents=True, exist_ok=False)
    trainer = WyckoffTrainer.from_config(config_dict, device, run_path=this_run_path, production_training=production_training, no_test=no_test)
    shutil.copy(
        Path(wyckoff_transformer.__file__).parent / WYCKOFF_MAPPINGS_FILENAME,
        this_run_path / WYCKOFF_MAPPINGS_FILENAME,
    )
    tokenizers_engineers = wandb.Artifact(name=f"processors_{wandb.run.id}", type="processors")
    processor_json = trainer.processor.save_pretrained(this_run_path)
    tokenizers_engineers.add_file(processor_json)
    tokenizers_engineers.add_file(this_run_path / WYCKOFF_MAPPINGS_FILENAME)
    wandb.log_artifact(tokenizers_engineers)
    config_save_path = this_run_path / "config.yaml"
    OmegaConf.save(config_dict, config_save_path)
    run_config_artifact = wandb.Artifact(name=f"run_config_{wandb.run.id}", type="config")
    run_config_artifact.add_file(config_save_path)
    wandb.log_artifact(run_config_artifact)
    trainer.train()
    config = OmegaConf.create(config_dict)
    if config.model.WyckoffTrainer_args.target == "NextToken" and \
        config.evaluation.get("n_structures_to_generate", 0) > 0:

        print("Training complete, loading the best model")
        load_model_weights(
            trainer.model, trainer.run_path / "best_model_params.pt", trainer.device)

        evaluator: Optional[StatisticalEvaluator] = None
        if not no_test:
            data_cache_path = Path(__file__).resolve().parents[2] / "cache" / config.dataset / "data.pkl.gz"
            with gzip.open(data_cache_path, "rb") as f:
                datasets_pd = pickle.load(f)
            datasets_pd.pop("train", None)
            datasets_pd.pop("val", None)
            if "test" not in datasets_pd:
                logger.warning(
                    "Test split not found in data cache at %s; skipping test-set evaluation but still generating structures.",
                    data_cache_path,
                )
            else:
                test_no_sites = datasets_pd['test']['site_symmetries'].map(len).values
                num_sites_bins = np.arange(0, 21)
                test_no_sites_hist = np.histogram(test_no_sites, bins=num_sites_bins)
                wandb.run.summary["num_sites"] = {"test": {"hist": wandb.Histogram(np_histogram=test_no_sites_hist)}}

                print(f"Test dataset size: {len(datasets_pd['test']['site_symmetries'])}")
                wandb.run.summary["test_dataset_size"] = len(datasets_pd["test"]["site_symmetries"])
                evaluator = StatisticalEvaluator(datasets_pd["test"])
                test_smact_validity = datasets_pd["test"]["composition"].map(smac_validity_from_counter).mean()
                print(f"SMAC-T validity on the test dataset: {test_smact_validity}")
                wandb.run.summary["smact_validity"] = {"test": test_smact_validity}
        wandb.run.summary["formal_validity"] = {}
        wandb.run.summary["wp"] = {}
        print("No calibration:")
        trainer.generate_evaluate_and_log_wp(
            "no_calibration", calibrate=False, n_structures=config.evaluation.n_structures_to_generate,
            evaluator=evaluator)
        print("Temperature calibration:")
        trainer.generate_evaluate_and_log_wp(
            "temperature_calibration", calibrate=True, n_structures=config.evaluation.n_structures_to_generate,
            evaluator=evaluator)
