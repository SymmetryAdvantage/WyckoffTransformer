from typing import Tuple, Dict, Optional, List, Any, Union, Set
import importlib
import math
import os
import random
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
from wandb.sdk.data_types._private import MEDIA_TMP


import wyckoff_transformer
from wyckoff_transformer.cascade.dataset import AugmentedCascadeDataset, AugmentedCascadeLoader, TargetClass
from wyckoff_transformer.cascade.model import CascadeTransformer
from wyckoff_transformer.censored import CensoredMinDiagnostics, CensoredMinLoss
from wyckoff_transformer.composition import (
    COMPOSITION_FIELD,
    attach_composition_vector,
    composition_conditioning_dim,
)
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


def normalise_condition_features(condition_feature) -> Tuple[str, ...]:
    """One name, a list of names, or nothing, as the tuple the trainer works with.

    A config may write `condition_feature: energy_above_hull` or
    `condition_feature: [energy_above_hull, delta_e_polymorph, max_force]`; both reach
    here. The tuple's ORDER is what every AdaLN weight is tied to, so it is preserved
    exactly as written and duplicates are refused -- a repeated name would give one
    feature two columns and shift every column after it.
    """
    if condition_feature is None:
        return ()
    if isinstance(condition_feature, str):
        return (condition_feature,)
    features = tuple(str(name) for name in condition_feature)
    duplicates = sorted({name for name in features if features.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"condition_feature repeats {duplicates}; each feature owns exactly one column")
    return features


def normalise_condition_scales(condition_scale, features: Tuple[str, ...]) -> Tuple[float, ...]:
    """A divisor per conditioning feature, applied before its transform.

    The conditioning channels are not commensurable. Energy above hull and the polymorph
    gap live on the same eV/atom scale as each other, but `max_force` is a hundred times
    smaller: its median over this archive is 0.0035 eV/A against 0.23 eV/atom for e_hull,
    so `log1p` leaves it a near-constant input and the AdaLN modulation has to make up
    the difference in the weight. Dividing it by 0.01 -- reading it in centi-eV/A -- puts
    the three channels within a factor of two of each other before the transform sees
    them, without introducing a fitted statistic that would then have to be persisted
    alongside the weights.

    Accepts one number for every feature, a list in `features` order, or a mapping keyed
    by feature name. The stored data and every value crossing the API stay in physical
    units; this is applied on the way into the model, exactly like the transform.
    """
    if condition_scale is None:
        scales = (1.0,) * len(features)
    elif isinstance(condition_scale, (int, float)):
        scales = (float(condition_scale),) * len(features)
    elif isinstance(condition_scale, (dict, DictConfig)):
        unknown = sorted(set(condition_scale) - set(features))
        if unknown:
            raise ValueError(
                f"condition_scale names {unknown}, which are not conditioning features "
                f"{list(features)}")
        scales = tuple(float(condition_scale.get(name, 1.0)) for name in features)
    else:
        scales = tuple(float(value) for value in condition_scale)
        if len(scales) != len(features):
            raise ValueError(
                f"condition_scale has {len(scales)} entries for {len(features)} conditioning "
                f"features {list(features)}; give one per feature, one number for all of "
                "them, or a mapping keyed by feature name")
    bad = [name for name, scale in zip(features, scales) if not scale > 0]
    if bad:
        raise ValueError(f"condition_scale must be positive; {bad} are not")
    return scales


def normalise_condition_transforms(
    condition_transform, features: Tuple[str, ...]) -> Tuple[Optional[str], ...]:
    """A transform name per conditioning feature.

    Accepts one name applied to every feature (the historical form, and the right one
    when the features share a support), a list in `features` order, or a mapping from
    feature name to transform for the case where they do not -- `log1p` is only defined
    on non-negative values, so a formation energy and an energy above hull cannot share
    it.
    """
    if condition_transform is None or isinstance(condition_transform, str):
        return (condition_transform,) * len(features)
    if isinstance(condition_transform, (dict, DictConfig)):
        unknown = sorted(set(condition_transform) - set(features))
        if unknown:
            raise ValueError(
                f"condition_transform names {unknown}, which are not conditioning features "
                f"{list(features)}")
        return tuple(condition_transform.get(name) for name in features)
    transforms = tuple(condition_transform)
    if len(transforms) != len(features):
        raise ValueError(
            f"condition_transform has {len(transforms)} entries for {len(features)} "
            f"conditioning features {list(features)}; give one per feature, one name for "
            "all of them, or a mapping keyed by feature name")
    return transforms


#: Prefix `torch.compile` adds to every parameter name of the module it wraps.
_COMPILE_PREFIX = "_orig_mod."

#: Everything `train()` needs to pick a crashed run up where it stopped: weights, optimiser
#: and schedule state, the RNG, the loaders' shuffle position and the early-stopping
#: bookkeeping. Distinct from `best_model_params.pt`, which holds only the weights that
#: scored best and is what generation and evaluation load.
CHECKPOINT_FILENAME = "last_checkpoint.pt"

#: Bumped when the checkpoint layout changes in a way that makes older files unreadable.
#: A resume that finds an older version fails loudly rather than restoring half a run.
CHECKPOINT_FORMAT_VERSION = 1


def ensure_wandb_media_directory() -> None:
    """Restore W&B's media staging directory if a long-running job lost it from /tmp."""
    Path(MEDIA_TMP.name).mkdir(parents=True, exist_ok=True)


def atomic_torch_save(obj: Any, path: Path) -> None:
    """`torch.save` to a temporary file in the same directory, then rename over the target.

    Saving straight to the destination leaves a truncated file behind if the process dies
    mid-write -- which is exactly the event a checkpoint exists to survive, and it would
    otherwise destroy the previous, good checkpoint on its way out. `os.replace` is atomic
    within a filesystem, so the destination is always one whole checkpoint or the other.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _match_compile_prefix(state_dict: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
    """Rename the keys of `state_dict` to the `_orig_mod.` convention `model` expects."""
    weights = {key.removeprefix(_COMPILE_PREFIX): value for key, value in state_dict.items()}
    if any(key.startswith(_COMPILE_PREFIX) for key in model.state_dict()):
        weights = {_COMPILE_PREFIX + key: value for key, value in weights.items()}
    return weights


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
    model.load_state_dict(_match_compile_prefix(state_dict, model))


def cascade_target_indices(
    cascade_order: Tuple[str, ...],
    cascade_is_target: Dict[str, bool],
    token_engineers: Dict) -> Tuple[int, ...]:
    """Positions of the cascade fields that have a prediction head, validating the rest.

    A non-target field is filled in by its engineer rather than predicted, so it need not
    be last -- and it must not be, when a later target has to see it.
    ``get_masked_multiclass_cascade_data`` masks every field from the predicted one onwards
    at the current position, so a non-target field placed after a head that needs it is
    invisible exactly where it matters. What is required instead is that its engineer's
    inputs are all decided before it: the sequence-level first input (the start token),
    plus cascade fields sitting earlier in the order.

    Returns positions rather than a count because the targets need not be a prefix.
    """
    is_target_in_order = [cascade_is_target[field] for field in cascade_order]
    if not any(is_target_in_order):
        raise ValueError("At least one cascade field must be a prediction target")
    for position, (field, is_target) in enumerate(zip(cascade_order, is_target_in_order)):
        if is_target:
            continue
        engineer = token_engineers.get(field)
        if engineer is None:
            raise ValueError(
                f"Cascade field {field} is not a target and has no engineer to fill it in, "
                "so nothing would ever set its value")
        for engineer_input in engineer.inputs[1:]:
            if engineer_input in cascade_order and \
                    list(cascade_order).index(engineer_input) > position:
                raise ValueError(
                    f"Non-target cascade field {field} at position {position} is filled "
                    f"from {engineer_input}, which comes later in the cascade; move it "
                    "after that field")
    return tuple(
        position for position, is_target in enumerate(is_target_in_order) if is_target)


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

    #: Which likelihood a Scalar target is fitted with, "mse" or "censored". A class
    #: attribute for the same reason as `scheduler_steps_per_batch`: the Scalar paths
    #: branch on it, and a trainer built by __new__ should read as the default.
    scalar_loss = "mse"

    #: Set to a CensoredMinDiagnostics when scalar_loss is "censored", and None otherwise.
    censored_diagnostics = None

    #: Whether the target composition is part of the conditioning vector, and the width
    #: of the element vocabulary it is expressed over. Class attributes for the same
    #: reason as `scheduler_steps_per_batch`: every conditioning path reads them.
    composition_conditioning = False
    condition_on_cell_size = True
    n_elements = None

    #: Set by `--resume`: train() continues from `last_checkpoint.pt` instead of epoch 0.
    resume = False

    #: Set by `--reschedule`: a resumed run is allowed to carry a different learning-rate
    #: horizon than the checkpoint was written under. A class default for the same reason as
    #: `resume`, and False so the horizon guards below hold unless something asks for them not to.
    reschedule = False

    #: How often, in epochs, train() writes that checkpoint. Class defaults for the same
    #: reason as the two above: trainers built by tests via __new__ skip __init__.
    checkpoint_period = 1

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
        condition_scale: Optional[float] = None,
        scalar_loss: str = "mse",
        censored_loss_args: Optional[dict] = None,
        composition_conditioning: bool = False,
        condition_on_cell_size: bool = True,
        resume: bool = False,
        reschedule: bool = False,
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
            condition_feature: Name of the feature to condition on via AdaLN, or a list of
                names for several. Each occupies one column of the conditioning vector, in
                the order given; that order is what the learned AdaLN weights are tied to,
                so reordering the list repoints every one of them.
            condition_transform: Name of a transform from CONDITION_TRANSFORMS applied to the
                conditioning features on their way into the model. One name applies to every
                feature; a list gives one per feature in `condition_feature` order; a mapping
                keyed by feature name does the same for the case where only some need one.
                The stored data and the values accepted by generate_structures stay in
                physical units.
            condition_scale: Divisor applied to each conditioning feature before its
                transform, in the same one-for-all / list / mapping forms as
                condition_transform. Defaults to 1. It exists to put channels of very
                different magnitude on a comparable footing -- see
                normalise_condition_scales.
            scalar_loss: Which likelihood a Scalar target is fitted with. "mse" regresses the
                conditional mean, which for a Wyckoff gene is E[E | gene]. "censored" reads
                every label as an upper bound and regresses min(E | gene) instead -- what CSP
                wants, since the gene fixes neither the coordinates nor the cell and the
                downstream reconstruction gets many attempts at the manifold it does fix.
                See wyckoff_transformer.censored. Requires CascadeTransformer_args.outputs=2
                unless censored_loss_args sets predict_scale=False.
            censored_loss_args: Keyword arguments for CensoredMinLoss when scalar_loss is
                "censored": noise, min_scale, predict_scale, init_scale.
            composition_conditioning: Condition on the target chemical formula, as a vector
                over the element vocabulary appended to whatever `condition_feature`
                supplies. Requires the tokeniser to emit the composition counters
                ('counters: {composition: elements}' under sequence_fields) and
                CascadeTransformer_args.condition_dim to equal `self.condition_dim`, which
                this constructor checks. See wyckoff_transformer.composition.
            condition_on_cell_size: Also feed the model log1p of the cell's atom count,
                as one more column of that vector. An input, not a prediction: the model
                is *told* the size, so every sampling call has to commit to one. Leave it
                on for de novo generation; turn it off for a model meant for CSP, where
                the cell size is what the model is being asked to choose and telling it
                the answer forces the decoder to work one z at a time.
                See docs/csp_mode.md.
            resume: Continue an interrupted run: train() restores weights, optimiser, schedule,
                RNG and loader position from `last_checkpoint.pt` in `run_path` and starts at
                the epoch after the one the checkpoint recorded. Mutually exclusive with
                `weights_path`, which starts a *new* run from someone else's weights.
        """
        if isinstance(target, str):
            target = TargetClass[target]
        if target != TargetClass.Scalar:
            self.cascade_target_indices = cascade_target_indices(
                cascade_order, cascade_is_target, token_engineers)
            self.cascade_target_count = len(self.cascade_target_indices)
        else:
            self.cascade_target_indices = ()
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
        # Order matters only in that both setters re-derive the per-feature transforms, so
        # whichever is assigned second sees a consistent pair.
        self.condition_feature = condition_feature
        self.condition_scale = condition_scale
        self.condition_transform = condition_transform
        extra_fields = list(self.condition_features) or None
        self.scalar_loss = scalar_loss
        self.censored_diagnostics = None
        self.composition_conditioning = composition_conditioning
        self.condition_on_cell_size = condition_on_cell_size
        self.n_elements = len(tokenisers["elements"]) if "elements" in tokenisers else None
        if composition_conditioning:
            if self.n_elements is None:
                raise ValueError(
                    "composition_conditioning needs an 'elements' tokeniser to size the vector")
            # Densify before the datasets are built: they keep only cascade_order and
            # extra_fields, and the ragged counters the tokeniser stores are neither.
            for raw in (train_dataset, val_dataset, test_dataset):
                if raw is not None:
                    attach_composition_vector(
                        raw, self.n_elements, condition_on_cell_size=condition_on_cell_size)
            extra_fields = (extra_fields or []) + [COMPOSITION_FIELD]

        if target == TargetClass.NextToken:
            # Sequences have difference lengths, so we need to make sure that
            # long sequences don't dominate the loss, so we don't average the loss
            self.criterion = nn.CrossEntropyLoss(reduction="sum")
        elif target == TargetClass.NumUniqueTokens:
            if not multiclass_next_token_with_order_permutation:
                raise NotImplementedError("NumUniqueTokens is not implemented without permutations")
            self.criterion = nn.MSELoss(reduction="none")
        elif target == TargetClass.Scalar:
            if scalar_loss == "mse":
                # Assumes the batch size is the same for all batches
                self.criterion = nn.MSELoss(reduction='mean')
                self.testing_criterion = nn.L1Loss(reduction='mean')
            elif scalar_loss == "censored":
                # Regressing min(E | gene) rather than E[E | gene]: every label is
                # an upper bound on the target, so the loss reads it as one. See
                # wyckoff_transformer.censored.
                self.criterion = CensoredMinLoss(
                    reduction='mean', **(censored_loss_args or {})).to(device)
                # Reported instead of an MAE, which is not the objective and is
                # floored by the mean excess. Selects checkpoints on the NLL.
                self.testing_criterion = self.criterion
                self.censored_diagnostics = CensoredMinDiagnostics(self.criterion)
            else:
                raise ValueError(f"Unknown scalar_loss: {scalar_loss}")
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
            list(self.trainable_parameters()), **optimisation_config.optimiser.config)
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

        # Pre-cast every conditioning feature to (num_examples, 1) float32 on self.device,
        # so the loss path can index without a per-step .to(device).to(float32).
        for name, transform in zip(self.condition_features, self.condition_transforms):
            for ds in (self.train_dataset, self.val_dataset, self.test_dataset):
                if ds is None:
                    continue
                cond_tensor = ds.data[name].to(self.device, dtype=torch.float32)
                if cond_tensor.dim() == 1:
                    cond_tensor = cond_tensor.unsqueeze(1)
                if cond_tensor.dim() != 2 or cond_tensor.shape[-1] != 1:
                    raise ValueError(
                        f"Conditioning feature {name!r} is {tuple(cond_tensor.shape)}; a "
                        "conditioning feature is one scalar per structure.")
                # `no_processing` copies a dataframe column into a tensor untouched, so a
                # feature that was NaN for some rows arrives NaN here and would otherwise
                # poison every AdaLN modulation it touches without ever raising.
                if bool(torch.isnan(cond_tensor).any()):
                    raise ValueError(
                        f"Conditioning feature {name!r} has NaN values. Drop or impute them "
                        "when the dataset is built; nothing downstream masks them.")
                ds.data[name] = cond_tensor
                # Stored in physical units; the transform is applied on the way into the model.
                # Validate once here rather than per step, which would force a device sync.
                self._validate_condition_column(name, transform, cond_tensor)
        if self.composition_conditioning:
            for ds in (self.train_dataset, self.val_dataset, self.test_dataset):
                if ds is None:
                    continue
                ds.data[COMPOSITION_FIELD] = ds.data[COMPOSITION_FIELD].to(
                    self.device, dtype=torch.float32)
        # Unconditionally, not only for composition_conditioning: a config declaring a
        # condition_dim wider than what build_cond produces used to pass every check here
        # and fail later inside nn.Linear with a bare shape mismatch.
        declared = getattr(self.model, "condition_dim", None)
        if declared != self.condition_dim:
            raise ValueError(
                f"The model was built with condition_dim={declared}, but this run's "
                f"conditioning is {self.condition_dim} wide: "
                f"{self.describe_condition_layout()}. Set "
                f"CascadeTransformer_args.condition_dim to {self.condition_dim}.")
    
        # Optional: omit, or set to null, to train without a norm constraint. The pre-clip norm
        # is logged either way, so a run can watch the gradient scale without being shaped by it.
        self.clip_grad_norm = optimisation_config.get("clip_grad_norm", None)
        self.cascade_len = len(cascade_order)
        self.cascade_order = cascade_order
        self.epochs = optimisation_config.epochs
        self.resume = resume
        self.reschedule = reschedule
        if resume and weights_path is not None:
            # Both write the model's weights, and the checkpoint also carries the optimiser
            # state that goes with them. Applying weights_path afterwards would leave an
            # optimiser whose moments belong to different weights.
            raise ValueError(
                "resume and weights_path are mutually exclusive: a resumed run takes its "
                "weights from its own checkpoint.")
        if weights_path is not None:
            load_model_weights(self.model, weights_path, device)

        self.validation_period = optimisation_config.validation_period
        # Validation epochs are the natural cadence -- the loop already pauses there, and the
        # early-stopping bookkeeping the checkpoint carries only moves then. A run with a long
        # validation_period can buy a tighter one: the cost is a file write, the benefit is
        # that a crash loses less. A period longer than the run itself is not an error: the
        # checkpoint train() writes when the loop ends is unconditional.
        self.checkpoint_period = int(
            optimisation_config.get("checkpoint_period", self.validation_period))
        if self.checkpoint_period < 1:
            raise ValueError(
                f"checkpoint_period must be at least 1 epoch, got {self.checkpoint_period}")
        self.early_stopping_patience_epochs = optimisation_config.early_stopping_patience_epochs
        self.target = target
        self.target_name = target_name
        self.multiclass_next_token_with_order_permutation = multiclass_next_token_with_order_permutation
        self.evaluation_samples = evaluation_samples
        self.start_token_distribution = start_token_distribution


    def trainable_parameters(self):
        """Every parameter the optimiser should move.

        Almost always just the model's. `CensoredMinLoss` with `predict_scale=False`
        holds the excess scale as a parameter of the criterion, and it has to be
        optimised alongside the weights or the likelihood is fitted at a fixed scale
        nobody chose.
        """
        yield from self.model.parameters()
        if isinstance(self.criterion, nn.Module):
            yield from self.criterion.parameters()


    @property
    def condition_feature(self):
        """What the config asked for: None, one name, or the tuple of names.

        `condition_features` is the canonical form and is what the rest of the class
        reads. This stays a single string for a single-feature run so that the CLIs,
        the diagnostics scripts and the saved configs keep round-tripping unchanged.
        """
        if not self._condition_features:
            return None
        if len(self._condition_features) == 1:
            return self._condition_features[0]
        return self._condition_features


    @condition_feature.setter
    def condition_feature(self, value):
        self._condition_features = normalise_condition_features(value)
        self._rebuild_condition_transforms()


    @property
    def condition_features(self) -> Tuple[str, ...]:
        """The conditioning features in column order; empty when unconditional."""
        return self._condition_features


    @property
    def condition_transform(self):
        """The transform spec as configured: a name, a list, a mapping, or None."""
        return self._condition_transform


    @condition_transform.setter
    def condition_transform(self, value):
        self._condition_transform = value
        self._rebuild_condition_transforms()


    @property
    def condition_scale(self):
        """The scale spec as configured: a number, a list, a mapping, or None."""
        return self._condition_scale


    @condition_scale.setter
    def condition_scale(self, value):
        self._condition_scale = value
        self._rebuild_condition_transforms()


    def _rebuild_condition_transforms(self):
        """Re-derive the per-feature transform names, functions and scales.

        Every setter calls this, so they stay consistent no matter which is assigned
        first -- the test skeletons around this class assign them in several orders.
        """
        features = getattr(self, "_condition_features", ())
        spec = getattr(self, "_condition_transform", None)
        self.condition_transforms = normalise_condition_transforms(spec, features)
        self._condition_transform_fns = tuple(
            map(get_condition_transform, self.condition_transforms))
        self.condition_scales = normalise_condition_scales(
            getattr(self, "_condition_scale", None), features)
        self._condition_is_identity = (
            all(fn is None for fn in self._condition_transform_fns)
            and all(scale == 1.0 for scale in self.condition_scales))
        # The training loop transforms the whole block in one call whenever every column
        # is treated identically, which is the usual case; the per-column path below is
        # for mixed transforms and mixed scales.
        self._condition_is_uniform = (
            len(set(map(id, self._condition_transform_fns))) == 1
            and len(set(self.condition_scales)) == 1)


    def describe_condition_layout(self) -> str:
        """The conditioning vector's columns, in order, for an error message."""
        parts = list(self.condition_features)
        if self.composition_conditioning:
            parts.append(
                f"{composition_conditioning_dim(self.n_elements, self.condition_on_cell_size)} "
                f"columns for the composition over {self.n_elements} element tokens")
        return ", ".join(parts) if parts else "nothing"


    @staticmethod
    def _validate_condition_column(name: str, transform: Optional[str], column: Tensor):
        """Reject values one transform cannot represent, with a legible message."""
        if transform == "log1p" and bool((column < 0).any()):
            raise ValueError(
                f"condition_transform='log1p' requires {name} >= 0, "
                "but negative values were supplied")


    def _validate_condition_values(self, values: Optional[Tensor]):
        """Reject a scalar conditioning block the transforms cannot represent."""
        if values is None or not self.condition_features:
            return
        block = values if values.dim() > 1 else values.unsqueeze(-1)
        if block.shape[-1] != len(self.condition_features):
            raise ValueError(
                f"The conditioning values are {block.shape[-1]} wide, but this run "
                f"conditions on {len(self.condition_features)} features "
                f"{list(self.condition_features)}. Give one column per feature, in that "
                "order.")
        for index, (name, transform) in enumerate(
                zip(self.condition_features, self.condition_transforms)):
            self._validate_condition_column(name, transform, block[..., index])


    def transform_condition(self, values: Optional[Tensor]) -> Optional[Tensor]:
        """Map the scalar conditioning block from physical units to what the model consumes.

        `values` carries one column per entry of `condition_features`, in that order.
        When every feature shares a transform -- the usual case, and the only one the
        training loop hits -- it is applied to the whole block in one call rather than
        column by column.
        """
        fns = self._condition_transform_fns
        if values is None or not fns or self._condition_is_identity:
            return values
        if self._condition_is_uniform:
            scaled = values if self.condition_scales[0] == 1.0 else values / self.condition_scales[0]
            return scaled if fns[0] is None else fns[0](scaled)
        block = values if values.dim() > 1 else values.unsqueeze(-1)
        if block.shape[-1] != len(fns):
            raise ValueError(
                f"The conditioning values are {block.shape[-1]} wide, but this run "
                f"conditions on {len(fns)} features {list(self.condition_features)}.")
        columns = []
        for fn, scale, column in zip(fns, self.condition_scales, block.split(1, dim=-1)):
            if scale != 1.0:
                column = column / scale
            columns.append(column if fn is None else fn(column))
        return torch.cat(columns, dim=-1)


    @property
    def condition_dim(self) -> Optional[int]:
        """Width of the vector this run feeds to AdaLN, or None when unconditional.

        Each `condition_feature` occupies one column, in the order they are configured,
        and the composition the rest. `CascadeTransformer_args.condition_dim` has to
        agree.
        """
        width = len(self.condition_features)
        if self.composition_conditioning:
            width += composition_conditioning_dim(
                self.n_elements, self.condition_on_cell_size)
        return width or None


    def build_cond(self, dataset: AugmentedCascadeDataset,
                   batch_selection: 'Tensor | slice' = slice(None)) -> Optional[Tensor]:
        """Assemble the conditioning vector for a batch, in the model's units.

        One place, because the scalars are stored in physical units and transformed on
        the way in while the composition is stored ready to use, and getting that
        order wrong in one of three call sites would be invisible until the
        conditioning quietly stopped meaning anything.
        """
        parts = []
        if self.condition_features:
            columns = [dataset.data[name][batch_selection] for name in self.condition_features]
            block = columns[0] if len(columns) == 1 else torch.cat(columns, dim=-1)
            parts.append(self.transform_condition(block))
        if self.composition_conditioning:
            parts.append(dataset.data[COMPOSITION_FIELD][batch_selection])
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


    def build_condition_from_values(
        self,
        values: 'float | Dict[str, float]',
        n_rows: int,
        device: Optional[torch.device] = None,
    ) -> Optional[Tensor]:
        """A [n_rows, len(condition_features)] block in physical units, one value per feature.

        Everything that generates at a fixed target -- both CLIs and the diagnostics
        scripts -- goes through here instead of `torch.full((n, condition_dim), x)`,
        which writes the same number into every column and so silently mis-conditions
        any model with more than one channel.

        `values` may be a single number only when the run conditions on exactly one
        feature; otherwise it is a mapping from feature name to value. The composition
        block is not built here: it depends on the target formula, which this does not
        know.
        """
        if not self.condition_features:
            raise ValueError("This model has no scalar conditioning features.")
        if isinstance(values, (int, float)):
            if len(self.condition_features) != 1:
                raise ValueError(
                    f"This model conditions on {list(self.condition_features)}; a single "
                    "value is ambiguous. Give one value per feature, by name.")
            values = {self.condition_features[0]: float(values)}
        missing = [name for name in self.condition_features if name not in values]
        unknown = sorted(set(values) - set(self.condition_features))
        if missing or unknown:
            raise ValueError(
                f"Conditioning values must name every feature exactly once. This model "
                f"conditions on {list(self.condition_features)}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unknown {unknown}" if unknown else "") + ".")
        row = torch.tensor(
            [float(values[name]) for name in self.condition_features],
            dtype=torch.float32, device=device if device is not None else self.device)
        return row.unsqueeze(0).expand(n_rows, -1).contiguous()


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
                    no_test: bool = False,
                    resume: bool = False,
                    reschedule: bool = False):
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
        # The conditioning width is derived data, not a design choice: it follows the list of
        # conditioning features and, when the composition is one of them, the element
        # vocabulary of whichever dataset the run uses. Fill it in rather than making every
        # config hardcode a number that silently rots when either changes.
        trainer_args = config.model.WyckoffTrainer_args
        condition_features = normalise_condition_features(trainer_args.get("condition_feature"))
        derived = len(condition_features)
        if trainer_args.get("composition_conditioning", False):
            derived += composition_conditioning_dim(
                len(tokenisers["elements"]),
                trainer_args.get("condition_on_cell_size", True))
        declared = config.model.CascadeTransformer_args.get("condition_dim")
        if derived:
            if declared is None:
                logger.info("Setting condition_dim to %d from the conditioning features", derived)
                config.model.CascadeTransformer_args.condition_dim = derived
            elif declared != derived:
                raise ValueError(
                    f"condition_dim is {declared} in the config, but conditioning on "
                    f"{list(condition_features)}"
                    + (" plus the composition" if trainer_args.get("composition_conditioning", False)
                       else "")
                    + f" makes it {derived}. Remove it and let it be derived, or fix it.")
        elif declared is not None:
            raise ValueError(
                f"condition_dim is {declared} in the config, but nothing conditions this "
                "model: set condition_feature and/or composition_conditioning, or remove "
                "condition_dim.")
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
        cascade_is_target = config.model.cascade.get("is_target")
        if cascade_is_target is None:
            cascade_is_target = {field: False for field in config.model.cascade.order}

        return cls(
            model, train_data, val_data, tokenisers, token_engineers, config.model.cascade.order,
            cascade_is_target,
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
            resume=resume,
            reschedule=reschedule,
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

        cond = self.build_cond(dataset, batch_selection)

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
            prediction = self.model(start_tokens, masked_data, padding_mask, None, cond=cond)
            if self.scalar_loss == "censored":
                # [batch, n_outputs]; the criterion splits the columns itself, and a
                # bare squeeze() would fuse them for a batch of one.
                prediction = prediction.reshape(-1, self.criterion.n_outputs)
            else:
                prediction = prediction.squeeze()
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
                    self.cascade_target_indices[randint(0, self.cascade_target_count - 1)]
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
                # the loss for each cascade field separately -- and so we can drop the
                # non-target columns here, which are filled in by an engineer rather than
                # predicted and so must not train a head. NextToken needs no such filter:
                # it computes one head per step, and train_epoch only ever draws a target.
                loss = loss[:, list(self.cascade_target_indices)].mean()
            loss.backward()
            # Measure the norm whether or not it is constrained: an infinite max_norm makes
            # clip_grad_norm_ a no-op that still returns the pre-clip norm. Worth logging even
            # when clipping is on -- a threshold that binds on every step is not catching
            # outliers, it is setting the step size, and only this metric shows the difference.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.trainable_parameters(),
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


    @property
    def cascade_target_order(self) -> Tuple[str, ...]:
        """The names of the cascade fields that have a head, in cascade order.

        The labels for a per-field loss vector. Derived rather than stored, so that it cannot
        drift from `cascade_order`, which generation rebinds.
        """
        return tuple(self.cascade_order[i] for i in self.cascade_target_indices)


    @torch.no_grad()
    def evaluate(self, dataset: AugmentedCascadeDataset, loader: Optional[AugmentedCascadeLoader] = None) -> Tensor:
        """
        Evaluates the model by calculating the average loss on the dataset.
        Args:
            dataset: The dataset to evaluate on.
            loader: The loader to use for batching.
        Returns:
            The average loss on the dataset: one entry per *target* cascade field, in the
            order of `self.cascade_target_order`, not one per cascade field. A non-target
            field is filled in by its engineer rather than predicted, so it has no head and
            no loss; it used to occupy a permanently-zero slot here, which reached wandb as
            `loss.epoch.<split>.<field> = 0` and read like a collapsed head.
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

        loss = torch.zeros(self.cascade_target_count, device=self.device)

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
                    # `loss` is indexed by target rank, `known_cascade_len` by cascade position:
                    # the two differ as soon as a non-target field sits among the targets.
                    for target_rank, known_cascade_len in enumerate(self.cascade_target_indices):
                        # get_loss already rescales a sampled batch to the whole viable set.
                        loss[target_rank] += self.get_loss(
                            dataset, known_seq_len, known_cascade_len, loader=loader,
                            no_batch=loader is None)
                else: # NumUniqueTokens
                    # One head per cascade field is predicted in a single pass here, so unlike
                    # NextToken the non-target columns are computed; drop them rather than
                    # carry them, matching what train_epoch backpropagates.
                    if loader is None:
                        loss += self.get_loss(
                            dataset, known_seq_len, 0, no_batch=True
                        )[:, list(self.cascade_target_indices)].sum(dim=0)
                    else:
                        batch_loss, n_samples = self.get_loss(
                            dataset, known_seq_len, 0, loader=loader, no_batch=False,
                            return_n_samples=True)
                        loss += batch_loss[:, list(self.cascade_target_indices)].sum(dim=0) * (
                            dataset.viable_count(known_seq_len) / n_samples)
            # ln(P) = ln p(t_n|t_n-1, ..., t_1) + ... + ln p(t_2|t_1)
            # We are minimising the negative log likelihood of the whole sequences
        return loss / self.evaluation_samples / len(dataset)


    @torch.no_grad()
    def scalar_diagnostics(
        self,
        dataset: AugmentedCascadeDataset,
        loader: AugmentedCascadeLoader) -> Dict[str, float]:
        """Calibration diagnostics for a censored Scalar fit, averaged over one pass.

        Returns an empty dict for every other configuration. See
        `wyckoff_transformer.censored.CensoredMinDiagnostics` for what the keys mean.
        """
        if self.censored_diagnostics is None:
            return {}
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()
        totals: Dict[str, Tensor] = {}
        for _ in range(loader.batches_per_epoch):
            batch_selection = loader.get_next_batch()
            start_tokens, masked_data, target, padding_mask = dataset.get_augmented_data(
                batch_selection=batch_selection)
            cond = self.build_cond(dataset, batch_selection)
            prediction = self.model(start_tokens, masked_data, padding_mask, None, cond=cond)
            prediction = prediction.reshape(-1, self.criterion.n_outputs)
            for key, value in self.censored_diagnostics(prediction, target).items():
                totals[key] = totals.get(key, 0.) + value
        return {key: (value / loader.batches_per_epoch).item() for key, value in totals.items()}


    @property
    def checkpoint_path(self) -> Path:
        """Where train() writes the resume checkpoint for this run."""
        if self.run_path is None:
            raise ValueError("Checkpointing requires a run_path")
        return self.run_path / CHECKPOINT_FILENAME

    def _loaders(self) -> Dict[str, AugmentedCascadeLoader]:
        """The loaders whose shuffle position is part of the run's state, by split name."""
        candidates = {"train": self.train_loader, "val": self.val_loader, "test": self.test_loader}
        return {name: loader for name, loader in candidates.items() if loader is not None}

    def save_training_checkpoint(
            self, epoch: int, best_val_loss: float, best_val_epoch: int) -> Path:
        """Write everything needed to re-enter the training loop at `epoch`.

        Weights and optimiser state are captured in the same instant and must stay that way.
        It matters most for the schedule-free optimisers: what sits in the parameters is `y`
        or `x` depending on whether `train()` or `eval()` was called last, `z` lives in the
        optimiser state, and the flag that says which is in the optimiser's param_groups.
        Saving the pair together makes the triple self-consistent whenever it was taken;
        saving them apart would silently resume from a mixture of two iterates.

        The RNG snapshot covers `random` and torch, the two the training loop draws from --
        `known_seq_len`, the augmentation choice and the batch draw. numpy is deliberately
        absent: nothing in the loop uses it, and its state does not survive
        `torch.load(weights_only=True)`, which is worth keeping.

        Args:
            epoch: The epoch a resume should start at, i.e. one past the last one completed.
            best_val_loss: Best total validation loss seen so far.
            best_val_epoch: Epoch that achieved it, which is what early stopping counts from.

        Returns:
            The path written.
        """
        checkpoint = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_epoch": int(best_val_epoch),
            "model": self.model.state_dict(),
            # Empty for every criterion except a CensoredMinLoss holding a global scale.
            "criterion": (self.criterion.state_dict()
                          if isinstance(self.criterion, nn.Module) else {}),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            # Checked on load: a schedule resumed against a different horizon is a different
            # schedule, and the mismatch is otherwise invisible until the decay misses.
            "scheduler_total_steps": self.scheduler_total_steps,
            "epochs": self.epochs,
            "loaders": {name: loader.state_dict() for name, loader in self._loaders().items()},
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                # Gated on the run's own device rather than on a GPU being present: reading
                # the CUDA RNG initialises a context on the default device, which a CPU run
                # has no business doing and which fails outright when someone else's job is
                # already filling that card.
                "cuda": (torch.cuda.get_rng_state_all()
                         if self.device.type == "cuda" else []),
            },
            # A plain string: the checkpoint is read back with weights_only=True, which
            # accepts primitives and tensors and nothing else.
            "wandb_run_id": None if wandb.run is None else str(wandb.run.id),
        }
        atomic_torch_save(checkpoint, self.checkpoint_path)
        logger.info("Wrote the resume checkpoint for epoch %d to %s", epoch, self.checkpoint_path)
        return self.checkpoint_path

    def load_training_checkpoint(self) -> Dict[str, Any]:
        """Restore the state saved by `save_training_checkpoint` into this trainer.

        Everything the training loop owns -- weights, optimiser, schedule, RNG, loader
        positions -- is restored in place. The epoch counters are returned instead, because
        they are locals of `train()`.

        Returns:
            The `epoch`, `best_val_loss` and `best_val_epoch` to resume the loop with.
        """
        path = self.checkpoint_path
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        version = checkpoint.get("format_version")
        if version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"{path} is a format {version} checkpoint; this version of the trainer writes "
                f"and reads format {CHECKPOINT_FORMAT_VERSION}. It cannot be resumed.")
        if checkpoint["epochs"] != self.epochs:
            message = (
                f"{path} was written by a {checkpoint['epochs']}-epoch run, but this one is "
                f"configured for {self.epochs}. Resuming would place the schedule and the "
                f"early-stopping budget on a horizon neither run has.")
            if not self.reschedule:
                raise ValueError(message)
            logger.warning("RESCHEDULING: %s Continuing anyway, as asked.", message)
        if checkpoint["scheduler_total_steps"] != self.scheduler_total_steps:
            message = (
                f"{path} was written under a schedule of {checkpoint['scheduler_total_steps']} "
                f"optimiser steps; this run's is {self.scheduler_total_steps}. The step budget "
                f"is usually the dataset changing size under a run.")
            if not self.reschedule:
                raise ValueError(message)
            logger.warning("RESCHEDULING: %s Continuing anyway, as asked.", message)
        self.model.load_state_dict(_match_compile_prefix(checkpoint["model"], self.model))
        # Absent from checkpoints written before the criterion could carry state, and empty
        # for every criterion that carries none, so a missing key is not an error.
        criterion_state = checkpoint.get("criterion") or {}
        if criterion_state:
            self.criterion.load_state_dict(criterion_state)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if (self.scheduler is None) != (checkpoint["scheduler"] is None):
            raise ValueError(
                "The checkpoint has a learning rate schedule and this run does not, or the "
                "other way round.")
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        for name, loader in self._loaders().items():
            if name in checkpoint["loaders"]:
                loader.load_state_dict(checkpoint["loaders"][name])
            else:
                logger.warning("The checkpoint holds no %s loader state; reshuffling it.", name)
        random.setstate(checkpoint["rng"]["python"])
        torch.set_rng_state(checkpoint["rng"]["torch"].to(torch.uint8).cpu())
        # `map_location` has moved these onto the GPU; the setter takes CPU ByteTensors only.
        cuda_state = [state.to(torch.uint8).cpu() for state in checkpoint["rng"]["cuda"]]
        if cuda_state and self.device.type == "cuda":
            if len(cuda_state) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_state)
            else:
                # Restoring a subset would leave the remaining devices on a state that has
                # already been used, which is worse than starting them fresh.
                logger.warning(
                    "The checkpoint holds CUDA RNG state for %d devices and this host has %d; "
                    "leaving the CUDA RNG as it is.", len(cuda_state), torch.cuda.device_count())
        return {
            "epoch": int(checkpoint["epoch"]),
            "best_val_loss": float(checkpoint["best_val_loss"]),
            "best_val_epoch": int(checkpoint["best_val_epoch"]),
        }


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
        start_epoch = 0
        self.run_path.mkdir(exist_ok=True)
        best_model_params_path = self.run_path / "best_model_params.pt"
        if self.resume:
            resumed = self.load_training_checkpoint()
            start_epoch = resumed["epoch"]
            best_val_loss = resumed["best_val_loss"]
            best_val_epoch = resumed["best_val_epoch"]
            if start_epoch >= self.epochs:
                logger.info(
                    "The checkpoint records all %d epochs as done; nothing left to train.",
                    self.epochs)
            else:
                logger.info(
                    "Resuming at epoch %d of %d; best validation loss %.4f at epoch %d.",
                    start_epoch, self.epochs, best_val_loss, best_val_epoch)
        self.save_start_token_distribution()

        wandb.define_metric("loss.epoch.val.total", step_metric="epoch", summary="min")
        wandb.define_metric("loss.epoch.train.total", step_metric="epoch", summary="min")
        wandb.define_metric("loss.epoch.val_best", step_metric="epoch", summary="min")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("known_seq_len", hidden=True)
        wandb.define_metric("known_cascade_len", hidden=True)

        for epoch in (train_tqdm := trange(start_epoch, self.epochs)):
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
                    metric_name = "nll" if self.scalar_loss == "censored" else "mae"
                    for name, loss in raw_losses.items():
                        loss_dict[name] = {metric_name: loss.item()}
                    if self.censored_diagnostics is not None:
                        # The NLL moves with the fitted scale and cannot be read as an
                        # error, so log the three quantities that can be: how often the
                        # predicted minimum is above an observation (should be ~0), the
                        # mean excess, and the scale the model thinks it has.
                        loss_dict["val"].update(
                            self.scalar_diagnostics(self.val_dataset, self.val_loader))
                else:
                    total_val_loss = raw_losses['val'].sum()
                    for name, loss in raw_losses.items():
                        # evaluate() returns one entry per target field, so this zips against
                        # cascade_target_order; a non-target field has no loss to report.
                        loss_dict[name] = {
                            field: loss[i] for i, field in enumerate(self.cascade_target_order)}
                        loss_dict[name]["total"] = loss.sum().item()
                logged = {"loss.epoch": loss_dict,
                          "lr": self.optimizer.param_groups[0]['lr'],
                          "epoch": epoch}
                lag_metrics = self.schedule_free_lag()
                if lag_metrics:
                    logged["schedule_free"] = lag_metrics
                wandb.log(logged, commit=False)
                if total_val_loss < best_val_loss:
                    # A float, not the 0-dim tensor: this is what the checkpoint carries
                    # across a resume, and what the comparison above is against on the
                    # first epoch of one.
                    best_val_loss = float(total_val_loss)
                    best_val_epoch = epoch
                    atomic_torch_save(self.model.state_dict(), best_model_params_path)
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
            if (epoch + 1) % self.checkpoint_period == 0:
                self.save_training_checkpoint(epoch + 1, best_val_loss, best_val_epoch)

        # Record that the loop is over, whether it ran to `epochs` or stopped early. A crash in
        # the generation and evaluation that follows then costs no training at all: resuming
        # from this checkpoint enters the loop with nothing to do and goes straight on to them.
        self.save_training_checkpoint(self.epochs, best_val_loss, best_val_epoch)
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
            composition_cond: Optional[torch.Tensor] = None,
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
            cond: Optional tensor of shape [n_structures, len(condition_features)] carrying
                the scalar conditioning features in their configured order, in physical
                units (any condition_transform is applied here, not by the caller).
                `WyckoffTrainer.build_condition_from_values` builds it from one value per
                named feature.
            composition_cond: Optional tensor of shape [n_structures, composition width]
                for a model with composition_conditioning, as
                wyckoff_transformer.composition builds it. Concatenated after `cond`.
        """
        # `stops` must be passed: without it the generator cannot tell a finished sequence
        # from a live one, and `compute_validity_per_known_sequence_length` then scores STOP
        # tokens as invalid Wyckoff positions.
        generator = WyckoffGenerator(
            self.model, self.cascade_order, self.cascade_is_target, self.token_engineers,
            self.masks_dict, self.max_sequence_length, stops=self.stops_dict)

        condition_features = self.condition_features

        if calibrate:
            if self.val_dataset is None:
                raise ValueError("Calibration requires a validation dataset")
            generator.calibrate(self.val_dataset, cond_builder=self.build_cond)
        if start_tensor is None:
            start_tensor = self._sample_start_tokens_from_distribution(n_structures)
        else:
            if start_tensor.size(0) != n_structures:
                raise ValueError("Custom start tensor must have the same number of samples as requested structures.")
            if hasattr(self, 'train_dataset') and self.train_dataset is not None:
                start_tensor = start_tensor.to(self.device).to(self.train_dataset.start_tokens.dtype)
            else:
                start_tensor = start_tensor.to(self.device).to(torch.int64 if self.model.start_type == "categorial" else torch.float32)

        if cond is not None:
            # Anything a caller passes in is the scalar block, in physical units.
            if not condition_features:
                raise ValueError(
                    "`cond` was given, but this model has no scalar conditioning features.")
            self._validate_condition_values(cond)
            cond = self.transform_condition(cond)
            if self.composition_conditioning:
                if composition_cond is None:
                    raise ValueError(
                        "composition_conditioning is on, so a caller supplying `cond` must "
                        "also supply `composition_cond`; otherwise the model is handed a "
                        "conditioning vector of the wrong width.")
                cond = torch.cat([cond, composition_cond.to(cond.device, torch.float32)], dim=-1)
        elif composition_cond is not None:
            if not self.composition_conditioning:
                raise ValueError("composition_cond was given, but this model is not "
                                 "conditioned on the composition.")
            cond = composition_cond.to(self.device, torch.float32)
            if condition_features:
                raise ValueError(
                    f"This model is also conditioned on {list(condition_features)}; pass "
                    "`cond` for those alongside `composition_cond`.")
        elif self.condition_dim is not None:
            # Nothing supplied: draw whole conditioning rows from the training data, which
            # keeps the scalar and the composition paired as they actually occur rather
            # than crossing an energy with an unrelated formula.
            if getattr(self, "train_dataset", None) is None:
                wanted = [repr(name) for name in condition_features]
                if self.composition_conditioning:
                    wanted.append("the target composition")
                raise ValueError(
                    f"This model is conditioned on {' and '.join(wanted)}, but no `cond` "
                    "was provided and no train_dataset is available to sample one from.")
            random_indices = torch.randint(
                0, self.train_dataset.num_examples, (n_structures,), device=self.device)
            cond = self.build_cond(self.train_dataset, random_indices)

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

        # Non-target fields are filled in by their engineers (harmonic_site_symmetries,
        # site_symmetry_ops_id, ...). They are inputs to the model, not part of the generated
        # structure, and a vector-valued one cannot be stacked with the [batch, length] token
        # fields anyway. Dropped back to front so the surviving indices stay valid.
        generated_cascade_order = list(self.cascade_order)
        for position in reversed(range(len(self.cascade_order))):
            if not self.cascade_is_target.get(self.cascade_order[position], False):
                del generated_tensors[position]
                del generated_cascade_order[position]
        generated_cascade_order = tuple(generated_cascade_order)
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
        # They are averaged over structures still placing real sites at that length; one that has
        # emitted STOP is excluded from that point on, rather than having its STOP scored as an
        # invalid Wyckoff position.
        ensure_wandb_media_directory()
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
        augmentation_samples: int = 1,
        cond: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict scalar targets for pre-tokenised data.

        Args:
            prediction_data: Tokenised cascade data in the same format as AugmentedCascadeDataset expects.
                Must include the start field (``self.start_name``), every field from ``self.cascade_order``,
                engineered fields, and augmented variants if applicable.
            augmentation_samples: Number of random augmentation draws to average over. Each draw samples
                a random augmented variant when available.
            cond: Scalar conditioning values in physical units, one row per
                prediction and one column per configured condition feature.
                The configured transform is applied here. This is intentionally
                the scalar block only; composition-conditioned scalar models
                need a composition vector assembled by their caller.

        Returns:
            A tuple (mean_predictions, all_predictions) where:
                - mean_predictions is a tensor of shape [num_examples] with the average prediction across
                  augmentation samples. Under scalar_loss="censored" this is the location head,
                  i.e. the estimate of min(E | gene), not of the energy of any one structure.
                - all_predictions is a tensor of shape [augmentation_samples, num_examples] with raw
                  predictions per augmentation sample.
        """
        if self.target != TargetClass.Scalar:
            raise ValueError("predict_scalars is only available for Scalar targets.")
        if augmentation_samples < 1:
            raise ValueError("augmentation_samples must be at least 1.")
        if self.composition_conditioning:
            raise ValueError(
                "predict_scalars does not build composition conditioning. Supply a scalar "
                "regressor without composition_conditioning for gene screening.")

        prediction_data = prediction_data.copy()
        dummy_target_name = "__scalar_prediction_dummy__"
        num_examples = prediction_data[self.start_name].shape[0]
        prediction_data[dummy_target_name] = torch.zeros(num_examples, dtype=torch.float32)
        if cond is not None:
            if not self.condition_features:
                raise ValueError(
                    "cond was supplied, but this scalar regressor has no conditioning features.")
            self._validate_condition_values(cond)
            cond = cond if cond.dim() > 1 else cond.unsqueeze(-1)
            if cond.shape[0] != num_examples:
                raise ValueError(
                    f"cond has {cond.shape[0]} rows for {num_examples} scalar predictions.")
            cond = self.transform_condition(cond.to(self.device, dtype=torch.float32))
        elif self.condition_features:
            raise ValueError(
                f"This scalar regressor is conditioned on {list(self.condition_features)}; "
                "pass cond in physical units.")

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
            start_dtype=(
                self.train_dataset.start_tokens.dtype
                if self.train_dataset is not None
                else (torch.int64 if self.model.start_type == "categorial" else torch.float32)
            ),
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
                preds = self.model(start_tokens, cascade_tokens, padding_mask, None, cond=cond)
                if self.scalar_loss == "censored":
                    # The location column is the estimate of min(E | gene); the scale column
                    # describes the spread above it and is not a prediction of the energy.
                    preds, _ = self.criterion.split(preds.reshape(-1, self.criterion.n_outputs))
                else:
                    preds = preds.squeeze()
                sample_predictions.append(preds)
            stacked_predictions = torch.stack(sample_predictions, dim=0)
            mean_predictions = stacked_predictions.mean(dim=0)
        finally:
            if was_training:
                self.model.train()
        return mean_predictions, stacked_predictions


#: Sentinel for "this key is not in that config at all", which None is a legitimate value for.
_MISSING = object()


def flatten_config(config: Any, prefix: str = "") -> Dict[str, Any]:
    """A config as a flat {dotted.key: leaf} mapping, for reporting where two of them differ."""
    if isinstance(config, dict):
        flat = {}
        for key, value in config.items():
            flat.update(flatten_config(value, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    if isinstance(config, list):
        flat = {}
        for index, value in enumerate(config):
            flat.update(flatten_config(value, f"{prefix}[{index}]"))
        return flat
    return {prefix: config}


#: The only keys `--reschedule` may change under a resumed run: the horizon the
#: learning-rate schedule is laid out on, and the shape it lays out there. They are safe to
#: change mid-run in a way `lr` or `train_batch_size` are not, because the schedule is a pure
#: function of (step, horizon) -- give it a new horizon and the next step follows the new
#: curve, with the optimiser state, the weights and the step counter all still the run's own.
#: Everything else stays refused: a resumed run that changed them would be neither the run on
#: disk nor the one on the command line.
RESCHEDULABLE_CONFIG_KEYS = ("optimisation.epochs", "optimisation.scheduler.config.")


def _is_reschedulable(key: str) -> bool:
    return any(key == k or key.startswith(k) for k in RESCHEDULABLE_CONFIG_KEYS)


def check_resume_config(
    config_dict: dict|DictConfig,
    saved_config_path: Path,
    reschedule: bool = False) -> list[str]:
    """Refuse to resume a run under a config other than the one it started under.

    A resumed run carries the optimiser, the schedule and its horizon forward from the
    checkpoint, so a changed learning rate, batch size or model shape would produce something
    that is neither the run on disk nor the one on the command line -- and the W&B run it
    logs into would claim to be the former. `load_training_checkpoint` catches the epoch count
    and the step budget; this catches everything else, before any training happens.

    Args:
        reschedule: Permit differences in RESCHEDULABLE_CONFIG_KEYS -- deliberately moving the
            run's horizon, e.g. to bring the decay forward and land the run on a deadline.
            Every other key is still refused.

    Returns:
        The reschedulable keys that differ, as human-readable lines. Empty unless `reschedule`.

    Raises:
        ValueError: The configs differ outside what `reschedule` allows, listing the keys.
    """
    if not saved_config_path.exists():
        raise FileNotFoundError(
            f"Cannot verify the config of the run being resumed: {saved_config_path} is missing.")
    saved = flatten_config(OmegaConf.to_container(OmegaConf.load(saved_config_path), resolve=True))
    current = flatten_config(OmegaConf.to_container(OmegaConf.create(config_dict), resolve=True))
    differences = []
    rescheduled = []
    for key in sorted(set(saved) | set(current)):
        if saved.get(key, _MISSING) != current.get(key, _MISSING):
            line = (f"  {key}: {saved.get(key, '<absent>')!r} (saved) != "
                    f"{current.get(key, '<absent>')!r} (given)")
            if reschedule and _is_reschedulable(key):
                rescheduled.append(line)
            else:
                differences.append(line)
    if differences:
        raise ValueError(
            f"The config given differs from the one {saved_config_path} recorded for this run, "
            f"so it cannot be resumed:\n" + "\n".join(differences))
    return rescheduled


def train_from_config(
    config_dict: dict,
    device: torch.device,
    run_path: Path = Path(__file__).resolve().parent.parent / "runs",
    production_training: bool = False,
    no_test: bool = False,
    resume: bool = False,
    reschedule: bool = False):

    if wandb.run is None:
        raise ValueError("W&B run must be initialized")
    this_run_path = run_path / wandb.run.id
    if resume:
        checkpoint_path = this_run_path / CHECKPOINT_FILENAME
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Asked to resume run {wandb.run.id}, but it has no checkpoint at "
                f"{checkpoint_path}. A run that died before writing one has to be started over.")
        rescheduled = check_resume_config(
            config_dict, this_run_path / "config.yaml", reschedule=reschedule)
        logger.info("Resuming run %s from %s", wandb.run.id, checkpoint_path)
        if rescheduled:
            # The run's own config.yaml is what every later link is held to, so it has to
            # become the schedule the run is actually on -- otherwise the next resume diffs
            # against a horizon nothing is following any more and needs --reschedule to get
            # past a change that already happened.
            logger.warning(
                "RESCHEDULING run %s onto a new horizon:\n%s",
                wandb.run.id, "\n".join(rescheduled))
            OmegaConf.save(config_dict, this_run_path / "config.yaml")
            wandb.run.summary["rescheduled_at_epoch"] = wandb.run.summary.get("epoch")
    else:
        this_run_path.mkdir(parents=True, exist_ok=False)
    trainer = WyckoffTrainer.from_config(config_dict, device, run_path=this_run_path, production_training=production_training, no_test=no_test, resume=resume, reschedule=reschedule)
    if not resume:
        # A resumed run wrote all of these on its first attempt, and their W&B artifacts with
        # them; the config one is what check_resume_config just held it to.
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
