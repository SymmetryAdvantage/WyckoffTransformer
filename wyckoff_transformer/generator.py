from typing import Tuple, List, Dict, Optional, Set, Union
from pathlib import Path
import logging
import torch
from torch import nn, Tensor
import numpy as np
from pymatgen.core.periodic_table import Element

from cascade_transformer.dataset import AugmentedCascadeDataset, TargetClass, jagged_batch_randperm
from wyckoff_transformer.tokenization import load_tensors_and_tokenisers, FeatureEngineer
from scripts.preprocess_wychoffs import inverse_series

logger = logging.getLogger(__name__)

class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        # Apply temperature scaling
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Temperature scaling of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def fit(self, logits, labels):
        # Fit the temperature parameter using the validation data
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_():
            optimizer.zero_grad()
            loss = criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        logger.info("Cross entropy before calibration: %f", criterion(logits, labels).item())
        optimizer.step(eval_)
        logger.info("Cross entropy after calibration: %f", criterion(self.temperature_scale(logits), labels).item())
        logger.info("Temperature: %f", self.temperature.item())
        return self


class WyckoffGenerator():
    @classmethod
    def from_wandb_run(
        cls,
        wandb_run_id,
        update_wandb: bool = False,
        device: torch.device = torch.device("cpu")):
        import wandb
        from omegaconf import OmegaConf
        from cascade_transformer.model import CascadeTransformer

        if update_wandb:
            wandb_run = wandb.init(project="WyckoffTransformer", id=wandb_run_id, resume=True)
        else:
            wandb_run = wandb.Api().run(f"WyckoffTransformer/{wandb_run_id}")

        config = OmegaConf.create(dict(wandb_run.config))

        # The start tokens will be sampled from the train+validation datasets,
        # to preserve the sanctity of the test dataset and ex nihilo generation.
        logger.warning("Loading current tensors and tokenisers for %s, not the saved ones", config.dataset)
        tensors, tokenisers, engineers = load_tensors_and_tokenisers(config.dataset, config.tokeniser.name)

        model = CascadeTransformer.from_config_and_tokenisers(config, tokenisers, device)
        model.load_state_dict(torch.load(Path("runs", wandb_run_id, "best_model_params.pt"), map_location=device))
        # We need to grab any tensor from the train dataset
        max_sequence_len = tensors["train"][config.model.cascade.order[0]].size(1)

        masks_dict = {field: tokenisers[field].mask_token for field in config.model.cascade.order}
        stops_dict = {field: tokenisers[field].stop_token for field in config.model.cascade.order}

        generator = WyckoffGenerator(model,
            config.model.cascade.order,
            config.model.cascade.is_target,
            engineers,
            masks_dict, max_sequence_len, stops=stops_dict)
        return generator, wandb_run, tensors, tokenisers, engineers


    def __init__(self,
                 model: nn.Module,
                 cascade_order: Tuple,
                 cascade_is_target: Dict,
                 token_engineers: Dict,
                 masks: dict,
                 max_sequence_len: int,
                 stops: Optional[Dict] = None):
        self.model = model
        self.max_sequence_len = int(max_sequence_len)
        self.cascade_order = cascade_order
        self.cascade_is_target = cascade_is_target
        self.masks = masks
        self.stops = stops
        self.calibrators = None
        self.tail_calibrators = None
        self.token_engineers = token_engineers


    def calibrate(self, dataset: AugmentedCascadeDataset, calibration_element_count_threshold: int = 100):
        """
        The calibraiton is going to be per cascade field.
        We will generate p_predicted and p_true for each cascade field for each
        known sequence length.
        """
        assert dataset.cascade_order == self.cascade_order
        with torch.no_grad():
            self.model.eval()
            self.calibrators = []
            self.tail_calibrators = []
            for known_cascade_len, cascade_name in enumerate(self.cascade_order):
                tail_predictions = []
                tail_targets = []
                self.calibrators.append([])
                if not self.cascade_is_target[cascade_name]:
                    logging.info("Cascade field %s is not a target doesn't need calibration", cascade_name)
                    continue
                logging.info("Calibrating cascade field %s", cascade_name)
                for known_seq_len in range(dataset.max_sequence_length):
                    if known_cascade_len == 0:
                        start_tokens, masked_data, target = dataset.get_masked_multiclass_cascade_data(
                            known_seq_len, known_cascade_len,
                            target_type=TargetClass.NextToken, multiclass_target=True)
                    else:
                        start_tokens, masked_data, target = dataset.get_masked_multiclass_cascade_data(
                            known_seq_len, known_cascade_len,
                            target_type=TargetClass.NextToken, multiclass_target=False)
                    model_output = self.model(start_tokens, masked_data, None, known_cascade_len)
                    # Enought data for separate calibration
                    if target.size(0) >= calibration_element_count_threshold:
                        # If model_output and target are on different devices, not our problem
                        self.calibrators[known_cascade_len].append(TemperatureScaling().to(target.device).fit(
                            model_output, target))
                    # Not enough data for separate calibration, so we gather all the data in the tail
                    else:
                        tail_predictions.append(model_output)
                        tail_targets.append(target)
                tail_predictions = torch.concatenate(tail_predictions, axis=0)
                tail_targets = torch.concatenate(tail_targets, axis=0)
                if tail_predictions.size(0) < calibration_element_count_threshold:
                    logger.warning("Tail too small, %i when requested %i", tail_predictions.size(0), calibration_element_count_threshold)
                self.tail_calibrators.append(TemperatureScaling().to(tail_targets.device).fit(tail_predictions, tail_targets))


    def compute_likelihoods(self,
        dataset: AugmentedCascadeDataset,
        n_permutations: int = 1,
        n_augmentations: int = 1) -> Tensor:
        """
        Computes the likelihood of the dataset.
        """
        with torch.no_grad():
            self.model.eval()
            log_likelihoods = torch.zeros(
                n_permutations, n_augmentations, dataset.start_tokens.size()[0],
                device=dataset.device, dtype=torch.float32)
            if n_augmentations > 1:
                augmentations = [dataset.get_augmentation() for _ in range(n_augmentations)]
            else:
                augmentations = [None]
            for permutation_idx in range(n_permutations):
                if n_permutations > 1:
                    full_permutation = jagged_batch_randperm(
                        dataset.pure_sequences_lengths, dataset.max_sequence_length)
                    applied_permutation = True
                else:
                    full_permutation = None
                    applied_permutation = False
                for augmentation_idx, augmented_data in enumerate(augmentations):
                    for known_seq_len in range(self.max_sequence_len):
                        for known_cascade_len in range(len(self.cascade_order)):
                            start, this_data, target, batch_target_is_viable = dataset.get_masked_multiclass_cascade_data(
                                known_seq_len, known_cascade_len, TargetClass.NextToken, multiclass_target=False,
                                augmented_data=augmented_data, full_permutation=full_permutation,
                                apply_permutation=applied_permutation, return_chosen_indices=True)
                            logits = self.model(start, this_data, None, known_cascade_len)
                            log_probas = torch.nn.functional.log_softmax(logits, dim=1)
                            log_likelihoods[permutation_idx, augmentation_idx, batch_target_is_viable] += torch.gather(
                                log_probas, 1, target.unsqueeze(1)).squeeze()
            return log_likelihoods

    @torch.no_grad()
    def generate_tensors(
        self,
        start: Tensor,
        temperature: float = 1) -> List[Tensor]:
        """
        Generates a sequence of tokens.

        Arguments:
            start: The start token. It should be a tensor of shape [batch_size].
            temperature: The temperature to use for the generation

        Returns:
            The generated sequence of tokens. It has shape [batch_size, max_len, len(cascade_order)].
            It doesn't include the start token.
        """
        self.model.eval()
        batch_size = start.size(0)
        # Since we are doing in-place operations, we can just pre-assign
        # everything to be a mask
        generated = []
        for field in self.cascade_order:
            # print(f"Generating {field} with mask shape {self.masks[field].shape}")
            if np.issubdtype(type(self.masks[field]), np.integer) or self.masks[field].ndim == 0:
                if np.issubdtype(type(self.masks[field]), np.integer):
                    dtype = torch.int64
                else:
                    dtype = self.masks[field].dtype
                generated.append(torch.full((batch_size, self.max_sequence_len), self.masks[field],
                                             dtype=dtype, device=start.device))
            else:
                unsqueezed_mask = self.masks[field].unsqueeze(0)
                if self.masks[field].dim() == 0:
                    generated.append(torch.tile(unsqueezed_mask, (batch_size, self.max_sequence_len)))
                elif self.masks[field].dim() == 1:
                    unsqueezed_mask = unsqueezed_mask.unsqueeze(0)
                    generated.append(torch.tile(unsqueezed_mask, (batch_size, self.max_sequence_len, 1)))
                else:
                    raise NotImplementedError("Mask should be a scalar or a vector")
            # print(f"Generated {field} with shape {generated[-1].shape}")
        # We have a problem. Engeineers by default can only work with data, not output of other engineers.
        if 'harmonic_site_symmetries' in self.cascade_order and 'sites_enumeration' not in self.cascade_order:
            cluster_to_enum = inverse_series(self.token_engineers["harmonic_cluster"].db)
            self.token_engineers['sites_enumeration'] = FeatureEngineer(
                cluster_to_enum, mask_token=None, stop_token=None, pad_token=None)
            
        cascade_index_by_name = {name: idx for idx, name in enumerate(self.cascade_order)}
        if len(start.size()) > 1:
            start_converted = list(map(tuple, start.tolist()))
        else:
            start_converted = start.tolist()
        stop_generated = np.zeros(batch_size, dtype=bool)
        for known_seq_len in range(self.max_sequence_len):
            for known_cascade_len, cascade_name in enumerate(self.cascade_order):
                if self.cascade_is_target[cascade_name]:
                    # +1 for MASK
                    this_generation_input = [generated_cascade[:, :known_seq_len + 1] for generated_cascade in generated]
                    #import pdb
                    #pdb.set_trace()
                    logits = self.model(start, this_generation_input, None, known_cascade_len)
                    if self.calibrators is not None:
                        if known_seq_len < len(self.calibrators[known_cascade_len]):
                            logits = self.calibrators[known_cascade_len][known_seq_len](logits)
                        else:
                            logits = self.tail_calibrators[known_cascade_len](logits)
                    logits = logits / temperature
                    print(f"[DEBUG] Cascade: {cascade_name}, Logits shape: {logits.shape}") # debug
                    print(f"[DEBUG] Cascade: {cascade_name}, Logits: {logits}") # debug
                    # binary/ternary hack
                    # if known_cascade_len == 0 and known_seq_len > 2:
                    #    logits *= 3.
                    # CONSIDER: remove probas for all special tokens aside from STOP
                    calibrated_probas = torch.nn.functional.softmax(logits, dim=1)
                    print(f"[DEBUG] Cascade: {cascade_name}, Calibrated probas shape: {calibrated_probas.shape}") # debug
                    print(f"[DEBUG] Cascade: {cascade_name}, Calibrated probas: {calibrated_probas}") # debug
                    # calibrated_probas = probas.numpy()
                    # calibrated_probas = calibrator.predict_proba(probas.numpy())
                    generated[known_cascade_len][:, known_seq_len] = \
                        torch.multinomial(calibrated_probas, num_samples=1).squeeze()
                    print(f"[DEBUG] Cascade: {cascade_name}, Sampled token shape: {generated[known_cascade_len][:, known_seq_len].shape}") # debug
                    print(f"[DEBUG] Cascade: {cascade_name}, Sampled token: {generated[known_cascade_len][:, known_seq_len]}") # debug
                    if self.stops is not None:
                        stop_generated |= generated[known_cascade_len][:, known_seq_len] == self.stops[cascade_name]
                else:
                    if known_cascade_len != len(self.cascade_order) - 1:
                        raise NotImplementedError("Only the last cascade field can be non-target")
                    if self.token_engineers[cascade_name].inputs[0] != 'spacegroup_number':
                        raise NotImplementedError("Only engineers with spacegroup_number first input are supported")
                    this_engineer_input = []
                    for input_field in self.token_engineers[cascade_name].inputs[1:]:
                        if input_field in cascade_index_by_name:
                            this_cascade_input = generated[cascade_index_by_name[input_field]][:, known_seq_len]
                        elif cascade_name == 'harmonic_site_symmetries' and input_field == 'sites_enumeration':
                            # Since we don't natively support either two engineers for one field or
                            # chainging engineers, we do this hack
                            enumerations = self.token_engineers['sites_enumeration'].get_feature_from_token_batch(
                                start_converted, [
                                    generated[cascade_index_by_name['site_symmetries']][:, known_seq_len].tolist(),
                                    generated[cascade_index_by_name['harmonic_cluster']][:, known_seq_len].tolist()])
                            this_cascade_input = enumerations
                            #import pdb
                            #pdb.set_trace()
                        else:
                            raise NotImplementedError(
                                f"Unknown input field {input_field} for engineer {cascade_name}")
                        this_engineer_input.append(this_cascade_input.tolist())
                    #print(this_engineer_input)
                    #print(len(this_engineer_input))
                    #print(this_generation_input[0].shape)
                    # print(f"Generating {cascade_name}")
                    feature_np = self.token_engineers[cascade_name].get_feature_from_token_batch(
                        start_converted, this_engineer_input)
                    if feature_np.dtype == "O": # Object, in this case array of array
                        # import pdb
                        # pdb.set_trace()
                        feature_np = np.stack(feature_np)
                    generated[known_cascade_len][:, known_seq_len] = \
                        torch.from_numpy(feature_np)
            ss_validitity = []
            enum_validity = []
            for structure_index, this_start in enumerate(start_converted):
                if stop_generated[structure_index]:
                    continue
                ss_validitity.append(
                    (this_start,
                     generated[cascade_index_by_name['site_symmetries']][structure_index, known_seq_len].item()
                     ) in self.token_engineers["multiplicity"].db)
                if 'sites_enumeration' in cascade_index_by_name:
                    enum_validity.append(
                        (this_start,
                            generated[cascade_index_by_name['site_symmetries']][structure_index, known_seq_len].item(),
                            generated[cascade_index_by_name['sites_enumeration']][structure_index, known_seq_len].item()
                        ) in self.token_engineers["multiplicity"].db)
                elif "harmonic_cluster" in cascade_index_by_name:
                    enum_validity.append(
                        (this_start,
                            generated[cascade_index_by_name['site_symmetries']][structure_index, known_seq_len].item(),
                            generated[cascade_index_by_name['harmonic_cluster']][structure_index, known_seq_len].item()
                        ) in self.token_engineers["sites_enumeration"].db)
            print(f"Known sequence length: {known_seq_len}")
            print(f"SS validity: {sum(ss_validitity) / len(ss_validitity)}")
            print(f"ENUM validity: {sum(enum_validity) / len(enum_validity)}")
            #import pdb
            #pdb.set_trace()
        return generated
    
    @torch.no_grad()
    def generate_csx_tensors(
        self,
        start: Tensor,
        required_element_set: Union[str, Set[int]],
        allowed_element_set: Union[str, Set[int]] = "all",
        *,
        temperature: float = 1.0,
        max_length: Optional[int] = None,
        elements_vocab: Optional[Dict] = None,
        delimiter: str = "-",
    ) -> List[Tensor]:
        """
        Generate a sequence of tokens in Chemical System eXploration mode (CSX).

        Parameters
        ----------
        start : Tensor
            The start token. It should be a tensor of shape [batch_size].
        required_element_set : Union[str, Set[int]]
            A set of required element token IDs or a dash-separated string
            (e.g., "Li-O") of elements that MUST be present.
        allowed_element_set : Union[str, Set[int]], default "all"
            Controls the pool of allowed elements. It has three modes:
            - "all": All elements in the `elements_vocab` are allowed.
            - "fix": Only elements from `required_element_set` are allowed.
            - str (e.g., "Li-Co-Mn-Ni-Fe-P-O"): A custom pool of allowed elements defined by
              a dash-separated string.
            - Set[int]: A custom pool of allowed elements defined by token IDs.
        temperature : float
            Softmax temperature used after optional calibration.
        max_length : Optional[int]
            Maximum number of sequence sites to generate. Defaults to self.max_sequence_len.
        elements_vocab : Optional[Dict]
            Vocabulary dict mapping element keys to token IDs.
        delimiter : str
            Delimiter for parsing the string form, default "-".

        Returns
        -------
        List[Tensor]
            Same layout as `generate_tensors`: one tensor per cascade, each of shape
            [batch_size, max_sequence_len, ...].
        """

        self.model.eval()
        device = start.device
        batch_size = start.size(0)
        if max_length is None:
            max_length = self.max_sequence_len

        if elements_vocab is not None and 'STOP' in elements_vocab:
            stop_id = elements_vocab['STOP']
        else:
            raise RuntimeError("STOP token for 'elements' cascade not defined.")

        def _symbol_to_id(sym: str) -> int:
            if elements_vocab is None:
                raise RuntimeError("elements_vocab must be provided for string-based element sets.")
            try:
                key = Element(sym)
                if key in elements_vocab: return elements_vocab[key]
            except Exception: pass
            k2 = f"Element {sym}"
            if k2 in elements_vocab: return elements_vocab[k2]
            if sym in elements_vocab: return elements_vocab[sym]
            raise KeyError(f"Element symbol '{sym}' not found in provided elements_vocab.")

        def _parse_string_to_ids(s: str) -> set:
            syms = [x.strip() for x in s.split(delimiter) if x.strip()]
            if not syms: raise ValueError("Empty element string provided.")
            return { _symbol_to_id(x) for x in syms }

        # Normalize required_element_set to a set of token IDs
        if isinstance(required_element_set, str):
            required_id_set = _parse_string_to_ids(required_element_set)
        else:
            required_id_set = set(required_element_set)

        if allowed_element_set == "all":
            # Option 1: Use all elements from the vocabulary.
            if elements_vocab is None:
                raise ValueError("elements_vocab must be provided when allowed_element_set is 'all'.")
            allowed_id_set = {v for k, v in elements_vocab.items() if isinstance(k, Element)}
        elif allowed_element_set == "fix":
            # Option 2: The allowed set is strictly the required set.
            allowed_id_set = set(required_id_set)
        elif isinstance(allowed_element_set, str):
            # Option 3: A custom set defined by a string.
            allowed_id_set = _parse_string_to_ids(allowed_element_set)
        elif isinstance(allowed_element_set, set):
            # Support direct set input for programmatic use.
            allowed_id_set = set(allowed_element_set)
        else:
            raise ValueError(f"Invalid value for allowed_element_set: {allowed_element_set}")

        # Always allow the STOP token in the allowed set.
        allowed_id_set.add(stop_id)

        # Validate that required elements are a subset of allowed elements.
        if not required_id_set.issubset(allowed_id_set):
            raise ValueError("The required_element_set must be a subset of the allowed_element_set.")

        generated: List[Tensor] = []
        for field in self.cascade_order:
            if np.issubdtype(type(self.masks[field]), np.integer) or self.masks[field].ndim == 0:
                dtype = torch.int64 if np.issubdtype(type(self.masks[field]), np.integer) else self.masks[field].dtype
                generated.append(
                    torch.full((batch_size, max_length), self.masks[field], dtype=dtype, device=device)
                )
            else:
                unsqueezed_mask = self.masks[field].unsqueeze(0)
                if self.masks[field].dim() == 0:
                    generated.append(torch.tile(unsqueezed_mask, (batch_size, max_length)))
                elif self.masks[field].dim() == 1:
                    generated.append(torch.tile(unsqueezed_mask.unsqueeze(0), (batch_size, max_length, 1)))
                else:
                    raise NotImplementedError("Mask should be a scalar or a vector")
        
        if 'harmonic_site_symmetries' in self.cascade_order and 'sites_enumeration' not in self.cascade_order:
            cluster_to_enum = inverse_series(self.token_engineers["harmonic_cluster"].db)
            self.token_engineers['sites_enumeration'] = FeatureEngineer(
                cluster_to_enum, mask_token=None, stop_token=None, pad_token=None)

        cascade_index_by_name = {name: idx for idx, name in enumerate(self.cascade_order)}
        if len(start.size()) > 1:
            start_converted = list(map(tuple, start.tolist()))
        else:
            start_converted = start.tolist()

        stop_generated = np.zeros(batch_size, dtype=bool)
        placed_required = [set() for _ in range(batch_size)]
        elements_idx = cascade_index_by_name["elements"]
        elements_stop_generated = np.zeros(batch_size, dtype=bool)

        for known_seq_len in range(max_length):
            for known_cascade_len, cascade_name in enumerate(self.cascade_order):
                if cascade_name == "elements" and self.cascade_is_target[cascade_name]:
                    model_inputs = [gen_cascade[:, : known_seq_len + 1] for gen_cascade in generated]
                    logits = self.model(start, model_inputs, None, known_cascade_len)

                    if self.calibrators is not None:
                        if known_seq_len < len(self.calibrators[known_cascade_len]):
                            logits = self.calibrators[known_cascade_len][known_seq_len](logits)
                        else:
                            logits = self.tail_calibrators[known_cascade_len](logits)

                    logits_masked = torch.full_like(logits, float("-inf"))
                    allowed_idx_tensor = torch.tensor(
                        sorted(list(allowed_id_set)), dtype=torch.long, device=device
                    )
                    logits_masked[:, allowed_idx_tensor] = logits[:, allowed_idx_tensor]
                    logits = logits_masked

                    logits = logits / temperature
                    calibrated_probas = torch.softmax(logits, dim=1)
                    next_tokens = torch.empty(batch_size, dtype=torch.long, device=device)

                    for b in range(batch_size):
                        if elements_stop_generated[b]:
                            next_tokens[b] = stop_id
                            continue

                        missing_required = required_id_set - placed_required[b]

                        if missing_required:
                            subset_idx = torch.tensor(sorted(list(missing_required)), dtype=torch.long, device=device)
                            chosen_local = torch.argmax(calibrated_probas[b, subset_idx]).item()
                            next_tokens[b] = subset_idx[chosen_local]
                        else:
                            next_tokens[b] = torch.multinomial(calibrated_probas[b], num_samples=1)

                    generated[elements_idx][:, known_seq_len] = next_tokens

                    for b in range(batch_size):
                        tok = next_tokens[b].item()
                        if tok == stop_id:
                            elements_stop_generated[b] = True
                        elif tok in required_id_set:
                            placed_required[b].add(tok)
                    
                    if self.stops is not None:
                        stop_generated |= generated[known_cascade_len][:, known_seq_len].cpu() == self.stops[cascade_name]

                elif self.cascade_is_target.get(cascade_name, False):
                    model_inputs = [gen_cascade[:, : known_seq_len + 1] for gen_cascade in generated]
                    logits = self.model(start, model_inputs, None, known_cascade_len)

                    if self.calibrators is not None:
                        if known_seq_len < len(self.calibrators[known_cascade_len]):
                            logits = self.calibrators[known_cascade_len][known_seq_len](logits)
                        else:
                            logits = self.tail_calibrators[known_cascade_len](logits)

                    logits = logits / temperature
                    probas = torch.softmax(logits, dim=1)
                    generated[known_cascade_len][:, known_seq_len] = torch.multinomial(probas, 1).squeeze()
                    
                    if self.stops is not None:
                        stop_generated |= generated[known_cascade_len][:, known_seq_len].cpu() == self.stops[cascade_name]

                else:
                    if known_cascade_len != len(self.cascade_order) - 1:
                        raise NotImplementedError("Only the last cascade field can be non-target")
                    if self.token_engineers[cascade_name].inputs[0] != "spacegroup_number":
                        raise NotImplementedError("Only engineers with spacegroup_number first input are supported")
                    
                    engineer_inputs = []
                    for input_field in self.token_engineers[cascade_name].inputs[1:]:
                        if input_field in cascade_index_by_name:
                            this_cascade_input = generated[cascade_index_by_name[input_field]][:, known_seq_len]
                        elif cascade_name == 'harmonic_site_symmetries' and input_field == 'sites_enumeration':
                            enumerations = self.token_engineers['sites_enumeration'].get_feature_from_token_batch(
                                start_converted, [
                                    generated[cascade_index_by_name['site_symmetries']][:, known_seq_len].tolist(),
                                    generated[cascade_index_by_name['harmonic_cluster']][:, known_seq_len].tolist()])
                            this_cascade_input = torch.tensor(enumerations)
                        else:
                            raise NotImplementedError(f"Unknown input field {input_field} for engineer {cascade_name}")
                        engineer_inputs.append(this_cascade_input.tolist())

                    feature_np = self.token_engineers[cascade_name].get_feature_from_token_batch(
                        start_converted, engineer_inputs
                    )
                    if feature_np.dtype == "O": feature_np = np.stack(feature_np)
                    generated[known_cascade_len][:, known_seq_len] = torch.from_numpy(feature_np).to(device)
            
            ss_validitity, enum_validity = [], []
            for structure_index, this_start in enumerate(start_converted):
                if stop_generated[structure_index]: continue
                ss_validitity.append((this_start, generated[cascade_index_by_name['site_symmetries']][structure_index, known_seq_len].item()) in self.token_engineers["multiplicity"].db)
                if 'sites_enumeration' in cascade_index_by_name:
                    enum_validity.append((this_start, generated[cascade_index_by_name['site_symmetries']][structure_index, known_seq_len].item(), generated[cascade_index_by_name['sites_enumeration']][structure_index, known_seq_len].item()) in self.token_engineers["multiplicity"].db)
                elif "harmonic_cluster" in cascade_index_by_name:
                    enum_validity.append((this_start, generated[cascade_index_by_name['site_symmetries']][structure_index, known_seq_len].item(), generated[cascade_index_by_name['harmonic_cluster']][structure_index, known_seq_len].item()) in self.token_engineers["sites_enumeration"].db)
            
            if len(ss_validitity) > 0: print(f"Known sequence length: {known_seq_len}, SS validity: {sum(ss_validitity) / len(ss_validitity)}")
            if len(enum_validity) > 0: print(f"ENUM validity: {sum(enum_validity) / len(enum_validity)}")

        return generated
