from typing import Tuple, List, Dict, Optional, Set, Union
from pathlib import Path
import logging
import torch
from torch import nn, Tensor
import numpy as np
from pymatgen.core.periodic_table import Element

from wyckoff_transformer.cascade.dataset import AugmentedCascadeDataset, TargetClass
from wyckoff_transformer.tokenization import load_tensors_and_tokenisers, FeatureEngineer, WyckoffProcessor
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


    @torch.no_grad()
    def generate_tensors(
        self,
        start: Tensor,
        temperature: float = 1,
        compute_validity: bool = False,
        required_element_set: Optional[Union[str, Set[int]]] = None,
        allowed_element_set: Union[str, Set[int]] = "all",
        max_length: Optional[int] = None,
        elements_vocab: Optional[Dict] = None,
        delimiter: str = "-",
    ) -> List[Tensor] | Tuple[List[Tensor], List[float], List[float]]:
        """
        Generates a sequence of tokens.

        Arguments:
            start: The start token. It should be a tensor of shape [batch_size].
            temperature: The temperature to use for the generation
            compute_validity: Whether to compute the validity of the generated sequences
            required_element_set : A set of required element token IDs or a dash-separated string (for CSX)
            allowed_element_set : Controls the pool of allowed elements. (for CSX)
            max_length : Maximum number of sequence sites to generate. Defaults to self.max_sequence_len.
            elements_vocab : Vocabulary dict mapping element keys to token IDs.
            delimiter : Delimiter for parsing the string form, default "-".
        Returns:
            The generated sequence of tokens. It has shape [batch_size, max_len, len(cascade_order)].
                It doesn't include the start token.
            If compute_validity is True, also returns the formal validity of the generated sequences
                for the site symmetries and the enumeration (if applicable) for each known sequence length.
        """
        is_csx = required_element_set is not None
        device = start.device
        batch_size = start.size(0)
        if max_length is None:
            max_length = self.max_sequence_len

        if is_csx:
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

            if isinstance(required_element_set, str):
                required_id_set = _parse_string_to_ids(required_element_set)
            else:
                required_id_set = set(required_element_set)

            if allowed_element_set == "all":
                if elements_vocab is None:
                    raise ValueError("elements_vocab must be provided when allowed_element_set is 'all'.")
                allowed_id_set = {v for k, v in elements_vocab.items() if isinstance(k, Element)}
            elif allowed_element_set == "fix":
                allowed_id_set = set(required_id_set)
            elif isinstance(allowed_element_set, str):
                allowed_id_set = _parse_string_to_ids(allowed_element_set)
            elif isinstance(allowed_element_set, set):
                allowed_id_set = set(allowed_element_set)
            else:
                raise ValueError(f"Invalid value for allowed_element_set: {allowed_element_set}")

            allowed_id_set.add(stop_id)

            if not required_id_set.issubset(allowed_id_set):
                raise ValueError("The required_element_set must be a subset of the allowed_element_set.")

            placed_required = [set() for _ in range(batch_size)]
            elements_stop_generated = np.zeros(batch_size, dtype=bool)

        self.model.eval()
        
        # Since we are doing in-place operations, we can just pre-assign everything to be a mask
        generated = []
        for field in self.cascade_order:
            if np.issubdtype(type(self.masks[field]), np.integer) or self.masks[field].ndim == 0:
                dtype = torch.int64 if np.issubdtype(type(self.masks[field]), np.integer) else self.masks[field].dtype
                generated.append(torch.full((batch_size, max_length), self.masks[field],
                                             dtype=dtype, device=device))
            else:
                unsqueezed_mask = self.masks[field].unsqueeze(0)
                if self.masks[field].dim() == 0:
                    generated.append(torch.tile(unsqueezed_mask, (batch_size, max_length)))
                elif self.masks[field].dim() == 1:
                    unsqueezed_mask = unsqueezed_mask.unsqueeze(0)
                    generated.append(torch.tile(unsqueezed_mask, (batch_size, max_length, 1)))
                else:
                    raise NotImplementedError("Mask should be a scalar or a vector")
                    
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
        if compute_validity:
            global_ss_validitity = []
            global_enum_validity = []
            
        for known_seq_len in range(max_length):
            for known_cascade_len, cascade_name in enumerate(self.cascade_order):
                if self.cascade_is_target.get(cascade_name, False):
                    # +1 for MASK
                    this_generation_input = [generated_cascade[:, :known_seq_len + 1] for generated_cascade in generated]
                    logits = self.model(start, this_generation_input, None, known_cascade_len)
                    if self.calibrators is not None:
                        if known_seq_len < len(self.calibrators[known_cascade_len]):
                            logits = self.calibrators[known_cascade_len][known_seq_len](logits)
                        else:
                            logits = self.tail_calibrators[known_cascade_len](logits)
                            
                    if is_csx and cascade_name == "elements":
                        logits_masked = torch.full_like(logits, float("-inf"))
                        allowed_idx_tensor = torch.tensor(
                            sorted(list(allowed_id_set)), dtype=torch.long, device=device
                        )
                        logits_masked[:, allowed_idx_tensor] = logits[:, allowed_idx_tensor]
                        logits = logits_masked

                    logits = logits / temperature
                    calibrated_probas = torch.nn.functional.softmax(logits, dim=1)
                    
                    if is_csx and cascade_name == "elements":
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
                                next_tokens[b] = torch.multinomial(calibrated_probas[b], num_samples=1).squeeze()

                        generated[known_cascade_len][:, known_seq_len] = next_tokens

                        for b in range(batch_size):
                            tok = next_tokens[b].item()
                            if tok == stop_id:
                                elements_stop_generated[b] = True
                            elif tok in required_id_set:
                                placed_required[b].add(tok)
                    else:
                        generated[known_cascade_len][:, known_seq_len] = \
                            torch.multinomial(calibrated_probas, num_samples=1).squeeze()
                            
                    if self.stops is not None:
                        stop_mask = (generated[known_cascade_len][:, known_seq_len] == self.stops[cascade_name]).cpu().numpy()
                        stop_generated |= stop_mask
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
                        else:
                            raise NotImplementedError(
                                f"Unknown input field {input_field} for engineer {cascade_name}")
                        this_engineer_input.append(this_cascade_input.tolist())
                    feature_np = self.token_engineers[cascade_name].get_feature_from_token_batch(
                        start_converted, this_engineer_input)
                    if feature_np.dtype == "O": # Object, in this case array of arrays
                        feature_np = np.stack(feature_np)
                    generated[known_cascade_len][:, known_seq_len] = \
                        torch.from_numpy(feature_np).to(device)
                        
            if compute_validity:
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
                if ss_validitity:
                    global_ss_validitity.append(np.mean(ss_validitity))
                if enum_validity:
                    global_enum_validity.append(np.mean(enum_validity))
        if compute_validity:
            return generated, global_ss_validitity, global_enum_validity
        return generated
    
