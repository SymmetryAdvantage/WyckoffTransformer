from typing import List, Tuple, Optional
from enum import Enum
import logging
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

def randint_tensor(high: torch.Tensor) -> Tensor:
    """
    Generates a random tensor of integers from 0 (inclusive) to high (exclusive).

    The probability distribution is not perfectly uniform in the case of
    torch.iinfo(dtype).max - torch.iinfo(dtype).min % high != 0
    but it's good enough for our purposes, where high < 50, and max-min ~ 2^64
    
    Arguments:
        high: The maximum value of the random tensor.
    Returns:
        A random tensor of integers from 0 to high - 1, size high.size.
    """
    dtype = high.dtype
    return torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, size=high.size(),
        device=high.device, dtype=dtype) % high


def jagged_batch_randperm_naive(permutation_lengths: Tensor, max_sequence_length: int):
    """
    Generates a random permutation of the integers from 0 to max_sequence_length
    for each batch in the batch_size. It's a random permutation of the first
    permutation_lengths[i] integers for the i-th batch, and the rest is
    the identity permutation (intended for padding).

    Arguments:
        permutation_lengths: A tensor of integers, shape [batch_size].
        max_sequence_length: The maximum sequence length.
    
    Returns:
        A tensor of integers, shape [batch_size, max_sequence_length].
    """
    batch_size = permutation_lengths.size(0)
    result = torch.arange(max_sequence_length, device=permutation_lengths.device).expand(batch_size, max_sequence_length).clone()

    for i in range(batch_size):
        perm_length = permutation_lengths[i].item()
        if perm_length > 0:
            perm = torch.randperm(perm_length, device=permutation_lengths.device)
            result[i, :perm_length] = perm

    return result


def jagged_batch_randperm(permutation_lengths: Tensor, max_sequence_length: int):
    """
    Generates a random permutation of the integers from 0 to max_sequence_length
    for each batch in the batch_size. It's a random permutation of the first
    permutation_lengths[i] integers for the i-th batch, the next index is retained
    for STOP, and the rest is arbitrary (intended for padding).

    Uses agrsort(random_tensor). There is a very small probability of collisions,
    introducing an unknown bias, as we don't ask for a stable argsort.
  
    Arguments:
        permutation_lengths: A tensor of integers, shape [batch_size].
        max_sequence_length: The maximum sequence length.
    
    Returns:
        A tensor of integers, shape [batch_size, max_sequence_length].
    """
    batch_size = permutation_lengths.size(0)
    random_tensor = torch.rand(
        batch_size, max_sequence_length, device=permutation_lengths.device, dtype=torch.float32,
        requires_grad=False)
    # Assign 3. to PAD
    padding_mask = torch.arange(
        max_sequence_length, device=permutation_lengths.device).unsqueeze(0) > permutation_lengths.unsqueeze(1)
    logger.debug("Padding mask size: %s", str(padding_mask.size()))
    # Fails for padding_mask.size()[0] == 0
    # logger.debug("Padding mask #0: %s", str(padding_mask[0]))
    random_tensor[padding_mask] = 3.
    # Assign 2. to retain STOP at every permutation_lengths
    stop_indices = permutation_lengths.unsqueeze(1)
    random_tensor.scatter_(1, stop_indices, 2.0)
    result = torch.argsort(random_tensor, dim=1)
    return result


def batched_bincount(x, dim, max_value, dtype=None):
    if dtype is None:
        dtype = x.dtype
    target = torch.zeros(x.shape[0], max_value, dtype=dtype, device=x.device)
    values = torch.ones(x.size(), dtype=dtype, device=x.device)
    target.scatter_add_(dim, x, values)
    return target


TargetClass = Enum("TargetClass", ['NextToken', 'NumUniqueTokens', 'Scalar'])

class AugmentedCascadeDataset():
    def __init__(
        self,
        data: dict[Tensor|List[Tensor]],
        cascade_order: Tuple[str, ...],
        masks: dict[str, int|Tensor],
        pads: dict[str, int|Tensor],
        stops: dict[str, int|Tensor],
        num_classes: dict[str, int],
        start_field: str,
        augmented_field: str,
        batch_size: Optional[int] = None,
        fix_batch_size: bool = True,
        dtype: torch.dtype = torch.int64,
        start_dtype: torch.dtype = torch.int64,
        device: str = "cpu",
        target_name = None):
        """
        A class for contaning augmented cascade datasets.
        Cascade means that a dataset is a list of tensors, where each tensor is a different field, with
        each subsequent field depending on the previous one. For use in the Wyckoff transformer. Augmented
        means that for up to one field we have multiple possible values, and we want to sample from them.

        Arguments:
            data: A dictionary of tensors (for normal fields) or lists of tensors (for augmented fields),
                with keys being the field names.
            cascade_order: A tuple of field names, in the order they should be passed to the model.
            masks: A dictionary of mask values for each field
            start_field: The name of the field that should be used as the start token.
            augmented_field: The name of the field that should be augmented.
            batch_size: The batch size of the dataset. It is applied before cutting the PAD targets,
                and there will be the end of the dataset, so the actual batches might be smaller.
            dtype: The dtype of the tensors.
            device: The device of the tensors.
        """
        self.device = device
        self.batch_size = batch_size
        self.num_examples = len(data[cascade_order[0]])
        if batch_size is not None:
            if batch_size > self.num_examples:
                raise ValueError("Batch size is larger than the dataset")
            self.this_shuffle_order = torch.randperm(self.num_examples, device=device)
            self.next_batch_index = 0
        self.fix_batch_size = fix_batch_size
        self.cascade_order = cascade_order
        self.cascade_index_from_field = {name: i for i, name in enumerate(cascade_order)}
        self.augmented_field = augmented_field
        self.data = {name: data[name].type(dtype).to(device) for name in cascade_order if name != augmented_field}
        self.max_sequence_length = next(iter(self.data.values())).size(1)
        self.masks = {name: torch.tensor(masks[name], dtype=dtype, device=device) for name in cascade_order}
        self.pads = {name: torch.tensor(pads[name], dtype=dtype, device=device) for name in cascade_order}
        self.stops = {name: torch.tensor(stops[name], dtype=dtype, device=device) for name in cascade_order}
        self.num_classes = tuple((num_classes[name] for name in cascade_order))
        if augmented_field is None:
            self.augmented_field_index = None
        else:
            # Ideally, we would like a nested tensor, but it doesn't support indexing we need
            # So we use a custom data store, and indices to it
            self.augmented_field_index = cascade_order.index(augmented_field)
            self.augmentation_variants = torch.tensor(
                list(map(len, data[f"{augmented_field}_augmented"])), dtype=dtype, device=device)
            self.augmentation_start_indices = (torch.cumsum(self.augmentation_variants, dim=0) -
                self.augmentation_variants) * self.max_sequence_length
            # It will be used in torch.gather
            self.augmentation_data_store = torch.cat([torch.cat(x, dim=0) for x in data[f"{augmented_field}_augmented"]],
                dim=0).type(dtype).to(device).unsqueeze(0).expand(self.augmentation_variants.size(0), -1)
        self.start_tokens = data[start_field].type(start_dtype).to(device)
        if "pure_sequence_length" in data:
            self.pure_sequences_lengths = data["pure_sequence_length"].type(dtype).to(device)
        else:
            # In principle, all the sequences should be 
            # T T T T T STOP PAD PAD
            # with mask not appearing in the data
            # Generated data, however, is not guaranteed to follow this pattern
            # especially since we don't train the model to generate PAD
            is_service_token = (self.data[cascade_order[0]] == self.pads[cascade_order[0]]) | \
                               (self.data[cascade_order[0]] == self.stops[cascade_order[0]]) | \
                               (self.data[cascade_order[0]] == self.masks[cascade_order[0]])
            self.pure_sequences_lengths = torch.max(is_service_token, dim=1).indices
        self.target = None if target_name is None else data[target_name].to(self.device)
        # padding length is the same for all cascade elements
        # prepending 0 to account for the start token
        self.padding_mask = torch.cat(
            (torch.zeros(self.data[cascade_order[0]].shape[0], 1, device=self.device, dtype=bool),
            (self.data[cascade_order[0]] == self.pads[cascade_order[0]])), dim=1)
        if batch_size is None:
            self.batches_per_epoch = 1
        else:
            # Round up, as the last batch might be smaller, but is still a batch
            self.batches_per_epoch = -(-self.num_examples // batch_size)


    def __len__(self):
        return self.num_examples


    # Compilation is safe since the function only ever uses the same data
    @torch.compile(fullgraph=True)
    def get_augmentation(self):
        augnementation_selection = randint_tensor(self.augmentation_variants)
        selection_start = augnementation_selection*self.max_sequence_length + self.augmentation_start_indices
        selection_indices = selection_start.unsqueeze(1) + torch.arange(
            self.max_sequence_length, device=selection_start.device, dtype=selection_start.dtype)
        return torch.gather(self.augmentation_data_store, 1, selection_indices)


    def get_masked_cascade_data(
        self,
        known_seq_len: int,
        known_cascade_len: int):
    
        if self.batch_size is not None:
            raise NotImplementedError("Batch size is not supported for get_masked_cascade_data")
        # assert known_seq_len < data.shape[1]
        # assert known_seq_len >= 0

        if self.augmented_field_index is not None:
            augmented_data = self.get_augmentation()
        if known_cascade_len == self.augmented_field_index:
            target = augmented_data[:, known_seq_len]
        else:
            target = self.data[self.cascade_order[known_cascade_len]][:, known_seq_len]
        target_is_viable = (target != self.pads[self.cascade_order[known_cascade_len]])
        
        # We are predicting the first cascade element also through the mask
        # It would be slightly more efficient to predict it using only the previous data
        res = []
        for cascade_index, name in enumerate(self.cascade_order):
            if name == self.augmented_field:
                cascade_vector = augmented_data[target_is_viable]
            else:
                cascade_vector = self.data[name][target_is_viable]
            if cascade_index < known_cascade_len:
                res.append(cascade_vector[:, :known_seq_len + 1])
            else:
                res.append(torch.cat([
                    cascade_vector[:, :known_seq_len],
                    self.masks[name].expand(cascade_vector.size(0), 1)], dim=1))
        return self.start_tokens[target_is_viable], res, target[target_is_viable]


    def get_next_batch(self) -> Tensor:
        """
        Advances the internal state to the next batch, and returns the indices of the current batch.    
        If self.fix_batch_size is False, the last batch might be smaller than self.batch_size,
        if self.fix_batch_size is True, the last smaller batch will be dropped.
        Shuffles the data if the end of the dataset is reached.
        Returns:
            The indices of the current batch.
        """
        batch_start = self.next_batch_index * self.batch_size
        batch_end = batch_start + self.batch_size
        if batch_end >= self.num_examples:
            self.this_shuffle_order = torch.randperm(self.num_examples, device=self.device)
            self.next_batch_index = 0
            # We have checked during the initialisation that batch_size <= num_examples
            if self.fix_batch_size:
                return self.get_next_batch()
        else:
            self.next_batch_index += 1
        batch_selection = self.this_shuffle_order[batch_start:batch_end]
        logging.debug("The current batch size is %i", len(batch_selection))
        return batch_selection


    def get_augmented_data(self, no_batch: bool) -> Tuple[Tensor, List[Tensor], Tensor, Tensor]:
        """
        Returns the full sequences, using a randon augmentation for the augmented field.
        """
        if self.batch_size is not None and not no_batch:
            batch_selection = self.get_next_batch()
        else:
            batch_selection = slice(None)
        res = []
        for name in self.cascade_order:
            if name == self.augmented_field:
                cascade_vector = self.get_augmentation()
            else:
                cascade_vector = self.data[name]
            res.append(cascade_vector[batch_selection])
        return self.start_tokens[batch_selection], res, self.target[batch_selection], self.padding_mask[batch_selection]


    def get_masked_multiclass_cascade_data(
        self,
        known_seq_len: int,
        known_cascade_len: int,
        target_type: TargetClass,
        multiclass_target: bool,
        no_batch: bool = False,
        augmented_data: Optional[Tensor] = None,
        full_permutation: Optional[Tensor] = None,
        apply_permutation: bool = True,
        truncate_invalid_targets: bool = True,
        return_chosen_indices: bool = False):

        if target_type == TargetClass.NumUniqueTokens:
            if multiclass_target:
                raise ValueError("Multiclass target doesn't make sense for NumUniqueTokens")
            if known_cascade_len != 0:
                # In principle, it's possible, but looks like a waste of compute
                raise NotImplementedError("NumUniqueTokens is only supported when no cascade elements are known")
        
        logger.debug("Known sequence length %i", known_seq_len)
        logger.debug("Known cascade length %i", known_cascade_len)
        if self.augmented_field_index is not None and augmented_data is None:
            augmented_data = self.get_augmentation()

        if self.batch_size is not None and not no_batch:
            if not truncate_invalid_targets:
                raise NotImplementedError("Not truncating invalid targets is not supported for batched data")
            # Duck typing...
            batch_target_is_viable = ()
            # A good research question would be optimising this by removing the invalid targets first,
            # and batching later, but then we'll need to do something to avoid bias
            # towards longer sequences
            while len(batch_target_is_viable) == 0:
                batch_start = self.next_batch_index * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_selection = self.this_shuffle_order[batch_start:batch_end]
                logging.debug("The current batch size is %i", len(batch_selection))
                # STOP is not included in the pure length, but is a viable target
                target_is_viable = self.pure_sequences_lengths[batch_selection] >= known_seq_len
                batch_target_is_viable = batch_selection[target_is_viable]
                if batch_end >= self.num_examples:
                    self.this_shuffle_order = torch.randperm(self.num_examples, device=self.device)
                    self.next_batch_index = 0
                else:
                    self.next_batch_index += 1
        elif truncate_invalid_targets:
            batch_target_is_viable = self.pure_sequences_lengths >= known_seq_len
        else:
            batch_target_is_viable = slice(None)

        chosen_pure_sequences_lengths = self.pure_sequences_lengths[batch_target_is_viable]
        # STOP is still not included in the pure length
        if apply_permutation:
            if full_permutation is None:
                permutation = jagged_batch_randperm(chosen_pure_sequences_lengths, self.max_sequence_length)
            else:
                permutation = full_permutation[batch_target_is_viable]
        logger.debug("Max sequence length %i", self.max_sequence_length)
        # logger.debug("Max queried permutation length %i", chosen_pure_sequences_lengths.max())
        # logger.debug("Min queried permutation length %i", chosen_pure_sequences_lengths.min())
        # logger.debug("Permutation size (%i, %i)", *permutation.size())
        # logger.debug("Permutation #0: %s", permutation[0])
        # Debug
        #for queried_length, this_permutation in zip(chosen_pure_sequences_lengths, permutation):
            #if not (this_permutation[queried_length:] == idenity_permutation[queried_length:]).all():
            #    logger.error("Queried length: %i", queried_length)
            #    logger.error("Permutation: %s", str(this_permutation))
            #    raise ValueError("Permutation is not identity after the end of the sequence")
            #assert (this_permutation[:queried_length] < queried_length).all()
            #assert (this_permutation[queried_length:] >= queried_length).all()
        cascade_result = []
        cascade_targets = []
        for cascade_index, name in enumerate(self.cascade_order):
            logger.debug("Processing cascade #%i aka %s", cascade_index, name)
            if name == self.augmented_field:
                cascade_vector = augmented_data[batch_target_is_viable]
            else:
                cascade_vector = self.data[name][batch_target_is_viable]
            if apply_permutation:
                if cascade_vector.dim() == 2:
                    permuted_cascade_vector = cascade_vector.gather(1, permutation)
                else:
                    expanded_permutation = permutation.unsqueeze(2).expand_as(cascade_vector)
                    permuted_cascade_vector = cascade_vector.gather(1, expanded_permutation)
            else:
                permuted_cascade_vector = cascade_vector
            # logger.debug("Permuted cascade size (%i, %i)", *permuted_cascade_vector.size())
            if cascade_index < known_cascade_len:
                cascade_result.append(permuted_cascade_vector[:, :known_seq_len + 1])
            else:
                if cascade_vector.dim() == 2:
                    mask = self.masks[name].expand(permuted_cascade_vector.size(0), 1)
                else:
                    #print(name)
                    #print(self.masks[name].size())
                    mask = self.masks[name].unsqueeze(0).expand(
                        permuted_cascade_vector.size(0), 1, self.masks[name].size(0))
                    #print(mask.size())
                cascade_result.append(torch.cat([permuted_cascade_vector[:, :known_seq_len], mask], dim=1))
                if target_type == TargetClass.NextToken:
                    if cascade_index == known_cascade_len:
                        if multiclass_target:
                            raw_targets = permuted_cascade_vector[:, known_seq_len:]
                            target_counts = batched_bincount(raw_targets, 1, self.num_classes[known_cascade_len])
                            # PAD is not a valid target
                            target_counts[:, self.pads[name]] = 0
                            # STOP is a valid target only for the last sequence element
                            target_is_not_stop = (known_seq_len != chosen_pure_sequences_lengths)
                            target_counts[target_is_not_stop, self.stops[name]] = 0
                            # Debug
                            # if not (target_counts[~target_is_not_stop, self.stops[name]] == 1).all():
                            #    raise ValueError("STOP missing")
                            # if (target_counts[~target_is_not_stop].sum(dim=1) > 1).any():
                            #    raise ValueError("STOP appears along other targets")
                            # logger.debug("Target counts size (%i, %i)", *target_counts.size())
                            target = target_counts / target_counts.sum(1, keepdim=True)
                            # logger.debug("Target #0: %s", target[0])
                        else:
                            target = permuted_cascade_vector[:, known_seq_len]
                            # Debug:
                            # if (target == self.pads[self.cascade_order[known_cascade_len]]).any():
                            #    raise ValueError("PAD is not a valid target")
                else:
                    # We need the number of unique values, as opposed to values themselves
                    # hence, we use bincount with bool dtype, as 
                    # true + true = true
                    present_classes = batched_bincount(
                        permuted_cascade_vector[:, :known_seq_len], 1,
                        self.num_classes[known_cascade_len], dtype=torch.bool)
                    cascade_targets.append(present_classes.sum(1))
                    # logger.debug("Unique tokens max: %i", cascade_targets[-1].max())
                    # logger.debug("Unique tokens min: %i", cascade_targets[-1].min())
                    # logger.debug("Unique tokens mean: %f", cascade_targets[-1].float().mean())
                    # logger.debug("Unique tokens var aka std**2: %f", cascade_targets[-1].float().var())
                    # logger.debug("No. of unique tokens in %s for #0: %i", name, cascade_targets[-1][0])
        
        if target_type == TargetClass.NumUniqueTokens:
            target = torch.stack(cascade_targets, dim=1)

        if return_chosen_indices:
            return self.start_tokens[batch_target_is_viable], cascade_result, target, batch_target_is_viable
        else:
            return self.start_tokens[batch_target_is_viable], cascade_result, target