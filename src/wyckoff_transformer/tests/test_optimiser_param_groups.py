"""WyckoffTrainer.optimiser_param_groups: the optional no-decay group for gains and biases."""
import torch
from torch import nn

from wyckoff_transformer.trainer import WyckoffTrainer


class _Stub:
    """Just enough of a trainer for optimiser_param_groups: a model and its parameters."""

    def __init__(self):
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))

    def trainable_parameters(self):
        yield from self.model.parameters()


def _one_step_of_pure_decay(exclude: bool):
    """Weights before and after an AdamW step with zero gradients: only decay moves them."""
    stub = _Stub()
    before = [p.detach().clone() for p in stub.model.parameters()]
    optimiser = torch.optim.AdamW(
        WyckoffTrainer.optimiser_param_groups(stub, exclude), lr=0.1, weight_decay=0.5)
    for p in stub.model.parameters():
        p.grad = torch.zeros_like(p)
    optimiser.step()
    return before, [p.detach() for p in stub.model.parameters()]


def test_default_is_one_plain_list_so_existing_checkpoints_still_load():
    stub = _Stub()
    groups = WyckoffTrainer.optimiser_param_groups(stub)
    assert groups == list(stub.model.parameters())
    assert len(torch.optim.AdamW(groups, weight_decay=0.01).param_groups) == 1


def test_gains_and_biases_are_exempt_and_matrices_still_decay():
    groups = WyckoffTrainer.optimiser_param_groups(_Stub(), True)
    # Linear weight | Linear bias, LayerNorm gain, LayerNorm bias
    assert [len(g["params"]) for g in groups] == [1, 3]
    before, after = _one_step_of_pure_decay(exclude=True)
    shrink = 1 - 0.1 * 0.5
    assert torch.allclose(after[0], before[0] * shrink)
    for b, a in zip(before[1:], after[1:]):
        assert torch.equal(a, b)
    before, after = _one_step_of_pure_decay(exclude=False)
    assert torch.allclose(after[2], before[2] * shrink)  # the gain decays when not exempt
