"""Training on N ranks is training on one with the global batch.

The property the distributed trainer is built around: `train_batch_size` is the global
batch, the ranks hold shards of it, and the gradient DDP averages is the gradient one
process would have taken of the mean over the whole of it -- including the steps where
the batch is a whole viable set that does not split evenly, and the ones where some rank
gets nothing. The multi-rank tests run two gloo processes on the CPU.
"""
from datetime import timedelta
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from wyckoff_transformer.cascade.dataset import (
    AugmentedCascadeDataset, AugmentedCascadeLoader, TargetClass, even_shard_bounds)
from wyckoff_transformer.distributed import SINGLE_PROCESS, DistributedContext
from wyckoff_transformer.trainer import CHECKPOINT_FILENAME, WyckoffTrainer
from wyckoff_transformer.tests.test_training_resume import (
    N_CLASSES, _TinyModel, _crash_after, _make_trainer, _run)

WORLD_SIZE = 2
SHARED_SEED = 20260917
MAX_SEQ = 6
PAD, MASK, STOP = 5, 6, 7


def _constant_row_dataset(lengths, batch_size) -> AugmentedCascadeDataset:
    """Every real token of a row is the same, so the multiclass target ignores the order.

    That makes a single-process reference loss computable over any set of rows, whatever
    permutation each rank drew for its own.
    """
    rows = []
    for index, length in enumerate(lengths):
        row = torch.full((MAX_SEQ,), PAD, dtype=torch.int64)
        row[:length] = index % 5
        row[length] = STOP
        rows.append(row)
    n = len(lengths)
    data = {"field1": torch.stack(rows),
            "spacegroup": torch.arange(n, dtype=torch.int64) % 3,
            "pure_sequence_length": torch.tensor(lengths, dtype=torch.int64)}
    return AugmentedCascadeDataset(
        data=data, cascade_order=("field1",), masks={"field1": MASK}, pads={"field1": PAD},
        stops={"field1": STOP}, num_classes={"field1": N_CLASSES}, start_field="spacegroup",
        augmented_fields=None, batch_size=batch_size)


#: Viable counts by known_seq_len: 16, 16, 10, 6, 3, 1. With a global batch of 4 on two
#: ranks, 0-3 are sampled and split evenly, 4 is a whole viable set split 2/1, and 5 is a
#: single example that leaves rank 1 with nothing.
LENGTHS = [1] * 6 + [2] * 4 + [3] * 3 + [4] * 2 + [5]


def _sharded_loaders(dataset, batch_size=None):
    return [AugmentedCascadeLoader(dataset, batch_size=batch_size or dataset._batch_size,
                                   rank=rank, world_size=WORLD_SIZE, seed=SHARED_SEED)
            for rank in range(WORLD_SIZE)]


class TestEvenShardBounds(unittest.TestCase):
    def test_the_shards_partition_the_range(self):
        for n_items in range(12):
            for world_size in (1, 2, 3, 4):
                bounds = [even_shard_bounds(n_items, world_size, rank)
                          for rank in range(world_size)]
                covered = [i for start, stop in bounds for i in range(start, stop)]
                self.assertEqual(covered, list(range(n_items)))
                sizes = [stop - start for start, stop in bounds]
                self.assertLessEqual(max(sizes) - min(sizes), 1)


class TestShardedLoader(unittest.TestCase):
    def setUp(self):
        self.dataset = _constant_row_dataset(LENGTHS, batch_size=4)

    def test_sampled_shards_are_the_single_generator_draw_split(self):
        loaders = _sharded_loaders(self.dataset)
        reference = torch.Generator().manual_seed(SHARED_SEED)
        # The loader drew its first shuffle order from the same stream when it was built.
        torch.randperm(len(LENGTHS), generator=reference)
        for known_seq_len in (0, 1, 2, 3) * 5:
            positions = self.dataset.draw_viable_positions(known_seq_len, 4, generator=reference)
            expected = self.dataset.length_sorted_indices[positions]
            shards = [loader.get_next_viable_batch(known_seq_len) for loader in loaders]
            self.assertEqual([len(shard) for shard in shards], [2, 2])
            torch.testing.assert_close(torch.cat(shards), expected, rtol=0, atol=0)
            for loader in loaders:
                self.assertEqual(loader.last_batch_share, 2)
                self.assertFalse(loader.last_batch_is_filler)

    def test_a_whole_viable_set_is_split_unevenly_and_weighted_by_share(self):
        loaders = _sharded_loaders(self.dataset)
        shards = [loader.get_next_viable_batch(4) for loader in loaders]
        self.assertEqual([len(shard) for shard in shards], [2, 1])
        self.assertEqual(sorted(torch.cat(shards).tolist()),
                         sorted(self.dataset.length_sorted_indices[:3].tolist()))
        self.assertEqual([loader.last_batch_share for loader in loaders], [1.5, 1.5])

    def test_a_rank_with_an_empty_share_gets_a_flagged_stand_in(self):
        loaders = _sharded_loaders(self.dataset)
        shards = [loader.get_next_viable_batch(5) for loader in loaders]
        self.assertEqual([len(shard) for shard in shards], [1, 1])
        self.assertEqual([loader.last_batch_is_filler for loader in loaders], [False, True])
        self.assertEqual([loader.last_batch_share for loader in loaders], [0.5, 0.5])

    def test_shuffled_batches_partition_an_epoch(self):
        loaders = _sharded_loaders(self.dataset)
        seen = []
        for _ in range(loaders[0].batches_per_epoch):
            shards = [loader.get_next_batch() for loader in loaders]
            self.assertEqual([len(shard) for shard in shards], [2, 2])
            seen.extend(torch.cat(shards).tolist())
        self.assertEqual(sorted(seen), list(range(len(LENGTHS))))

    def test_a_restored_loader_continues_the_same_draws(self):
        loader, = _sharded_loaders(self.dataset)[:1]
        for _ in range(3):
            loader.get_next_batch()
            loader.get_next_viable_batch(2)
        state = loader.state_dict()
        expected = [(loader.get_next_batch(), loader.get_next_viable_batch(1)) for _ in range(6)]
        restored = AugmentedCascadeLoader(
            self.dataset, batch_size=4, rank=0, world_size=WORLD_SIZE, seed=SHARED_SEED + 1)
        restored.load_state_dict(state)
        for batch, viable in expected:
            torch.testing.assert_close(restored.get_next_batch(), batch, rtol=0, atol=0)
            torch.testing.assert_close(restored.get_next_viable_batch(1), viable, rtol=0, atol=0)

    def test_unsupported_configurations_are_refused(self):
        with self.assertRaisesRegex(ValueError, "needs a batch size"):
            AugmentedCascadeLoader(self.dataset, None, rank=0, world_size=2, seed=1)
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            AugmentedCascadeLoader(self.dataset, 5, rank=0, world_size=2, seed=1)
        with self.assertRaises(NotImplementedError):
            AugmentedCascadeLoader(self.dataset, 4, fix_batch_size=False,
                                   rank=0, world_size=2, seed=1)

    def test_a_single_process_loader_is_not_sharded(self):
        loader = AugmentedCascadeLoader.from_dataset(self.dataset)
        self.assertFalse(loader.sharded)
        self.assertNotIn("generator", loader.state_dict())


def _join(rank: int, out_dir: str) -> DistributedContext:
    # A file rendezvous rather than TCP: no port to race for, and some torch builds lack the
    # libuv the TCP store asks for by default.
    dist.init_process_group(
        "gloo", init_method=f"file://{Path(out_dir) / 'rendezvous'}", rank=rank,
        world_size=WORLD_SIZE, timeout=timedelta(seconds=120))
    context = DistributedContext(rank=rank, local_rank=rank, world_size=WORLD_SIZE,
                                 shared_seed=SHARED_SEED, backend="gloo")
    torch.manual_seed(context.rank_seed())
    return context


def _distribute(trainer: WyckoffTrainer, context: DistributedContext) -> WyckoffTrainer:
    """What `WyckoffTrainer.__init__` does for a DDP run, applied to a test skeleton."""
    trainer.distributed = context
    torch.manual_seed(context.rank_seed())
    trainer.step_rng = random.Random(context.shared_seed)
    trainer.train_loader = AugmentedCascadeLoader.from_dataset(
        trainer.train_dataset, rank=context.rank, world_size=context.world_size,
        seed=context.shared_seed)
    trainer.ddp_model = DistributedDataParallel(trainer.model, find_unused_parameters=True)
    return trainer


def _spawn(worker, out_dir: str):
    mp.start_processes(worker, args=(out_dir,), nprocs=WORLD_SIZE,
                       start_method="spawn", join=True)


def _gradient_worker(rank: int, out_dir: str):
    """Record, per step, each rank's batch and the gradient the optimiser was handed."""
    context = _join(rank, out_dir)
    try:
        torch.manual_seed(0)
        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.target = TargetClass.NextToken
        trainer.multiclass_next_token_with_order_permutation = True
        trainer.condition_feature = None
        trainer.cascade_target_count = 1
        trainer.cascade_target_indices = (0,)
        trainer.cascade_order = ("field1",)
        trainer.device = torch.device("cpu")
        trainer.clip_grad_norm = None
        trainer.criterion = nn.CrossEntropyLoss(reduction="sum")
        trainer.model = _TinyModel()
        trainer.train_dataset = _constant_row_dataset(LENGTHS, batch_size=4)
        # lr=0: the weights stay where the reference computes its gradients.
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.)
        trainer.scheduler = None
        _distribute(trainer, context)

        # Every known_seq_len, including the uneven and the empty shard, in a fixed cycle.
        cycle = iter(list(range(MAX_SEQ)) * 4)
        trainer.train_dataset.sample_known_seq_len = lambda rng=None: next(cycle)
        records = []
        draw = trainer.train_loader.get_next_viable_batch

        def recording_draw(known_seq_len):
            indices = draw(known_seq_len)
            records.append({"known_seq_len": known_seq_len, "indices": indices.clone(),
                            "filler": trainer.train_loader.last_batch_is_filler})
            return indices

        trainer.train_loader.get_next_viable_batch = recording_draw
        step = trainer.optimizer.step

        def recording_step():
            records[-1]["grads"] = [p.grad.clone() for p in trainer.model.parameters()]
            step()

        trainer.optimizer.step = recording_step
        with patch("wyckoff_transformer.trainer.wandb"):
            for _ in range(5):
                trainer.train_epoch()
        torch.save({"records": records, "weights": trainer.model.state_dict()},
                   Path(out_dir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


class TestDistributedGradients(unittest.TestCase):
    def test_the_averaged_gradient_is_that_of_the_global_batch(self):
        with tempfile.TemporaryDirectory() as out_dir:
            _spawn(_gradient_worker, out_dir)
            ranks = [torch.load(Path(out_dir) / f"rank{rank}.pt", weights_only=True)
                     for rank in range(WORLD_SIZE)]

        dataset = _constant_row_dataset(LENGTHS, batch_size=4)
        reference_model = _TinyModel()
        reference_model.load_state_dict(ranks[0]["weights"])
        criterion = nn.CrossEntropyLoss(reduction="sum")
        steps = list(zip(*(rank["records"] for rank in ranks)))
        self.assertEqual(len(steps), 20)
        covered = set()
        for step in steps:
            known_seq_len = step[0]["known_seq_len"]
            self.assertTrue(all(r["known_seq_len"] == known_seq_len for r in step))
            union = torch.cat([r["indices"] for r in step if not r["filler"]])
            self.assertEqual(len(union), len(set(union.tolist())), "the shards overlap")
            self.assertEqual(len(union), min(4, dataset.viable_count(known_seq_len)))
            covered.add((known_seq_len, tuple(len(r["indices"]) for r in step),
                         tuple(r["filler"] for r in step)))

            reference_model.zero_grad()
            start, masked, target = dataset.get_masked_multiclass_cascade_data(
                known_seq_len, 0, TargetClass.NextToken, multiclass_target=True,
                batch_target_is_viable=union)
            loss = criterion(reference_model(start, masked, None, 0), target) / len(union)
            loss.backward()
            for rank_record in step:
                for got, parameter in zip(rank_record["grads"], reference_model.parameters()):
                    torch.testing.assert_close(got, parameter.grad, rtol=1e-5, atol=1e-6)
        # The cases the weighting exists for were actually exercised.
        self.assertIn((4, (2, 1), (False, False)), covered)
        self.assertIn((5, (1, 1), (False, True)), covered)


class _ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 1)

    def forward(self, start_tokens, masked_data, padding_mask, known_cascade_len, cond=None):
        return self.head(start_tokens.float().unsqueeze(-1))


def _scalar_dataset() -> AugmentedCascadeDataset:
    lengths = LENGTHS
    dataset = _constant_row_dataset(lengths, batch_size=8)
    generator = torch.Generator().manual_seed(3)
    data = {"field1": dataset.data["field1"].clone(),
            "spacegroup": torch.arange(len(lengths), dtype=torch.int64) % 7,
            "pure_sequence_length": torch.tensor(lengths, dtype=torch.int64),
            "energy": torch.randn(len(lengths), generator=generator)}
    return AugmentedCascadeDataset(
        data=data, cascade_order=("field1",), masks={"field1": MASK}, pads={"field1": PAD},
        stops={"field1": STOP}, num_classes={"field1": N_CLASSES}, start_field="spacegroup",
        augmented_fields=None, batch_size=8, target_name="energy")


def _scalar_gradient_worker(rank: int, out_dir: str):
    """The Scalar target: shuffled batches rather than viable ones, and a mean loss."""
    context = _join(rank, out_dir)
    try:
        torch.manual_seed(0)
        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.target = TargetClass.Scalar
        trainer.multiclass_next_token_with_order_permutation = False
        trainer.condition_feature = None
        trainer.device = torch.device("cpu")
        trainer.clip_grad_norm = None
        trainer.criterion = nn.MSELoss(reduction="mean")
        trainer.model = _ScalarModel()
        trainer.train_dataset = _scalar_dataset()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.)
        trainer.scheduler = None
        _distribute(trainer, context)
        records = []
        draw = trainer.train_loader.get_next_batch

        def recording_draw():
            indices = draw()
            records.append({"indices": indices.clone()})
            return indices

        trainer.train_loader.get_next_batch = recording_draw
        step = trainer.optimizer.step

        def recording_step():
            records[-1]["grads"] = [p.grad.clone() for p in trainer.model.parameters()]
            step()

        trainer.optimizer.step = recording_step
        with patch("wyckoff_transformer.trainer.wandb"):
            for _ in range(3):
                trainer.train_epoch()
        torch.save({"records": records, "weights": trainer.model.state_dict()},
                   Path(out_dir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


class TestDistributedScalarGradients(unittest.TestCase):
    def test_the_averaged_gradient_is_that_of_the_global_batch(self):
        with tempfile.TemporaryDirectory() as out_dir:
            _spawn(_scalar_gradient_worker, out_dir)
            ranks = [torch.load(Path(out_dir) / f"rank{rank}.pt", weights_only=True)
                     for rank in range(WORLD_SIZE)]
        dataset = _scalar_dataset()
        reference_model = _ScalarModel()
        reference_model.load_state_dict(ranks[0]["weights"])
        steps = list(zip(*(rank["records"] for rank in ranks)))
        self.assertEqual(len(steps), 3 * 2)
        epoch = []
        for step in steps:
            union = torch.cat([r["indices"] for r in step])
            self.assertEqual(len(union), 8)
            epoch.extend(union.tolist())
            reference_model.zero_grad()
            prediction = reference_model(dataset.start_tokens[union], None, None, None).squeeze()
            nn.MSELoss()(prediction, dataset.target[union]).backward()
            for rank_record in step:
                for got, parameter in zip(rank_record["grads"], reference_model.parameters()):
                    torch.testing.assert_close(got, parameter.grad, rtol=1e-5, atol=1e-6)
        # Each epoch of two global batches visits every example once.
        self.assertEqual(sorted(epoch[:16]), list(range(16)))


def _resume_worker(rank: int, out_dir: str):
    """An uninterrupted two-rank run, and one that crashes and resumes."""
    context = _join(rank, out_dir)
    root = Path(out_dir)
    try:
        def make(path, resume=False):
            trainer = _make_trainer(path, epochs=6, resume=resume)
            return _distribute(trainer, context)

        uninterrupted = make(root / "uninterrupted")
        _run(uninterrupted)

        crashed = make(root / "crashed")
        _crash_after(crashed, 3)
        try:
            _run(crashed)
        except RuntimeError:
            pass
        else:
            raise AssertionError("the crash did not happen")
        resumed = make(root / "crashed", resume=True)
        _run(resumed)
        torch.save({"uninterrupted": uninterrupted.model.state_dict(),
                    "resumed": resumed.model.state_dict()}, root / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


class TestDistributedResume(unittest.TestCase):
    def test_a_resumed_run_matches_the_uninterrupted_one_on_every_rank(self):
        with tempfile.TemporaryDirectory() as out_dir:
            root = Path(out_dir)
            for name in ("uninterrupted", "crashed"):
                (root / name).mkdir()
            _spawn(_resume_worker, out_dir)
            weights = [torch.load(root / f"rank{rank}.pt", weights_only=True)
                       for rank in range(WORLD_SIZE)]
            checkpoint = torch.load(root / "crashed" / CHECKPOINT_FILENAME, weights_only=True)
            written = sorted(path.name for path in (root / "crashed").iterdir())

        self.assertEqual(len(checkpoint["per_rank"]), WORLD_SIZE)
        self.assertIn("generator", checkpoint["loaders"]["train"])
        # Rank 0 alone writes: two ranks racing through atomic_torch_save share a .tmp name.
        self.assertIn("best_model_params.pt", written)
        self.assertFalse([name for name in written if name.endswith(".tmp")], written)
        for key, value in weights[0]["uninterrupted"].items():
            for rank in range(WORLD_SIZE):
                # The ranks agree, and the resumed run is the uninterrupted one exactly.
                torch.testing.assert_close(
                    weights[rank]["uninterrupted"][key], value, rtol=0, atol=0)
                torch.testing.assert_close(weights[rank]["resumed"][key], value, rtol=0, atol=0)


class TestSingleProcessContext(unittest.TestCase):
    def test_collectives_are_identities(self):
        tensor = torch.arange(3.)
        self.assertIs(SINGLE_PROCESS.all_reduce_mean(tensor), tensor)
        self.assertEqual(SINGLE_PROCESS.broadcast_object("x"), "x")
        self.assertEqual(SINGLE_PROCESS.all_gather_object("x"), ["x"])
        self.assertTrue(SINGLE_PROCESS.is_main)
        self.assertFalse(SINGLE_PROCESS.enabled)

    def test_the_trainer_defaults_to_one_process(self):
        self.assertIs(WyckoffTrainer.distributed, SINGLE_PROCESS)
        self.assertIsNone(WyckoffTrainer.ddp_model)
        self.assertIs(WyckoffTrainer.step_rng, random)


class _TinyPredictStartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, N_CLASSES)
        self.start_prediction_head = nn.Linear(1, 3)
        self.start_query = nn.Parameter(torch.randn(1))
        self.predict_start = True

    def forward(self, start_tokens, masked_data, padding_mask, known_cascade_len, cond=None,
                start_cond=None, start_batch_size=None):
        out = self.head(start_tokens.float().unsqueeze(-1))
        if start_cond is not None or start_batch_size is not None:
            n_start = start_cond.size(0) if start_cond is not None else start_batch_size
            start_pred = self.forward_start(n_start, cond=start_cond)
            return out, start_pred
        return out

    def forward_start(self, batch_size, cond=None):
        return self.start_prediction_head(self.start_query.expand(batch_size, 1))


def _predict_start_worker(rank: int, out_dir: str):
    context = _join(rank, out_dir)
    try:
        torch.manual_seed(0)
        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.target = TargetClass.NextToken
        trainer.multiclass_next_token_with_order_permutation = True
        trainer.predict_start = True
        trainer.start_loss_weight = 1.0
        trainer.condition_feature = None
        trainer.cascade_target_count = 1
        trainer.cascade_target_indices = (0,)
        trainer.cascade_order = ("field1",)
        trainer.device = torch.device("cpu")
        trainer.clip_grad_norm = None
        trainer.criterion = nn.CrossEntropyLoss(reduction="sum")
        trainer.model = _TinyPredictStartModel()
        trainer.train_dataset = _constant_row_dataset(LENGTHS, batch_size=4)
        trainer.train_dataset.start_classes = trainer.train_dataset.start_tokens
        trainer.build_cond = lambda dataset, selection, **kwargs: None
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        trainer.scheduler = None
        trainer.scheduler_steps_per_batch = False
        trainer.trainable_parameters = lambda: trainer.model.parameters()
        _distribute(trainer, context)
        with patch("wyckoff_transformer.trainer.wandb"):
            for _ in range(2):
                trainer.train_epoch()
        torch.save(trainer.model.state_dict(), Path(out_dir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


class TestDistributedPredictStart(unittest.TestCase):
    def test_predict_start_trains_under_ddp(self):
        with tempfile.TemporaryDirectory() as out_dir:
            _spawn(_predict_start_worker, out_dir)
            weights = [torch.load(Path(out_dir) / f"rank{rank}.pt", weights_only=True)
                       for rank in range(WORLD_SIZE)]
            for key, val in weights[0].items():
                torch.testing.assert_close(weights[1][key], val)


if __name__ == "__main__":
    unittest.main()
