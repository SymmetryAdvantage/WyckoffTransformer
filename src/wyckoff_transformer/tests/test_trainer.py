import unittest
from types import SimpleNamespace

import torch
from unittest.mock import patch, MagicMock

from ..cascade.dataset import TargetClass
from ..trainer import WyckoffTrainer, cascade_target_indices

# Intentionally tests a private helper to validate serialized distribution semantics.
# pylint: disable=protected-access


class TestCascadeTargetIndices(unittest.TestCase):
    """Which cascade positions get a head, and where an engineer-filled field may sit."""
    ORDER = ("elements", "site_symmetries", "site_symmetry_ops_id", "sites_enumeration")
    IS_TARGET = {"elements": True, "site_symmetries": True,
                 "site_symmetry_ops_id": False, "sites_enumeration": True}

    @staticmethod
    def _engineers(*inputs):
        return {"site_symmetry_ops_id": SimpleNamespace(inputs=list(inputs))}

    def test_targets_need_not_be_a_prefix(self):
        self.assertEqual(
            cascade_target_indices(
                self.ORDER, self.IS_TARGET,
                self._engineers("spacegroup_number", "site_symmetries")),
            (0, 1, 3))

    def test_rejects_a_field_filled_from_a_later_one(self):
        order = ("elements", "site_symmetry_ops_id", "site_symmetries", "sites_enumeration")
        is_target = dict(self.IS_TARGET)
        with self.assertRaisesRegex(ValueError, "comes later in the cascade"):
            cascade_target_indices(
                order, is_target, self._engineers("spacegroup_number", "site_symmetries"))

    def test_rejects_a_non_target_with_nothing_to_fill_it(self):
        with self.assertRaisesRegex(ValueError, "no engineer"):
            cascade_target_indices(self.ORDER, self.IS_TARGET, {})

    def test_rejects_a_cascade_with_no_targets(self):
        with self.assertRaisesRegex(ValueError, "at least one|At least one"):
            cascade_target_indices(
                ("site_symmetry_ops_id",), {"site_symmetry_ops_id": False},
                self._engineers("spacegroup_number"))

    def test_an_all_target_cascade_is_every_position(self):
        order = ("elements", "site_symmetries", "sites_enumeration")
        self.assertEqual(
            cascade_target_indices(order, {f: True for f in order}, {}), (0, 1, 2))


class TestBuildStartTokenDistribution(unittest.TestCase):
    def test_build_start_token_distribution_categorial(self):
        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.train_dataset = SimpleNamespace(
            start_tokens=torch.tensor([0, 2, 2], dtype=torch.int64)
        )
        trainer.val_dataset = SimpleNamespace(
            start_tokens=torch.tensor([1, 2], dtype=torch.int64)
        )
        trainer.model = SimpleNamespace(
            start_type="categorial",
            start_embedding=SimpleNamespace(num_embeddings=5),
        )
        trainer.start_name = "spacegroup_number"
        trainer.max_sequence_length = 13
        trainer.production_training = False

        distribution = trainer._build_start_token_distribution()

        self.assertEqual(distribution["start_name"], "spacegroup_number")
        self.assertEqual(distribution["start_type"], "categorial")
        self.assertEqual(distribution["max_sequence_length"], 13)
        self.assertEqual(distribution["counts"], [1, 1, 3, 0, 0])

    def test_build_start_token_distribution_one_hot(self):
        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.train_dataset = SimpleNamespace(
            start_tokens=torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        trainer.val_dataset = SimpleNamespace(
            start_tokens=torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )
        trainer.model = SimpleNamespace(start_type="one_hot")
        trainer.start_name = "spacegroup_number"
        trainer.max_sequence_length = 7
        trainer.production_training = False

        distribution = trainer._build_start_token_distribution()

        self.assertEqual(distribution["start_name"], "spacegroup_number")
        self.assertEqual(distribution["start_type"], "one_hot")
        self.assertEqual(distribution["max_sequence_length"], 7)

        vector_count_map = {
            tuple(vector): count
            for vector, count in zip(distribution["vectors"], distribution["counts"])
        }
        self.assertEqual(vector_count_map[(1.0, 0.0, 0.0)], 2)
        self.assertEqual(vector_count_map[(0.0, 1.0, 0.0)], 2)
        self.assertEqual(vector_count_map[(0.0, 0.0, 1.0)], 1)

class TestWyckoffTrainerGeneration(unittest.TestCase):
    def setUp(self):
        self.trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        self.trainer.model = MagicMock()
        self.trainer.model.start_type = "categorial"
        self.trainer.model.start_embedding.num_embeddings = 5
        self.trainer.cascade_order = ["spacegroup", "harmonic_site_symmetries"]
        self.trainer.cascade_is_target = {"spacegroup": True}
        self.trainer.token_engineers = {}
        self.trainer.tokenisers = {'elements': MagicMock()}
        self.trainer.masks_dict = {}
        self.trainer.stops_dict = {"spacegroup": 7}
        self.trainer.start_name = "spacegroup_number"
        self.trainer.start_token_distribution = None
        
        self.trainer.train_dataset = MagicMock()
        self.trainer.train_dataset.start_tokens = torch.tensor([0, 1])
        self.trainer.train_dataset.masks = {}
        self.trainer.train_dataset.max_sequence_length = 10
        
        self.trainer.val_dataset = MagicMock()
        self.trainer.val_dataset.start_tokens = torch.tensor([1, 2])
        self.trainer.max_sequence_length = 10
        self.trainer.device = torch.device("cpu")
        self.trainer.processor = MagicMock()
        self.trainer.production_training = False
        self.trainer.condition_feature = None
        
        # Setup processor to return a mock pyxtal-like string/object
        self.trainer.processor.tensor_to_pyxtal.return_value = "pyxtal_mock"
        self.trainer.run_path = None

    @patch("wyckoff_transformer.trainer.WyckoffGenerator")
    @patch("wyckoff_transformer.trainer.load_wyckoff_mappings")
    @patch("wyckoff_transformer.trainer.get_wp_index")
    def test_generate_structures(self, mock_get_wp_index, mock_load_wyckoff_mappings, MockWyckoffGenerator):
        # Setup mocks
        mock_generator_instance = MockWyckoffGenerator.return_value
        # generated tensors: mock what generator.generate_tensors returns
        mock_generator_instance.generate_tensors.return_value = [torch.zeros((2, 5)), torch.ones((2, 5))]
        mock_load_wyckoff_mappings.return_value.ss_from_letter = "ss_from_letter_mock"

        # Test basic generation
        structures = self.trainer.generate_structures(
            n_structures=2,
            calibrate=False,
            compute_validity_per_known_sequence_length=False
        )
        
        # Assertions
        mock_generator_instance.generate_tensors.assert_called_once()
        # Without `stops` the generator cannot tell a finished sequence from a live one, and
        # the per-length validity metric degenerates into the stop rate.
        self.assertEqual(
            MockWyckoffGenerator.call_args.kwargs.get("stops"), self.trainer.stops_dict)
        self.assertEqual(len(structures), 2)
        self.assertEqual(structures[0], "pyxtal_mock")
        # Ensure that harmonic_site_symmetries is deleted from tensors during processing 
        # (which means tensor_to_pyxtal is called with 1-element cascade_order)
        self.assertTrue(self.trainer.processor.tensor_to_pyxtal.called)

    @patch("wyckoff_transformer.trainer.WyckoffGenerator")
    @patch("wyckoff_transformer.trainer.load_wyckoff_mappings")
    @patch("wyckoff_transformer.trainer.get_wp_index")
    def test_trailing_non_target_field_is_dropped_by_role_not_by_name(
            self, mock_get_wp_index, mock_load_wyckoff_mappings, MockWyckoffGenerator):
        """The trailing field the engineer fills in is an input, not part of the decoded
        structure. Any non-target field qualifies -- site_symmetry_ops_id as much as
        harmonic_site_symmetries -- so the drop keys off is_target."""
        self.trainer.cascade_order = ["spacegroup", "site_symmetry_ops_id"]
        self.trainer.cascade_is_target = {"spacegroup": True, "site_symmetry_ops_id": False}
        mock_generator_instance = MockWyckoffGenerator.return_value
        mock_generator_instance.generate_tensors.return_value = [
            torch.zeros((2, 5)), torch.ones((2, 5))]
        mock_load_wyckoff_mappings.return_value.ss_from_letter = "ss_from_letter_mock"

        self.trainer.generate_structures(
            n_structures=2, calibrate=False,
            compute_validity_per_known_sequence_length=False)

        kwargs = self.trainer.processor.tensor_to_pyxtal.call_args.kwargs
        self.assertEqual(kwargs["cascade_order"], ("spacegroup",))

    @patch("wyckoff_transformer.trainer.WyckoffGenerator")
    @patch("wyckoff_transformer.trainer.load_wyckoff_mappings")
    @patch("wyckoff_transformer.trainer.get_wp_index")
    def test_all_target_cascade_keeps_every_field(
            self, mock_get_wp_index, mock_load_wyckoff_mappings, MockWyckoffGenerator):
        self.trainer.cascade_order = ["spacegroup", "elements"]
        self.trainer.cascade_is_target = {"spacegroup": True, "elements": True}
        mock_generator_instance = MockWyckoffGenerator.return_value
        mock_generator_instance.generate_tensors.return_value = [
            torch.zeros((2, 5)), torch.ones((2, 5))]
        mock_load_wyckoff_mappings.return_value.ss_from_letter = "ss_from_letter_mock"

        self.trainer.generate_structures(
            n_structures=2, calibrate=False,
            compute_validity_per_known_sequence_length=False)

        kwargs = self.trainer.processor.tensor_to_pyxtal.call_args.kwargs
        self.assertEqual(kwargs["cascade_order"], ("spacegroup", "elements"))

    @patch("wyckoff_transformer.trainer.WyckoffGenerator")
    @patch("wyckoff_transformer.trainer.load_wyckoff_mappings")
    @patch("wyckoff_transformer.trainer.get_wp_index")
    def test_a_non_target_field_in_the_middle_is_dropped(
            self, mock_get_wp_index, mock_load_wyckoff_mappings, MockWyckoffGenerator):
        """A field the engineer fills in belongs directly after its inputs, which can be
        the middle of the cascade. Decoding must drop it wherever it sits."""
        self.trainer.cascade_order = ["spacegroup", "site_symmetry_ops_id", "elements"]
        self.trainer.cascade_is_target = {
            "spacegroup": True, "site_symmetry_ops_id": False, "elements": True}
        mock_generator_instance = MockWyckoffGenerator.return_value
        mock_generator_instance.generate_tensors.return_value = [
            torch.zeros((2, 5)), torch.full((2, 5), 9.), torch.ones((2, 5))]
        mock_load_wyckoff_mappings.return_value.ss_from_letter = "ss_from_letter_mock"

        self.trainer.generate_structures(
            n_structures=2, calibrate=False,
            compute_validity_per_known_sequence_length=False)

        kwargs = self.trainer.processor.tensor_to_pyxtal.call_args.kwargs
        self.assertEqual(kwargs["cascade_order"], ("spacegroup", "elements"))
        # The dropped column carried 9; what survives is the two target columns.
        passed = self.trainer.processor.tensor_to_pyxtal.call_args.args[1]
        self.assertEqual(tuple(passed.shape), (5, 2))
        self.assertFalse((passed == 9.).any())

    @patch("wyckoff_transformer.trainer.WyckoffGenerator")
    @patch("wyckoff_transformer.trainer.load_wyckoff_mappings")
    @patch("wyckoff_transformer.trainer.get_wp_index")
    def test_generate_element_constrained_structures(self, mock_get_wp_index, mock_load_wyckoff_mappings, MockWyckoffGenerator):
        # Setup mocks
        mock_generator_instance = MockWyckoffGenerator.return_value
        mock_generator_instance.generate_tensors.return_value = [torch.zeros((2, 5)), torch.ones((2, 5))]
        mock_load_wyckoff_mappings.return_value.ss_from_letter = "ss_from_letter_mock"

        start_tensor = torch.tensor([0, 1])

        structures = self.trainer.generate_structures(
            n_structures=2,
            calibrate=False,
            required_element_set="Li-O",
            allowed_element_set="all",
            start_tensor=start_tensor
        )
        
        # Assertions
        mock_generator_instance.generate_tensors.assert_called_once()
        call_kwargs = mock_generator_instance.generate_tensors.call_args[1]
        self.assertEqual(call_kwargs["required_element_set"], "Li-O")
        self.assertEqual(call_kwargs["allowed_element_set"], "all")
        self.assertEqual(len(structures), 2)
        self.assertEqual(structures[0], "pyxtal_mock")

    @patch("wyckoff_transformer.trainer.WyckoffGenerator")
    @patch("wyckoff_transformer.trainer.load_wyckoff_mappings")
    @patch("wyckoff_transformer.trainer.get_wp_index")
    def test_generate_allowed_elements_only(self, mock_get_wp_index, mock_load_wyckoff_mappings, MockWyckoffGenerator):
        """--allowed-elements without --required-elements: required_element_set=set(), allowed_element_set forwarded."""
        mock_generator_instance = MockWyckoffGenerator.return_value
        mock_generator_instance.generate_tensors.return_value = [torch.zeros((2, 5)), torch.ones((2, 5))]
        mock_load_wyckoff_mappings.return_value.ss_from_letter = "ss_from_letter_mock"

        structures = self.trainer.generate_structures(
            n_structures=2,
            calibrate=False,
            required_element_set=set(),
            allowed_element_set="Li-O-P",
            start_tensor=torch.tensor([0, 1]),
        )

        mock_generator_instance.generate_tensors.assert_called_once()
        call_kwargs = mock_generator_instance.generate_tensors.call_args[1]
        self.assertEqual(call_kwargs["required_element_set"], set())
        self.assertEqual(call_kwargs["allowed_element_set"], "Li-O-P")
        self.assertEqual(len(structures), 2)


class _CountingDataset:
    """The only things evaluate() asks of a dataset, with every example viable everywhere."""
    def __init__(self, max_sequence_length: int, n_structures: int):
        self.max_sequence_length = max_sequence_length
        self._n_structures = n_structures

    def __len__(self):
        return self._n_structures

    def viable_count(self, known_seq_len: int) -> int:  # pylint: disable=unused-argument
        return self._n_structures


class TestEvaluateReportsOnlyTargets(unittest.TestCase):
    """A non-target field has no head, so it must get no entry in the loss vector.

    It used to get a permanently-zero one, which reached wandb as
    `loss.epoch.<split>.site_symmetry_ops_id = 0` and is indistinguishable from a head that
    has collapsed."""
    ORDER = ("elements", "site_symmetries", "site_symmetry_ops_id", "sites_enumeration")
    TARGET_INDICES = (0, 1, 3)

    def _trainer(self):
        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.target = TargetClass.NextToken
        trainer.model = MagicMock()
        # Not a schedule-free optimiser: evaluate() only calls .eval() when there is one.
        trainer.optimizer = None
        trainer.evaluation_samples = 1
        trainer.device = torch.device("cpu")
        trainer.cascade_order = self.ORDER
        trainer.cascade_len = len(self.ORDER)
        trainer.cascade_target_indices = self.TARGET_INDICES
        trainer.cascade_target_count = len(self.TARGET_INDICES)
        # One unit of loss per cascade position, so an entry landing in the wrong slot shows up
        # as a wrong value rather than as a coincidence.
        trainer.get_loss = lambda dataset, known_seq_len, known_cascade_len, **kwargs: \
            torch.tensor(float(known_cascade_len) + 1.)
        return trainer

    def test_the_loss_vector_has_one_entry_per_target(self):
        trainer = self._trainer()
        loss = trainer.evaluate(_CountingDataset(max_sequence_length=2, n_structures=1))
        self.assertEqual(tuple(loss.shape), (3,))
        # Two known_seq_len passes, each contributing known_cascade_len + 1.
        torch.testing.assert_close(loss, torch.tensor([2., 4., 8.]))

    def test_the_non_target_field_is_not_labelled(self):
        trainer = self._trainer()
        loss = trainer.evaluate(_CountingDataset(max_sequence_length=2, n_structures=1))
        logged = dict(zip(trainer.cascade_target_order, loss.tolist()))
        self.assertNotIn("site_symmetry_ops_id", logged)
        # And what survives is labelled by field, not shifted by the hole the drop leaves.
        self.assertEqual(
            logged, {"elements": 2., "site_symmetries": 4., "sites_enumeration": 8.})


if __name__ == "__main__":
    unittest.main()
