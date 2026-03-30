import unittest
import inspect
import io
import json
import pickle
import sys
import trace
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from .. import tokenization as tok
from ..tokenization import argsort_multiple, SpaceGroupEncoder


class TestArgsortMultiple(unittest.TestCase):
    @torch.no_grad()
    def test_argsort_multiple_single_tensor_respects_dim(self):
        tensor1 = torch.tensor([[3, 1, 2], [6, 5, 4]])
        expected_output = torch.tensor([[0, 0, 0], [1, 1, 1]])
        output = argsort_multiple(tensor1, dim=0)
        self.assertTrue(torch.equal(output, expected_output))

    @torch.no_grad()
    def test_argsort_multiple_two_tensors_lexicographic_no_collision(self):
        # Crafted so radix=max(second_key) would collide, but radix=max+1 stays injective.
        tensor1 = torch.tensor([[1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.uint8)
        tensor2 = torch.tensor([[0, 4, 0, 4, 0, 4, 0, 4]], dtype=torch.uint8)
        expected_output = torch.tensor([[1, 0, 3, 2, 5, 4, 7, 6]])
        output = argsort_multiple(tensor1, tensor2, dim=1)
        self.assertTrue(torch.equal(output, expected_output))

    @torch.no_grad()
    def test_argsort_multiple_two_tensors_respects_dim_zero(self):
        tensor1 = torch.tensor([[2, 1, 0], [1, 1, 1]], dtype=torch.uint8)
        tensor2 = torch.tensor([[0, 2, 3], [5, 1, 0]], dtype=torch.uint8)
        expected_output = torch.tensor([[1, 1, 0], [0, 0, 1]])
        output = argsort_multiple(tensor1, tensor2, dim=0)
        self.assertTrue(torch.equal(output, expected_output))

    @torch.no_grad()
    def test_argsort_multiple_rejects_non_uint8_two_tensor_input(self):
        tensor1 = torch.tensor([[3, 1, 2], [6, 5, 4]])
        tensor2 = torch.tensor([[9, 8, 7], [6, 5, 4]])
        with self.assertRaises(NotImplementedError):
            argsort_multiple(tensor1, tensor2, dim=1)

    @torch.no_grad()
    def test_argsort_multiple_invalid_arity(self):
        tensor1 = torch.tensor([[3, 1, 2], [6, 5, 4]], dtype=torch.uint8)
        tensor2 = torch.tensor([[9, 8, 7], [6, 5, 4]], dtype=torch.uint8)
        tensor3 = torch.tensor([[3, 2, 1], [4, 5, 6]], dtype=torch.uint8)
        with self.assertRaises(NotImplementedError):
            argsort_multiple(tensor1, tensor2, tensor3, dim=1)

class TestSpaceGroupEncoder(unittest.TestCase):
    def test_space_group_encoding_uniqueness(self):
        """
        By design, all space groups must have uniqe encodings
        """
        all_group_encoder = SpaceGroupEncoder.from_sg_set(range(1, 231))
        all_groups = set()
        for group_number in range(1, 231):
            group = tuple(all_group_encoder[group_number])
            if group in all_groups:
                raise ValueError(f"Duplicate group: {group}")
            all_groups.add(group)

    def test_encode_spacegroups_shape(self):
        encoder = SpaceGroupEncoder.from_sg_set({1, 2, 3})
        encoded = encoder.encode_spacegroups([1, 2], dtype=torch.float32)
        self.assertEqual(encoded.ndim, 2)
        self.assertEqual(encoded.shape[0], 2)
        self.assertEqual(encoded.dtype, torch.float32)


class TestTupleAndTokenisers(unittest.TestCase):
    def test_tuple_dict_accepts_non_tuple_keys(self):
        d = tok.TupleDict({("a", "b"): 7})
        self.assertEqual(d[("a", "b")], 7)
        self.assertEqual(d[["a", "b"]], 7)

    def test_enumerating_tokeniser_roundtrip_and_padding(self):
        tokeniser = tok.EnumeratingTokeniser.from_token_set({"Li", "Na"}, include_stop=True)
        seq = tokeniser.tokenise_sequence(["Li"], original_max_len=3, dtype=torch.int64)
        self.assertEqual(seq.tolist()[0], tokeniser["Li"])
        self.assertIn(tokeniser.stop_token, seq.tolist())
        self.assertEqual(seq.tolist().count(tokeniser.pad_token), 2)
        single = tokeniser.tokenise_single("Na", dtype=torch.int64)
        self.assertEqual(single.item(), tokeniser["Na"])

    def test_enumerating_tokeniser_special_token_rejected(self):
        with self.assertRaises(ValueError):
            tok.EnumeratingTokeniser.from_token_set({"MASK", "Li"})

    def test_enumerating_tokeniser_max_tokens_rejected(self):
        with self.assertRaises(ValueError):
            tok.EnumeratingTokeniser.from_token_set({"A", "B"}, max_tokens=1)

    def test_dummy_item_getter_and_pass_through(self):
        getter = tok.DummyItemGetter()
        self.assertEqual(getter[5], 5)

        passthrough = tok.PassThroughTokeniser(values_count=9, stop_token=6, pad_token=7, mask_token=8)
        self.assertEqual(len(passthrough), 9)
        self.assertEqual(passthrough[42], 42)
        self.assertEqual(passthrough.to_token[3], 3)


class TestFeatureEngineer(unittest.TestCase):
    def test_init_with_series_sorts_multiindex(self):
        index = pd.MultiIndex.from_tuples([(2, "b"), (1, "a")], names=["a", "b"])
        series = pd.Series([20, 10], index=index, name="f")
        engineer = tok.FeatureEngineer(series)
        self.assertTrue(engineer.db.index.is_monotonic_increasing)

    def test_init_inconsistent_object_shapes_raises(self):
        data = {
            (1, "x"): np.array([1, 2]),
            (1, "y"): np.array([1, 2, 3]),
        }
        with self.assertRaises(ValueError):
            tok.FeatureEngineer(data=data, inputs=("a", "b"), name="f")

    def test_pad_and_stop_vector_feature(self):
        data = {
            (1, "x"): np.array([1, 2]),
            (1, "y"): np.array([3, 4]),
        }
        engineer = tok.FeatureEngineer(
            data=data,
            inputs=("a", "b"),
            name="f",
            stop_token=np.array([-1, -1]),
            pad_token=np.array([0, 0]),
        )
        padded = engineer.pad_and_stop([np.array([1, 2])], original_max_len=2, dtype=torch.int64)
        self.assertEqual(tuple(padded.shape), (3, 2))

    def test_get_feature_tensor_from_series(self):
        data = {
            (1, "m", "a"): 10,
            (1, "n", "b"): 20,
        }
        engineer = tok.FeatureEngineer(
            data=data,
            inputs=("sg", "ss", "enum"),
            name="feat",
            stop_token=99,
            pad_token=0,
            include_stop=True,
        )
        record = pd.Series({"sg": 1, "ss": ["m", "n"], "enum": ["a", "b"]})
        tensor = engineer.get_feature_tensor_from_series(record, original_max_len=3, dtype=torch.int64)
        self.assertEqual(tensor.tolist(), [10, 20, 99, 0])

    def test_get_feature_tensor_from_series_mismatch_raises(self):
        data = {(1, "m", "a"): 10}
        engineer = tok.FeatureEngineer(
            data=data,
            inputs=("sg", "ss", "enum"),
            name="feat",
            stop_token=99,
            pad_token=0,
            include_stop=True,
        )
        record = pd.Series({"sg": 1, "ss": ["m"], "enum": ["a"], "feat": [11]})
        with self.assertRaises(ValueError):
            engineer.get_feature_tensor_from_series(record, original_max_len=1, dtype=torch.int64)

    def test_get_feature_from_augmented_series(self):
        data = {
            (1, "m", "a"): 1,
            (1, "n", "b"): 2,
            (1, "m", "b"): 3,
            (1, "n", "a"): 4,
        }
        engineer = tok.FeatureEngineer(
            data=data,
            inputs=("sg", "ss", "sites_enumeration"),
            name="feat",
            stop_token=9,
            pad_token=0,
            include_stop=True,
        )
        record = pd.Series(
            {
                "sg": 1,
                "ss": ["m", "n"],
                "sites_enumeration_augmented": [["a", "b"], ["b", "a"]],
            }
        )
        tensors = engineer.get_feature_from_augmented_series(
            record=record,
            augmented_field_orginal_name="sites_enumeration",
            original_max_len=2,
            dtype=torch.int64,
        )
        self.assertEqual(len(tensors), 2)
        self.assertEqual(tensors[0].tolist(), [1, 2, 9])
        self.assertEqual(tensors[1].tolist(), [3, 4, 9])

    def test_get_feature_from_token_batch(self):
        data = {
            (1, 2, 3): 5,
            (2, 2, 3): 6,
        }
        engineer = tok.FeatureEngineer(
            data=data,
            inputs=("a", "b", "c"),
            name="feat",
            default_value=-1,
        )
        values = engineer.get_feature_from_token_batch(
            level_0=[1, 9],
            levels_plus=[[2, 2], [3, 3]],
        )
        self.assertEqual(values.tolist(), [5, -1])


class TestTokeniseEngineer(unittest.TestCase):
    def test_tokenise_engineer_success(self):
        raw_engineer = tok.FeatureEngineer(
            data={("x", "a"): 1, ("y", "b"): 2},
            inputs=("f1", "f2"),
            name="feat",
            stop_token=9,
            pad_token=8,
            mask_token=7,
            include_stop=True,
        )
        tokenisers = {
            "f1": tok.EnumeratingTokeniser.from_token_set({"x", "y"}, include_stop=True),
            "f2": tok.EnumeratingTokeniser.from_token_set({"a", "b"}, include_stop=True),
        }
        token_engineer = tok.tokenise_engineer(raw_engineer, tokenisers)
        self.assertEqual(token_engineer.db.name, "feat")
        self.assertIn((tokenisers["f1"]["x"], tokenisers["f2"]["a"]), token_engineer.db.index)

    def test_tokenise_engineer_inconsistent_include_stop_raises(self):
        raw_engineer = tok.FeatureEngineer(
            data={("x", "a"): 1},
            inputs=("f1", "f2"),
            name="feat",
            include_stop=True,
        )
        tokenisers = {
            "f1": tok.EnumeratingTokeniser.from_token_set({"x"}, include_stop=True),
            "f2": tok.EnumeratingTokeniser.from_token_set({"a"}, include_stop=False),
        }
        with self.assertRaises(ValueError):
            tok.tokenise_engineer(raw_engineer, tokenisers)


class TestTokeniseDataset(unittest.TestCase):
    @staticmethod
    def _load_fixture() -> dict[str, pd.DataFrame]:
        fixture_path = Path(__file__).resolve().parent / "data" / "tokenization_subsample.json"
        payload = json.loads(fixture_path.read_text())
        return {name: pd.DataFrame(rows) for name, rows in payload.items()}

    @patch.object(tok.pandarallel, "initialize")
    def test_tokenise_dataset_basic_and_augmented(self, _mock_init):
        datasets_pd = self._load_fixture()
        config = OmegaConf.create(
            {
                "dtype": "int64",
                "include_stop": True,
                "token_fields": {
                    "pure_categorical": ["elements"],
                },
                "sequence_fields": {
                    "pure_categorical": ["formation_bin"],
                    "space_group": ["spacegroup_number"],
                    "no_processing": ["float_feature"],
                    "counters": {"elem_counter": "elements"},
                },
                "augmented_token_fields": ["elements"],
            }
        )

        tensors, tokenisers, token_engineers = tok.tokenise_dataset(datasets_pd, config, n_jobs=1)

        self.assertIn("train", tensors)
        self.assertIn("elements", tensors["train"])
        self.assertIn("elements_augmented", tensors["train"])
        self.assertIn("formation_bin", tensors["train"])
        self.assertIn("spacegroup_number", tensors["train"])
        self.assertIn("float_feature", tensors["train"])
        self.assertIn("elem_counter_tokens", tensors["train"])
        self.assertIn("elem_counter_counts", tensors["train"])
        self.assertIn("pure_sequence_length", tensors["train"])
        self.assertIn("elements", tokenisers)
        self.assertIn("spacegroup_number", tokenisers)
        self.assertEqual(token_engineers, {})

    @patch.object(tok.pandarallel, "initialize")
    def test_tokenise_dataset_token_sort_and_augmented_not_supported(self, _mock_init):
        datasets_pd = self._load_fixture()
        config = OmegaConf.create(
            {
                "dtype": "int64",
                "token_fields": {"pure_categorical": ["elements"]},
                "sequence_fields": {"space_group": ["spacegroup_number"]},
                "augmented_token_fields": ["elements"],
                "token_sort": ["elements"],
            }
        )
        with self.assertRaises(NotImplementedError):
            tok.tokenise_dataset(datasets_pd, config, n_jobs=1)

    @patch.object(tok.pandarallel, "initialize")
    def test_tokenise_dataset_uses_existing_tokeniser_file(self, _mock_init):
        datasets_pd = self._load_fixture()
        config = OmegaConf.create(
            {
                "dtype": "int64",
                "token_fields": {"pure_categorical": ["elements"]},
                "sequence_fields": {"space_group": ["spacegroup_number"]},
            }
        )
        _, saved_tokenisers, _ = tok.tokenise_dataset(datasets_pd, config, n_jobs=1)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            tokeniser_path = Path(tmp_file.name)
        try:
            tok.WyckoffProcessor(
                config=config,
                tokenisers=saved_tokenisers,
                token_engineers={},
            ).save_pretrained(tokeniser_path)
            _, loaded_tokenisers, _ = tok.tokenise_dataset(
                datasets_pd,
                config,
                tokenizer_path=tokeniser_path,
                n_jobs=1,
            )
            self.assertEqual(set(saved_tokenisers.keys()), set(loaded_tokenisers.keys()))
        finally:
            tokeniser_path.unlink(missing_ok=True)


class TestWyckoffProcessor(unittest.TestCase):
    @staticmethod
    def _load_fixture() -> dict[str, pd.DataFrame]:
        fixture_path = Path(__file__).resolve().parent / "data" / "tokenization_subsample.json"
        payload = json.loads(fixture_path.read_text())
        return {name: pd.DataFrame(rows) for name, rows in payload.items()}

    @patch.object(tok.pandarallel, "initialize")
    def test_processor_from_config_and_tokenise_dataset(self, _mock_init):
        datasets_pd = self._load_fixture()
        config = OmegaConf.create(
            {
                "dtype": "int64",
                "include_stop": True,
                "token_fields": {"pure_categorical": ["elements"]},
                "sequence_fields": {
                    "pure_categorical": ["formation_bin"],
                    "space_group": ["spacegroup_number"],
                },
            }
        )
        processor = tok.WyckoffProcessor.from_config(config)
        tensors, tokenisers, token_engineers = processor.tokenise_dataset(datasets_pd, n_jobs=1)

        self.assertIn("train", tensors)
        self.assertIn("elements", tokenisers)
        self.assertIn("spacegroup_number", tokenisers)
        self.assertEqual(token_engineers, {})
        self.assertIs(processor.tokenisers, tokenisers)
        self.assertIs(processor.token_engineers, token_engineers)

    def test_processor_save_and_load_pretrained_json(self):
        enum = tok.EnumeratingTokeniser.from_token_set({"Na", "Cl"}, include_stop=True)
        passthrough = tok.PassThroughTokeniser(values_count=8, stop_token=5, pad_token=6, mask_token=7)

        sg = tok.SpaceGroupEncoder()
        sg[1] = (1.0, 0.0)
        sg.np_dict[1] = np.array([1.0, 0.0])
        sg.to_token = tok.TupleDict({(1.0, 0.0): 1})

        engineer = tok.FeatureEngineer(
            data={(1, "m", 0): 3},
            inputs=("spacegroup_number", "site_symmetries", "sites_enumeration"),
            name="sites_enumeration",
            stop_token=9,
            pad_token=8,
            mask_token=7,
        )

        processor = tok.WyckoffProcessor(
            config=OmegaConf.create({"dtype": "int64", "token_fields": {"pure_categorical": ["elements"]}}),
            tokenisers={
                "elements": enum,
                "spacegroup_number": sg,
                "dummy": passthrough,
            },
            token_engineers={"sites_enumeration": engineer},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = processor.save_pretrained(Path(tmpdir))
            loaded = tok.WyckoffProcessor.from_pretrained(saved_path)

        self.assertIn("elements", loaded.tokenisers)
        self.assertIn("spacegroup_number", loaded.tokenisers)
        self.assertIn("dummy", loaded.tokenisers)
        self.assertIsInstance(loaded.tokenisers["elements"], tok.EnumeratingTokeniser)
        self.assertIsInstance(loaded.tokenisers["spacegroup_number"], tok.SpaceGroupEncoder)
        self.assertIsInstance(loaded.tokenisers["dummy"], tok.PassThroughTokeniser)
        self.assertEqual(loaded.tokenisers["elements"]["Na"], enum["Na"])
        self.assertTrue(np.array_equal(loaded.tokenisers["spacegroup_number"].np_dict[1], sg.np_dict[1]))
        self.assertEqual(loaded.token_engineers["sites_enumeration"].db.loc[(1, "m", 0)], 3)

    def test_processor_save_pretrained_handles_numpy_scalar_tokens(self):
        processor = tok.WyckoffProcessor(
            config=OmegaConf.create({"dtype": "int64"}),
            tokenisers={
                "dummy": tok.PassThroughTokeniser(
                    values_count=np.int64(8),
                    stop_token=np.int64(5),
                    pad_token=np.int64(6),
                    mask_token=np.int64(7),
                )
            },
            token_engineers={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = processor.save_pretrained(Path(tmpdir))
            loaded = tok.WyckoffProcessor.from_pretrained(saved_path)

        self.assertEqual(loaded.tokenisers["dummy"].values_count, 8)
        self.assertEqual(loaded.tokenisers["dummy"].stop_token, 5)
        self.assertEqual(loaded.tokenisers["dummy"].pad_token, 6)
        self.assertEqual(loaded.tokenisers["dummy"].mask_token, 7)

    def test_processor_tensor_to_pyxtal(self):
        tokenisers = {
            "elements": tok.EnumeratingTokeniser.from_token_set({"Na"}),
            "site_symmetries": tok.EnumeratingTokeniser.from_token_set({"m"}),
            "sites_enumeration": tok.EnumeratingTokeniser.from_token_set({0}),
        }
        sg = tok.PassThroughTokeniser(values_count=2)
        sg.to_token = tok.TupleDict({(1, 0): 1})
        tokenisers["spacegroup_number"] = sg

        processor = tok.WyckoffProcessor(config={}, tokenisers=tokenisers, token_engineers={})
        element_idx = tokenisers["elements"]["Na"]
        ss_idx = tokenisers["site_symmetries"]["m"]
        enum_idx = tokenisers["sites_enumeration"][0]
        result = processor.tensor_to_pyxtal(
            space_group_tensor=torch.tensor([1, 0], dtype=torch.int64),
            wp_tensor=torch.tensor([[element_idx, ss_idx, enum_idx]], dtype=torch.int64),
            cascade_order=("elements", "site_symmetries", "sites_enumeration"),
            letter_from_ss_enum_idx={1: {"m": {enum_idx: "a"}}},
            ss_from_letter={1: {"a": "m"}},
            wp_index={1: {"m": {"a": (1, 1)}}},
        )
        self.assertEqual(result["group"], 1)
        self.assertEqual(result["species"], ["Na"])


class TestLoadTensorsAndHelpers(unittest.TestCase):
    def test_load_tensors_and_tokenisers_from_cache_pt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            ds_dir = base / "toy"
            (ds_dir / "tokenisers").mkdir(parents=True)
            (ds_dir / "tensors").mkdir(parents=True)

            tokenisers = {
                "elements": tok.EnumeratingTokeniser.from_token_set({"Na"}, include_stop=True),
            }
            token_engineers = {}
            tok.WyckoffProcessor(
                config=OmegaConf.create({"dtype": "int64", "token_fields": {"pure_categorical": ["elements"]}}),
                tokenisers=tokenisers,
                token_engineers=token_engineers,
            ).save_pretrained(ds_dir / "tokenisers" / "cfg.json")
            torch.save({"train": {"x": torch.tensor([1])}}, ds_dir / "tensors" / "cfg.pt")

            tensors, loaded_tokenisers, loaded_token_engineers = tok.load_tensors_and_tokenisers(
                dataset="toy",
                config_name="cfg",
                use_cached_tensors=True,
                cache_path=base,
            )

            self.assertIn("train", tensors)
            self.assertEqual(set(loaded_tokenisers.keys()), set(tokenisers.keys()))
            self.assertEqual(loaded_token_engineers, token_engineers)

    def test_load_tensors_and_tokenisers_missing_pt_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            ds_dir = base / "toy"
            (ds_dir / "tokenisers").mkdir(parents=True)
            (ds_dir / "tensors").mkdir(parents=True)

            tok.WyckoffProcessor(config={}, tokenisers={}, token_engineers={}).save_pretrained(
                ds_dir / "tokenisers" / "cfg.json"
            )

            with self.assertRaises(FileNotFoundError):
                tok.load_tensors_and_tokenisers(
                    dataset="toy",
                    config_name="cfg",
                    use_cached_tensors=True,
                    cache_path=base,
                )

    def test_load_tensors_and_tokenisers_without_cache_calls_tokenise_dataset(self):
        fake_datasets = {"train": pd.DataFrame([{"elements": ["H"]}])}
        with patch.object(tok, "tokenise_dataset", return_value=("tensors", "tokenisers", "engineers")) as mock_tok, \
             patch.object(tok.OmegaConf, "load", return_value=OmegaConf.create({})) as mock_cfg, \
             patch.object(tok.gzip, "open") as mock_gzip_open:
            in_memory = io.BytesIO()
            pickle.dump(fake_datasets, in_memory)
            in_memory.seek(0)
            mock_gzip_open.return_value.__enter__.return_value = in_memory

            result = tok.load_tensors_and_tokenisers(
                dataset="toy",
                config_name="cfg",
                use_cached_tensors=False,
            )
            self.assertEqual(result, ("tensors", "tokenisers", "engineers"))
            mock_cfg.assert_called_once()
            mock_tok.assert_called_once()

    def test_get_wp_index_with_fake_group(self):
        class FakeWP:
            def __init__(self, letter, multiplicity, dof, site_symm):
                self.letter = letter
                self.multiplicity = multiplicity
                self._dof = dof
                self._site_symm = site_symm

            def get_site_symmetry(self):
                self.site_symm = self._site_symm

            def get_dof(self):
                return self._dof

        class FakeGroup:
            def __init__(self, number):
                self.number = number
                self.Wyckoff_positions = [FakeWP("a", 2, 1, "m")]

        with patch.object(tok, "Group", side_effect=FakeGroup):
            wp_index = tok.get_wp_index()
        self.assertEqual(wp_index[1]["m"]["a"], (2, 1))
        self.assertEqual(len(wp_index), 230)

    def test_get_letter_from_ss_enum_idx(self):
        letter_from_ss_enum = {
            1: {
                "m": {
                    0: "a",
                    99: "b",
                }
            }
        }
        mock_mappings = tok.WyckoffMappings(
            enum_from_ss_letter={},
            letter_from_ss_enum=letter_from_ss_enum,
            ss_from_letter={},
        )

        with patch.object(tok, "load_wyckoff_mappings", return_value=mock_mappings):
            enum_tokeniser = tok.EnumeratingTokeniser.from_token_set({0}, include_stop=True)
            result = enum_tokeniser.get_letter_from_ss_enum_idx()

        enum_token = enum_tokeniser[0]
        self.assertEqual(result[1]["m"][enum_token], "a")
        self.assertNotIn(99, result[1]["m"])


class TestTensorToPyxtal(unittest.TestCase):
    @staticmethod
    def _build_tokenisers_for_modes():
        tokenisers = {
            "elements": tok.EnumeratingTokeniser.from_token_set({"Na", "Cl"}),
            "site_symmetries": tok.EnumeratingTokeniser.from_token_set({"m"}),
            "sites_enumeration": tok.EnumeratingTokeniser.from_token_set({0}),
            "wyckoff_letters": tok.EnumeratingTokeniser.from_token_set({"a"}),
            "harmonic_cluster": tok.EnumeratingTokeniser.from_token_set({7}),
        }
        sg = tok.PassThroughTokeniser(values_count=2)
        sg.to_token = tok.TupleDict({(1, 0): 1})
        tokenisers["spacegroup_number"] = sg
        return tokenisers

    def test_tensor_to_pyxtal_site_symmetry_mode(self):
        tokenisers = self._build_tokenisers_for_modes()
        processor = tok.WyckoffProcessor(config={}, tokenisers=tokenisers, token_engineers={})
        element = tokenisers["elements"]["Na"]
        ss = tokenisers["site_symmetries"]["m"]
        enum = tokenisers["sites_enumeration"][0]

        wp_tensor = torch.tensor(
            [
                [element, ss, enum],
                [
                    tokenisers["elements"].stop_token,
                    tokenisers["site_symmetries"].stop_token,
                    tokenisers["sites_enumeration"].stop_token,
                ],
            ],
            dtype=torch.int64,
        )
        result = processor.tensor_to_pyxtal(
            space_group_tensor=torch.tensor([1, 0], dtype=torch.int64),
            wp_tensor=wp_tensor,
            cascade_order=("elements", "site_symmetries", "sites_enumeration"),
            letter_from_ss_enum_idx={1: {"m": {enum: "a"}}},
            ss_from_letter={1: {"a": "m"}},
            wp_index={1: {"m": {"a": (2, 1)}}},
        )
        self.assertEqual(result["group"], 1)
        self.assertEqual(result["sites"], [["2a"]])
        self.assertEqual(result["species"], ["Na"])
        self.assertEqual(result["numIons"], [2])

    def test_tensor_to_pyxtal_wyckoff_letters_mode(self):
        tokenisers = self._build_tokenisers_for_modes()
        processor = tok.WyckoffProcessor(config={}, tokenisers=tokenisers, token_engineers={})
        element = tokenisers["elements"]["Cl"]
        letter = tokenisers["wyckoff_letters"]["a"]
        wp_tensor = torch.tensor(
            [[element, letter]],
            dtype=torch.int64,
        )
        result = processor.tensor_to_pyxtal(
            space_group_tensor=torch.tensor([1, 0], dtype=torch.int64),
            wp_tensor=wp_tensor,
            cascade_order=("elements", "wyckoff_letters"),
            letter_from_ss_enum_idx={1: {"m": {0: "a"}}},
            ss_from_letter={1: {"a": "m"}},
            wp_index={1: {"m": {"a": (4, 1)}}},
        )
        self.assertEqual(result["species"], ["Cl"])
        self.assertEqual(result["numIons"], [4])

    def test_tensor_to_pyxtal_harmonic_mode(self):
        tokenisers = self._build_tokenisers_for_modes()
        element = tokenisers["elements"]["Na"]
        ss_idx = tokenisers["site_symmetries"]["m"]
        cluster_idx = tokenisers["harmonic_cluster"][7]
        enum_token = tokenisers["sites_enumeration"][0]

        fe = tok.FeatureEngineer(
            data={((1, 0), ss_idx, cluster_idx): enum_token},
            inputs=("spacegroup_number", "site_symmetries", "harmonic_cluster"),
            name="sites_enumeration",
        )
        processor = tok.WyckoffProcessor(
            config={},
            tokenisers=tokenisers,
            token_engineers={"sites_enumeration": fe},
        )
        result = processor.tensor_to_pyxtal(
            space_group_tensor=torch.tensor([1, 0], dtype=torch.int64),
            wp_tensor=torch.tensor([[element, ss_idx, cluster_idx]], dtype=torch.int64),
            cascade_order=("elements", "site_symmetries", "harmonic_cluster"),
            letter_from_ss_enum_idx={1: {"m": {enum_token: "a"}}},
            ss_from_letter={1: {"a": "m"}},
            wp_index={1: {"m": {"a": (1, 1)}}},
        )
        self.assertEqual(result["group"], 1)
        self.assertEqual(result["sites"], [["1a"]])

    def test_tensor_to_pyxtal_rejects_invalid_tokens_and_constraints(self):
        tokenisers = self._build_tokenisers_for_modes()
        processor = tok.WyckoffProcessor(config={}, tokenisers=tokenisers, token_engineers={})
        element = tokenisers["elements"]["Na"]
        ss = tokenisers["site_symmetries"]["m"]
        enum = tokenisers["sites_enumeration"][0]

        # PAD in sequence is invalid.
        invalid_pad = processor.tensor_to_pyxtal(
            space_group_tensor=torch.tensor([1, 0], dtype=torch.int64),
            wp_tensor=torch.tensor([[element, ss, tokenisers["sites_enumeration"].pad_token]], dtype=torch.int64),
            cascade_order=("elements", "site_symmetries", "sites_enumeration"),
            letter_from_ss_enum_idx={1: {"m": {enum: "a"}}},
            ss_from_letter={1: {"a": "m"}},
            wp_index={1: {"m": {"a": (2, 1)}}},
        )
        self.assertIsNone(invalid_pad)

        # Element-count constraint can invalidate otherwise correct sample.
        valid_tensor = torch.tensor([[element, ss, enum]], dtype=torch.int64)
        too_many = processor.tensor_to_pyxtal(
            space_group_tensor=torch.tensor([1, 0], dtype=torch.int64),
            wp_tensor=valid_tensor,
            cascade_order=("elements", "site_symmetries", "sites_enumeration"),
            letter_from_ss_enum_idx={1: {"m": {enum: "a"}}},
            ss_from_letter={1: {"a": "m"}},
            wp_index={1: {"m": {"a": (2, 1)}}},
            enforced_min_elements=2,
        )
        self.assertIsNone(too_many)

    def test_tensor_to_pyxtal_unsupported_cascade_raises(self):
        tokenisers = self._build_tokenisers_for_modes()
        processor = tok.WyckoffProcessor(config={}, tokenisers=tokenisers, token_engineers={})
        with self.assertRaises(NotImplementedError):
            processor.tensor_to_pyxtal(
                space_group_tensor=torch.tensor([1, 0], dtype=torch.int64),
                wp_tensor=torch.tensor([[0]], dtype=torch.int64),
                cascade_order=("bad",),
                letter_from_ss_enum_idx={},
                ss_from_letter={},
                wp_index={},
            )


class TestTokenizationCoverageSmoke(unittest.TestCase):
    def _exercise_entrypoints(self):
        # TupleDict
        tuple_dict = tok.TupleDict({("x", "y"): 1})
        _ = tuple_dict[["x", "y"]]

        # SpaceGroupEncoder
        spg_encoder = tok.SpaceGroupEncoder.from_sg_set({1, 2})
        _ = spg_encoder.encode_spacegroups([1], dtype=torch.float32)

        # EnumeratingTokeniser
        enum_tok = tok.EnumeratingTokeniser.from_token_set({"Na", "Cl"}, include_stop=True)
        _ = enum_tok.tokenise_sequence(["Na"], original_max_len=2, dtype=torch.int64)
        _ = enum_tok.tokenise_single("Cl", dtype=torch.int64)

        # FeatureEngineer and tokenise_engineer
        engineer = tok.FeatureEngineer(
            data={(1, "m", "a"): 10},
            inputs=("sg", "ss", "enum"),
            name="feat",
            stop_token=99,
            pad_token=0,
            include_stop=True,
        )
        _ = engineer.pad_and_stop([10], original_max_len=2, dtype=torch.int64)
        _ = engineer.get_feature_tensor_from_series(
            pd.Series({"sg": 1, "ss": ["m"], "enum": ["a"]}),
            original_max_len=1,
            dtype=torch.int64,
        )
        _ = engineer.get_feature_from_token_batch(level_0=[1], levels_plus=[["m"], ["a"]])

        aug_engineer = tok.FeatureEngineer(
            data={(1, "m", "a"): 1, (1, "m", "b"): 2},
            inputs=("sg", "ss", "sites_enumeration"),
            name="feat_aug",
            stop_token=9,
            pad_token=0,
            include_stop=True,
        )
        _ = aug_engineer.get_feature_from_augmented_series(
            record=pd.Series(
                {
                    "sg": 1,
                    "ss": ["m"],
                    "sites_enumeration_augmented": [["a"], ["b"]],
                }
            ),
            augmented_field_orginal_name="sites_enumeration",
            original_max_len=1,
            dtype=torch.int64,
        )

        tokenisers_for_engineer = {
            "sg": tok.EnumeratingTokeniser.from_token_set({1}, include_stop=True),
            "ss": tok.EnumeratingTokeniser.from_token_set({"m"}, include_stop=True),
            "enum": tok.EnumeratingTokeniser.from_token_set({"a"}, include_stop=True),
        }
        _ = tok.tokenise_engineer(engineer, tokenisers_for_engineer)

        # DummyItemGetter and PassThroughTokeniser
        getter = tok.DummyItemGetter()
        _ = getter[3]
        passthrough = tok.PassThroughTokeniser(values_count=4, stop_token=1, pad_token=2, mask_token=3)
        _ = len(passthrough)
        _ = passthrough[0]

        # argsort_multiple
        _ = tok.argsort_multiple(torch.tensor([[1, 0]], dtype=torch.uint8), dim=1)

        # tokenise_dataset
        small_df = pd.DataFrame(
            [
                {
                    "elements": ["Na"],
                    "spacegroup_number": 1,
                    "formation_bin": "low",
                    "elem_counter": {"Na": 1},
                }
            ]
        )
        config = OmegaConf.create(
            {
                "dtype": "int64",
                "token_fields": {"pure_categorical": ["elements"]},
                "sequence_fields": {
                    "pure_categorical": ["formation_bin"],
                    "space_group": ["spacegroup_number"],
                    "counters": {"elem_counter": "elements"},
                },
            }
        )
        with patch.object(tok.pandarallel, "initialize"):
            _ = tok.tokenise_dataset({"train": small_df}, config, n_jobs=1)

        # load_tensors_and_tokenisers
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            ds_dir = base / "toy"
            (ds_dir / "tokenisers").mkdir(parents=True)
            (ds_dir / "tensors").mkdir(parents=True)
            tok.WyckoffProcessor(config={}, tokenisers={}, token_engineers={}).save_pretrained(
                ds_dir / "tokenisers" / "cfg.json"
            )
            torch.save({"train": {"x": torch.tensor([1])}}, ds_dir / "tensors" / "cfg.pt")
            _ = tok.load_tensors_and_tokenisers(
                dataset="toy",
                config_name="cfg",
                use_cached_tensors=True,
                cache_path=base,
            )

        # get_wp_index with patched Group
        class FakeWP:
            def __init__(self):
                self.letter = "a"
                self.multiplicity = 1

            def get_site_symmetry(self):
                self.site_symm = "m"

            def get_dof(self):
                return 1

        class FakeGroup:
            def __init__(self, _number):
                self.Wyckoff_positions = [FakeWP()]

        with patch.object(tok, "Group", side_effect=FakeGroup):
            _ = tok.get_wp_index()

        # get_letter_from_ss_enum_idx
        payload = pickle.dumps((None, {1: {"m": {0: "a"}}}))
        with patch.object(tok.gzip, "open") as mock_open:
            mock_open.return_value.__enter__.return_value = io.BytesIO(payload)
            _ = tok.EnumeratingTokeniser.from_token_set({0}, include_stop=True).get_letter_from_ss_enum_idx()

        # tensor_to_pyxtal
        tks = {
            "elements": tok.EnumeratingTokeniser.from_token_set({"Na"}),
            "site_symmetries": tok.EnumeratingTokeniser.from_token_set({"m"}),
            "sites_enumeration": tok.EnumeratingTokeniser.from_token_set({0}),
        }
        spg_tok = tok.PassThroughTokeniser(values_count=2)
        spg_tok.to_token = tok.TupleDict({(1, 0): 1})
        tks["spacegroup_number"] = spg_tok
        processor = tok.WyckoffProcessor(config={}, tokenisers=tks, token_engineers={})

        element_idx = tks["elements"]["Na"]
        ss_idx = tks["site_symmetries"]["m"]
        enum_idx = tks["sites_enumeration"][0]
        _ = processor.tensor_to_pyxtal(
            space_group_tensor=torch.tensor([1, 0], dtype=torch.int64),
            wp_tensor=torch.tensor(
                [
                    [element_idx, ss_idx, enum_idx],
                    [
                        tks["elements"].stop_token,
                        tks["site_symmetries"].stop_token,
                        tks["sites_enumeration"].stop_token,
                    ],
                ],
                dtype=torch.int64,
            ),
            cascade_order=("elements", "site_symmetries", "sites_enumeration"),
            letter_from_ss_enum_idx={1: {"m": {enum_idx: "a"}}},
            ss_from_letter={1: {"a": "m"}},
            wp_index={1: {"m": {"a": (1, 1)}}},
        )

    def test_tokenization_entrypoints_are_touched(self):
        tracer = trace.Trace(
            count=True,
            trace=False,
            ignoredirs=[sys.prefix, sys.exec_prefix],
        )
        tracer.runfunc(self._exercise_entrypoints)
        counts = tracer.results().counts

        targets = {
            "TupleDict.__getitem__": tok.TupleDict.__getitem__,
            "SpaceGroupEncoder.from_sg_set": tok.SpaceGroupEncoder.from_sg_set,
            "SpaceGroupEncoder.encode_spacegroups": tok.SpaceGroupEncoder.encode_spacegroups,
            "EnumeratingTokeniser.from_token_set": tok.EnumeratingTokeniser.from_token_set,
            "EnumeratingTokeniser.tokenise_sequence": tok.EnumeratingTokeniser.tokenise_sequence,
            "EnumeratingTokeniser.tokenise_single": tok.EnumeratingTokeniser.tokenise_single,
            "FeatureEngineer._lexsort_db": tok.FeatureEngineer._lexsort_db,
            "FeatureEngineer.__init__": tok.FeatureEngineer.__init__,
            "FeatureEngineer.pad_and_stop": tok.FeatureEngineer.pad_and_stop,
            "FeatureEngineer.get_feature_tensor_from_series": tok.FeatureEngineer.get_feature_tensor_from_series,
            "FeatureEngineer.get_feature_from_augmented_series": tok.FeatureEngineer.get_feature_from_augmented_series,
            "FeatureEngineer.get_feature_from_token_batch": tok.FeatureEngineer.get_feature_from_token_batch,
            "DummyItemGetter.__getitem__": tok.DummyItemGetter.__getitem__,
            "PassThroughTokeniser.__init__": tok.PassThroughTokeniser.__init__,
            "PassThroughTokeniser.__len__": tok.PassThroughTokeniser.__len__,
            "PassThroughTokeniser.__getitem__": tok.PassThroughTokeniser.__getitem__,
            "tokenise_engineer": tok.tokenise_engineer,
            "argsort_multiple": tok.argsort_multiple,
            "tokenise_dataset": tok.tokenise_dataset,
            "load_tensors_and_tokenisers": tok.load_tensors_and_tokenisers,
            "get_wp_index": tok.get_wp_index,
            "EnumeratingTokeniser.get_letter_from_ss_enum_idx": tok.EnumeratingTokeniser.from_token_set({0}, include_stop=True).get_letter_from_ss_enum_idx,
            "WyckoffProcessor.tensor_to_pyxtal": tok.WyckoffProcessor.tensor_to_pyxtal,
        }

        executed_lines_by_file = {}
        for (filename, line), _count in counts.items():
            executed_lines_by_file.setdefault(Path(filename).resolve(), set()).add(line)

        missing = []
        for name, obj in targets.items():
            source_lines, start_line = inspect.getsourcelines(obj)
            obj_file = Path(inspect.getfile(obj)).resolve()
            obj_executed = executed_lines_by_file.get(obj_file, set())
            line_window = range(start_line, start_line + len(source_lines) + 1)
            if not any(line in obj_executed for line in line_window):
                missing.append(name)

        self.assertEqual(missing, [], f"Untouched tokenization entrypoints: {missing}")


if __name__ == '__main__':
    unittest.main()
