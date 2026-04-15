"""Tests for the wyformer-cryspr CLI entry point."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from wyckoff_transformer.cli.relax import main


_NACL_GENE = {
    "group": 225,
    "species": ["Na", "Cl"],
    "numIons": [4, 4],
    "sites": [["4a"], ["4b"]],
}


class TestCLIArgumentParsing(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.input_file = self.tmp_path / "genes.json"
        self.input_file.write_text(json.dumps([_NACL_GENE]))

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_model_exits(self):
        sys.argv = ["wyformer-cryspr", str(self.input_file)]
        with self.assertRaises(SystemExit):
            main()

    @patch("wyckoff_transformer.cli.relax.build_mace_calculator")
    @patch("wyckoff_transformer.cli.relax.func_run")
    def test_basic_run_writes_csv(self, mock_func_run, mock_build_calc):
        mock_func_run.return_value = (MagicMock(), "NaCl", -10.0, -1.25)
        mock_build_calc.return_value = MagicMock()

        out_dir = self.tmp_path / "output"
        sys.argv = [
            "wyformer-cryspr",
            str(self.input_file),
            "--model", "/fake/model.model",
            "--output-dir", str(out_dir),
            "--n-trials", "1",
        ]
        main()

        csv_files = list(out_dir.glob("*.csv"))
        self.assertEqual(len(csv_files), 1)
        df = pd.read_csv(csv_files[0])
        self.assertIn("energy", df.columns)
        self.assertEqual(len(df), 1)

    @patch("wyckoff_transformer.cli.relax.build_mace_calculator")
    @patch("wyckoff_transformer.cli.relax.func_run")
    def test_model_name_used_as_csv_stem(self, mock_func_run, mock_build_calc):
        mock_func_run.return_value = (None, None, None, None)
        mock_build_calc.return_value = MagicMock()

        out_dir = self.tmp_path / "out2"
        sys.argv = [
            "wyformer-cryspr",
            str(self.input_file),
            "--model", "/fake/model.model",
            "--model-name", "mymace",
            "--output-dir", str(out_dir),
        ]
        main()

        self.assertTrue((out_dir / "mymace_results.csv").exists())

    @patch("wyckoff_transformer.cli.relax.build_mace_calculator")
    @patch("wyckoff_transformer.cli.relax.func_run")
    def test_start_end_slices_input(self, mock_func_run, mock_build_calc):
        # Write a 3-element input file
        input3 = self.tmp_path / "genes3.json"
        input3.write_text(json.dumps([_NACL_GENE] * 3))
        mock_func_run.return_value = (None, None, None, None)
        mock_build_calc.return_value = MagicMock()

        out_dir = self.tmp_path / "out3"
        sys.argv = [
            "wyformer-cryspr",
            str(input3),
            "--model", "/fake/model.model",
            "--output-dir", str(out_dir),
            "--start", "1",
            "--end", "2",
        ]
        main()

        # Only one gene in range [1, 2)
        self.assertEqual(mock_func_run.call_count, 1)
        called_id = mock_func_run.call_args.kwargs["id_gene"]
        self.assertEqual(called_id, 1)

    @patch("wyckoff_transformer.cli.relax.build_mace_calculator")
    @patch("wyckoff_transformer.cli.relax.func_run")
    def test_url_model_name_derived_from_stem(self, mock_func_run, mock_build_calc):
        mock_func_run.return_value = (None, None, None, None)
        mock_build_calc.return_value = MagicMock()

        out_dir = self.tmp_path / "out4"
        url = "https://example.com/my-mace-model.model"
        sys.argv = [
            "wyformer-cryspr",
            str(self.input_file),
            "--model", url,
            "--output-dir", str(out_dir),
        ]
        main()

        self.assertTrue((out_dir / "my-mace-model_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
