"""Tests for ``wyformer-protocol-wandb`` (no network, no W&B)."""
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from wyckoff_transformer.cli import protocol_wandb as pw


class TestFlattenFunnel(unittest.TestCase):
    def test_prefixes_and_drops_none_and_bool(self):
        funnel = {
            "sampled": 1000,
            "metasun_per_sampled_gene": 0.31,
            "no_hull_energy": None,
            "some_flag": True,
        }
        flat = pw.flatten_funnel(funnel)
        self.assertEqual(flat["protocol/sampled"], 1000)
        self.assertAlmostEqual(flat["protocol/metasun_per_sampled_gene"], 0.31)
        self.assertNotIn("protocol/no_hull_energy", flat)
        self.assertNotIn("protocol/some_flag", flat)


class TestBuildStageArgs(unittest.TestCase):
    def test_carries_every_field_the_stages_read(self):
        args = pw.build_parser().parse_args(
            ["r1", "--output-dir", "/out", "--cores", "8", "--no-rattle"]
        )
        gene_file = Path("/out/wyckoff_genes.json.gz")
        stage_args = pw.build_stage_args(args, gene_file)
        self.assertEqual(stage_args.input, gene_file)
        self.assertEqual(stage_args.output_dir, Path("/out"))
        self.assertEqual(stage_args.cores, 8)
        self.assertFalse(stage_args.rattle)
        self.assertEqual(stage_args.n_trials, "0:1,2:2,*:3")
        # Every attribute the protocol stage functions touch must exist.
        for name in (
            "input", "output_dir", "mlip", "cores", "devices", "workers_per_device",
            "n_trials", "pyxtal_cores", "pyxtal_timeout", "fmax", "relax_timeout",
            "release_symmetry", "rattle", "limit", "resume",
            "reference_cache", "reference_splits", "reference_fingerprint_cache",
            "lemat_cif_csv", "debug",
        ):
            self.assertTrue(hasattr(stage_args, name), name)


class TestParserDefaults(unittest.TestCase):
    def test_defaults(self):
        args = pw.build_parser().parse_args(["run42", "--output-dir", "x"])
        self.assertEqual(args.n_genes, 1000)
        self.assertTrue(args.upload)
        self.assertTrue(args.rattle)
        self.assertFalse(args.skip_generate)

    def test_resume_is_on_by_default(self):
        args = pw.build_parser().parse_args(["r", "--output-dir", "x"])
        self.assertTrue(args.resume)

    def test_no_upload(self):
        args = pw.build_parser().parse_args(["r", "--output-dir", "x", "--no-upload"])
        self.assertFalse(args.upload)

    def test_stages_and_from_artifact_defaults(self):
        args = pw.build_parser().parse_args(["r", "--output-dir", "x"])
        self.assertEqual(args.stages, "screen,generate,relax,score")
        self.assertIsNone(args.from_artifact)

    def test_from_artifact_takes_latest_or_a_pinned_version(self):
        bare = pw.build_parser().parse_args(
            ["r", "--output-dir", "x", "--from-artifact"]
        )
        self.assertEqual(bare.from_artifact, "latest")
        pinned = pw.build_parser().parse_args(
            ["r", "--output-dir", "x", "--from-artifact", "v2"]
        )
        self.assertEqual(pinned.from_artifact, "v2")


class TestEnsureRunFiles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_downloads_only_missing(self):
        (self.run_dir / "best_model_params.pt").write_bytes(b"x")
        run = MagicMock()
        run.id = "r1"
        pw.ensure_run_files(run, self.run_dir)
        downloaded = {c.args[0] for c in run.file.call_args_list}
        self.assertEqual(downloaded, {"wyckoff_processor.json", "spacegroup_distribution.json"})

    def test_missing_download_raises_with_message(self):
        run = MagicMock()
        run.id = "r1"
        run.file.return_value.download.side_effect = RuntimeError("404")
        run.logged_artifacts.return_value = []
        with self.assertRaises(FileNotFoundError):
            pw.ensure_run_files(run, self.run_dir)

    def test_downloads_missing_file_from_latest_artifact(self):
        (self.run_dir / "wyckoff_processor.json").write_text("{}")
        (self.run_dir / "spacegroup_distribution.json").write_text("{}")
        run = MagicMock()
        run.id = "r1"
        run.file.return_value.download.side_effect = RuntimeError("404")
        older = MagicMock()
        older.name = "best_model_r1:v0"
        older_file = MagicMock()
        older_file.name = "wrong_file"
        older.files.return_value = [older_file]
        latest = MagicMock()
        latest.name = "best_model_r1:v1"
        latest_file = MagicMock()
        latest_file.name = "best_model_params.pt"
        latest.files.return_value = [latest_file]
        run.logged_artifacts.return_value = [older, latest]

        pw.ensure_run_files(run, self.run_dir)

        latest.download.assert_called_once_with(root=str(self.run_dir))
        older.download.assert_not_called()


class TestMainSkipGenerate(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        with gzip.open(self.out / pw.GENES_FILE, "wt", encoding="utf-8") as handle:
            json.dump([{"group": 225}], handle)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_funnel(self, *_args):
        (self.out / "funnel.json").write_text(
            json.dumps({"sampled": 1, "metasun_per_sampled_gene": 0.0}), encoding="utf-8"
        )

    def test_runs_every_stage_and_uploads(self):
        with patch.object(pw.protocol_cli, "stage_screen") as screen, \
             patch.object(pw.protocol_cli, "stage_generate") as generate, \
             patch.object(pw.protocol_cli, "stage_relax") as relax, \
             patch.object(pw.protocol_cli, "stage_score", side_effect=self._write_funnel) as score, \
             patch.object(pw, "upload") as upload:
            sys.argv = [
                "wyformer-protocol-wandb", "run7",
                "--output-dir", str(self.out),
                "--skip-generate",
            ]
            pw.main()
        screen.assert_called_once()
        generate.assert_called_once()
        relax.assert_called_once()
        score.assert_called_once()
        upload.assert_called_once()

    def test_no_upload_skips_writeback(self):
        with patch.object(pw.protocol_cli, "stage_screen"), \
             patch.object(pw.protocol_cli, "stage_generate"), \
             patch.object(pw.protocol_cli, "stage_relax"), \
             patch.object(pw.protocol_cli, "stage_score", side_effect=self._write_funnel), \
             patch.object(pw, "upload") as upload:
            sys.argv = [
                "wyformer-protocol-wandb", "run7",
                "--output-dir", str(self.out),
                "--skip-generate", "--no-upload",
            ]
            pw.main()
        upload.assert_not_called()

    def test_stages_subset_runs_only_those(self):
        with patch.object(pw.protocol_cli, "stage_screen") as screen, \
             patch.object(pw.protocol_cli, "stage_generate") as generate, \
             patch.object(pw.protocol_cli, "stage_relax") as relax, \
             patch.object(pw.protocol_cli, "stage_score", side_effect=self._write_funnel) as score, \
             patch.object(pw, "upload"):
            sys.argv = [
                "wyformer-protocol-wandb", "run7", "--output-dir", str(self.out),
                "--skip-generate", "--stages", "score",
            ]
            pw.main()
        screen.assert_not_called()
        generate.assert_not_called()
        relax.assert_not_called()
        score.assert_called_once()

    def test_from_artifact_downloads_then_scores(self):
        with patch.object(pw, "download_protocol_artifact") as download, \
             patch.object(pw.protocol_cli, "stage_score", side_effect=self._write_funnel) as score, \
             patch.object(pw, "upload") as upload:
            sys.argv = [
                "wyformer-protocol-wandb", "run7", "--output-dir", str(self.out),
                "--from-artifact", "--stages", "score",
            ]
            pw.main()
        download.assert_called_once()
        score.assert_called_once()
        upload.assert_called_once()

    def test_skip_generate_without_file_raises(self):
        (self.out / pw.GENES_FILE).unlink()
        sys.argv = [
            "wyformer-protocol-wandb", "run7",
            "--output-dir", str(self.out), "--skip-generate",
        ]
        with self.assertRaises(FileNotFoundError):
            pw.main()


class TestGenerateGenes(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name) / "genes.json.gz"

    def tearDown(self):
        self._tmp.cleanup()

    def test_truncates_to_n_genes(self):
        trainer = MagicMock()
        trainer.condition_features = ()
        trainer.generate_structures.return_value = [{"i": i} for i in range(20)]
        with patch.object(pw, "load_trainer", return_value=trainer), \
             patch.object(pw, "ensure_run_files"), \
             patch("wandb.Api"):
            n = pw.generate_genes(
                run_id="r", entity="e", project="p", n_genes=10,
                oversample=1.15, device="cpu", output_path=self.out,
            )
        self.assertEqual(n, 10)
        self.assertIsNone(trainer.generate_structures.call_args.kwargs["cond"])
        with gzip.open(self.out, "rt") as handle:
            self.assertEqual(len(json.load(handle)), 10)

    def test_too_few_valid_raises(self):
        trainer = MagicMock()
        trainer.condition_features = ()
        trainer.generate_structures.return_value = [{"i": 0}] * 3
        with patch.object(pw, "load_trainer", return_value=trainer), \
             patch.object(pw, "ensure_run_files"), \
             patch("wandb.Api"):
            with self.assertRaises(ValueError):
                pw.generate_genes(
                    run_id="r", entity="e", project="p", n_genes=10,
                    oversample=1.15, device="cpu", output_path=self.out,
                )

    def test_conditional_run_needs_a_target(self):
        trainer = MagicMock()
        trainer.condition_features = ("energy_above_hull",)
        with patch.object(pw, "load_trainer", return_value=trainer), \
             patch.object(pw, "ensure_run_files"), \
             patch("wandb.Api"):
            with self.assertRaises(ValueError):
                pw.generate_genes(
                    run_id="r", entity="e", project="p", n_genes=10,
                    oversample=1.15, device="cpu", output_path=self.out,
                )

    def test_condition_value_is_built_and_passed(self):
        trainer = MagicMock()
        trainer.condition_features = ("energy_above_hull",)
        trainer.build_condition_from_values.return_value = "COND"
        trainer.generate_structures.return_value = [{"i": i} for i in range(20)]
        with patch.object(pw, "load_trainer", return_value=trainer), \
             patch.object(pw, "ensure_run_files"), \
             patch("wandb.Api"):
            pw.generate_genes(
                run_id="r", entity="e", project="p", n_genes=10,
                oversample=1.15, device="cpu", output_path=self.out,
                condition=["energy_above_hull=0"],
            )
        values, n_rows = trainer.build_condition_from_values.call_args.args[:2]
        self.assertEqual(values, {"energy_above_hull": 0.0})
        self.assertEqual(trainer.generate_structures.call_args.kwargs["cond"], "COND")


if __name__ == "__main__":
    unittest.main()
