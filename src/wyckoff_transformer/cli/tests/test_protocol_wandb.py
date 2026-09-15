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

    def test_hierarchical_funnel(self):
        funnel = {
            "gene": {
                "sampled": 1000,
                "valid_gene": 900,
                "none_val": None,
            },
            "fixed_symmetry": {
                "structure": 800,
                "sun_per_sampled_gene": 0.5,
            },
            "free": {
                "structure": 750,
                "sun_per_sampled_gene": 0.45,
            },
        }
        flat = pw.flatten_funnel(funnel)
        self.assertEqual(flat["protocol/gene/sampled"], 1000)
        self.assertEqual(flat["protocol/gene/valid_gene"], 900)
        self.assertNotIn("protocol/gene/none_val", flat)
        self.assertEqual(flat["protocol/fixed_symmetry/structure"], 800)
        self.assertEqual(flat["protocol/fixed_symmetry/sun_per_sampled_gene"], 0.5)
        self.assertEqual(flat["protocol/free/structure"], 750)
        self.assertEqual(flat["protocol/free/sun_per_sampled_gene"], 0.45)


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
            "n_trials", "pyxtal_cores", "pyxtal_timeout", "pyxtal_tol_factor",
            "fmax", "relax_timeout",
            "release_symmetry", "rattle", "limit", "resume", "allow_incomplete",
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
        self.assertEqual(downloaded, {"wyckoff_processor.json", "spacegroup_distribution.json",
                                      "wyckoffs_enumerated_by_ss.json"})

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
        (self.run_dir / "wyckoffs_enumerated_by_ss.json").write_text("{}")
        (self.run_dir / "engineers").mkdir()
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


class TestEnsureSystemPrior(unittest.TestCase):
    """A chemsys run's prior comes from the run itself, and its absence is not an error."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name)
        self.prior = self.run_dir / "system_prior.npz"

    def test_a_local_copy_is_not_fetched_again(self):
        self.prior.write_bytes(b"npz")
        run = MagicMock()
        self.assertEqual(pw.ensure_system_prior(run, self.run_dir), self.prior)
        run.file.assert_not_called()
        run.logged_artifacts.assert_not_called()

    def test_downloads_from_the_run_files(self):
        run = MagicMock()
        run.id = "r1"
        run.file.return_value.download.side_effect = (
            lambda root, replace: self.prior.write_bytes(b"npz"))
        self.assertEqual(pw.ensure_system_prior(run, self.run_dir), self.prior)
        run.file.assert_called_once_with("system_prior.npz")
        run.logged_artifacts.assert_not_called()

    def test_falls_back_to_the_artifact(self):
        run = MagicMock()
        run.id = "r1"
        run.file.return_value.download.side_effect = RuntimeError("404")
        other = MagicMock()
        other.name = "best_model_r1:v3"
        other_file = MagicMock()
        other_file.name = "best_model_params.pt"
        other.files.return_value = [other_file]
        prior_artifact = MagicMock()
        prior_artifact.name = "system_prior_r1:v0"
        prior_file = MagicMock()
        prior_file.name = "system_prior.npz"
        prior_artifact.files.return_value = [prior_file]
        prior_artifact.download.side_effect = lambda root: self.prior.write_bytes(b"npz")
        run.logged_artifacts.return_value = [prior_artifact, other]

        self.assertEqual(pw.ensure_system_prior(run, self.run_dir), self.prior)
        prior_artifact.download.assert_called_once_with(root=str(self.run_dir))
        other.download.assert_not_called()

    def test_a_run_that_carries_none_returns_none(self):
        """Runs trained before the prior was saved have one nowhere; the caller says so."""
        run = MagicMock()
        run.id = "r1"
        run.file.return_value.download.side_effect = RuntimeError("404")
        run.logged_artifacts.return_value = []
        self.assertIsNone(pw.ensure_system_prior(run, self.run_dir))


class TestEnsureRunEngineers(unittest.TestCase):
    """A run's own engineers come from its processors artifact, when it logged them."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name)

    @staticmethod
    def _artifact(type_, *names):
        artifact = MagicMock()
        artifact.type = type_
        artifact.name = f"{type_}_r1:v0"
        files = []
        for name in names:
            artifact_file = MagicMock()
            artifact_file.name = name
            files.append(artifact_file)
        artifact.files.return_value = files
        return artifact

    def test_downloads_the_processors_artifact_carrying_engineers(self):
        model = self._artifact("model", "best_model_params.pt")
        processors = self._artifact(
            "processors", "wyckoff_processor.json", "engineers/multiplicity.json")
        run = MagicMock()
        run.logged_artifacts.return_value = [model, processors]
        pw.ensure_run_engineers(run, self.run_dir)
        processors.download.assert_called_once_with(root=str(self.run_dir))
        model.download.assert_not_called()

    def test_keeps_a_local_copy(self):
        (self.run_dir / "engineers").mkdir()
        run = MagicMock()
        pw.ensure_run_engineers(run, self.run_dir)
        run.logged_artifacts.assert_not_called()

    def test_a_run_that_predates_engineers_only_warns(self):
        processors = self._artifact("processors", "wyckoff_processor.json")
        run = MagicMock()
        run.id = "r1"
        run.logged_artifacts.return_value = [processors]
        with self.assertLogs(pw.logger, level="WARNING"):
            pw.ensure_run_engineers(run, self.run_dir)
        processors.download.assert_not_called()


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

    def test_a_stage_subset_without_score_exits_cleanly(self):
        """A screen-only arm writes no funnel; that is not a failure."""
        with patch.object(pw.protocol_cli, "stage_screen") as screen, \
             patch.object(pw, "upload") as upload:
            sys.argv = [
                "wyformer-protocol-wandb", "run7", "--output-dir", str(self.out),
                "--skip-generate", "--stages", "screen", "--no-upload",
            ]
            pw.main()
        screen.assert_called_once()
        upload.assert_not_called()

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

    def test_a_failed_rescore_uploads_nothing(self):
        """The artifact holds a complete funnel; a partial one must not replace it."""
        from wyckoff_transformer.evaluation.protocol import GeneScreen

        pw.protocol_cli.write_screen(
            GeneScreen(n_sampled=1, valid=[0], counts={0: 1}, novel=[0]),
            self.out / pw.protocol_cli.SCREEN_FILE,
        )
        with patch.object(pw, "download_protocol_artifact"), \
             patch.object(pw.protocol_cli, "stage_screen"), \
             patch.object(pw.protocol_cli, "stage_score",
                          side_effect=MemoryError("reference")), \
             patch.object(pw, "upload") as upload:
            sys.argv = [
                "wyformer-protocol-wandb", "run7", "--output-dir", str(self.out),
                "--from-artifact", "--stages", "screen,score",
            ]
            with self.assertRaises(MemoryError):
                pw.main()
        upload.assert_not_called()
        self.assertFalse((self.out / pw.protocol_cli.FUNNEL_FILE).exists())

    def test_skip_generate_without_file_raises(self):
        (self.out / pw.GENES_FILE).unlink()
        sys.argv = [
            "wyformer-protocol-wandb", "run7",
            "--output-dir", str(self.out), "--skip-generate",
        ]
        with self.assertRaises(FileNotFoundError):
            pw.main()

    def test_all_workers_fail_reports_computed_metrics_then_raises(self):
        def _screen_and_fail_relax(stage, stage_args):
            if stage == "relax":
                raise RuntimeError("All 5 relaxation worker(s) failed: BrokenProcessPool")

        from wyckoff_transformer.evaluation.protocol import GeneScreen

        screen = GeneScreen(
            n_sampled=10, valid=[0, 1], invalid=[], counts={0: 6, 1: 4}, novel=[0], known=[1]
        )
        pw.protocol_cli.write_screen(screen, self.out / pw.protocol_cli.SCREEN_FILE)

        with patch.object(pw.protocol_cli, "run_stage", side_effect=_screen_and_fail_relax), \
             patch.object(pw, "upload") as mock_upload:
            sys.argv = [
                "wyformer-protocol-wandb", "run7",
                "--output-dir", str(self.out),
                "--skip-generate",
            ]
            with self.assertRaises(RuntimeError) as ctx:
                pw.main()
            self.assertIn("All 5 relaxation worker(s) failed", str(ctx.exception))

        mock_upload.assert_called_once()
        uploaded_funnel = mock_upload.call_args[0][2]
        self.assertEqual(uploaded_funnel["sampled"], 10)
        self.assertEqual(uploaded_funnel["valid_gene"], 2)
        self.assertIsNone(uploaded_funnel["structure"])
        self.assertIsNone(uploaded_funnel["metastable"])
        self.assertIsNone(uploaded_funnel["metasun_per_sampled_gene"])

        funnel_on_disk = json.loads(
            (self.out / pw.protocol_cli.FUNNEL_FILE).read_text(encoding="utf-8")
        )
        self.assertEqual(funnel_on_disk["gene"]["sampled"], 10)
        self.assertEqual(funnel_on_disk["gene"]["valid_gene"], 2)
        self.assertIsNone(funnel_on_disk["free"]["metastable"])
        self.assertIsNone(funnel_on_disk["fixed_symmetry"]["metastable"])


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

    def test_conditional_run_needs_a_target_for_non_default_features(self):
        trainer = MagicMock()
        trainer.condition_features = ("band_gap",)
        with patch.object(pw, "load_trainer", return_value=trainer), \
             patch.object(pw, "ensure_run_files"), \
             patch("wandb.Api"):
            with self.assertRaisesRegex(ValueError, "conditions on .*band_gap"):
                pw.generate_genes(
                    run_id="r", entity="e", project="p", n_genes=10,
                    oversample=1.15, device="cpu", output_path=self.out,
                )

    def test_default_conditions_when_unspecified(self):
        trainer = MagicMock()
        trainer.condition_features = ("energy_above_hull", "delta_e_polymorph", "max_force")
        trainer.build_condition_from_values.return_value = "COND"
        trainer.generate_structures.return_value = [{"i": i} for i in range(20)]
        with patch.object(pw, "load_trainer", return_value=trainer), \
             patch.object(pw, "ensure_run_files"), \
             patch("wandb.Api"):
            pw.generate_genes(
                run_id="r", entity="e", project="p", n_genes=10,
                oversample=1.15, device="cpu", output_path=self.out,
            )
        values, n_rows = trainer.build_condition_from_values.call_args.args[:2]
        self.assertEqual(
            values,
            {"energy_above_hull": 0.0, "delta_e_polymorph": 0.0, "max_force": 0.0},
        )
        self.assertEqual(trainer.generate_structures.call_args.kwargs["cond"], "COND")

    def test_partial_condition_fills_remaining_defaults(self):
        trainer = MagicMock()
        trainer.condition_features = ("energy_above_hull", "delta_e_polymorph", "max_force")
        trainer.build_condition_from_values.return_value = "COND"
        trainer.generate_structures.return_value = [{"i": i} for i in range(20)]
        with patch.object(pw, "load_trainer", return_value=trainer), \
             patch.object(pw, "ensure_run_files"), \
             patch("wandb.Api"):
            pw.generate_genes(
                run_id="r", entity="e", project="p", n_genes=10,
                oversample=1.15, device="cpu", output_path=self.out,
                condition=["energy_above_hull=0.05"],
            )
        values, n_rows = trainer.build_condition_from_values.call_args.args[:2]
        self.assertEqual(
            values,
            {"energy_above_hull": 0.05, "delta_e_polymorph": 0.0, "max_force": 0.0},
        )

    def test_condition_value_on_multi_channel_model(self):
        trainer = MagicMock()
        trainer.condition_features = ("energy_above_hull", "delta_e_polymorph", "max_force")
        trainer.build_condition_from_values.return_value = "COND"
        trainer.generate_structures.return_value = [{"i": i} for i in range(20)]
        with patch.object(pw, "load_trainer", return_value=trainer), \
             patch.object(pw, "ensure_run_files"), \
             patch("wandb.Api"):
            pw.generate_genes(
                run_id="r", entity="e", project="p", n_genes=10,
                oversample=1.15, device="cpu", output_path=self.out,
                condition_value=0.05,
            )
        values, n_rows = trainer.build_condition_from_values.call_args.args[:2]
        self.assertEqual(
            values,
            {"energy_above_hull": 0.05, "delta_e_polymorph": 0.0, "max_force": 0.0},
        )

    def test_single_channel_defaults_to_zero(self):
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
            )
        values, n_rows = trainer.build_condition_from_values.call_args.args[:2]
        self.assertEqual(values, {"energy_above_hull": 0.0})

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

    def test_chemical_system_conditioning_uses_prior(self):
        trainer = MagicMock()
        trainer.condition_features = ()
        trainer.chemical_system_conditioning = True
        trainer.start_name = "spacegroup_number"
        trainer.tokenisers = {"elements": MagicMock(), "spacegroup_number": MagicMock()}
        trainer.generate_structures.return_value = [{"i": i} for i in range(20)]
        prior = MagicMock()
        draws = MagicMock()
        draws.conditioning_block.return_value = "COMP_COND"
        draws.start_tensor.return_value = "START_T"
        draws.element_mask.return_value = "ELEM_MASK"
        prior.sample.return_value = draws

        run_mock = MagicMock()
        run_mock.config = {"dataset": "lemat_bulk_fmax1"}
        api_mock = MagicMock()
        api_mock.run.return_value = run_mock

        with patch.object(pw, "load_trainer", return_value=trainer), \
             patch.object(pw, "ensure_run_files"), \
             patch("wandb.Api", return_value=api_mock), \
             patch("wyckoff_transformer.system_prior.SystemSpaceGroupPrior.load", return_value=prior), \
             patch("pathlib.Path.is_file", return_value=True):
            pw.generate_genes(
                run_id="r", entity="e", project="p", n_genes=10,
                oversample=1.15, device="cpu", output_path=self.out,
            )
        kwargs = trainer.generate_structures.call_args.kwargs
        self.assertEqual(kwargs["composition_cond"], "COMP_COND")
        self.assertEqual(kwargs["start_tensor"], "START_T")
        self.assertEqual(kwargs["allowed_element_mask"], "ELEM_MASK")

    def _chemsys_trainer(self):
        trainer = MagicMock()
        trainer.condition_features = ()
        trainer.chemical_system_conditioning = True
        trainer.start_name = "spacegroup_number"
        trainer.tokenisers = {"elements": MagicMock(), "spacegroup_number": MagicMock()}
        trainer.generate_structures.return_value = [{"i": i} for i in range(20)]
        return trainer

    def test_the_prior_is_taken_from_the_run_before_the_cache(self):
        """The run's own is the only one guaranteed to share the checkpoint's element tokens."""
        trainer = self._chemsys_trainer()
        prior = MagicMock()
        prior.sample.return_value = MagicMock()
        run_mock = MagicMock()
        run_mock.config = {"dataset": "lemat_bulk_fmax1_stress"}
        api_mock = MagicMock()
        api_mock.run.return_value = run_mock
        from_run = Path("/runs/r/system_prior.npz")

        with patch.object(pw, "load_trainer", return_value=trainer), \
             patch.object(pw, "ensure_run_files"), \
             patch.object(pw, "ensure_system_prior", return_value=from_run) as ensure, \
             patch("wandb.Api", return_value=api_mock), \
             patch("wyckoff_transformer.system_prior.SystemSpaceGroupPrior.load",
                   return_value=prior) as load, \
             patch("pathlib.Path.is_file", return_value=True):
            pw.generate_genes(
                run_id="r", entity="e", project="p", n_genes=10,
                oversample=1.15, device="cpu", output_path=self.out,
            )
        ensure.assert_called_once()
        self.assertEqual(load.call_args.args[0], from_run)

    def test_a_run_with_no_prior_anywhere_says_how_to_build_one(self):
        trainer = self._chemsys_trainer()
        run_mock = MagicMock()
        run_mock.config = {"dataset": "lemat_bulk_fmax1_stress"}
        api_mock = MagicMock()
        api_mock.run.return_value = run_mock

        with patch.object(pw, "load_trainer", return_value=trainer), \
             patch.object(pw, "ensure_run_files"), \
             patch.object(pw, "ensure_system_prior", return_value=None), \
             patch("wandb.Api", return_value=api_mock), \
             patch("pathlib.Path.is_file", return_value=False):
            with self.assertRaises(ValueError) as caught:
                pw.generate_genes(
                    run_id="r", entity="e", project="p", n_genes=10,
                    oversample=1.15, device="cpu", output_path=self.out,
                )
        self.assertIn("wyformer-system-prior build lemat_bulk_fmax1_stress", str(caught.exception))

    def test_a_prior_over_another_vocabulary_is_refused(self):
        """A cached prior from a different dataset would decode systems into other elements."""
        trainer = self._chemsys_trainer()
        trainer.tokenisers["elements"].to_token = ["Li", "O", "Na"]
        prior = MagicMock()
        prior.element_symbols = ["Li", "O"]
        prior.n_elements = 2
        run_mock = MagicMock()
        run_mock.config = {"dataset": "lemat_bulk_fmax1_stress"}
        api_mock = MagicMock()
        api_mock.run.return_value = run_mock

        with patch.object(pw, "load_trainer", return_value=trainer), \
             patch.object(pw, "ensure_run_files"), \
             patch.object(pw, "ensure_system_prior", return_value=Path("/p/system_prior.npz")), \
             patch("wandb.Api", return_value=api_mock), \
             patch("wyckoff_transformer.system_prior.SystemSpaceGroupPrior.load",
                   return_value=prior), \
             patch("pathlib.Path.is_file", return_value=True):
            with self.assertRaises(ValueError) as caught:
                pw.generate_genes(
                    run_id="r", entity="e", project="p", n_genes=10,
                    oversample=1.15, device="cpu", output_path=self.out,
                )
        self.assertIn("2 element tokens", str(caught.exception))
        trainer.generate_structures.assert_not_called()

    def test_temperature_reaches_the_generator(self):
        """The sweep is only a sweep if the cohort was actually drawn at T."""
        trainer = MagicMock()
        trainer.condition_features = ()
        trainer.generate_structures.return_value = [{"i": i} for i in range(20)]
        with patch.object(pw, "load_trainer", return_value=trainer), \
             patch.object(pw, "ensure_run_files"), \
             patch("wandb.Api"):
            pw.generate_genes(
                run_id="r", entity="e", project="p", n_genes=10,
                oversample=1.15, device="cpu", output_path=self.out,
                temperature=0.7,
            )
        self.assertEqual(
            trainer.generate_structures.call_args.kwargs["temperature"], 0.7)

    def test_the_manifest_records_the_temperature_and_the_raw_validity(self):
        """The kept cohort is truncated, so the rejected fraction lives here or nowhere."""
        trainer = MagicMock()
        trainer.condition_features = ()
        trainer.generate_structures.return_value = [{"i": i} for i in range(11)]
        manifest = self.out.parent / "manifest.json"
        with patch.object(pw, "load_trainer", return_value=trainer), \
             patch.object(pw, "ensure_run_files"), \
             patch("wandb.Api"):
            pw.generate_genes(
                run_id="r", entity="e", project="p", n_genes=10,
                oversample=1.4, device="cpu", output_path=self.out,
                temperature=0.7, manifest_path=manifest,
            )
        recorded = json.loads(manifest.read_text())
        self.assertEqual(recorded["sampling_temperature"], 0.7)
        self.assertEqual(recorded["generation_attempted"], 14)
        self.assertEqual(recorded["generation_formally_valid"], 11)
        self.assertAlmostEqual(recorded["formal_gene_validity"], 11 / 14, places=4)


if __name__ == "__main__":
    unittest.main()


class TestStagesWithoutScore(unittest.TestCase):
    """A run that does not score has nothing to report, and must not crash.

    Only the score stage writes funnel.json. ``--stages`` need not include it:
    the wide-then-narrow arm draws and pre-screens on CPU cores and then relaxes
    and scores on a GPU, which is two invocations with different hardware, and
    the first of them has no funnel yet. Reading it unconditionally turned a
    finished pre-screen into a FileNotFoundError.
    """

    def test_a_run_without_the_score_stage_reports_nothing_and_returns(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from wyckoff_transformer.cli import protocol_wandb

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            genes = out / protocol_wandb.GENES_FILE
            import gzip

            with gzip.open(genes, "wt", encoding="utf-8") as handle:
                json.dump([], handle)
            argv = [
                "wyformer-protocol-wandb", "someid", "--output-dir", str(out),
                "--stages", "generate", "--skip-generate", "--no-upload",
            ]
            with patch("sys.argv", argv), \
                    patch.object(protocol_wandb.protocol_cli, "run_stage") as run_stage, \
                    patch.object(protocol_wandb, "upload") as upload:
                protocol_wandb.main()
            run_stage.assert_called_once()
            upload.assert_not_called()

    def test_a_scoring_run_still_uploads(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from wyckoff_transformer.cli import protocol_wandb

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            import gzip

            with gzip.open(out / protocol_wandb.GENES_FILE, "wt", encoding="utf-8") as handle:
                json.dump([], handle)
            (out / protocol_wandb.protocol_cli.FUNNEL_FILE).write_text(
                json.dumps({"metasun_per_sampled_gene": 0.1}), encoding="utf-8"
            )
            argv = [
                "wyformer-protocol-wandb", "someid", "--output-dir", str(out),
                "--stages", "score", "--skip-generate",
            ]
            with patch("sys.argv", argv), \
                    patch.object(protocol_wandb.protocol_cli, "run_stage"), \
                    patch.object(protocol_wandb, "upload") as upload:
                protocol_wandb.main()
            upload.assert_called_once()


class TestIncompleteStagesAreNotUploaded(unittest.TestCase):
    """Trials waiting for --resume are not a result, partial or otherwise."""

    def test_an_incomplete_relax_uploads_nothing(self):
        import gzip
        import tempfile

        from wyckoff_transformer.cli import protocol_wandb

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with gzip.open(out / protocol_wandb.GENES_FILE, "wt", encoding="utf-8") as handle:
                json.dump([], handle)
            (out / protocol_wandb.protocol_cli.SCREEN_FILE).write_text("{}", encoding="utf-8")
            argv = [
                "wyformer-protocol-wandb", "someid", "--output-dir", str(out),
                "--stages", "relax,score", "--skip-generate",
            ]
            with patch("sys.argv", argv), \
                    patch.object(
                        protocol_wandb.protocol_cli, "run_stage",
                        side_effect=protocol_wandb.protocol_cli.IncompleteStageError("holes"),
                    ), \
                    patch.object(protocol_wandb, "upload") as upload:
                with self.assertRaises(protocol_wandb.protocol_cli.IncompleteStageError):
                    protocol_wandb.main()
            upload.assert_not_called()


class TestRefuseToResampleUnderResume(unittest.TestCase):
    """Sampling over a gene file whose logs would be resumed is how
    protocol_ehull5x-20260904-213346 v1 scored one cohort's draws as another's."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)

    def _args(self, resume):
        from types import SimpleNamespace

        return SimpleNamespace(output_dir=self.out, resume=resume)

    def _log(self, name, rows):
        header = "index,trial,status\n"
        (self.out / name).write_text(header + "".join(f"{i},0,ok\n" for i in range(rows)))

    def test_refused_with_resume_and_a_log(self):
        from wyckoff_transformer.cli.protocol import PYXTAL_TRIALS_FILE, StaleOutputError

        self._log(PYXTAL_TRIALS_FILE, 1)
        with self.assertRaisesRegex(StaleOutputError, "--skip-generate"):
            pw.refuse_to_resample_under_resume(self._args(True), self.out / pw.GENES_FILE)

    def test_allowed_with_no_resume(self):
        from wyckoff_transformer.cli.protocol import PYXTAL_TRIALS_FILE

        self._log(PYXTAL_TRIALS_FILE, 1)
        pw.refuse_to_resample_under_resume(self._args(False), self.out / pw.GENES_FILE)

    def test_allowed_on_a_fresh_or_empty_directory(self):
        from wyckoff_transformer.cli.protocol import RELAXATIONS_FILE

        pw.refuse_to_resample_under_resume(self._args(True), self.out / pw.GENES_FILE)
        self._log(RELAXATIONS_FILE, 0)
        pw.refuse_to_resample_under_resume(self._args(True), self.out / pw.GENES_FILE)

    def test_main_refuses_before_sampling(self):
        from wyckoff_transformer.cli.protocol import PYXTAL_TRIALS_FILE, StaleOutputError

        self._log(PYXTAL_TRIALS_FILE, 1)
        argv = ["wyformer-protocol-wandb", "run-id", "--output-dir", str(self.out)]
        with patch.object(sys, "argv", argv), \
                patch.object(pw, "generate_genes") as generate:
            with self.assertRaises(StaleOutputError):
                pw.main()
        generate.assert_not_called()


class TestArms(unittest.TestCase):
    """A sweep arm must not overwrite the run's headline cohort, nor another run's."""

    def test_the_headline_cohort_keeps_its_names(self):
        self.assertEqual(pw.protocol_artifact_name("r1"), "protocol_r1")
        self.assertEqual(pw.summary_prefix(), "protocol/")

    def test_an_arm_gets_its_own_artifact_and_summary_prefix(self):
        self.assertEqual(pw.protocol_artifact_name("r1", "cfg-w2"), "protocol_r1.cfg-w2")
        self.assertEqual(pw.summary_prefix("cfg-w2"), "protocol_cfg-w2/")
        flat = pw.flatten_funnel({"free": {"structure": 3}}, prefix=pw.summary_prefix("w2"))
        self.assertEqual(flat, {"protocol_w2/free/structure": 3})

    def test_an_arm_cannot_collide_with_another_runs_headline(self):
        # Run ids contain '-' and '_', so neither may separate the arm.
        self.assertNotEqual(pw.protocol_artifact_name("a_b"), pw.protocol_artifact_name("a", "b"))
        for bad in ("w.2", "w/2", "w 2", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pw.protocol_artifact_name("r1", bad)

    def test_parser_defaults(self):
        args = pw.build_parser().parse_args(["r", "--output-dir", "x"])
        self.assertIsNone(args.arm)
        self.assertEqual(args.guidance_scale, 1.0)
        args = pw.build_parser().parse_args(
            ["r", "--output-dir", "x", "--arm", "w3", "--guidance-scale", "3"])
        self.assertEqual((args.arm, args.guidance_scale), ("w3", 3.0))

    def test_main_rejects_a_bad_arm_before_doing_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["wyformer-protocol-wandb", "run-id", "--output-dir", tmp, "--arm", "w.2"]
            with patch.object(sys, "argv", argv), \
                    patch.object(pw, "generate_genes") as generate:
                with self.assertRaises(SystemExit):
                    pw.main()
            generate.assert_not_called()

    def test_main_forwards_the_guidance_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["wyformer-protocol-wandb", "run-id", "--output-dir", tmp,
                    "--guidance-scale", "2.5", "--stages", "screen", "--no-upload"]
            with patch.object(sys, "argv", argv), \
                    patch.object(pw, "generate_genes") as generate, \
                    patch.object(pw.protocol_cli, "run_stage"):
                pw.main()
            self.assertEqual(generate.call_args.kwargs["guidance_scale"], 2.5)
