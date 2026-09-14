"""Where data, cache, runs and W&B's directory resolve to.

Every test runs with `XDG_CONFIG_HOME` pointed at a temporary directory and the four
variables removed: on a configured machine the real `~/.config/wyformer/paths.env`
would otherwise decide the outcome of tests about the fallback.
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wyckoff_transformer import paths
from wyckoff_transformer.paths import (
    CACHE_ENV_VAR,
    CONFIG_KEYS,
    DATA_ENV_VAR,
    REPO_MARKER,
    RUNS_ENV_VAR,
    WANDB_DIR_ENV_VAR,
    StoreConfigError,
    StoreNotFoundError,
    cache_root,
    config_path,
    data_glob,
    data_path,
    data_roots,
    data_store,
    read_config,
    repo_root,
    resolve_store_path,
    runs_path,
    runs_root,
    shadowed_datasets,
    wandb_dir,
)

SHELL_HELPER = Path(__file__).resolve().parents[3] / "scripts" / "wyformer_paths.sh"


class _Isolated(unittest.TestCase):
    """A temporary directory, no store variables, and a config home nobody has written."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.config_home = self.root / "config-home"
        env = {k: v for k, v in os.environ.items() if k not in CONFIG_KEYS}
        env["XDG_CONFIG_HOME"] = str(self.config_home)
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_config(self, text: str) -> Path:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def configure(self, store: Path, **overrides) -> dict:
        values = {DATA_ENV_VAR: store / "data", CACHE_ENV_VAR: store / "cache",
                  RUNS_ENV_VAR: store / "runs", WANDB_DIR_ENV_VAR: store}
        values.update(overrides)
        self.write_config("".join(f"{k}={v}\n" for k, v in values.items() if v is not None))
        return values

    def make_repo(self, name: str = "repo") -> Path:
        repo = self.root / name
        repo.mkdir()
        (repo / REPO_MARKER).touch()
        return repo


class TestConfigPath(_Isolated):
    def test_honours_xdg_config_home(self):
        self.assertEqual(config_path(), self.config_home / "wyformer" / "paths.env")

    def test_defaults_to_dot_config(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XDG_CONFIG_HOME")
            self.assertEqual(config_path(), Path.home() / ".config" / "wyformer" / "paths.env")


class TestReadConfig(_Isolated):
    def test_absent_is_none(self):
        self.assertIsNone(read_config())

    def test_docker_env_file_format(self):
        self.write_config(
            "# a comment\n\n  WYFORMER_DATA=/a/data  \n"
            "WANDB_API_KEY\n"            # docker: pass the host's value through
            "WYFORMER_CACHE=/a/cache\nWYFORMER_CACHE=/b/cache\n")
        self.assertEqual(read_config(), {DATA_ENV_VAR: "/a/data", CACHE_ENV_VAR: "/b/cache"})

    def test_export_prefix_is_refused(self):
        """The form a converted shell profile would naturally have; docker rejects it too."""
        self.write_config("export WYFORMER_DATA=/a/data\n")
        with self.assertRaisesRegex(StoreConfigError, "drop the 'export'"):
            read_config()

    def test_whitespace_in_a_name_is_refused(self):
        self.write_config("WYFORMER DATA=/a\n")
        with self.assertRaisesRegex(StoreConfigError, "whitespace"):
            read_config()


class TestPrecedence(_Isolated):
    def test_config_file_is_used(self):
        store = self.root / "store"
        self.configure(store)
        self.assertEqual(cache_root(start=self.root), store / "cache")
        self.assertEqual(runs_root(start=self.root), store / "runs")

    def test_environment_beats_config(self):
        self.configure(self.root / "store")
        elsewhere = self.root / "elsewhere"
        with mock.patch.dict(os.environ, {CACHE_ENV_VAR: str(elsewhere)}):
            self.assertEqual(cache_root(start=self.root), elsewhere)

    def test_environment_expands_user(self):
        with mock.patch.dict(os.environ, {CACHE_ENV_VAR: "~/some-store"}):
            self.assertEqual(cache_root(start=self.root), Path.home() / "some-store")

    def test_empty_environment_value_is_unset(self):
        store = self.root / "store"
        self.configure(store)
        with mock.patch.dict(os.environ, {CACHE_ENV_VAR: ""}):
            self.assertEqual(cache_root(start=self.root), store / "cache")

    def test_an_existing_config_never_falls_back_to_the_checkout(self):
        """The failure this whole scheme exists to prevent: a quiet cache/ in the repo."""
        repo = self.make_repo()
        (repo / "cache").mkdir()
        self.configure(self.root / "store", **{CACHE_ENV_VAR: None})
        with self.assertRaisesRegex(StoreConfigError, "does not set WYFORMER_CACHE"):
            cache_root(start=repo)

    def test_values_must_be_literal_absolute_paths(self):
        for value in ("relative/cache", "$HOME/cache", "~/cache", "/home/$USER/cache", ""):
            with self.subTest(value=value):
                self.configure(self.root / "store", **{CACHE_ENV_VAR: value})
                with self.assertRaisesRegex(StoreConfigError, "literal absolute path"):
                    cache_root(start=self.root)


class TestFallbackWithoutConfig(_Isolated):
    def test_existing_directory_beside_cwd(self):
        """What the previous `Path.cwd() / "cache"` sites relied on."""
        (self.root / "cache").mkdir()
        self.assertEqual(cache_root(start=self.root), self.root / "cache")

    def test_repo_from_a_subdirectory(self):
        repo = self.make_repo()
        (repo / "scripts").mkdir()
        self.assertEqual(cache_root(start=repo / "scripts"), repo / "cache")

    def test_repo_location_need_not_exist_yet(self):
        """Writers such as scripts/cache_a_dataset.py create it -- on a fresh clone."""
        repo = self.make_repo()
        self.assertEqual(runs_root(start=repo), repo / "runs")
        self.assertFalse((repo / "runs").exists())

    def test_raises_outside_a_checkout(self):
        nested = self.root / "scratch"
        nested.mkdir()
        if repo_root(nested) is not None:
            self.skipTest("temp directory sits inside a checkout")
        with self.assertRaises(StoreNotFoundError) as caught:
            runs_root(start=nested)
        self.assertIn(str(config_path()), str(caught.exception))

    def test_wandb_default_is_left_to_wandb(self):
        self.assertIsNone(wandb_dir())


class TestDataLookup(_Isolated):
    """Tracked datasets stay in the checkout; untracked ones live in the store."""

    def setUp(self):
        super().setUp()
        self.repo = self.make_repo()
        self.store = self.root / "store"
        self.configure(self.store)
        self.tracked = self.repo / "data"
        self.untracked = self.store / "data"
        for dataset in ("mp_20", "perov_5"):
            (self.tracked / dataset).mkdir(parents=True)
        for dataset in ("lemat-bulk", "lemat_bulk_fmax1"):
            (self.untracked / dataset).mkdir(parents=True)

    def test_roots_are_the_store_then_the_checkout(self):
        self.assertEqual(data_roots(start=self.repo), [self.untracked, self.tracked])

    def test_a_tracked_dataset_resolves_to_the_checkout(self):
        self.assertEqual(data_path("mp_20", "train.csv", start=self.repo),
                         self.tracked / "mp_20" / "train.csv")

    def test_an_untracked_dataset_resolves_to_the_store(self):
        self.assertEqual(data_path("lemat-bulk", "lemat_pbe.csv.gz", start=self.repo),
                         self.untracked / "lemat-bulk" / "lemat_pbe.csv.gz")

    def test_a_multi_part_string_is_split_on_its_first_component(self):
        self.assertEqual(data_path("lemat-bulk/raw/data.parquet", start=self.repo),
                         self.untracked / "lemat-bulk" / "raw" / "data.parquet")

    def test_a_new_dataset_resolves_to_the_store(self):
        """Where it will be written."""
        self.assertEqual(data_path("lemat_bulk_fmax2", start=self.repo),
                         self.untracked / "lemat_bulk_fmax2")

    def test_the_store_shadows_the_checkout_for_the_whole_dataset(self):
        """Resolved per dataset, not per file: a split never mixes the two places."""
        (self.untracked / "mp_20").mkdir()
        (self.tracked / "mp_20" / "test.csv").touch()   # only the checkout has this file
        self.assertEqual(data_path("mp_20", "test.csv", start=self.repo),
                         self.untracked / "mp_20" / "test.csv")
        self.assertEqual(shadowed_datasets(start=self.repo), ["mp_20"])

    def test_no_parts_is_the_store(self):
        self.assertEqual(data_path(start=self.repo), self.untracked)
        self.assertEqual(data_store(start=self.repo), self.untracked)

    def test_glob_spans_both_places_with_the_same_shadowing(self):
        (self.tracked / "lemat_bulk_fmax1").mkdir()      # hidden by the store's copy
        (self.tracked / "lemat_bulk_tracked").mkdir()
        found = data_glob("lemat*", start=self.repo)
        self.assertEqual(found, [self.untracked / "lemat-bulk",
                                 self.untracked / "lemat_bulk_fmax1",
                                 self.tracked / "lemat_bulk_tracked"])

    def test_unconfigured_there_is_one_root(self):
        config_path().unlink()
        self.assertEqual(data_roots(start=self.repo), [self.tracked])
        self.assertEqual(data_path("lemat-bulk", start=self.repo), self.tracked / "lemat-bulk")


class TestCacheRunsWandb(_Isolated):
    def test_runs_path_joins(self):
        store = self.root / "store"
        self.configure(store)
        self.assertEqual(runs_path("abc123", "config.yaml"), store / "runs" / "abc123" / "config.yaml")

    def test_wandb_dir_from_config(self):
        store = self.root / "store"
        self.configure(store)
        self.assertEqual(wandb_dir(), store)

    def test_wandb_dir_is_required_once_there_is_a_config(self):
        """Otherwise wandb would quietly write wandb/ into whatever directory it ran in."""
        self.configure(self.root / "store", **{WANDB_DIR_ENV_VAR: None})
        with self.assertRaisesRegex(StoreConfigError, "WANDB_DIR"):
            wandb_dir()


class TestResolveStorePath(_Isolated):
    def setUp(self):
        super().setUp()
        self.repo = self.make_repo()
        self.store = self.root / "store"
        self.configure(self.store)
        (self.repo / "data" / "mp_20").mkdir(parents=True)
        os.chdir(self.repo)
        self.addCleanup(os.chdir, Path(__file__).resolve().parents[3])

    def test_data_goes_through_the_lookup(self):
        self.assertEqual(resolve_store_path("data/mp_20/train.csv"),
                         self.repo / "data" / "mp_20" / "train.csv")
        self.assertEqual(resolve_store_path(Path("data/lemat-bulk/lemat_pbe.csv.gz")),
                         self.store / "data" / "lemat-bulk" / "lemat_pbe.csv.gz")

    def test_cache_and_runs_reroot(self):
        self.assertEqual(resolve_store_path("cache/x/data.pkl.gz"), self.store / "cache" / "x" / "data.pkl.gz")
        self.assertEqual(resolve_store_path("runs/formula_energy/ensemble.pt"),
                         self.store / "runs" / "formula_energy" / "ensemble.pt")

    def test_absolute_and_unprefixed_paths_pass_through(self):
        explicit = self.root / "somewhere" / "else.csv.gz"
        self.assertEqual(resolve_store_path(explicit), explicit)
        self.assertEqual(resolve_store_path("generated/run.csv.gz"), Path("generated/run.csv.gz"))


class TestShellParity(_Isolated):
    """`scripts/wyformer_paths.sh` and this module must agree, or a launcher and the Python
    it starts disagree about where a run lives."""

    def shell(self, key: str, fallback: str = "/fallback") -> tuple[int, str]:
        result = subprocess.run(
            ["bash", "-c", f'. "{SHELL_HELPER}"; wyformer_path {key} {fallback}'],
            capture_output=True, text=True, env=dict(os.environ), check=False)
        return result.returncode, result.stdout.strip()

    def python(self, key: str) -> tuple[int, str]:
        try:
            configured = paths._configured(key)
        except StoreConfigError:
            return 1, ""
        return 0, ("/fallback" if configured is None else str(configured[0]))

    def test_agree(self):
        cases = {
            "complete": "WYFORMER_DATA=/s/data\nWYFORMER_CACHE=/s/cache\nWYFORMER_RUNS=/s/runs\nWANDB_DIR=/s\n",
            "comments and passthrough": "# c\n\nWANDB_API_KEY\n  WYFORMER_CACHE=/s/cache  \nWYFORMER_CACHE=/t/cache\n",
            "key missing": "WYFORMER_DATA=/s/data\n",
            "relative": "WYFORMER_CACHE=s/cache\n",
            "expansion": "WYFORMER_CACHE=$HOME/cache\n",
            "tilde": "WYFORMER_CACHE=~/cache\n",
            "empty": "WYFORMER_CACHE=\n",
        }
        for name, text in cases.items():
            self.write_config(text)
            with self.subTest(case=name):
                self.assertEqual(self.shell(CACHE_ENV_VAR), self.python(CACHE_ENV_VAR))

    def test_agree_without_a_config_file(self):
        self.assertEqual(self.shell(CACHE_ENV_VAR), (0, "/fallback"))
        self.assertEqual(self.python(CACHE_ENV_VAR), (0, "/fallback"))

    def test_environment_wins_in_both(self):
        self.write_config("WYFORMER_CACHE=/s/cache\n")
        with mock.patch.dict(os.environ, {CACHE_ENV_VAR: "/from/env"}):
            self.assertEqual(self.shell(CACHE_ENV_VAR), (0, "/from/env"))
            self.assertEqual(self.python(CACHE_ENV_VAR), (0, "/from/env"))


class TestMain(_Isolated):
    def test_reports_a_complete_config(self):
        self.configure(self.root / "store")
        with mock.patch("builtins.print"):
            self.assertEqual(paths.main(), 0)

    def test_fails_on_an_incomplete_config(self):
        self.write_config("WYFORMER_DATA=/s/data\n")
        with mock.patch("builtins.print"):
            self.assertEqual(paths.main(), 1)


if __name__ == "__main__":
    unittest.main()
