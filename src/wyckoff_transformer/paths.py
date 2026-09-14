"""Locating the ``data``, ``cache`` and ``runs`` roots, and W&B's directory.

Where these live differs per machine, and is recorded once per machine in a
config file the code reads itself::

    ${XDG_CONFIG_HOME:-~/.config}/wyformer/paths.env

    WYFORMER_DATA=/home/kna/.local/share/wyformer/data
    WYFORMER_CACHE=/home/kna/.local/share/wyformer/cache
    WYFORMER_RUNS=/home/kna/.local/share/wyformer/runs
    WANDB_DIR=/home/kna/.local/share/wyformer

The format is docker's ``--env-file``: ``KEY=VALUE`` per line, ``#`` comments,
and **no expansion** -- ``$HOME`` or ``~`` would reach docker literally, so they
are rejected here rather than half-supported. The same file is read by
``scripts/wyformer_paths.sh`` for the shell launchers and ``store_sync.sh``, and
can be handed to a container with ``docker run --env-file``.

A file, rather than variables exported from a shell profile, because a profile
is only read by the shells that happen to source it: cron, Jupyter kernels, IDE
run configurations, batch schedulers and containers each miss it in their own
way, and a process that missed it used to fall back to the checkout without a
word -- writing a fresh ``cache/`` or ``runs/`` there beside the real one.

Resolution, for each key:

1. the environment variable, if set -- for containers, CI and tests;
2. the config file. **When the file exists it is authoritative**: a key it does
   not set is an error, never a fallback;
3. only when there is no config file at all: ``<cwd>/<name>`` if that directory
   exists, else ``<repo>/<name>``, where ``<repo>`` is found by searching upward
   from the working directory for ``pyproject.toml``. This is what a fresh clone
   of the public repository, or CI, runs on.

``data`` is looked up in two places. Datasets tracked by git -- plainly or
through LFS -- are versioned and already on every machine that has the
repository, so they stay in ``<repo>/data``. Only untracked datasets live in
the store. :func:`data_path` looks a dataset up by its top-level name, store
first and then the repository, and a dataset that is in neither resolves to the
store, which is where it will be written. ``cache`` and ``runs`` are single
locations. ``runs`` is per-machine and not replicated; W&B holds what matters
in it (see ``AGENTS.md``).

Nothing here is derived from ``__file__``. Worktrees share one venv whose
editable install points at the main checkout, so a ``__file__``-relative path
resolves into the main checkout whichever worktree the process runs in.

These roots are unrelated to ``.cache/`` (run-local hull caches), to
``~/.cache/wyckoff_transformer/`` (downloaded model weights), and to
package-internal resource directories such as
``wyckoff_transformer/evaluation/data``, which are correctly ``__file__``-relative.
"""

import os
import sys
from pathlib import Path

#: Environment variable and config key naming the store of untracked datasets.
DATA_ENV_VAR = "WYFORMER_DATA"
#: Environment variable and config key naming the ``cache`` store.
CACHE_ENV_VAR = "WYFORMER_CACHE"
#: Environment variable and config key naming the ``runs`` directory.
RUNS_ENV_VAR = "WYFORMER_RUNS"
#: W&B's own variable. wandb creates ``$WANDB_DIR/wandb/`` beneath it.
WANDB_DIR_ENV_VAR = "WANDB_DIR"
#: Every key a config file must set.
CONFIG_KEYS = (DATA_ENV_VAR, CACHE_ENV_VAR, RUNS_ENV_VAR, WANDB_DIR_ENV_VAR)
#: File whose presence marks a checkout root.
REPO_MARKER = "pyproject.toml"


class StoreNotFoundError(RuntimeError):
    """No environment variable, config file or checkout locates a root."""


class StoreConfigError(StoreNotFoundError):
    """The config file exists but is malformed or incomplete."""


def config_path() -> Path:
    """This machine's config file, whether or not it exists."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "wyformer" / "paths.env"


def read_config(path: Path | None = None) -> dict[str, str] | None:
    """Parse a config file, or return ``None`` if there is none.

    Follows docker's env-file rules so one file serves both: a line without
    ``=`` names a variable docker passes through from the host, and is skipped.
    """
    path = config_path() if path is None else Path(path)
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    values: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        if key.startswith("export "):
            raise StoreConfigError(
                f"{path}:{lineno}: drop the 'export' -- this file is docker's env-file "
                f"format, not a shell script")
        if any(character.isspace() for character in key):
            raise StoreConfigError(f"{path}:{lineno}: variable name {key!r} contains whitespace")
        values[key] = value
    return values


def repo_root(start: Path | None = None) -> Path | None:
    """Return the checkout containing ``start``, or ``None`` if there is none."""
    origin = (Path.cwd() if start is None else Path(start)).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / REPO_MARKER).is_file():
            return candidate
    return None


def _configured(key: str) -> tuple[Path, str] | None:
    """The explicitly configured value of *key* and where it came from.

    ``None`` only when neither the environment nor a config file exists to ask;
    an existing config file that omits *key* raises instead.
    """
    from_env = os.environ.get(key)
    if from_env:
        return Path(from_env).expanduser(), "environment"
    path = config_path()
    config = read_config(path)
    if config is None:
        return None
    if key not in config:
        raise StoreConfigError(
            f"{path} exists, so it is authoritative on this machine, but it does not set "
            f"{key}. Add {key}=/absolute/path -- see docs/data_store.md.")
    value = config[key]
    if not value.startswith("/") or "$" in value:
        raise StoreConfigError(
            f"{path}: {key}={value!r} must be a literal absolute path. Nothing in this "
            f"file is expanded -- docker's --env-file would not expand it either.")
    return Path(value), str(path)


def _single_root(name: str, key: str, start: Path | None = None) -> Path:
    configured = _configured(key)
    if configured is not None:
        return configured[0]
    origin = Path.cwd() if start is None else Path(start)
    beside_cwd = origin / name
    if beside_cwd.is_dir():
        return beside_cwd
    repo = repo_root(origin)
    if repo is not None:
        return repo / name
    raise StoreNotFoundError(
        f"Cannot locate {name!r}: ${key} is not set, {config_path()} does not exist, "
        f"and {origin} is not inside a checkout. Create the config file -- see "
        f"docs/data_store.md, and docs/platforms/<host>/environment.md for this host.")


def data_store(start: Path | None = None) -> Path:
    """Where untracked datasets live, and where new datasets are written."""
    return _single_root("data", DATA_ENV_VAR, start)


def data_roots(start: Path | None = None) -> list[Path]:
    """The places a dataset is looked up in, in order: the store, then the checkout."""
    store = data_store(start)
    roots = [store]
    repo = repo_root(start)
    if repo is not None:
        tracked = repo / "data"
        if os.path.realpath(tracked) != os.path.realpath(store):
            roots.append(tracked)
    return roots


def data_path(*parts: str | Path, start: Path | None = None) -> Path:
    """Look a dataset up by its top-level name, store first, then the checkout.

    The whole dataset comes from one place: resolving by top-level name rather than
    per file is what stops a split, say, from mixing a store copy of ``train`` with a
    repository copy of ``test``. A dataset found in neither resolves to the store.
    """
    relative = Path(*parts) if parts else Path()
    roots = data_roots(start)
    if not relative.parts:
        return roots[0]
    head = relative.parts[0]
    for root in roots:
        if (root / head).exists():
            return root / relative
    return roots[0] / relative


def data_glob(pattern: str, start: Path | None = None) -> list[Path]:
    """Glob over both data locations, honouring the same shadowing as :func:`data_path`.

    A dataset present in the store hides its namesake in the checkout.
    """
    matches: dict[Path, Path] = {}
    claimed: set[str] = set()
    for root in data_roots(start):
        if not root.is_dir():
            continue
        for match in root.glob(pattern):
            relative = match.relative_to(root)
            if relative.parts and relative.parts[0] not in claimed:
                matches.setdefault(relative, match)
        claimed.update(child.name for child in root.iterdir())
    return [matches[relative] for relative in sorted(matches)]


def shadowed_datasets(start: Path | None = None) -> list[str]:
    """Top-level names present in more than one data location."""
    seen: dict[str, int] = {}
    for root in data_roots(start):
        if root.is_dir():
            for child in root.iterdir():
                seen[child.name] = seen.get(child.name, 0) + 1
    return sorted(name for name, count in seen.items() if count > 1)


def cache_root(start: Path | None = None) -> Path:
    """Return the ``cache`` store."""
    return _single_root("cache", CACHE_ENV_VAR, start)


def cache_path(*parts: str | Path) -> Path:
    """Return a path inside the ``cache`` store."""
    return cache_root().joinpath(*parts)


def runs_root(start: Path | None = None) -> Path:
    """Return the ``runs`` directory: per-machine, not replicated."""
    return _single_root("runs", RUNS_ENV_VAR, start)


def runs_path(*parts: str | Path) -> Path:
    """Return a path inside the ``runs`` directory."""
    return runs_root().joinpath(*parts)


def wandb_dir() -> Path | None:
    """The directory to pass to ``wandb.init(dir=...)``.

    ``None`` -- wandb's own default, the working directory -- only when nothing is
    configured. Passing it explicitly rather than relying on ``WANDB_DIR`` being in
    the environment is the point: the environment is what cannot be relied on.
    """
    configured = _configured(WANDB_DIR_ENV_VAR)
    return None if configured is None else configured[0]


def resolve_store_path(path: str | Path) -> Path:
    """Re-root a ``data/``, ``cache/`` or ``runs/`` relative path onto its location.

    ``data/...`` goes through :func:`data_path`'s two-place lookup. Absolute paths,
    and relative paths that start with none of those names, are returned unchanged.

    This is the migration path for module-level constants and argparse defaults that
    spell a default as a store-relative literal. Call it on the value in the function
    body, not on the constant: resolving at import would make importing the package
    raise wherever no root can be located, breaking test collection repo-wide.
    """
    candidate = Path(path)
    if candidate.is_absolute() or not candidate.parts:
        return candidate
    head, rest = candidate.parts[0], candidate.parts[1:]
    if head == "data":
        return data_path(*rest) if rest else data_store()
    if head == "cache":
        return cache_root().joinpath(*rest)
    if head == "runs":
        return runs_root().joinpath(*rest)
    return candidate


def main() -> int:
    """Print what this machine resolves to, and from where: ``python -m wyckoff_transformer.paths``."""
    path = config_path()
    print(f"config file : {path} ({'present' if path.is_file() else 'absent'})")
    try:
        for key, label in ((DATA_ENV_VAR, "data store"), (CACHE_ENV_VAR, "cache"),
                           (RUNS_ENV_VAR, "runs"), (WANDB_DIR_ENV_VAR, "wandb dir")):
            configured = _configured(key)
            source = "fallback, no config file" if configured is None else configured[1]
            print(f"{label:<12}: ", end="")
            if key == WANDB_DIR_ENV_VAR:
                print(f"{wandb_dir() or '(wandb default: working directory)'}  [{source}]")
            else:
                resolver = {DATA_ENV_VAR: data_store, CACHE_ENV_VAR: cache_root,
                            RUNS_ENV_VAR: runs_root}[key]
                print(f"{resolver()}  [{source}]")
        print("data lookup : " + "  ->  ".join(str(root) for root in data_roots()))
        shadowed = shadowed_datasets()
        if shadowed:
            print("in both     : " + ", ".join(shadowed) + "  (the store's copy wins)")
    except StoreNotFoundError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
