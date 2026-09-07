"""Make uv aware of packages that live in the container image.

Without this, `uv pip install` resolves torch from PyPI and installs a second
copy into the venv -- on luna that meant torch 2.14.0 with **CUDA 13** wheels
(nvidia-cublas 13.x, nvidia-cudnn-cu13), which cannot run on the host's CUDA
12.4 driver. The container's torch 2.14.0+cu126 is the one that works.

`uv pip install` inspects only the target venv's own site-packages; it ignores
`include-system-site-packages`. So the venv is created with
--system-site-packages (which makes the container's modules *importable*), and
this script additionally copies the `*.dist-info` metadata of the container's
torch, and everything it transitively requires, into the venv's site-packages
(which is what makes uv consider them *installed*).

The metadata is copied rather than symlinked, deliberately: uv enumerates
site-packages without following symlinks, so a symlinked `foo.dist-info` is
invisible to it, and uv then tries to install over the symlink and dies with
"Read-only file system". The directories are metadata only, a few hundred KB.

A constraints file is also written, pinning each of these to the exact version
the container ships. Pass it to `uv pip install -c` so the resolver cannot
quietly pick a newer torch and drag in the wrong CUDA wheels; the local version
segment (+cu126) means a mismatch fails loudly instead.

Run with the *venv's* python, from inside the container.
"""

import argparse
import fnmatch
import shutil
import sys
import sysconfig
from importlib.metadata import Distribution, distributions
from pathlib import Path

# Only packages matching these are reused from the container. Keep the set as
# small as it can be: it is the CUDA stack, where a mismatch with the host
# driver is fatal, plus torch itself. Everything else torch happens to depend on
# (fsspec, jinja2, networkx, sympy, ...) is left to uv, so it can pick versions
# that satisfy the rest of WyFormer's requirements. Pinning those too made the
# resolution unsatisfiable: the container's fsspec 2026.7.0 is newer than the
# <=2026.6.0 that `datasets` allows.
DEFAULT_REUSE = ["torch", "triton", "pytorch-triton", "nvidia-*", "cuda-*"]


def normalise(name: str) -> str:
    return name.lower().replace("_", "-")


def venv_site_packages() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def closure(roots: set[str]) -> dict[str, Distribution]:
    """Root packages plus everything they transitively require."""
    by_name: dict[str, Distribution] = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        if name:
            by_name.setdefault(normalise(name), dist)

    seen: dict[str, Distribution] = {}
    queue = list(roots)
    while queue:
        name = normalise(queue.pop())
        if name in seen or name not in by_name:
            continue
        dist = by_name[name]
        seen[name] = dist
        for raw in dist.requires or []:
            # "nvidia-cublas-cu12==1.2.3 ; extra == 'opt'" -> skip optional extras
            if ";" in raw and "extra ==" in raw.split(";", 1)[1]:
                continue
            req = raw.split(";")[0].strip()
            for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "[", "(", " "):
                req = req.split(sep)[0]
            if req:
                queue.append(req.strip())
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", default=["torch"],
                        help="Packages whose dependency closure is considered for reuse.")
    parser.add_argument("--reuse", nargs="+", default=DEFAULT_REUSE,
                        help="Name globs within that closure to actually reuse.")
    parser.add_argument("--constraints-out", type=Path,
                        help="Write a constraints file pinning the reused versions.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if sys.prefix == sys.base_prefix:
        print("error: run this with the venv's python, not the container's", file=sys.stderr)
        return 1

    venv = venv_site_packages()
    copied, skipped, constraints = [], [], []

    # The container's site-packages is wherever the distributions actually are;
    # do not compute it. The image may use Debian's dist-packages layout, conda's
    # site-packages, or something else again.
    for name, dist in sorted(closure(set(args.roots)).items()):
        if not any(fnmatch.fnmatch(name, pattern) for pattern in args.reuse):
            skipped.append(f"{name} (not in --reuse; uv will resolve it)")
            continue
        info = Path(str(dist._path))  # the .dist-info directory
        constraints.append(f"{name}=={dist.version}")
        if info.parent == venv:
            skipped.append(f"{name} (already installed in the venv)")
            continue
        target = venv / info.name
        if target.exists():
            skipped.append(f"{name} (already present in the venv)")
            continue
        if not args.dry_run:
            if target.is_symlink():  # left over from an earlier approach
                target.unlink()
            shutil.copytree(info, target)
        copied.append(f"{name}=={dist.version}  <- {info.parent}")

    if args.constraints_out and not args.dry_run:
        args.constraints_out.write_text(
            "# Versions provided by the container image; do not let the\n"
            "# resolver move off them. Generated by reuse_container_packages.py.\n"
            + "\n".join(sorted(constraints)) + "\n"
        )

    print(f"venv site-packages: {venv}")
    print(f"\ncopied {len(copied)} metadata dir(s) so uv treats them as installed:")
    for item in copied:
        print(f"  + {item}")
    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for item in skipped:
            print(f"  - {item}")
    if args.constraints_out:
        print(f"\nconstraints written to {args.constraints_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
