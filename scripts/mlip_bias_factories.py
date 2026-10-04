"""ASE calculator factories for leaderboard models outside the WyFormer registry."""

from __future__ import annotations

import os
import json
from pathlib import Path

from wyckoff_transformer.cryspr.calculator import resolve_model_path
from wyckoff_transformer.paths import cache_root

PROPHET_OAME_MBD = (
    "https://huggingface.co/kairosmaterial/prophet/resolve/"
    "f9a54df874ba52b8b35e7d3f7f348b46e0135563/prophet-oame-mbd.pt"
)


def build_prophet(*, device: str):
    from prophet import KairosCalculator

    return KairosCalculator(model_path=str(resolve_model_path(PROPHET_OAME_MBD)),
                            use_kernel=False, use_compile=False, device=device)


def build_nequip(*, device: str):
    from nequip.integrations.ase import NequIPCalculator

    # NequIP's compiled routes need either Torch <2.10 (TorchScript) or Triton
    # (AOTInductor). The eager package loader works with the custom Torch here.
    cache = Path(os.environ.setdefault("NEQUIP_CACHE_DIR", str(cache_root() / "nequip")))
    model_id = "mir-group/NequIP-OAM-XL:0.1"
    cached_packages = []
    for metadata in cache.glob("*.metadata.json"):
        try:
            info = json.loads(metadata.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        package = metadata.with_name(metadata.name.replace(".metadata.json", ".nequip.zip"))
        if info.get("model_id") == model_id and package.is_file():
            cached_packages.append(package)
    locator = str(max(cached_packages, key=lambda path: path.stat().st_mtime)) \
        if cached_packages else f"nequip.net:{model_id}"
    return NequIPCalculator._from_saved_model(
        locator, device=device,
        chemical_species_to_atom_type_map=True, neighborlist_backend="matscipy",
    )
