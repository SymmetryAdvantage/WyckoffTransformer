"""Registry of foundation MLIPs usable as the CrySPR relaxation calculator.

Each entry corresponds to a model on the `Matbench Discovery
<https://matbench-discovery.materialsproject.org/>`_ leaderboard and knows how
to construct an ASE calculator for it.

The backends live in mutually incompatible dependency sets — several pin exact
CUDA-suffixed ``torch``/``torch-scatter`` builds and GRACE runs on TensorFlow
rather than PyTorch — so a given environment can normally import only one of
them.  Every backend import is therefore deferred until :func:`build_calculator`
actually needs it, and the registry itself stays importable everywhere.
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MlipSpec:
    """Description of one leaderboard MLIP.

    Attributes:
        name: Matbench Discovery leaderboard model name.
        backend: Key selecting the builder function in :data:`_BUILDERS`.
        checkpoint: Default checkpoint locator.  Its meaning is
            backend-specific: a URL, a local path, or a tag the backend
            resolves and downloads itself.
        default_dtype: Precision the model was released at; passed on only to
            backends that accept a dtype.
        pip: Requirement string that provides the backend.
    """

    name: str
    backend: str
    checkpoint: Optional[str] = None
    default_dtype: str = "float32"
    pip: str = ""


# Ordered by Matbench Discovery CPS rank as of 2026-08-14.  TACE-OAM-RRA-Preview
# (CPS 0.905) sits between TECE and EquiformerV3 on the leaderboard but is hidden
# from the default view as a preview entry, and shares the TACE backend with
# TECE-OAM-RRA-1.0.
MLIP_REGISTRY: dict[str, MlipSpec] = {
    "TECE-OAM-RRA-1.0": MlipSpec(
        name="TECE-OAM-RRA-1.0",
        backend="tace",
        checkpoint="https://huggingface.co/xvzemin/tace-foundations/resolve/main/TECE-OAM-RRA-1.0.pt",
        pip="tace",
    ),
    "EquFlashV2": MlipSpec(
        name="EquFlashV2",
        backend="equflash",
        checkpoint="https://figshare.com/ndownloader/files/65435007",
        pip="equflash",
    ),
    "TACE-OAM-RRA-Preview": MlipSpec(
        name="TACE-OAM-RRA-Preview",
        backend="tace",
        checkpoint="https://huggingface.co/xvzemin/tace-foundations/resolve/main/TACE-OAM-RRA-Preview.pt",
        pip="tace",
    ),
    "EquiformerV3+DeNS-OAM": MlipSpec(
        name="EquiformerV3+DeNS-OAM",
        backend="equiformer_v3",
        checkpoint=(
            "https://huggingface.co/mirror-physics/equiformer_v3/resolve/main/"
            "checkpoint/omat24-mptrj-salex_gradient.pt"
        ),
        pip="equiformer-v3",
    ),
    "GRACE-3L-OAM-L": MlipSpec(
        name="GRACE-3L-OAM-L",
        # grace_fm resolves and caches the checkpoint from this tag itself.
        backend="grace",
        checkpoint="GRACE-3L-OMAT-large-ft-AM",
        pip="tensorpotential",
    ),
    "PET-OAM-XL": MlipSpec(
        name="PET-OAM-XL",
        backend="pet",
        checkpoint="https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-oam-xl-v1.0.0.ckpt",
        pip="upet",
    ),
    # The MACE family predates this registry and keeps its own URL table; it is
    # listed here so that --mlip can select it uniformly.
    "MACE-MPA-0-medium": MlipSpec(
        name="MACE-MPA-0-medium",
        backend="mace",
        checkpoint="MACE-MPA-0-medium",
        default_dtype="float64",
        pip="mace-torch",
    ),
    "MACE-OMAT-0-medium": MlipSpec(
        name="MACE-OMAT-0-medium",
        backend="mace",
        checkpoint="MACE-OMAT-0-medium",
        default_dtype="float64",
        pip="mace-torch",
    ),
}


def _resolve_device(device: str) -> str:
    """Turn ``"auto"`` into a concrete device string."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        # Torch-free backends (GRACE) manage device placement themselves.
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _local_checkpoint(checkpoint: str) -> Path:
    """Return a local path for *checkpoint*, downloading it if it is a URL."""
    from .calculator import resolve_model_path

    return resolve_model_path(checkpoint)


def _build_mace(spec: MlipSpec, checkpoint: str, device: str, dtype: str) -> Calculator:
    from .calculator import build_mace_calculator

    return build_mace_calculator(model=checkpoint, device=device, dtype=dtype)


def _build_tace(spec: MlipSpec, checkpoint: str, device: str, dtype: str) -> Calculator:
    from tace.interface.ase.calculator import TACEAseCalc

    # TACE resolves its own foundation-model tags; only fetch when we were
    # handed a URL, so that a bare tag keeps using TACE's own cache.
    model = checkpoint if not checkpoint.startswith("http") else str(_local_checkpoint(checkpoint))
    # cuEQ is a CUDA-only acceleration path; enabling it on CPU makes the
    # backend allocate on the GPU anyway.  Same trap as in calculator.py.
    return TACEAseCalc(
        model=model,
        device=device,
        dtype=dtype,
        enable_cue=device.startswith("cuda") and _cueq_importable(),
    )


def _cueq_importable() -> bool:
    try:
        import cuequivariance_torch  # noqa: F401

        return True
    except Exception:
        return False


def _build_equflash(spec: MlipSpec, checkpoint: str, device: str, dtype: str) -> Calculator:
    from GGNN.common.calculator import UCalculator

    return UCalculator(checkpoint_path=str(_local_checkpoint(checkpoint)), cpu=device == "cpu")


def _build_equiformer_v3(spec: MlipSpec, checkpoint: str, device: str, dtype: str) -> Calculator:
    from equiformer_v3.core.common.relaxation.ase_utils import OCPCalculator

    return OCPCalculator(checkpoint_path=str(_local_checkpoint(checkpoint)), cpu=device == "cpu")


#: Hard per-process GPU cap for TensorFlow, in MiB. See _build_grace.
GRACE_GPU_MEMORY_LIMIT_MIB = int(os.environ.get("GRACE_GPU_MEMORY_LIMIT_MIB", "4096"))


def _build_grace(spec: MlipSpec, checkpoint: str, device: str, dtype: str) -> Calculator:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        gpus = []
    for gpu in gpus:
        # set_memory_growth alone is NOT enough under multiprocessing: it only
        # stops TF from claiming the whole card up front, it does not bound what
        # one process eventually grows to. Eight workers each growing without a
        # ceiling exhausted a 46 GB card, after which TF's cuda_executor logged
        # "failed to allocate ... CUDA_ERROR_OUT_OF_MEMORY" and retried forever
        # instead of raising -- the workers hung and the log grew to 789 GB.
        # A hard logical-device limit makes exhaustion impossible by construction.
        try:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=GRACE_GPU_MEMORY_LIMIT_MIB)],
            )
        except RuntimeError:  # device already initialised; fall back to growth
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    from tensorpotential.calculator import grace_fm

    return grace_fm(checkpoint)


def _build_pet(spec: MlipSpec, checkpoint: str, device: str, dtype: str) -> Calculator:
    from upet.calculator import UPETCalculator

    # upet keeps local files on a separate kwarg from its HuggingFace model tags,
    # and recovers model/size/version from the standard checkpoint filename.
    if checkpoint.startswith("http") or Path(checkpoint).exists():
        return UPETCalculator(checkpoint_path=str(_local_checkpoint(checkpoint)), device=device)
    return UPETCalculator(model=checkpoint, device=device)


_BUILDERS: dict[str, Callable[[MlipSpec, str, str, str], Calculator]] = {
    "mace": _build_mace,
    "tace": _build_tace,
    "equflash": _build_equflash,
    "equiformer_v3": _build_equiformer_v3,
    "grace": _build_grace,
    "pet": _build_pet,
}


def build_calculator(
    mlip: str,
    device: str = "auto",
    dtype: Optional[str] = None,
    checkpoint: Optional[Union[str, Path]] = None,
) -> Calculator:
    """Build the ASE calculator for a registered MLIP.

    Args:
        mlip: Key into :data:`MLIP_REGISTRY`.
        device: ``"cpu"``, ``"cuda"``, or ``"auto"``.
        dtype: Override the model's released precision.
        checkpoint: Override the registry's checkpoint locator with a local
            path or URL.

    Returns:
        An ASE :class:`~ase.calculators.calculator.Calculator`.

    Raises:
        KeyError: If *mlip* is not registered.
    """
    if mlip not in MLIP_REGISTRY:
        raise KeyError(f"Unknown MLIP {mlip!r}. Known: {', '.join(MLIP_REGISTRY)}")
    spec = MLIP_REGISTRY[mlip]
    device = _resolve_device(device)
    dtype = dtype or spec.default_dtype
    locator = str(checkpoint) if checkpoint is not None else spec.checkpoint
    if locator is None:
        raise ValueError(f"{mlip} has no default checkpoint; pass one explicitly.")

    calc = _BUILDERS[spec.backend](spec, locator, device, dtype)
    logger.info("Built %s calculator (backend=%s) on %s [%s]", spec.name, spec.backend, device, dtype)
    return calc
