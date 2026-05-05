"""MACE calculator factory with URL-based model download and caching."""
import hashlib
import logging
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import torch
from ase.calculators.calculator import Calculator

_DTYPE_BY_NAME = {"float32": torch.float32, "float64": torch.float64}


@contextmanager
def _scoped_default_dtype(dtype: torch.dtype):
    """Set ``torch``'s default dtype for the duration of the block.

    MACE's calculator both flips ``torch.set_default_dtype`` during
    construction and relies on it inside ``calculate`` (via dtype-less
    ``torch.zeros``), so we set the dtype on entry and restore it on exit.
    """
    saved = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(saved)

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "wyckoff_transformer" / "mace_models"


def resolve_model_path(model: Union[str, Path]) -> Path:
    """Return a local path to the model, downloading it first if a URL is given.

    Args:
        model: Local filesystem path or an ``http(s)://`` URL pointing to a
            MACE model file.

    Returns:
        A :class:`~pathlib.Path` to a local copy of the model.
    """
    s = str(model)
    if s.startswith("https://") or s.startswith("http://"):
        return _download_and_cache(s)
    return Path(model)


def _download_and_cache(url: str, cache_dir: Optional[Path] = None) -> Path:
    """Download *url* to *cache_dir* and return the local path.

    The destination filename is derived from a hash of the URL so that
    re-downloading is skipped on subsequent calls with the same URL.
    A temporary file is used during download so an interrupted transfer
    does not leave a partial file in the cache.

    Args:
        url: HTTPS/HTTP URL of the model file.
        cache_dir: Directory to store cached models.

    Returns:
        Path to the cached model file.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    suffix = Path(url.split("?")[0]).suffix or ".model"
    dest = cache_dir / f"{url_hash}{suffix}"
    if not dest.exists():
        logger.info("Downloading MACE model from %s → %s", url, dest)
        tmp_dest = dest.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(url, tmp_dest)
            tmp_dest.rename(dest)
        except Exception:
            if tmp_dest.exists():
                tmp_dest.unlink()
            raise
    else:
        logger.info("Using cached MACE model from %s", dest)
    return dest


def _cueq_available() -> bool:
    """Return True if cuequivariance_torch can be imported (cuEQ is usable)."""
    try:
        import cuequivariance_torch  # noqa: F401
        return True
    except Exception:
        return False


def build_mace_calculator(
    model: Union[str, Path],
    device: str = "auto",
    dtype: str = "float64",
) -> Calculator:
    """Build a MACE ASE calculator, enabling cuEQ when available.

    Args:
        model: Local path or HTTPS URL to a MACE model file.
        device: PyTorch device string, or ``"auto"`` to pick CUDA when
            available and fall back to CPU otherwise.
        dtype: Floating-point precision passed to :class:`~mace.calculators.MACECalculator`.

    Returns:
        An ASE :class:`~ase.calculators.calculator.Calculator` backed by MACE.
    """
    from mace.calculators import MACECalculator  # optional dependency

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = resolve_model_path(model)

    # Probe availability before constructing the calculator so that we never
    # partially initialise MACECalculator (which can allocate GPU memory and
    # spawn helper processes) only to tear it down and build a second instance.
    enable_cueq = _cueq_available()

    torch_dtype = _DTYPE_BY_NAME[dtype]

    class _IsolatedMACECalculator(MACECalculator):
        """MACECalculator that confines its ``torch.set_default_dtype`` change.

        MACE flips the global default dtype on construction and depends on
        it during ``calculate`` (a dtype-less ``torch.zeros``). Wrapping
        both with :func:`_scoped_default_dtype` keeps the side effect from
        leaking to other code in the same process (e.g. our model, whose
        ``nn.Linear`` weights would otherwise be created as float64).
        """

        def calculate(self, *args, **kwargs):
            with _scoped_default_dtype(torch_dtype):
                return super().calculate(*args, **kwargs)

    with _scoped_default_dtype(torch_dtype):
        calc = _IsolatedMACECalculator(
            model_paths=model_path,
            device=device,
            enable_cueq=enable_cueq,
            default_dtype=dtype,
        )
    logger.info(
        "MACE calculator loaded %s cuEQ support on %s.",
        "with" if enable_cueq else "without",
        device,
    )
    return calc
