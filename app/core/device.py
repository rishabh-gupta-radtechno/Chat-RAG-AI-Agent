"""Hardware device resolution for optional GPU acceleration.

One place decides "GPU or CPU?" so every model path agrees — Docling/TableFormer,
PaddleOCR, transformers (BLIP captioning), and sentence-transformers. Controlled
by the ``DEVICE`` setting (auto | cpu | cuda):

* ``auto`` (default) — use CUDA when it is actually available, else CPU.
* ``cpu``            — force CPU even on a GPU host.
* ``cuda``           — prefer CUDA; warn and fall back to CPU if none is present.

This keeps the stack runnable unchanged on CPU-only hosts: with a CPU-only torch
build (or no visible GPU), ``auto`` simply resolves to CPU.
"""

from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _preference() -> str:
    return (get_settings().device or "auto").strip().lower()


@lru_cache(maxsize=1)
def use_cuda() -> bool:
    """Whether torch-backed models (Docling, transformers, sentence-transformers)
    should run on CUDA. Cached: hardware availability is constant per process."""
    pref = _preference()
    if pref == "cpu":
        return False

    try:
        import torch

        available = bool(torch.cuda.is_available())
    except Exception as exc:  # torch missing or broken — stay on CPU
        logger.warning("Could not query torch CUDA availability (%s); using CPU.", exc)
        available = False

    if pref == "cuda" and not available:
        logger.warning(
            "DEVICE=cuda requested but no CUDA device is available; falling back to CPU."
        )
    if available:
        logger.info("CUDA enabled for torch-backed models (DEVICE=%s).", pref)
    return available


def torch_device() -> str:
    """Return the torch device string ("cuda" or "cpu")."""
    return "cuda" if use_cuda() else "cpu"


@lru_cache(maxsize=1)
def paddle_use_gpu() -> bool:
    """Whether PaddleOCR should run on GPU.

    Independent of :func:`use_cuda` because PaddlePaddle ships separate CPU and
    GPU builds — ``torch`` seeing a GPU does not mean ``paddlepaddle-gpu`` is
    installed. Requires the GPU-compiled paddle wheel AND a visible CUDA device,
    so with the default CPU paddle build this stays False.
    """
    if _preference() == "cpu":
        return False

    try:
        import paddle

        if not paddle.is_compiled_with_cuda():
            return False
        return paddle.device.cuda.device_count() > 0
    except Exception as exc:
        logger.debug("PaddlePaddle GPU not available (%s); using CPU OCR.", exc)
        return False
