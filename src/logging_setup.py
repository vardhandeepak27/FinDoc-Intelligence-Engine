"""Logging configuration shared by every entry point."""
import logging
import os
import sys

from config import settings

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger once, at program start.

    Call this only from entry points (``if __name__ == "__main__"`` blocks,
    and later the FastAPI startup). Library code should only do
    ``logger = logging.getLogger(__name__)`` and never configure logging.
    """
    logging.basicConfig(
        level=(level or settings.log_level).upper(),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stdout,
        force=True,  # replace any handlers set up earlier (e.g. by a library)
    )

    # Third-party libraries can be very chatty at INFO level
    for noisy in (
        "sentence_transformers", "httpx", "httpx2", "urllib3", "filelock",
        "huggingface_hub",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Windows without Developer Mode can't create symlinks; the Hugging Face
    # cache still works (it copies instead), so the warning is just noise.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
