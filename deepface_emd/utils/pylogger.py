import logging
from rich.logging import RichHandler

def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%m/%d %H:%M:%S]",
        handlers=[RichHandler()]
    )

    logger = logging.getLogger(name)
    return logger