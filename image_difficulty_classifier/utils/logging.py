import logging
import sys
import os
import time
from typing import Optional


def setup_logging(level: int = logging.INFO, name: Optional[str] = None, log_dir: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Always log to stdout
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Optionally also log to a file in the model directory
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            # Absolute epoch time integer appended to filename
            ts = int(time.time())
            log_path = os.path.join(log_dir, f"train_{ts}.log")
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


