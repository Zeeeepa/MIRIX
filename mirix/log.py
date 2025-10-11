import logging
from typing import Optional

from mirix.settings import settings

selected_log_level = logging.DEBUG if settings.debug else logging.INFO


def get_logger(name: Optional[str] = None) -> "logging.Logger":
    logger = logging.getLogger("Mirix")
    logger.setLevel(logging.INFO)
    return logger
