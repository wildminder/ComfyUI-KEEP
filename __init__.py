import logging
import sys

# Configure a logger for the entire custom node package
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a handler if none exist
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(f"[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


from .modules import utils
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']