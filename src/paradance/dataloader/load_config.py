import logging
import os
from typing import Any, Dict, Optional, cast

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract from 'Lova' project (https://github.com/yinsn/Lova/blob/develop/src/lova/dataloaders/load_config.py).
    Load the configuration from a YAML file.

    Args:
        file_path (str, optional): The path to the YAML file. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
    """
    config: Dict[str, Any] = {}
    if file_path is None:
        logger.info(
            "No configuration file path provided, using default 'config.yml' file in current directory."
        )
        file_path = os.path.abspath("config.yml")

    else:
        file_path = os.path.abspath(file_path)

    logger.info(f"Loading configuration ...")
    with open(file_path, "r") as file:
        loaded_config = yaml.safe_load(file)

    if loaded_config is None:
        config = {}
    elif isinstance(loaded_config, dict):
        config = cast(Dict[str, Any], loaded_config)
    else:
        raise TypeError(
            "Configuration file must contain a mapping at the top level, "
            f"got {type(loaded_config).__name__} instead."
        )

    return config
