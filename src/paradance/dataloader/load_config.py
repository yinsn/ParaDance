import logging
import os
from typing import Dict, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(file_path: Optional[str] = None) -> Dict:
    """
    Extract from 'Lova' project (https://github.com/yinsn/Lova/blob/develop/src/lova/dataloaders/load_config.py).
    Load the configuration from a YAML file.

    Args:
        file_path (str, optional): The path to the YAML file. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
    """
    config = {}
    if file_path is None:
        logger.info(
            "No configuration file path provided, using default 'config.yml' file in current directory."
        )
        file_path = os.path.abspath("config.yml")

    else:
        file_path = os.path.abspath(file_path)

    logger.info(f"Loading configuration ...")
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    return config
