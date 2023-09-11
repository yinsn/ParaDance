import os
from datetime import datetime
from typing import Optional


def ensure_study_directory(study_path: Optional[str], study_name: Optional[str]) -> str:
    """
    Ensure that a directory with the given study_name exists in the specified study_path.

    Args:
        study_path (str, optional): The base directory path. Defaults to the current directory.
        study_name (str, optional): The name of the study directory. If not provided, the current
                                    system time in the format 'YYYY_MM_DD_HH_MM' will be used.

    Returns:
        str: The absolute path to the study directory.
    """

    if study_path is None:
        study_path = os.getcwd()

    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M")

    if study_name is None:
        study_name = date_string
    else:
        study_name = f"{date_string}_{study_name}"

    full_path = os.path.join(study_path, study_name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return os.path.abspath(full_path)
