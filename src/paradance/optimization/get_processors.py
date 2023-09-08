import os
import platform
import subprocess


def get_logical_processors_count() -> int:
    """
    Fetch the count of logical processors based on the system type.

    Returns:
        int: Number of logical processors.

    Raises:
        ValueError: If the system is not Windows, Darwin (MacOS), or Linux.
    """

    system: str = platform.system()

    if system == "Windows":
        return int(os.environ["NUMBER_OF_PROCESSORS"])
    elif system == "Darwin":
        return int(
            subprocess.check_output(["sysctl", "-n", "hw.ncpu"]).decode("utf-8").strip()
        )
    elif system == "Linux":
        command: str = "grep -c ^processor /proc/cpuinfo"
        return int(subprocess.check_output(command, shell=True).decode("utf-8").strip())
    else:
        raise ValueError(f"Unsupported OS: {system}")
