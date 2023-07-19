from abc import ABCMeta, abstractmethod
from typing import Optional

import optuna


class BaseObjective(metaclass=ABCMeta):
    """
    This class provides methods to optimize obejective.
    """

    def __init__(
        self,
        direction: str,
        formula: str,
        dirichlet: bool = True,
        log_file: Optional[str] = None,
    ) -> None:
        self.study = optuna.create_study(direction=direction)
