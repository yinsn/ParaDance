from abc import ABCMeta, abstractmethod

import optuna


class BaseObjective(metaclass=ABCMeta):
    """
    This class provides methods to optimize obejective.
    """

    def __init__(self, direction: str, formula: str, dirichlet: bool = True) -> None:
        self.study = optuna.create_study(direction=direction)
