import logging
from abc import ABCMeta, abstractmethod
from typing import Optional

import optuna

from ..evaluation.calculator import Calculator
from .set_path import ensure_study_directory


class BaseObjective(metaclass=ABCMeta):
    """
    This class provides methods to optimize obejective.
    """

    def __init__(
        self,
        calculator: Calculator,
        direction: str,
        weights_num: int,
        formula: str,
        first_order: bool = False,
        power: bool = True,
        dirichlet: bool = True,
        study_name: Optional[str] = None,
        study_path: Optional[str] = None,
    ) -> None:
        self.calculator = calculator
        self.direction = direction
        self.weights_num = weights_num
        self.formula = formula
        self.first_order = first_order
        self.power = power
        self.dirichlet = dirichlet
        self.study_name = study_name
        self.study_path = study_path
        self.full_path = ensure_study_directory(study_path, study_name)
        storage_path = f"sqlite:///{self.full_path}/paradance_storage.db"
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage_path,
            load_if_exists=True,
        )

    def build_logger(self, process_id: Optional[int] = None) -> None:
        """Build logger for paradance.

        Args:
            log_file (str): log file path.
            process_id (int, optional): ID of the process. Used to create unique log filenames in parallel execution.
        """

        if process_id is not None:
            log_filename = f"{self.full_path}/paradance_{process_id}.log"
        else:
            log_filename = f"{self.full_path}/paradance.log"

        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        self.logger = optuna.logging.get_logger(
            f"paradance_{process_id}" if process_id else "optuna"
        )
        self.logger.addHandler(file_handler)

    @abstractmethod
    def objective(self, trial: optuna.trial.Trial) -> float:
        """Objective function to optimize."""
        raise NotImplementedError

    def optimize(self, n_trials: int) -> None:
        """Optimize the objective."""
        self.build_logger()
        self.study.optimize(self.objective, n_trials=n_trials)
