import logging
from abc import ABCMeta, abstractmethod
from typing import Optional

import optuna

from ..evaluation.calculator import Calculator
from .set_path import ensure_study_directory


class BaseObjective(metaclass=ABCMeta):
    """
    BaseObjective serves as an abstract base class for objective optimization.
    It provides core functionalities for setting up and optimizing objectives
    using the `optuna` library.

    Attributes:
        calculator (Calculator): An instance of Calculator for various evaluations.
        direction (str): Direction for optimization (e.g., "minimize" or "maximize").
        formula (str): Mathematical formula representing the objective.
        first_order (bool): Whether to use first order optimization or not. Default is False.
        power (bool): If True, includes power in the optimization. Default is True.
        dirichlet (bool): If True, uses Dirichlet distribution for optimization. Default is True.
        weights_num (Optional[int]): Number of weights for optimization.
        study_name (Optional[str]): Name of the study.
        study_path (Optional[str]): Path to the study directory.
        full_path (str): Full path combining study_path and study_name.
        study (Study): Optuna study object for optimization.
        logger (logging.Logger): Logger object for logging optimization progress.

    Methods:
        build_logger(process_id) -> None:
            Constructs a logger for the optimization process.
        objective(trial) -> float:
            Abstract method for objective function to be overridden in derived classes.
        optimize(n_trials) -> None:
            Optimize the objective for a specified number of trials.
    """

    def __init__(
        self,
        calculator: Calculator,
        direction: str,
        formula: str,
        first_order: bool = False,
        power: bool = True,
        dirichlet: bool = True,
        weights_num: Optional[int] = None,
        study_name: Optional[str] = None,
        study_path: Optional[str] = None,
        save_study: bool = True,
    ) -> None:
        """
        Initializes the BaseObjective class with necessary parameters.

        Args:
            calculator (Calculator): Calculator for evaluations.
            direction (str): Direction of optimization.
            weights_num (int): Number of weights.
            formula (str): Formula for objective.
            first_order (bool, optional): Use first order optimization or not. Defaults to False.
            power (bool, optional): Include power in optimization. Defaults to True.
            dirichlet (bool, optional): Use Dirichlet distribution. Defaults to True.
            study_name (Optional[str], optional): Name of the study. Defaults to None.
            study_path (Optional[str], optional): Path to the study directory. Defaults to None.
        """
        self.calculator = calculator
        self.direction = direction
        self.formula = formula
        self.first_order = first_order
        self.power = power
        self.dirichlet = dirichlet
        self.study_name = study_name
        self.study_path = study_path
        self.save_study = save_study
        self.full_path = ensure_study_directory(study_path, study_name)
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{self.full_path}/paradance_storage.db",
            engine_kwargs={"connect_args": {"timeout": 60}},
        )
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
        if weights_num is not None:
            self.weights_num = weights_num
        else:
            self.weights_num = self.get_weights_num()

    def build_logger(self, process_id: Optional[int] = None) -> None:
        """
        Constructs a logger for the optimization process.

        Args:
            process_id (Optional[int], optional): ID of the process, used for creating unique log filenames.
                                                  Defaults to None.
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
        """
        Abstract method for the objective function. Must be overridden in derived classes.

        Args:
            trial (Trial): Optuna trial instance.

        Returns:
            float: Objective value for the given trial.
        """
        raise NotImplementedError

    def get_weights_num(self) -> int:
        """
        Returns the number of weights.

        Returns:
            int: Number of weights.
        """
        return len(self.calculator.selected_columns)

    def optimize(self, n_trials: int) -> None:
        """
        Optimizes the objective for a set number of trials.

        Args:
            n_trials (int): Number of trials for optimization.
        """
        self.build_logger()
        self.study.optimize(self.objective, n_trials=n_trials)
