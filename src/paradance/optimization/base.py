import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import optuna
from pydantic import BaseModel

from ..evaluation import Calculator, LogarithmPCACalculator
from .set_path import ensure_study_directory


class BaseObjectiveConfig(BaseModel):
    """
    Base configuration class for defining optimization objectives.

    Attributes:
        direction (Optional[str]): Specifies the optimization direction. This can be either 'minimize' or 'maximize'.
                                   Default is None, which should be overridden in subclass or instance.
        formula (Optional[str]): The mathematical formula or expression used in the objective function. Default is None.
        first_order (Optional[bool]): Indicates whether to use first-order optimization. Defaults to False.
        power (Optional[bool]): Indicates whether to apply a power transform in the objective. Defaults to True.
        dirichlet (Optional[bool]): Specifies if a Dirichlet process should be used. Defaults to False.
        weights_num (Optional[int]): The number of weights or parameters to optimize. Default is None.
        study_name (Optional[str]): The name of the optimization study. Default is None.
        study_path (Optional[str]): Filesystem path where study results are stored. Default is None.
        save_study (Optional[bool]): Flag indicating whether to persist the study to disk. Defaults to True.
    """

    direction: Optional[str] = None
    formula: Optional[str] = None
    first_order: Optional[bool] = False
    power: Optional[bool] = True
    dirichlet: Optional[bool] = False
    weights_num: Optional[int] = None
    study_name: Optional[str] = None
    study_path: Optional[str] = None
    save_study: Optional[bool] = True


class BaseObjective(metaclass=ABCMeta):
    """
    BaseObjective serves as an abstract base class for objective optimization.
    It provides core functionalities for setting up and optimizing objectives
    using the `optuna` library.

    Attributes:
        calculator (Calculator): An instance of Calculator for various evaluations.
        direction (Optional[str]): Direction for optimization (e.g., "minimize" or "maximize").
        formula (Optional[str]): Mathematical formula representing the objective.
        first_order (bool): Whether to use first order optimization or not. Default is False.
        power (bool): If True, includes power in the optimization. Default is True.
        dirichlet (bool): If True, uses Dirichlet distribution for optimization. Default is False.
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
        calculator: Union[Calculator, LogarithmPCACalculator],
        direction: Optional[str] = None,
        formula: Optional[str] = None,
        first_order: Optional[bool] = False,
        power: Optional[bool] = True,
        dirichlet: Optional[bool] = False,
        weights_num: Optional[int] = None,
        study_name: Optional[str] = None,
        study_path: Optional[str] = None,
        save_study: Optional[bool] = True,
        config: Optional[Dict] = None,
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
            dirichlet (bool, optional): Use Dirichlet distribution. Defaults to False.
            study_name (Optional[str], optional): Name of the study. Defaults to None.
            study_path (Optional[str], optional): Path to the study directory. Defaults to None.
        """
        self.calculator = calculator
        if config is not None:
            self.config = BaseObjectiveConfig(**config)
        else:
            self.config = BaseObjectiveConfig(
                direction=direction,
                formula=formula,
                first_order=first_order,
                power=power,
                dirichlet=dirichlet,
                weights_num=weights_num,
                study_name=study_name,
                study_path=study_path,
                save_study=save_study,
            )

        self.direction = self.config.direction
        self.formula = self.config.formula
        self.first_order = self.config.first_order
        self.power = self.config.power
        self.dirichlet = self.config.dirichlet
        self.weights_num = self.config.weights_num
        self.study_name = self.config.study_name
        self.study_path = self.config.study_path
        self.save_study = self.config.save_study
        self._prepare_study()

    def _prepare_study(self) -> None:
        """
        Prepares the study by setting up the study directory, storage, and the study object itself.

        This method configures the study by ensuring the study directory exists and initializing the study with the
        specified direction, name, and storage backend. If `weights_num` is not already defined, it determines the number
        of weights. Finally, it initializes the `best_params` attribute as a zero array of size `weights_num`.

        Side effects:
            - Creates or ensures the existence of a directory for the study.
            - Initializes or loads an Optuna study with the specified configuration.
            - Updates `self.full_path` with the path to the study directory.
            - Initializes `self.study` with the created or loaded Optuna study.
            - Determines `self.weights_num` if it is not already specified.
            - Initializes `self.best_params` as a NumPy zero array of size `self.weights_num`.
        """
        self.full_path = ensure_study_directory(self.study_path, self.study_name)
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{self.full_path}/paradance_storage.db",
            engine_kwargs={"connect_args": {"timeout": 120}},
        )
        self.study = optuna.create_study(
            direction=self.direction,
            study_name=self.study_name,
            storage=storage,
            load_if_exists=True,
        )
        if self.weights_num is None:
            self.weights_num = self.get_weights_num()
        self.best_params: np.ndarray = np.zeros(self.weights_num)

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
    def evaluate_custom_weights(self, weights: List[float]) -> List[float]:
        """
        Evaluates the custom weights for the given list of weights.

        Args:
            weights (List[float]): List of weights to evaluate.

        Returns:
            float: Evaluation score for the given weights.
        """
        raise NotImplementedError

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
