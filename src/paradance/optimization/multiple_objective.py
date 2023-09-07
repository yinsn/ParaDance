import logging
from functools import partialmethod
from typing import List, Optional

import optuna
from optuna.trial import Trial

from ..evaluation.calculator import Calculator
from .base import BaseObjective
from .construct_weights import construct_weights
from .evaluate_targets import evaluate_targets


class MultipleObjective(BaseObjective):
    """
    This class provides methods to optimize the portfolio objective.
    """

    construct_weights = partialmethod(construct_weights)

    def __init__(
        self,
        calculator: Calculator,
        direction: str,
        weights_num: int,
        formula: str,
        power: bool = False,
        first_order: bool = False,
        first_order_lower_bound: float = 1e-3,
        first_order_upper_bound: float = 1e6,
        power_lower_bound: float = 0,
        power_upper_bound: float = 1,
        dirichlet: bool = True,
        log_file: Optional[str] = None,
    ) -> None:
        """
        Initialize with direction, weights_num, formula and dirichlet.

        Args:
            direction (str ["minimize", "maximize"]): direction to optimize.
            weights_num (int): numbers of weights to search.
            formula (str): formula of targets to calculate the objective.
            dirichlet (bool, optional): Use dirichlet distribution or not. Defaults to True.
        """
        super().__init__(direction, formula, first_order, dirichlet)
        self.power = power
        self.formula = formula
        self.first_order = first_order
        self.weights_num = weights_num
        self.target_columns: List[str] = []
        self.evaluator_flags: List[str] = []
        self.groupbys: List[Optional[str]] = []
        self.calculator: Calculator = calculator
        self.power_lower_bound = power_lower_bound
        self.power_upper_bound = power_upper_bound
        self.hyperparameters: List[Optional[float]] = []
        self.first_order_lower_bound = first_order_lower_bound
        self.first_order_upper_bound = first_order_upper_bound
        if self.power_lower_bound < 0:
            self.dirichlet = False
        else:
            self.dirichlet = dirichlet
        if log_file:
            self.build_logger(log_file)

    def build_logger(self, log_file: str) -> None:
        """Build logger for optuna.

        Args:
            log_file (str): log file path.
        """
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        self.logger = optuna.logging.get_logger("optuna")
        self.logger.addHandler(file_handler)

    def add_evaluator(
        self,
        flag: str,
        target_column: str,
        hyperparameter: Optional[float] = None,
        groupby: Optional[str] = None,
    ) -> None:
        """Add calculators to the objective.

        Args:
            calculator (Calculator): calculator building blocks.
            flag (str ["wuauc", "portfolio", "logmse", ..., ect.]): type of calculator.
            target_column (str): target column to calculate.
        """
        self.evaluator_flags.append(flag)
        self.target_columns.append(target_column)
        if hyperparameter is not None:
            self.hyperparameters.append(hyperparameter)
        else:
            self.hyperparameters.append(None)
        if groupby is not None:
            self.groupbys.append(groupby)
        else:
            self.groupbys.append(None)

    def objective(
        self,
        trial: Trial,
    ) -> float:
        """Objective function for optuna.

        Args:
            trial (Trial): optuna trial.

        Returns:
            float: objective value.
        """
        weights = construct_weights(self, trial)
        targets = evaluate_targets(
            self.calculator,
            self.evaluator_flags,
            self.hyperparameters,
            self.groupbys,
            self.target_columns,
            weights,
        )

        local_vars = {"targets": targets, "sum": sum, "max": max, "min": min}
        result = float(eval(self.formula, {"__builtins__": None}, local_vars))
        if self.logger:
            self.logger.info(f"Trial {trial.number} finished with result: {result}")
            self.logger.info(f"targets: {targets}")
            self.logger.info(f"weights: {weights}")
        return result
