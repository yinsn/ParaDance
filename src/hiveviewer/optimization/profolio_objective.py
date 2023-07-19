import logging
from typing import List, Literal, Optional

import numpy as np
import optuna
from optuna.trial import Trial

from ..evaluation.calculate_auc import Calculator
from .base import BaseObjective


class ProfolioObjective(BaseObjective):
    """
    This class provides methods to optimize the profolio objective.
    """

    def __init__(
        self,
        direction: Literal["minimize", "maximize"],
        weights_num: int,
        formula: str,
        dirichlet: bool = True,
        log_file: Optional[str] = None,
    ) -> None:
        """
        Initialize with direction, weights_num, formula and dirichlet.

        Args:
            direction (Literal["minimize", "maximize"]): direction to optimize.
            weights_num (int): numbers of weights to search.
            formula (str): formula of targets to calculate the objective.
            dirichlet (bool, optional): Use dirichlet distribution or not. Defaults to True.
        """
        super().__init__(direction, formula, dirichlet)
        self.calculators: List[Calculator] = []
        self.calculator_flags: List[Literal["wuauc", "profolio"]] = []
        self.hyperparameters: List[Optional[float]] = []
        self.target_columns: List[str] = []
        self.weights_num = weights_num
        self.formula = formula
        self.dirichlet = dirichlet
        if log_file:
            self.build_logger(log_file)

    def build_logger(self, log_file: str) -> None:
        """Build logger to log the results of each trial."""
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        self.logger = optuna.logging.get_logger("optuna")
        self.logger.addHandler(file_handler)

    def add_calculator(
        self,
        calculator: Calculator,
        flag: Literal["wuauc", "profolio"],
        hyperparameter: Optional[float],
        target_column: str,
    ) -> None:
        """Add calculators to the objective.

        Args:
            calculator (Calculator): calculator building blocks.
            flag (Literal["wuauc", "profolio"]): type of calculation.
            target_column (str): target column to calculate.
        """
        self.calculators.append(calculator)
        self.calculator_flags.append(flag)
        self.target_columns.append(target_column)
        if hyperparameter:
            self.hyperparameters.append(hyperparameter)
        else:
            self.hyperparameters.append(None)

    def objective(
        self,
        trial: Trial,
    ) -> float:
        """
        Calculate the objective value.
        """

        weights: List[float] = []
        if self.dirichlet:
            for i in range(self.weights_num - 1):
                weights.append(
                    trial.suggest_float(f"w{i+1}", 0, max((1 - sum(weights), 0.1)))
                )
            weights.append(1 - sum(weights))
        else:
            for i in range(self.weights_num):
                weights.append(trial.suggest_float(f"w{i+1}", 0, 1))

        targets: List[float] = []

        for calculator, flag, hyperparameter, target_column in zip(
            self.calculators,
            self.calculator_flags,
            self.hyperparameters,
            self.target_columns,
        ):
            calculator.get_overall_score(np.array(weights))
            if flag == "profolio":
                _, concentration = calculator.calculate_portfolio_concentration(
                    target_column=target_column,
                    expected_return=hyperparameter,
                )
                targets.append(concentration)
            elif flag == "wuauc":
                wuauc = calculator.calculate_wuauc(
                    groupby=target_column,
                    weights_for_equation=np.array(weights),
                )
                targets.append(wuauc)

        local_vars = {"targets": targets, "sum": sum}
        result = float(eval(self.formula, {"__builtins__": None}, local_vars))
        if self.logger:
            self.logger.info(f"Trial {trial.number} finished with result: {result}")
            self.logger.info(f"targets: {targets}")
            self.logger.info(f"weights: {weights}")
        return result
