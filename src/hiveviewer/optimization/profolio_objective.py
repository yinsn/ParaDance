import logging
from typing import List, Optional

import optuna
from optuna.trial import Trial

from ..evaluation.calculator import Calculator
from .base import BaseObjective


class ProfolioObjective(BaseObjective):
    """
    This class provides methods to optimize the profolio objective.
    """

    def __init__(
        self,
        direction: str,
        weights_num: int,
        formula: str,
        first_order: bool = False,
        first_order_lower_bound: float = 1e-3,
        first_order_upper_bound: float = 1e6,
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
        self.calculators: List[Calculator] = []
        self.calculator_flags: List[str] = []
        self.hyperparameters: List[Optional[float]] = []
        self.target_columns: List[str] = []
        self.weights_num = weights_num
        self.formula = formula
        self.first_order = first_order
        self.first_order_lower_bound = first_order_lower_bound
        self.first_order_upper_bound = first_order_upper_bound
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
        flag: str,
        hyperparameter: Optional[float],
        target_column: str,
    ) -> None:
        """Add calculators to the objective.

        Args:
            calculator (Calculator): calculator building blocks.
            flag (str ["wuauc", "profolio"]): type of calculation.
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

        if self.first_order:
            first_order_weights = []
            for i in range(self.weights_num):
                first_order_weights.append(
                    trial.suggest_float(
                        f"w{self.weights_num+i+1}",
                        self.first_order_lower_bound,
                        self.first_order_upper_bound,
                        log=True,
                    )
                )
        else:
            first_order_weights = None

        targets: List[float] = []
        for calculator, flag, hyperparameter, target_column in zip(
            self.calculators,
            self.calculator_flags,
            self.hyperparameters,
            self.target_columns,
        ):
            calculator.get_overall_score(
                powers_for_equation=weights,
                first_order_weights=first_order_weights,
            )
            if flag == "profolio":
                _, concentration = calculator.calculate_portfolio_concentration(
                    target_column=target_column,
                    expected_return=hyperparameter,
                )
                targets.append(concentration)
            elif flag == "wuauc":
                wuauc = calculator.calculate_wuauc(
                    groupby=target_column,
                    weights_for_equation=weights,
                )
                targets.append(wuauc)

        local_vars = {"targets": targets, "sum": sum}
        result = float(eval(self.formula, {"__builtins__": None}, local_vars))
        if self.logger:
            self.logger.info(f"Trial {trial.number} finished with result: {result}")
            self.logger.info(f"targets: {targets}")
            self.logger.info(f"weights: {weights}")
            if self.first_order:
                self.logger.info(f"first_order_weights: {first_order_weights}")
        return result
