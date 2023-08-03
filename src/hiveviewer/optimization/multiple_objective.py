import logging
from typing import List, Optional

import optuna
from optuna.trial import Trial

from ..evaluation.calculator import Calculator
from .base import BaseObjective


class MultipleObjective(BaseObjective):
    """
    This class provides methods to optimize the profolio objective.
    """

    def __init__(
        self,
        direction: str,
        weights_num: int,
        formula: str,
        power: bool = False,
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
        self.groupbys: List[Optional[str]] = []
        self.power = power
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
        target_column: str,
        hyperparameter: Optional[float] = None,
        groupby: Optional[str] = None,
    ) -> None:
        """Add calculators to the objective.

        Args:
            calculator (Calculator): calculator building blocks.
            flag (str ["wuauc", "profolio", "logmse"]): type of calculation.
            target_column (str): target column to calculate.
        """
        self.calculators.append(calculator)
        self.calculator_flags.append(flag)
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
        """
        Calculate the objective value.
        """
        power_weights: List[float] = []
        first_order_weights: List[float] = []

        if self.power:
            if self.dirichlet:
                for i in range(self.weights_num - 1):
                    power_weights.append(
                        trial.suggest_float(
                            f"w_po_{i+1}", 0, max((1 - sum(power_weights), 0.1))
                        )
                    )
                power_weights.append(1 - sum(power_weights))
            else:
                for i in range(self.weights_num):
                    power_weights.append(trial.suggest_float(f"w{i+1}", 0, 1))

        if self.first_order:
            for i in range(self.weights_num):
                first_order_weights.append(
                    trial.suggest_float(
                        f"w_fo_{i+1}",
                        self.first_order_lower_bound,
                        self.first_order_upper_bound,
                        log=True,
                    )
                )

        targets: List[float] = []
        for calculator, flag, hyperparameter, groupby, target_column in zip(
            self.calculators,
            self.calculator_flags,
            self.hyperparameters,
            self.groupbys,
            self.target_columns,
        ):
            weights: List[float] = []
            if not (self.first_order):
                weights = power_weights
            elif calculator.equation_type == "product" and self.power:
                weights = power_weights + first_order_weights
            elif calculator.equation_type == "sum":
                weights = first_order_weights
            calculator.get_overall_score(
                weights_for_equation=weights,
            )
            if flag == "profolio":
                _, concentration = calculator.calculate_portfolio_concentration(
                    target_column=target_column,
                    expected_return=hyperparameter,
                )
                targets.append(concentration)
            elif flag == "wuauc":
                wuauc = calculator.calculate_wuauc(
                    groupby=groupby,
                    label_column=target_column,
                    weights_for_equation=weights,
                )
                targets.append(wuauc)
            elif flag == "auc":
                auc = calculator.calculate_wuauc(
                    groupby=groupby,
                    label_column=target_column,
                    weights_for_equation=weights,
                    auc=True,
                )
                targets.append(auc)
            elif flag == "woauc":
                woauc = calculator.calculate_woauc(
                    weights_for_equation=weights,
                )
                targets.append(sum(woauc))
            elif flag == "logmse":
                mse = calculator.calculate_log_mse(
                    target_column=target_column,
                )
                targets.append(mse)

        local_vars = {"targets": targets, "sum": sum, "max": max, "min": min}
        result = float(eval(self.formula, {"__builtins__": None}, local_vars))
        if self.logger:
            self.logger.info(f"Trial {trial.number} finished with result: {result}")
            self.logger.info(f"targets: {targets}")
            self.logger.info(f"weights: {weights}")
            if self.first_order:
                self.logger.info(f"first_order_weights: {first_order_weights}")
        return result