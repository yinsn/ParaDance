from typing import List, Literal

import numpy as np
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
        self.target_columns: List[str] = []
        self.weights_num = weights_num
        self.formula = formula
        self.dirichlet = dirichlet

    def add_calculator(
        self,
        calculator: Calculator,
        flag: Literal["wuauc", "profolio"],
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

        for calculator, flag, target_column in zip(
            self.calculators, self.calculator_flags, self.target_columns
        ):
            calculator.get_overall_score(np.array(weights))
            if flag == "profolio":
                _, concentration = calculator.calculate_portfolio_concentration(
                    target_column=target_column,
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
        return result
