from functools import partialmethod
from typing import Any, List, Optional

from optuna.trial import Trial

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
        first_order_lower_bound: float = 1e-3,
        first_order_upper_bound: float = 1e6,
        power_lower_bound: float = -1,
        power_upper_bound: float = 1,
        **kwargs: Any,
    ) -> None:
        """
        Initialize with direction, weights_num, formula, and dirichlet.

        Args:
            first_order_lower_bound (float, optional): Lower bound for first order value. Defaults to 1e-3.
            first_order_upper_bound (float, optional): Upper bound for first order value. Defaults to 1e6.
            power_lower_bound (float, optional): Lower bound for power value. Defaults to -1.
            power_upper_bound (float, optional): Upper bound for power value. Defaults to 1.
            **kwargs: Arbitrary keyword arguments for the parent class.
        """
        super().__init__(**kwargs)
        self.target_columns: List[str] = []
        self.evaluator_flags: List[str] = []
        self.groupbys: List[Optional[str]] = []
        self.power_lower_bound = power_lower_bound
        self.power_upper_bound = power_upper_bound
        self.hyperparameters: List[Optional[float]] = []
        self.evaluator_propertys: List[Optional[str]] = []
        self.first_order_lower_bound = first_order_lower_bound
        self.first_order_upper_bound = first_order_upper_bound
        if self.power_lower_bound < 0:
            self.dirichlet = False

    def add_evaluator(
        self,
        flag: str,
        target_column: str,
        hyperparameter: Optional[float] = None,
        evaluator_property: Optional[str] = None,
        groupby: Optional[str] = None,
    ) -> None:
        """
        Adds evaluators to the objective.

        Args:
            flag (str): Type of calculator. Expected values include ["wuauc", "portfolio", "logmse", ...].
            target_column (str): The target column for calculation.
            hyperparameter (Optional[float], optional): Hyperparameter for the calculator. Defaults to None.
            evaluator_property (Optional[str], optional): Property of the evaluator. Defaults to None.
            groupby (Optional[str], optional): Grouping criteria. Defaults to None.
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
        if evaluator_property is not None:
            self.evaluator_propertys.append(evaluator_property)
        else:
            self.evaluator_propertys.append(None)

    def objective(
        self,
        trial: Trial,
    ) -> float:
        """
        Objective function to be optimized by optuna.

        Args:
            trial (Trial): Optuna trial instance.

        Returns:
            float: Computed objective value based on the provided trial.
        """
        weights = construct_weights(self, trial)
        targets = evaluate_targets(
            calculator=self.calculator,
            evaluator_flags=self.evaluator_flags,
            hyperparameters=self.hyperparameters,
            evaluator_propertys=self.evaluator_propertys,
            groupbys=self.groupbys,
            target_columns=self.target_columns,
            weights=weights,
        )
        local_vars = {"targets": targets, "sum": sum, "max": max, "min": min}
        result = float(eval(self.formula, {"__builtins__": None}, local_vars))
        if self.logger:
            self.logger.info(f"Trial {trial.number} finished with result: {result}")
            self.logger.info(f"targets: {targets}")
            self.logger.info(f"weights: {weights}")
        return result
