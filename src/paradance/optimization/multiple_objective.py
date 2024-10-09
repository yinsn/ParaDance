from functools import partialmethod
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from optuna.trial import Trial

from ..evaluation import Calculator, LogarithmPCACalculator
from .base import BaseObjective, BaseObjectiveConfig
from .construct_weights import construct_weights
from .evaluate_targets import evaluate_targets


class MultipleObjectiveConfig(BaseObjectiveConfig):
    """Configuration for handling multiple objectives in optimization.

    Attributes:
        first_order_with_scales (bool): Whether to use scales-control in the first-order objective. Disable 'first_order_lower_bound' and 'first_order_upper_bound' when 'first_order_with_scales' is true and use automatic configuration instead.
        first_order_lower_bound (float): The lower bound for the first-order objective.
        first_order_upper_bound (float): The upper bound for the first-order objective.
        free_style_lower_bound (Union[float, List[float]]): The lower bound for the free-style objective.
        free_style_upper_bound (Union[float, List[float]]): The upper bound for the free-style objective.
        base_weights (Optional[List[float]]): The base weights for the first-order objective.
        base_weights_offset_ratio (float): The offset ratio for the base weights.
        max_min_scale_ratio (Optional[float]): The maximum to minimum scale ratio. None indicates no specific ratio.
        first_order_scale_upper_bound (float): The upper scale bound for the first-order objective.
        first_order_scale_lower_bound (float): The lower scale bound for the first-order objective.
        power_lower_bound (Union[float, List[float]]): The lower bound for the power objective.
        power_upper_bound (Union[float, List[float]]): The upper bound for the power objective.
        pca_importance_lower_bound (float): The lower bound for PCA importance.
        pca_importance_upper_bound (float): The upper bound for PCA importance.
    """

    first_order_with_scales: bool = True
    first_order_lower_bound: float = 1e-3
    first_order_upper_bound: float = 1e6
    free_style_lower_bound: Union[float, List[float]] = 1e-3
    free_style_upper_bound: Union[float, List[float]] = 1e6
    base_weights: Optional[List[float]] = None
    base_weights_offset_ratio: float = 0.1
    max_min_scale_ratio: Optional[float] = None
    first_order_scale_upper_bound: Union[float, List[float]] = 1
    first_order_scale_lower_bound: Union[float, List[float]] = 1
    power_lower_bound: Union[float, List[float]] = -1
    power_upper_bound: Union[float, List[float]] = 1
    pca_importance_lower_bound: float = 0
    pca_importance_upper_bound: float = 10


class MultipleObjective(BaseObjective):
    """
    This class provides methods to optimize the portfolio objective.
    """

    construct_weights = partialmethod(construct_weights)

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
        first_order_with_scales: bool = True,
        first_order_lower_bound: float = 1e-3,
        first_order_upper_bound: float = 1e6,
        free_style_lower_bound: Union[float, List[float]] = 1e-3,
        free_style_upper_bound: Union[float, List[float]] = 1e6,
        base_weights: Optional[List[float]] = None,
        base_weights_offset_ratio: float = 0.1,
        max_min_scale_ratio: Optional[float] = None,
        first_order_scale_upper_bound: Union[float, List[float]] = 1,
        first_order_scale_lower_bound: Union[float, List[float]] = 1,
        power_lower_bound: Union[float, List[float]] = -1,
        power_upper_bound: Union[float, List[float]] = 1,
        pca_importance_lower_bound: float = 0,
        pca_importance_upper_bound: float = 10,
        config: Optional[Dict] = None,
    ) -> None:
        """
        Initialize with direction, weights_num, formula, and dirichlet.

        Args:
            first_order_with_scales (bool, optional): Whether to use scales-control in the first-order objective. Defaults to True.
            first_order_lower_bound (float, optional): Lower bound for first order value. Defaults to 1e-3.
            first_order_upper_bound (float, optional): Upper bound for first order value. Defaults to 1e6.
            power_lower_bound (Union[float, List[float]]): Lower bound for power value. Defaults to -1.
            power_upper_bound (Union[float, List[float]]): Upper bound for power value. Defaults to 1.
            pca_importance_lower_bound (float, optional): Lower bound for pca importance value. Defaults to 0.
            pca_importance_upper_bound (float, optional): Upper bound for pca importance value. Defaults to 10.
            first_order_scale_bound (Optional[float], optional): Scale bound for first order value. Defaults to None.
        """

        if config is not None:
            self.config = MultipleObjectiveConfig(**config)
        else:
            self.config = MultipleObjectiveConfig(
                direction=direction,
                formula=formula,
                first_order=first_order,
                power=power,
                dirichlet=dirichlet,
                weights_num=weights_num,
                study_name=study_name,
                study_path=study_path,
                save_study=save_study,
                first_order_with_scales=first_order_with_scales,
                first_order_lower_bound=first_order_lower_bound,
                first_order_upper_bound=first_order_upper_bound,
                free_style_lower_bound=free_style_lower_bound,
                free_style_upper_bound=free_style_upper_bound,
                base_weights=base_weights,
                base_weights_offset_ratio=base_weights_offset_ratio,
                max_min_scale_ratio=max_min_scale_ratio,
                first_order_scale_upper_bound=first_order_scale_upper_bound,
                first_order_scale_lower_bound=first_order_scale_lower_bound,
                power_lower_bound=power_lower_bound,
                power_upper_bound=power_upper_bound,
                pca_importance_lower_bound=pca_importance_lower_bound,
                pca_importance_upper_bound=pca_importance_upper_bound,
            )
        self.calculator = calculator
        self.direction = self.config.direction
        self.formula = self.config.formula
        self.first_order = self.config.first_order
        self.power = self.config.power
        self.dirichlet = self.config.dirichlet
        self.weights_num = self.config.weights_num
        self.study_name = self.config.study_name
        self.study_path = self.config.study_path
        self.save_study = self.config.save_study
        self.first_order_lower_bound = self.config.first_order_lower_bound
        self.first_order_upper_bound = self.config.first_order_upper_bound
        self.first_order_with_scales = self.config.first_order_with_scales
        self.free_style_lower_bound = self.config.free_style_lower_bound
        self.free_style_upper_bound = self.config.free_style_upper_bound
        self.base_weights = self.config.base_weights
        self.base_weights_offset_ratio = self.config.base_weights_offset_ratio
        self.max_min_scale_ratio = self.config.max_min_scale_ratio
        self.first_order_scale_upper_bound = self.config.first_order_scale_upper_bound
        self.first_order_scale_lower_bound = self.config.first_order_scale_lower_bound
        self.power_lower_bound = self.config.power_lower_bound
        self.power_upper_bound = self.config.power_upper_bound
        self.pca_importance_lower_bound = self.config.pca_importance_lower_bound
        self.pca_importance_upper_bound = self.config.pca_importance_upper_bound

        self.target_columns: List[str] = []
        self.mask_columns: List[Optional[str]] = []
        self.evaluator_flags: List[str] = []
        self.groupbys: List[Optional[str]] = []
        self.group_weights: List[Optional[pd.Series]] = []
        self.hyperparameters: List[Optional[float]] = []
        self.evaluator_propertys: List[Optional[str]] = []

        if self.calculator.equation_type not in ["free_style", "json"] and isinstance(
            self.calculator, Calculator
        ):
            self.calculator.value_scale()

        self._prepare_study()

    def add_evaluator(
        self,
        flag: str,
        target_column: str,
        mask_column: Optional[str] = None,
        hyperparameter: Optional[float] = None,
        evaluator_property: Optional[str] = None,
        groupby: Optional[str] = None,
        weights_for_groups: Optional[pd.Series] = None,
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
        if mask_column is not None:
            self.mask_columns.append(mask_column)
        else:
            self.mask_columns.append(None)
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
        if weights_for_groups is not None:
            self.group_weights.append(weights_for_groups)
        else:
            self.group_weights.append(None)

    def evaluate_custom_weights(self, weights: List[float]) -> List[float]:
        """
        Evaluate the objective function with custom weights.

        Args:
            weights (List[float]): Custom weights to evaluate.
        """
        self.calculator.get_overall_score(
            weights_for_equation=weights,
        )

        targets = evaluate_targets(
            calculator=self.calculator,
            evaluator_flags=self.evaluator_flags,
            mask_columns=self.mask_columns,
            hyperparameters=self.hyperparameters,
            evaluator_propertys=self.evaluator_propertys,
            groupbys=self.groupbys,
            target_columns=self.target_columns,
            group_weights=self.group_weights,
        )

        return targets

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
        targets = self.evaluate_custom_weights(weights)
        local_vars = {"targets": targets, "sum": sum, "max": max, "min": min}
        result = float(eval(str(self.formula), {"__builtins__": None}, local_vars))
        if self.logger:
            self.logger.info(f"Trial {trial.number} finished with result: {result}")
            self.logger.info(f"targets: {targets}")
            self.logger.info(f"weights: {weights}")
        return result

    def export_completed_formulas(self, weights: Optional[np.ndarray] = None) -> None:
        """Exports the completed formulas by replacing weight placeholders in the formulas
        with actual values from the provided or default weights.

        If `weights` is not provided, it defaults to `self.best_params`. The method updates
        `self.completed_formulas` by replacing occurrences of `weights[i]` in the stored
        equations with corresponding values from the weight array.

        Args:
            weights (Optional[np.ndarray]): An optional numpy array containing weight values
                to substitute in the formulas. If None, `self.best_params` is used.

        Returns:
            None: This method does not return a value but updates `self.completed_formulas`
            with the substituted equations.
        """
        json_equations = {}
        if weights is None:
            weights = self.best_params
        if isinstance(self.calculator, Calculator) and hasattr(
            self.calculator, "equation_json"
        ):
            json_equations = self.calculator.equation_json.formula
            for key, expr in json_equations.items():
                for i, param in enumerate(weights):
                    expr = expr.replace(f"weights[{i}]", str(param))
                json_equations[key] = expr
        self.completed_formulas = json_equations
