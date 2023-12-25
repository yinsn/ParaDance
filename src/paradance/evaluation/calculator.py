from functools import partialmethod
from typing import List, Optional

import numpy as np
import pandas as pd

from .auc_triple_parameters_evaluator import calculate_auc_triple_parameters
from .distinct_portfolio_evaluator import (
    calculate_distinct_count_portfolio_concentration,
)
from .inverse_pair_evaluator import calculate_inverse_pair
from .log_mse_evaluator import calculate_log_mse
from .neg_rank_ratio_evaluator import calculate_neg_rank_ratio
from .portfolio_evaluator import calculate_portfolio_concentration
from .tau_evaluator import calculate_tau
from .woauc_evaluator import calculate_woauc
from .wuauc_evaluator import calculate_wuauc


class Calculator:
    """Calculator class for calculating various metrics."""

    calculate_auc_triple_parameters = partialmethod(calculate_auc_triple_parameters)
    calculate_distinct_count_portfolio_concentration = partialmethod(
        calculate_distinct_count_portfolio_concentration
    )
    calculate_inverse_pair = partialmethod(calculate_inverse_pair)
    calculate_log_mse = partialmethod(calculate_log_mse)
    calculate_neg_rank_ratio = partialmethod(calculate_neg_rank_ratio)
    calculate_portfolio_concentration = partialmethod(calculate_portfolio_concentration)
    calculate_tau = partialmethod(calculate_tau)
    calculate_woauc = partialmethod(calculate_woauc)
    calculate_wuauc = partialmethod(calculate_wuauc)

    def __init__(
        self,
        df: pd.DataFrame,
        selected_columns: List[str],
        equation_type: str = "product",
        weights_for_groups: Optional[pd.Series] = None,
        equation_eval_str: Optional[str] = None,
    ) -> None:
        """Initialize Calculator.

        :param df: dataframe
        :param selected_columns: selected columns
        :param weights_for_groups: weights for group
        :param equation_eval_str: equation eval string
        """
        self.df = df
        self.df_len = len(self.df)
        self.equation_eval_str = equation_eval_str
        self.equation_type = equation_type
        self.samplers: dict = {}
        self.selected_columns = selected_columns
        self.selected_values = self.df[selected_columns].values
        self.woauc_dict: dict = {}
        self.bin_mappings: dict = {}
        if weights_for_groups is None:
            self.weights_for_groups = pd.Series(
                np.ones(len(self.df)), index=self.df.index
            )
        else:
            self.weights_for_groups = weights_for_groups

    def get_overall_score(
        self,
        weights_for_equation: List[float],
    ) -> None:
        """Calculate overall score.

        :param powers_for_equation: powers for equation
        :param first_order_weights: first order weights
        """
        if len(weights_for_equation) == 2 * len(self.selected_columns):
            powers_for_equation = weights_for_equation[: len(self.selected_columns)]
            first_order_weights = weights_for_equation[len(self.selected_columns) :]
            self.df["overall_score"] = np.product(
                (1 + np.asarray(first_order_weights) * np.asarray(self.selected_values))
                ** powers_for_equation,
                axis=1,
            )
        elif self.equation_type == "product":
            self.df["overall_score"] = np.product(
                self.selected_values**weights_for_equation, axis=1
            )
        elif self.equation_type == "sum":
            weights_array = np.array(weights_for_equation).reshape(-1, 1)
            self.df["overall_score"] = self.selected_values @ weights_array

        elif self.equation_type == "free_style":
            columns = [
                self.selected_values[:, i] for i in range(self.selected_values.shape[1])
            ]
            local_dict = {"weights": weights_for_equation, "columns": columns}
            if self.equation_eval_str is not None:
                self.df["overall_score"] = eval(
                    self.equation_eval_str, globals(), local_dict
                )
            else:
                raise ValueError("equation_eval_str is not defined.")

    def create_score_columns(
        self, boundary_dict: dict, score_column: str = "score"
    ) -> None:
        """Create score columns.

        :param boundary_dict: boundary dict
        :param score_column: score column to be converted
        """
        for k, _ in boundary_dict.items():
            self.df[f"{score_column}_lt_{k}"] = (self.df[score_column] >= k).astype(int)

    def initialize_fq_sampler(
        self,
        sample_size: int,
        score_column: str,
        slice_from: Optional[float] = None,
        slice_to: Optional[float] = None,
        log_scale: Optional[bool] = True,
        laplace_smoothing: Optional[bool] = True,
    ) -> None:
        """Initialize frequency sampler."""
        from ..sampling.frequency_sampler import FrequencySampler

        sampler = FrequencySampler(
            sample_size=sample_size,
            data=self.df[score_column],
            slice_from=slice_from,
            slice_to=slice_to,
            log_scale=log_scale,
            laplace_smoothing=laplace_smoothing,
        )
        self.create_score_columns(
            boundary_dict=sampler.sample(), score_column=score_column
        )
        self.samplers[score_column] = sampler
        slice_from_condition = (
            pd.Series(True, index=self.df.index)
            if slice_from is None
            else (self.df[score_column] >= slice_from)
        )
        slice_to_condition = (
            pd.Series(True, index=self.df.index)
            if slice_to is None
            else (self.df[score_column] <= slice_to)
        )
        self.woauc_dict[score_column] = self.df[
            slice_from_condition & slice_to_condition
        ].index
