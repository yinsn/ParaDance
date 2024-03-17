from typing import List, Optional

import numpy as np
import pandas as pd

from .base_calculator import BaseCalculator


class Calculator(BaseCalculator):
    """Calculator class for calculating various metrics."""

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
        self.value_scale()

    def value_scale(self) -> None:
        """
        Calculates the negative average log10 magnitude of absolute values for selected columns in the dataframe,
        storing the result in `self.value_scales`.
        """
        dataframe = self.df[self.selected_columns].abs()
        magnitudes = np.log10(dataframe.values)

        avg_magnitude = np.nanmean(magnitudes, axis=0)
        magnitudes = [-magnitude for magnitude in avg_magnitude]

        self.value_scales = np.asarray(magnitudes)

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
            self.df["overall_score"] = np.prod(
                (1 + np.asarray(first_order_weights) * np.asarray(self.selected_values))
                ** powers_for_equation,
                axis=1,
            )
        elif self.equation_type == "product":
            self.df["overall_score"] = np.prod(
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
