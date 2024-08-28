from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base_calculator import BaseCalculator
from .calculate_json_formula import JSONFormula, calculate_formula_scores


class Calculator(BaseCalculator):
    """A calculator for processing and analyzing data within a DataFrame based on specified equations and methods.

    Attributes:
        df (pd.DataFrame): The DataFrame to perform calculations on.
        df_len (int): The length of the DataFrame.
        equation_eval_str (Optional[str]): A string representing a custom equation to evaluate.
        equation_type (str): The type of equation to use for calculations ("product", "sum", "free_style", or "json").
        selected_columns (List[str]): Columns selected for calculations.
        selected_values (np.ndarray): The values of the selected columns in the DataFrame.
        value_scales (np.ndarray): The negative average log10 magnitude of absolute values for selected columns.
        weights_for_groups (pd.Series): A Series containing weights for different groups within the DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        selected_columns: List[str],
        equation_type: str = "product",
        weights_for_groups: Optional[pd.Series] = None,
        equation_eval_str: Optional[str] = None,
        equation_json: Optional[Dict] = None,
        delimiter: Optional[str] = "#",
    ) -> None:
        """Initializes the Calculator object.

        Args:
            df (pd.DataFrame): The DataFrame to perform calculations on.
            selected_columns (List[str]): The names of the columns to include in calculations.
            equation_type (str, optional): The type of equation to use for score calculation. Defaults to "product".
            weights_for_groups (Optional[pd.Series], optional): A Series containing weights for different groups. Defaults to None, which sets equal weights.
            equation_eval_str (Optional[str], optional): A string representing a custom equation for free-style calculations. Defaults to None.
        """
        super().__init__(
            selected_columns=selected_columns,
        )
        self.df = df
        self.df_len = len(self.df)
        self.equation_eval_str = equation_eval_str

        if equation_json is not None:
            self.equation_json = JSONFormula(**equation_json)

        self.delimiter = delimiter
        self.equation_type = equation_type
        self.selected_columns = selected_columns
        self.selected_values = self.df[selected_columns].values

        if weights_for_groups is None:
            self.weights_for_groups = pd.Series(
                np.ones(len(self.df)), index=self.df.index
            )
        else:
            self.weights_for_groups = weights_for_groups

    def value_scale(self) -> None:
        """
        Calculates the negative average log10 magnitude of absolute values for selected columns in the dataframe,
        storing the result in `self.value_scales`.
        """
        dataframe = self.df[self.selected_columns].abs()
        magnitudes = np.log10(dataframe.values + 1e-10)

        avg_magnitude = np.nanmean(magnitudes, axis=0)
        magnitudes = [-magnitude for magnitude in avg_magnitude]

        self.value_scales = np.asarray(magnitudes)

    def get_overall_score(
        self,
        weights_for_equation: List[float],
    ) -> None:
        """Calculates the overall score for each row in the DataFrame based on the specified equation type and weights.

        Args:
            weights_for_equation (List[float]): A list of weights to apply to each selected column for the calculation.
        """
        if self.equation_type == "product" and (
            len(weights_for_equation) == len(self.selected_columns)
        ):
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
            local_dict = self.initialize_local_dict(weights_for_equation, columns)
            if self.equation_eval_str is not None:
                self.df["overall_score"] = eval(
                    self.equation_eval_str, {"__builtins__": None}, local_dict
                )
            else:
                raise ValueError("equation_eval_str is not defined.")

        elif (self.equation_type == "json") and (self.equation_json is not None):
            self.df["overall_score"] = calculate_formula_scores(
                equation_json=self.equation_json,
                selected_values=self.df[self.selected_columns],
                weights=weights_for_equation,
                delimiter=self.delimiter,
            )

        elif len(weights_for_equation) == 2 * len(self.selected_columns):
            powers_for_equation = weights_for_equation[: len(self.selected_columns)]
            first_order_weights = weights_for_equation[len(self.selected_columns) :]
            self.df["overall_score"] = np.prod(
                (1 + np.asarray(first_order_weights) * np.asarray(self.selected_values))
                ** powers_for_equation,
                axis=1,
            )

    def create_score_columns(
        self, boundary_dict: dict, score_column: str = "score"
    ) -> None:
        """Creates new columns in the DataFrame to categorize rows based on score boundaries.

        Args:
            boundary_dict (Dict): A dictionary with score boundaries as keys and conditions as values.
            score_column (str, optional): The name of the column to apply the boundaries to. Defaults to "score".
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
        """Initializes a frequency sampler for a given score column and applies sampling results to create new columns.

        Args:
            sample_size (int): The size of the sample to generate.
            score_column (str): The name of the score column to sample from.
            slice_from (Optional[float], optional): The lower bound of the score range to sample. Defaults to None.
            slice_to (Optional[float], optional): The upper bound of the score range to sample. Defaults to None.
            log_scale (Optional[bool], optional): Whether to use logarithmic scaling for sampling. Defaults to True.
            laplace_smoothing (Optional[bool], optional): Whether to apply Laplace smoothing to the sampling. Defaults to True.
        """
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
