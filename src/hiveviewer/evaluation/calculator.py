from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class Calculator:
    """Calculator class for calculating various metrics."""

    def __init__(
        self,
        df: pd.DataFrame,
        selected_columns: List[str],
        equation_type: str = "product",
        weights_for_groups: Optional[pd.Series] = None,
    ) -> None:
        """Initialize Calculator.

        :param df: dataframe
        :param selected_columns: selected columns
        :param weights_for_groups: weights for group
        """
        self.df = df
        self.df_len = len(self.df)
        self.selected_columns = selected_columns
        self.selected_values = self.df[selected_columns].values
        self.equation_type = equation_type
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

    def calculate_wuauc(
        self,
        groupby: Optional[str],
        weights_for_equation: List,
        weights_for_groups: Optional[pd.Series] = None,
        label_column: str = "label",
        auc: bool = False,
    ) -> float:
        """Calculate weighted user AUC.

        :param groupby: groupby column
        :param weights_for_equation: weights for equation
        :param weights_for_groups: weights for group
        :param label_column: label column
        :return AUC/WUAUC/UAUC
        """
        self.get_overall_score(weights_for_equation)
        if auc:
            result = float(
                roc_auc_score(self.df[label_column], self.df["overall_score"])
            )
        else:
            if groupby is not None:
                grouped = self.df.groupby(groupby).apply(
                    lambda x: float(roc_auc_score(x[label_column], x["overall_score"]))
                )
                if weights_for_groups is not None:
                    counts_sorted = weights_for_groups.loc[grouped.index]
                    result = float(np.average(grouped, weights=counts_sorted.values))
                else:
                    result = float(np.mean(grouped))
        return result

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

        self.sampler = FrequencySampler(
            sample_size=sample_size,
            data=self.df[score_column],
            slice_from=slice_from,
            slice_to=slice_to,
            log_scale=log_scale,
            laplace_smoothing=laplace_smoothing,
        )
        self.score_column = score_column
        self.create_score_columns(
            boundary_dict=self.sampler.sample(), score_column=score_column
        )

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
        self.woauc_indices = self.df[slice_from_condition & slice_to_condition].index

    def calculate_woauc(
        self,
        weights_for_equation: List,
    ) -> List[float]:
        """Calculate weighted ordinal user AUC.

        :param weights_for_equation: weights for equation
        :param score_column: score column
        """
        self.get_overall_score(weights_for_equation)
        woauc = []
        for k, _ in self.sampler.boundary_dict.items():
            paritial_auc = float(
                roc_auc_score(
                    self.df.loc[self.woauc_indices][f"{self.score_column}_lt_{k}"],
                    self.df.loc[self.woauc_indices]["overall_score"],
                )
            )
            woauc.append(paritial_auc)
        return woauc

    def calculate_log_mse(self, target_column: str) -> float:
        """Calculate log mean squared error.

        :param target_column: target column
        """
        log_true = np.log(self.df[target_column] + 1)
        log_pred = np.log(self.df["overall_score"] + 1)
        mse = np.mean((log_true - log_pred) ** 2)
        return float(mse)

    def calculate_portfolio_concentration(
        self, target_column: str, expected_return: Optional[float] = 0.95
    ) -> Tuple[float, float]:
        """Calculate portfolio concentration.

        :param target_column: target column
        :param expected_return: expected return
        """
        df = self.df.sort_values("overall_score", ascending=False)
        sum_all = df[target_column].sum()
        df["cumulative_sum"] = df[target_column].cumsum()
        df["cumulative_ratio"] = df["cumulative_sum"] / sum_all
        threshold = df[df["cumulative_ratio"] > expected_return]["overall_score"].max()
        concentration = df[df["overall_score"] > threshold].shape[0] / self.df_len
        return threshold, concentration

    def calculate_auc_triple_parameters(self, grid_interval: int) -> Tuple:
        """Calculate AUC triple parameters.

        :param grid_interval: grid interval
        :return: tuple of W1, W2, WUAUC
        """
        w1_values = np.linspace(0, 1, grid_interval)
        w2_values = np.linspace(0, 1, grid_interval)
        W1, W2 = np.meshgrid(w1_values, w2_values)
        WUAUC = np.zeros_like(W1)

        for i in tqdm(range(W1.shape[0]), desc="Progress"):
            for j in range(W1.shape[1]):
                w1 = W1[i, j]
                w2 = W2[i, j]
                w3 = 1 - w1 - w2
                if w3 < 0:
                    WUAUC[i, j] = np.nan
                else:
                    WUAUC[i, j] = self.calculate_wuauc(
                        groupby="user_id",
                        weights_for_equation=[w1, w2, w3],
                        weights_for_groups=self.weights_for_groups,
                    )
        return W1, W2, WUAUC
