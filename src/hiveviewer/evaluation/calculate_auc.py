from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class CalculatorAUC:
    """CalculatorAUC class for calculating AUC."""

    def __init__(self, df: pd.DataFrame, selected_columns: List[str]) -> None:
        """Initialize CalculatorAUC.

        :param df: dataframe
        :param selected_columns: selected columns
        """
        self.df = df
        self.selected_values = self.df[selected_columns].values

    def get_overall_score(self, weights_for_equation: np.ndarray) -> None:
        """Calculate overall score.

        :param weights_for_equation: weights for equation
        """
        self.df["overall_score"] = np.product(
            self.selected_values**weights_for_equation, axis=1
        )

    def calculate_wuauc(
        self,
        groupby: str,
        weights_for_equation: np.ndarray,
        weights_for_groups: pd.Series,
        label_column: str = "label",
    ) -> float:
        """Calculate weighted user AUC.

        :param groupby: groupby column
        :param weights_for_equation: weights for equation
        :param weights_for_groups: weights for group
        :param label_column: label column
        """
        self.get_overall_score(weights_for_equation)
        grouped = self.df.groupby(groupby).apply(
            lambda x: float(roc_auc_score(x[label_column], x["overall_score"]))
        )
        counts_sorted = weights_for_groups.loc[grouped.index]
        wuauc = float(np.average(grouped, weights=counts_sorted.values))
        return wuauc
