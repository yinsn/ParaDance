from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    from .calculator import Calculator


def calculate_score(x: pd.DataFrame, target_column: str, k: str) -> float:
    """
    Helper function to calculate roc_auc_score and handle exceptions.

    :param x: Input DataFrame.
    :param target_column: Column name to use for target values in AUC calculation.
    :param k: Key used for the boundary dictionary and to format column names.
    :return: Calculated score or 0.5 in case of exceptions.
    """
    try:
        return float(roc_auc_score(x[f"{target_column}_lt_{k}"], x["overall_score"]))
    except:
        return 0.5


def calculate_woauc(
    calculator: "Calculator",
    groupby: Optional[str],
    target_column: str,
    weights_for_equation: List[float],
    weights_for_groups: Optional[pd.Series] = None,
) -> List[float]:
    """
    Calculate weighted ordinal user AUC.

    :param calculator: Calculator object that contains the data and methods for calculation.
    :param groupby: Column name to group data by, or None for no grouping.
    :param target_column: Column name to use for target values in AUC calculation.
    :param weights_for_equation: List of weights for equation.
    :param weights_for_groups: Series of weights for groups, or None.
    :return: List of calculated weighted ordinal user AUC values.
    """
    woauc_indices = calculator.woauc_dict[target_column]
    woauc = []
    sampler = calculator.samplers[target_column]

    for k, _ in sampler.boundary_dict.items():
        if groupby is not None:
            grouped = (
                calculator.df.loc[woauc_indices]
                .groupby(groupby)
                .apply(lambda x: calculate_score(x, target_column, k))
            )

            if weights_for_groups is not None:
                counts_sorted = weights_for_groups.loc[grouped.index]
                paritial_auc = float(np.average(grouped, weights=counts_sorted.values))
            else:
                paritial_auc = float(np.mean(grouped))
        else:
            paritial_auc = float(
                roc_auc_score(
                    calculator.df.loc[woauc_indices][f"{target_column}_lt_{k}"],
                    calculator.df.loc[woauc_indices]["overall_score"],
                )
            )
        woauc.append(paritial_auc)
    return woauc
