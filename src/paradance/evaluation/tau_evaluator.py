from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

if TYPE_CHECKING:
    from .calculator import Calculator

from typing import List

import numpy as np


def map_to_bins(
    data: Union[List[float], np.ndarray], num_bins: int = 100
) -> np.ndarray:
    """
    Map data to equal-frequency bins.

    :param data: Union of list of floats or np.ndarray. The input data to be binned.
    :param num_bins: Optional integer, default 100. The number of bins to use for the mapping.
    :returns: np.ndarray. The binned data.
    """
    data = np.array(data, dtype=float)

    if len(data) == 0:
        return np.array([])

    if np.all(data == 0):
        return np.zeros_like(data)

    non_zero_data: np.ndarray = data[data != 0]
    num_bins = min(num_bins, len(non_zero_data))
    quantiles = np.linspace(0, 1, num_bins + 1)[1:-1]
    non_zero_bins = np.digitize(non_zero_data, np.quantile(non_zero_data, quantiles))

    bins = np.zeros_like(data)
    bins[data != 0] = non_zero_bins + 1

    return bins


def calculate_tau(
    calculator: "Calculator",
    target_column: str,
    groupby: Optional[str],
    weights_for_groups: Optional[pd.Series] = None,
    num_bins: Optional[Union[int, float]] = 100,
) -> float:
    """
    Calculate the Kendall's Tau using binned data.

    :param calculator: Calculator object. Stores the DataFrame and any existing bin mappings.
    :param target_column: String. The column name in the DataFrame that you want to target.
    :param groupby: Optional string. The column name to group by.
    :param weights_for_groups: Optional pd.Series. Weights for each group.
    :param num_bins: Optional integer, default 100. The number of bins to use for the mapping.
    :returns: float. The calculated Kendall's Tau. Normalized to [0, 1].
    """
    if num_bins is None:
        num_bins = 100
    else:
        num_bins = int(num_bins)
    if target_column in calculator.bin_mappings:
        label_bins = calculator.bin_mappings[target_column]
    else:
        label_bins = map_to_bins(pd.Series(calculator.df[target_column]), num_bins)
        calculator.bin_mappings[target_column] = label_bins
    if groupby is not None:
        calculator.df[f"{target_column}_bin"] = label_bins
        grouped = calculator.df.groupby(groupby).apply(
            lambda x: float(
                kendalltau(x[f"{target_column}_bin"], x["overall_score"])[0]
            )
        )
        if weights_for_groups is not None:
            counts_sorted = weights_for_groups.loc[grouped.index]
            tau = float(np.average(grouped, weights=counts_sorted.values))
        else:
            tau = float(np.mean(grouped))
    else:
        overall_score_bins = map_to_bins(calculator.df["overall_score"], num_bins)
        tau, _ = kendalltau(label_bins, overall_score_bins)
        normalized_tau = tau * 0.5 + 0.5
    return normalized_tau
