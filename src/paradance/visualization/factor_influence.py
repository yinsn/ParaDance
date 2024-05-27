from typing import Dict, List

import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.stats import kendalltau

from ..evaluation import map_to_bins


def factor_influence_across_percentiles(
    dataframe: DataFrame,
    overall_score_column: str,
    selected_columns: List[str],
    num_percentiles: int = 10,
    only_top_part: bool = False,
) -> Dict[str, List[float]]:
    """
    Evaluates the influence of selected factors on an overall score across defined percentiles of the data.

    The function computes Kendall's tau correlation coefficient for each factor within the given percentiles
    of the overall score. It visualizes these influences using a line plot where the X-axis represents
    the percentiles and the Y-axis shows the tau values for each factor.

    Args:
        dataframe (DataFrame): The DataFrame containing the data.
        overall_score_column (str): The column name of the DataFrame representing the overall score.
        selected_columns (List[str]): A list of column names representing the factors to be evaluated.
        num_percentiles (int, optional): The number of equally-sized percentiles to divide the data into. Default is 10.
        only_top_part (bool, optional): If True, only the top percentile is considered for the analysis. Default is False.

    Returns:
        Dict[str, List[float]]: A dictionary where keys are the column names from `selected_columns`
        and values are lists of tau correlation coefficients for each percentile.
    """
    n_rows = len(dataframe)
    tau_dict = {}

    dataframe = dataframe.sort_values(by=overall_score_column, ascending=False)

    for target in selected_columns:
        taus = []
        for i in range(num_percentiles):
            threshold = i / num_percentiles
            start = int(threshold * n_rows)
            end = int((threshold + 1 / num_percentiles) * n_rows)
            subset = dataframe.iloc[start:end].copy()
            subset["overall_score_bin"] = map_to_bins(subset[overall_score_column], 100)
            subset[target + "_bin"] = map_to_bins(subset[target], 100)
            tau, _ = kendalltau(subset["overall_score_bin"], subset[target + "_bin"])
            taus.append(tau)
            if only_top_part and i == 0:
                break
        tau_dict[target] = taus

    if not only_top_part:
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.tab20.colors)

        percentiles = [f"{num_percentiles*i}%" for i in range(1, num_percentiles + 1)]
        plt.figure(figsize=(10, 6))
        for key, values in tau_dict.items():
            plt.plot(percentiles, values, marker=".", label=key)

        plt.axhline(0, color="red", linewidth=1.5, linestyle=":")

        plt.title("Factor Importance in Overall Ranking Score")
        plt.xlabel("Ranking Percentiles")
        plt.ylabel("Tau values")
        plt.legend()
        plt.show()

    return tau_dict
