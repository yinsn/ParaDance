from typing import TYPE_CHECKING, Optional

import numpy as np

from .base_evaluator import evaluation_preprocessor

if TYPE_CHECKING:
    from .calculator import Calculator


@evaluation_preprocessor
def calculate_cumulative_deviation(
    calculator: "Calculator",
    target_column: str,
    mask_column: Optional[str] = None,
    n_quantiles: Optional[int] = 10,
) -> float:
    """
    Calculate the cumulative quantile deviation between the sorted values of a target column
    and the 'overall_score' column in quantiles.

    Args:
        calculator (Calculator): An instance of a `Calculator` that contains an evaluated dataframe.
        target_column (str): The column in the dataframe whose deviation will be compared to 'overall_score'.
        mask_column (Optional[str], optional): A column used as a mask (not currently utilized in the logic). Defaults to None.
        n_quantiles (Optional[int], optional): The number of quantiles to split the data into. Defaults to 10.

    Returns:
        float: The cumulative deviation between the target column and the 'overall_score' column
               across the specified quantiles.
    """
    if n_quantiles is None:
        n_quantiles = 10

    df = calculator.evaluated_dataframe
    sorted_col1 = np.sort(df[target_column])[::-1]
    sorted_col2 = np.sort(df["overall_score"])[::-1]

    quantiles = np.linspace(1 / n_quantiles, 1, n_quantiles)
    indices = np.floor(quantiles * len(sorted_col1)).astype(int)

    diffs = np.zeros(n_quantiles - 1)
    for i in range(n_quantiles - 1):
        avg1 = np.mean(sorted_col1[: indices[i + 1]])
        avg2 = np.mean(sorted_col2[: indices[i + 1]])
        diffs[i] = abs(avg1 - avg2) / max(avg1, avg2)

    return float(sum(diffs))
