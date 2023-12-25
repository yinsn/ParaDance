from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .calculator import Calculator


def calculate_portfolio_concentration(
    calculator: "Calculator",
    target_column: str,
    expected_return: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculate the threshold and concentration of a portfolio based on a target column and expected return.

    This function sorts the DataFrame within a Calculator object based on 'overall_score' and calculates
    the cumulative sum and ratio of the target column. It determines the maximum 'overall_score' where
    the cumulative ratio exceeds the expected return. The concentration is the proportion of data points
    with an 'overall_score' higher than this threshold.

    Args:
        calculator: An instance of Calculator, expected to contain a DataFrame and an attribute 'df_len' for the length of the DataFrame.
        target_column: The column name in the DataFrame for which concentration is calculated.
        expected_return: The expected cumulative ratio, defaults to 0.95. If None, it's set to 0.95.

    Returns:
        A tuple containing:
        - threshold (float): The 'overall_score' value above which the expected return is met or exceeded.
        - concentration (float): The proportion of data points with 'overall_score' greater than the threshold.
    """
    if expected_return is None:
        expected_return = 0.95
    df = calculator.df.sort_values("overall_score", ascending=False)
    sum_all = df[target_column].sum()
    df["cumulative_sum"] = df[target_column].cumsum()
    df["cumulative_ratio"] = df["cumulative_sum"] / sum_all
    threshold = df[df["cumulative_ratio"] > expected_return]["overall_score"].max()
    concentration = df[df["overall_score"] > threshold].shape[0] / calculator.df_len
    concentration = 1 if concentration == 0 else concentration
    return threshold, concentration
