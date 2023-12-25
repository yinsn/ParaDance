from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .calculator import Calculator


def calculate_distinct_count_portfolio_concentration(
    calculator: "Calculator",
    target_column: str,
    expected_coverage: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculate the threshold and concentration of a portfolio based on a specified target column and expected coverage.

    This function takes a Calculator object, which is expected to contain a DataFrame. It computes the unique count of
    values in the target column and determines a threshold in 'overall_score' such that a certain proportion
    (expected_coverage) of unique values is covered. The concentration is calculated as the proportion of rows
    exceeding this threshold.

    Args:
        calculator: An instance of Calculator which contains a DataFrame.
        target_column: The name of the column in the DataFrame to analyze.
        expected_coverage: The expected proportion of unique values to be covered. Defaults to 0.95 if not provided.

    Returns:
        A tuple containing:
        - threshold (float): The minimum 'overall_score' to cover the expected proportion of unique values.
        - concentration (float): The proportion of rows in the DataFrame with 'overall_score' above the threshold.
    """
    if expected_coverage is None:
        expected_coverage = 0.95
    df = calculator.df.copy()
    total_ids = df[target_column].nunique()

    df_sorted = df.sort_values(by="overall_score", ascending=False).reset_index(
        drop=True
    )

    df_sorted["is_unique"] = ~df_sorted[target_column].duplicated()
    df_sorted["cumulative_coverage"] = df_sorted["is_unique"].cumsum() / total_ids
    df_filtered = df_sorted[df_sorted["cumulative_coverage"] > expected_coverage]
    if not df_filtered.empty:
        threshold_row = df_filtered.iloc[0]
        threshold = threshold_row["overall_score"]
        concentration = len(
            df_sorted[df_sorted["overall_score"] > threshold][target_column]
        ) / len(df_sorted)
    else:
        threshold = 1
        concentration = 1
    return threshold, concentration
