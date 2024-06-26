from typing import TYPE_CHECKING, Optional

from .base_evaluator import evaluation_preprocessor

if TYPE_CHECKING:
    from .calculator import Calculator


@evaluation_preprocessor
def calculate_top_coverage(
    calculator: "Calculator",
    target_column: str,
    mask_column: Optional[str] = None,
    head_percentage: Optional[float] = None,
) -> float:
    """Calculates the top coverage ratio of a specified column in a DataFrame.

    Args:
        calculator (Calculator): An object that provides access to the evaluated DataFrame.
        target_column (str): The name of the column to calculate the coverage for.
        mask_column (Optional[str], optional): The name of the column to apply a mask on. Defaults to None.
        head_percentage (Optional[float], optional): The percentage of the top rows to consider. Defaults to 0.05.

    Returns:
        float: The ratio of the sum of the top `head_percentage` rows to the total sum of the `target_column`.
    """
    if head_percentage is None:
        head_percentage = 0.05

    df = calculator.evaluated_dataframe
    df = df.sort_values("overall_score", ascending=False)
    total_sum = df[target_column].sum()
    top_rows_count = int(len(df) * head_percentage)
    top_sum = df.iloc[:top_rows_count][target_column].sum()
    top_coverage_ratio = top_sum / total_sum

    return float(top_coverage_ratio)


@evaluation_preprocessor
def calculate_distinct_top_coverage(
    calculator: "Calculator",
    target_column: str,
    mask_column: Optional[str] = None,
    head_percentage: Optional[float] = None,
) -> float:
    """Calculates the distinct top coverage ratio of a specified column in a DataFrame.

    Args:
        calculator (Calculator): An object that provides access to the evaluated DataFrame.
        target_column (str): The name of the column to calculate the distinct coverage for.
        mask_column (Optional[str], optional): The name of the column to apply a mask on. Defaults to None.
        head_percentage (Optional[float], optional): The percentage of the top rows to consider. Defaults to 0.05.

    Returns:
        float: The ratio of the number of unique top `head_percentage` rows to the total number of unique `target_column` values.
    """
    if head_percentage is None:
        head_percentage = 0.05

    df = calculator.evaluated_dataframe
    total_ids = df[target_column].nunique()

    df_sorted = df.sort_values(by="overall_score", ascending=False).reset_index(
        drop=True
    )
    df_sorted["is_unique"] = ~df_sorted[target_column].duplicated()
    top_rows_count = int(len(df) * head_percentage)
    top_sum = df_sorted.iloc[:top_rows_count]["is_unique"].sum()

    top_coverage_ratio = top_sum / total_ids

    return float(top_coverage_ratio)
