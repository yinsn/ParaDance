from typing import TYPE_CHECKING, Optional

from .base_evaluator import evaluation_preprocessor

if TYPE_CHECKING:
    from .calculator import Calculator


@evaluation_preprocessor
def calculate_top_coverage(
    calculator: "Calculator",
    target_column: str,
    mask_column: Optional[str] = None,
    head_percentage: float = 0.05,
) -> float:
    df = calculator.evaluated_dataframe
    df = df.sort_values("overall_score", ascending=False)
    total_sum = df[target_column].sum()
    print(head_percentage)
    top_rows_count = int(len(df) * head_percentage)
    top_sum = df.iloc[:top_rows_count][target_column].sum()
    top_coverage_ratio = top_sum / total_sum

    return float(top_coverage_ratio)
