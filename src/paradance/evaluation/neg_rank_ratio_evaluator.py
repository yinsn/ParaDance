from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .calculator import Calculator


def calculate_neg_rank_ratio(
    calculator: "Calculator", weights_for_equation: List, label_column: str = "label"
) -> float:
    """Calculate the rank ratio of negative target
    :param weights_for_equation: weights for equation
    :param label_column: target column, its values must be 0 or 1
    """
    neg_targets_rows = calculator.df[label_column].sum()
    total_rows = calculator.df.shape[0]
    neg_rank_sum = (
        calculator.df["overall_score"].rank(ascending=False, method="first")
        * calculator.df[label_column]
    ).sum()
    ratio = float(
        neg_rank_sum * 2 / ((total_rows * 2 - neg_targets_rows + 1) * neg_targets_rows)
    )
    return ratio
