from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .calculator import Calculator


def calculate_portfolio_concentration(
    calculator: "Calculator",
    target_column: str,
    expected_return: Optional[float] = 0.95,
) -> Tuple[float, float]:
    """Calculate portfolio concentration.

    :param target_column: target column
    :param expected_return: expected return
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
