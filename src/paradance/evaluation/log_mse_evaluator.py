from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .calculator import Calculator


def calculate_log_mse(
    calculator: "Calculator",
    target_column: str,
    laplace_smoothing: float = 1.0,
    use_rerank: bool = True,
) -> float:
    """Calculate log mean squared error.

    :param target_column: target column
    :param laplace_smoothing: Laplace smoothing
    :param use_rerank: whether to use rerank
    """
    log_true = np.log(calculator.df[target_column] + laplace_smoothing)
    if use_rerank:
        log_pred = np.log(calculator.df["overall_score"] + laplace_smoothing)
    else:
        log_pred = np.log(
            calculator.df["overall_score_before_rerank"] + laplace_smoothing
        )
    mse = np.mean((log_true - log_pred) ** 2)
    return float(mse)
