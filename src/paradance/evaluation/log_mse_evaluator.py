from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .calculator import Calculator


def calculate_log_mse(calculator: "Calculator", target_column: str) -> float:
    """Calculate log mean squared error.

    :param target_column: target column
    """
    log_true = np.log(calculator.df[target_column] + 1)
    log_pred = np.log(calculator.df["overall_score"] + 1)
    mse = np.mean((log_true - log_pred) ** 2)
    return float(mse)
