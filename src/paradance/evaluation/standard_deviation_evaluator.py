from typing import TYPE_CHECKING, Optional

import numpy as np

from .base_evaluator import evaluation_preprocessor

if TYPE_CHECKING:
    from .calculator import Calculator


@evaluation_preprocessor
def calculate_standard_deviation(
    calculator: "Calculator",
    target_column: str,
    mask_column: Optional[str] = None,
    target_std: float = 0.0,
    log_scale: bool = True,
    laplace_smoothing: float = 1.0,
    use_rerank: bool = True,
) -> float:
    """Calculate the adjusted std of a specified target column, with optional log-scaling and smoothing.

    Args:
        calculator (Calculator): The calculator instance containing the relevant DataFrame with scores.
        target_column (str): The name of the column in `calculator.df` whose std will be calculated.
        mask_column (Optional[str]): An optional column to mask values within `target_column` before calculation. Defaults to None.
        target_std (float): The target std value to calculate the deviation from. Defaults to 0.0.
        log_scale (bool): If True, applies logarithmic scaling to scores for std calculation. Defaults to True.
        laplace_smoothing (float): A constant added to scores for numerical stability in log-scale calculations. Defaults to 1.0.
        use_rerank (bool): If True, uses the 'overall_score' column; if False, uses 'overall_score_before_rerank'. Defaults to True.

    Returns:
        float: The absolute deviation of the calculated std from `target_std`.
    """
    if use_rerank:
        scores = calculator.df["overall_score"]
    else:
        scores = calculator.df["overall_score_before_rerank"]

    if log_scale:
        std_value = np.log(scores + laplace_smoothing).std()
    else:
        std_value = scores.std()

    return float(abs(std_value - target_std))
