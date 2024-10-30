from typing import TYPE_CHECKING, Optional

import numpy as np

from .base_evaluator import evaluation_preprocessor

if TYPE_CHECKING:
    from .calculator import Calculator


@evaluation_preprocessor
def calculate_mean(
    calculator: "Calculator",
    target_column: str,
    mask_column: Optional[str] = None,
    target_mean: float = 0.0,
    log_scale: bool = True,
    laplace_smoothing: float = 1.0,
    use_rerank: bool = True,
) -> float:
    """Calculate the adjusted mean of a specified target column, with optional log-scaling and smoothing.

    Args:
        calculator (Calculator): The calculator instance containing the relevant DataFrame with scores.
        target_column (str): The name of the column in `calculator.df` whose mean will be calculated.
        mask_column (Optional[str]): An optional column to mask values within `target_column` before calculation. Defaults to None.
        target_mean (float): The target mean value to calculate the deviation from. Defaults to 0.0.
        log_scale (bool): If True, applies logarithmic scaling to scores for mean calculation. Defaults to True.
        laplace_smoothing (float): A constant added to scores for numerical stability in log-scale calculations. Defaults to 1.0.
        use_rerank (bool): If True, uses the 'overall_score' column; if False, uses 'overall_score_before_rerank'. Defaults to True.

    Returns:
        float: The absolute deviation of the calculated mean from `target_mean`.
    """
    if use_rerank:
        scores = calculator.df["overall_score"]
    else:
        scores = calculator.df["overall_score_before_rerank"]

    if log_scale:
        mean_value = np.log(scores + laplace_smoothing).mean()
    else:
        mean_value = scores.mean()

    return float(abs(mean_value - target_mean))
