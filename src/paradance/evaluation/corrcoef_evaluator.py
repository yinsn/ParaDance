from typing import TYPE_CHECKING, Optional

import numpy as np

from .base_evaluator import evaluation_preprocessor

if TYPE_CHECKING:
    from .calculator import Calculator


@evaluation_preprocessor
def calculate_corrcoef(
    calculator: "Calculator",
    target_column: str,
    mask_column: Optional[str] = None,
) -> float:
    """
    Calculates the Pearson correlation coefficient between the 'overall_score' column
    and a specified target column in the dataframe.

    Args:
        calculator (Calculator): An instance of the Calculator class containing
                                 the evaluated dataframe.
        target_column (str): The name of the target column to compute the correlation with.
        mask_column (Optional[str], optional): The name of the column to use as a mask.
                                               Defaults to None.

    Returns:
        float: The Pearson correlation coefficient between 'overall_score' and
               the target column.
    """
    df = calculator.evaluated_dataframe
    correlation_matrix = np.corrcoef(df["overall_score"], df[target_column])
    pearson_correlation = float(correlation_matrix[0, 1])
    return pearson_correlation
