import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class JSONFormula(BaseModel):
    """Model to represent a JSON formula."""

    formula: Dict[str, str]

    class Config:
        extra = "forbid"

    @field_validator("formula")
    def check_formula_not_empty(cls, value: Dict[str, str]) -> Dict[str, str]:
        if not value:
            raise ValueError("The 'formula' field must not be empty.")
        return value


def if_else(cond: bool, true_val: Any, false_val: Any) -> Any:
    """Evaluates and returns `true_val` if `cond` is True, otherwise `false_val`.

    Args:
        cond (bool): Condition to evaluate.
        true_val (Any): Value to return if condition is True.
        false_val (Any): Value to return if condition is False.

    Returns:
        Any: `true_val` if `cond` is True, otherwise `false_val`.
    """
    return true_val if cond else false_val


def safe_eval(expression: str, context: Dict[str, Any]) -> Optional[float]:
    """Safely evaluates a mathematical expression in a given context.

    Args:
        expression (str): The expression to evaluate.
        context (Dict[str, Any]): A dictionary providing values for variables used in the expression.

    Returns:
        Optional[float]: The evaluated result as a float, or None if an error occurs.
    """
    expression = expression.replace("^", "**")
    expression = re.sub(
        r"if\(([^,]+),([^,]+),([^)]+)\)",
        lambda m: f"if_else({m.group(1)}, {m.group(2)}, {m.group(3)})",
        expression,
    )
    sorted_keys = sorted(context.keys(), key=len, reverse=True)
    for var in sorted_keys:
        if var in expression:
            expression = expression.replace(var, str(context[var]))

    try:
        return float(
            eval(
                expression,
                {"__builtins__": None},
                {"if_else": if_else, "min": min, "max": max, "abs": abs, "log": np.log},
            )
        )
    except Exception as e:
        logger.info(f"Error evaluating expression: {expression}", exc_info=True)
        return None


def calculate_row(
    row: pd.Series,
    equation_json: JSONFormula,
    tuning_weights: Dict[str, float],
    delimiter: Optional[str],
) -> Optional[float]:
    """Calculates the final score for a row based on the provided formula JSON and tuning weights.

    Args:
        row (pd.Series): A row of data.
        equation_json (JSONFormula): The JSON formula object containing the expressions to calculate scores.
        tuning_weights (Dict[str, float]): Dictionary of weights for tuning the calculations.
        delimiter (Optional[str]): Delimiter to split the keys in the formula.

    Returns:
        Optional[float]: The final calculated score for the row, or None if an error occurs.
    """
    variable_values = row.to_dict()
    variable_values.update(tuning_weights)
    row_scores: Dict[str, float] = {}
    for key, formula in equation_json.formula.items():
        base_key = key.split(delimiter)[0]
        result = safe_eval(formula, {**variable_values, **row_scores})
        if result is not None:
            row_scores[key] = result
            variable_values[base_key] = result

    final_score_key = list(equation_json.formula.keys())[-1]
    return safe_eval(
        equation_json.formula.get(final_score_key, "0"),
        {**variable_values, **row_scores},
    )


def calculate_formula_scores(
    equation_json: JSONFormula,
    selected_values: pd.DataFrame,
    weights: List[float],
    delimiter: Optional[str] = "#",
) -> pd.Series:
    """Calculates scores for each row in the DataFrame based on the provided formula JSON and weights.

    Args:
        equation_json (JSONFormula): The JSON formula object containing the expressions to calculate scores.
        data (pd.DataFrame): The data on which to apply the formula.
        weights (List[float]): List of weights for tuning the calculations.
        delimiter (Optional[str], optional): Delimiter to split the keys in the formula. Defaults to '#'.

    Returns:
        pd.Series: A series containing the calculated scores for each row in the DataFrame.
    """
    tuning_weights: Dict[str, float] = {
        f"weights[{index}]": value for index, value in enumerate(weights)
    }
    return selected_values.apply(
        lambda row: calculate_row(row, equation_json, tuning_weights, delimiter), axis=1
    )
