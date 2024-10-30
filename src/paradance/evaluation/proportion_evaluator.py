from typing import TYPE_CHECKING, Optional

from .base_evaluator import evaluation_preprocessor

if TYPE_CHECKING:
    from .calculator import Calculator


@evaluation_preprocessor
def calculate_proportion(
    calculator: "Calculator",
    target_column: str,
    mask_column: Optional[str] = None,
    target_value: float = 0.0,
    use_rerank: bool = True,
) -> float:
    if use_rerank:
        proportion = (calculator.df["overall_score"] == target_value).mean()
    else:
        proportion = (
            calculator.df["overall_score_before_rerank"] == target_value
        ).mean()

    return float(proportion)
