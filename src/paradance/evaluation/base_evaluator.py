from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, cast

if TYPE_CHECKING:
    from .calculator import Calculator


R = TypeVar("R")
F = Callable[..., R]


def evaluation_preprocessor(func: F) -> F:
    """
    A decorator to preprocess a dataframe before evaluation.

    Args:
        func (F): The evaluation function to be wrapped by this preprocessor. This function should accept a Calculator
            instance, a target column name, and optionally other arguments and keyword arguments.

    Returns:
        F: A wrapped evaluation function that first preprocesses the dataframe before calling the original evaluation
        function with the preprocessed dataframe and any additional arguments.
    """

    def wrapper(
        calculator: "Calculator",
        mask_column: Optional[str],
        target_column: str,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        if mask_column and mask_column in calculator.df.columns:
            masked_indices = calculator.df[calculator.df[mask_column] != 0].index
        else:
            masked_indices = calculator.df.index
        calculator.evaluated_dataframe = calculator.df.loc[masked_indices]
        return func(calculator, target_column, *args, **kwargs)

    return cast(F, wrapper)
