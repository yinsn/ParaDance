from typing import TYPE_CHECKING, List

from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    from .calculator import Calculator


def calculate_woauc(
    calculator: "Calculator",
    target_column: str,
    weights_for_equation: List,
) -> List[float]:
    """Calculate weighted ordinal user AUC.

    :param weights_for_equation: weights for equation
    :param target_column: score column
    """
    woauc_indices = calculator.woauc_dict[target_column]
    woauc = []
    sampler = calculator.samplers[target_column]
    for k, _ in sampler.boundary_dict.items():
        paritial_auc = float(
            roc_auc_score(
                calculator.df.loc[woauc_indices][f"{target_column}_lt_{k}"],
                calculator.df.loc[woauc_indices]["overall_score"],
            )
        )
        woauc.append(paritial_auc)
    return woauc
