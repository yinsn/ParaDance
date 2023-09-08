from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    from .calculator import Calculator


def calculate_wuauc(
    calculator: "Calculator",
    groupby: Optional[str],
    weights_for_equation: List,
    weights_for_groups: Optional[pd.Series] = None,
    label_column: str = "label",
    auc: bool = False,
) -> float:
    """Calculate weighted user AUC.

    :param groupby: groupby column
    :param weights_for_equation: weights for equation
    :param weights_for_groups: weights for group
    :param label_column: label column
    :param auc: bool, optional, default: False
    :return: AUC/WUAUC/UAUC
    """
    if auc:
        result = float(
            roc_auc_score(calculator.df[label_column], calculator.df["overall_score"])
        )
    else:
        if groupby is not None:
            grouped = calculator.df.groupby(groupby).apply(
                lambda x: float(roc_auc_score(x[label_column], x["overall_score"]))
            )
            if weights_for_groups is not None:
                counts_sorted = weights_for_groups.loc[grouped.index]
                result = float(np.average(grouped, weights=counts_sorted.values))
            else:
                result = float(np.mean(grouped))
    return result
