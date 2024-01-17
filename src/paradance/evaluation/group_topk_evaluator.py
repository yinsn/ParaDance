from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    from .calculator import Calculator


def calculate_group_topk(
    calculator: "Calculator",
    groupby: Optional[str],
    weights_for_groups: Optional[pd.Series] = None,
    label_column: str = "label",
    group_top_k: int = 1,
) -> float:
    """Calculate weighted user AUC.

    :param groupby: groupby column
    :param weights_for_equation: weights for equation
    :param weights_for_groups: weights for group
    :param label_column: label column
    :param auc: bool, optional, default: False
    :return: AUC/WUAUC/UAUC
    """
    if groupby is not None:
        grouped = calculator.df.groupby(groupby).apply(
            #lambda x: float(roc_auc_score(x[label_column], x["overall_score"]))
        lambda x: float(sum(sorted_row[0] for sorted_row in sorted(zip(x[label_column], x["overall_score"]), key=lambda row: row[1], reverse=True)[:group_top_k]))
        )
        if weights_for_groups is not None:
            counts_sorted = weights_for_groups.loc[grouped.index]
            result = float(np.average(grouped, weights=counts_sorted.values))
        else:
            result = float(np.mean(grouped))
    else:
        return 0.0
    return result
