from typing import TYPE_CHECKING, Tuple

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from .calculator import Calculator


def calculate_auc_triple_parameters(
    calculator: "Calculator", grid_interval: int
) -> Tuple:
    """Calculate AUC triple parameters.

    :param grid_interval: grid interval
    :return: tuple of W1, W2, WUAUC
    """
    w1_values = np.linspace(0, 1, grid_interval)
    w2_values = np.linspace(0, 1, grid_interval)
    W1, W2 = np.meshgrid(w1_values, w2_values)
    WUAUC = np.zeros_like(W1)

    for i in tqdm(range(W1.shape[0]), desc="Progress"):
        for j in range(W1.shape[1]):
            w1 = W1[i, j]
            w2 = W2[i, j]
            w3 = 1 - w1 - w2
            if w3 < 0:
                WUAUC[i, j] = np.nan
            else:
                WUAUC[i, j] = calculator.calculate_wuauc(
                    groupby="user_id",
                    weights_for_equation=[w1, w2, w3],
                    weights_for_groups=calculator.weights_for_groups,
                )
    return W1, W2, WUAUC
