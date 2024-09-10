from typing import TYPE_CHECKING, List

import numpy as np
import optuna

from ..evaluation import Calculator

if TYPE_CHECKING:
    from .multiple_objective import MultipleObjective


def construct_power_weights(
    ob: "MultipleObjective", trial: optuna.Trial
) -> List[float]:
    """
    Construct power weights based on the attributes of the MultipleObjective instance and the current trial.

    Args:
        ob (MultipleObjective): An instance of the MultipleObjective class, guiding the construction of the power weights.
        trial (optuna.Trial): The current optuna trial from which to suggest values for the weights.

    Returns:
        List[float]: A list of power weights constructed based on the given MultipleObjective instance and the current trial.
    """
    power_weights: List[float] = []
    if ob.weights_num is None:
        ob.weights_num = ob.get_weights_num()

    if ob.dirichlet:
        for i in range(ob.weights_num - 1):
            power_weights.append(
                trial.suggest_float(
                    f"w_po_{i+1}", 0, max((1 - sum(power_weights), 0.1))
                )
            )
        power_weights.append(1 - sum(power_weights))

    lower_bounds = (
        [ob.power_lower_bound] * ob.weights_num
        if isinstance(ob.power_lower_bound, float)
        else ob.power_lower_bound
    )
    upper_bounds = (
        [ob.power_upper_bound] * ob.weights_num
        if isinstance(ob.power_upper_bound, float)
        else ob.power_upper_bound
    )

    for i in range(ob.weights_num):
        power_weights.append(
            trial.suggest_float(f"w{i+1}", lower_bounds[i], upper_bounds[i])
        )

    return power_weights


def construct_first_order_weights(
    ob: "MultipleObjective", trial: optuna.Trial
) -> List[float]:
    """
    Construct first order weights based on the attributes of the MultipleObjective instance and the current trial.

    Args:
        ob (MultipleObjective): An instance of the MultipleObjective class, guiding the construction of the first order weights.
        trial (optuna.Trial): The current optuna trial from which to suggest values for the weights.

    Returns:
        List[float]: A list of first order weights constructed based on the given MultipleObjective instance and the current trial.
    """
    first_order_weights: List[float] = []
    log = ob.first_order_lower_bound >= 0

    if ob.weights_num is None:
        ob.weights_num = ob.get_weights_num()

    first_order_scale_upper_bound = ob.first_order_scale_upper_bound
    first_order_scale_lower_bound = ob.first_order_scale_lower_bound

    if isinstance(ob.calculator, Calculator) and ob.first_order_with_scales:
        value_scales = ob.calculator.value_scales
        for i in range(ob.weights_num):
            first_order_weights.append(
                trial.suggest_float(
                    f"w_fo_{i+1}",
                    np.power(10, value_scales[i] - first_order_scale_lower_bound),
                    np.power(10, value_scales[i] + first_order_scale_upper_bound),
                    log=False,
                )
            )
        if ob.max_min_scale_ratio is not None:
            scales_with_weights = [
                weight * np.power(10, -value_scale)
                for weight, value_scale in zip(first_order_weights, value_scales)
            ]
            scales_with_weights_ratio = max(scales_with_weights) / (
                min(scales_with_weights) + 1e-6
            )
            first_order_weights = [
                weight * ob.max_min_scale_ratio / scales_with_weights_ratio
                for weight in first_order_weights
            ]
    else:
        for i in range(ob.weights_num):
            first_order_weights.append(
                trial.suggest_float(
                    f"w_fo_{i+1}",
                    ob.first_order_lower_bound,
                    ob.first_order_upper_bound,
                    log=log,
                )
            )

    return first_order_weights


def construct_free_style_weights(
    ob: "MultipleObjective", trial: optuna.Trial
) -> List[float]:
    """Constructs a list of free-style weights for a given multiple objective.

    Args:
        ob (MultipleObjective): The multiple objective instance with the weight configurations.
        trial (optuna.Trial): The optuna trial object used to suggest the weights.

    Returns:
        List[float]: A list of weights suggested by the optuna trial.
    """
    free_style_weights: List[float] = []
    if ob.weights_num is None:
        ob.weights_num = ob.get_weights_num()
    if ob.base_weights is not None:
        ob.free_style_lower_bound = [
            (1 - ob.base_weights_offset_ratio) * weight for weight in ob.base_weights
        ]
        ob.free_style_upper_bound = [
            (1 + ob.base_weights_offset_ratio) * weight for weight in ob.base_weights
        ]
    if isinstance(ob.free_style_lower_bound, list) and isinstance(
        ob.free_style_upper_bound, list
    ):
        for i in range(ob.weights_num):
            free_style_weights.append(
                trial.suggest_float(
                    f"w{i+1}",
                    ob.free_style_lower_bound[i],
                    ob.free_style_upper_bound[i],
                )
            )
        return free_style_weights
    elif isinstance(ob.free_style_lower_bound, float) and isinstance(
        ob.free_style_upper_bound, float
    ):
        for i in range(ob.weights_num):
            free_style_weights.append(
                trial.suggest_float(
                    f"w{i+1}", ob.free_style_lower_bound, ob.free_style_upper_bound
                )
            )
    else:
        raise ValueError("Invalid free style bounds.")
    return free_style_weights


def construct_log_pca_weights(
    ob: "MultipleObjective", trial: optuna.Trial
) -> List[float]:
    """
    Constructs a list of weights for log PCA components based on the optimization trial.

    This function generates a list of floating-point numbers representing the weights
    for log PCA components. Each weight is suggested by the trial within the bounds
    defined in the MultipleObjective instance.

    """
    log_pca_weights: List[float] = []
    if ob.weights_num is None:
        ob.weights_num = ob.get_weights_num()
    for i in range(ob.weights_num):
        log_pca_weights.append(
            trial.suggest_float(
                f"w{i+1}", ob.pca_importance_lower_bound, ob.pca_importance_upper_bound
            )
        )
    return log_pca_weights


def construct_weights(ob: "MultipleObjective", trial: optuna.Trial) -> List[float]:
    """
    Construct weights by combining power and first order weights as required by the MultipleObjective instance.

    Args:
        ob (MultipleObjective): An instance of the MultipleObjective class, guiding the construction of the weights.
        trial (optuna.Trial): The current optuna trial from which to suggest values for the weights.

    Returns:
        List[float]: A list of weights constructed based on the given MultipleObjective instance and the current trial.
    """
    weights = []
    if ob.calculator.equation_type == "sum":
        weights = construct_first_order_weights(ob, trial)
    elif (ob.calculator.equation_type == "free_style") or (
        ob.calculator.equation_type == "json"
    ):
        weights = construct_free_style_weights(ob, trial)
    elif ob.calculator.equation_type == "log_pca":
        weights = construct_log_pca_weights(ob, trial)
    elif ob.calculator.equation_type == "product" and ob.first_order:
        weights = construct_power_weights(ob, trial) + construct_first_order_weights(
            ob, trial
        )
    elif not (ob.first_order):
        weights = construct_power_weights(ob, trial)

    return weights
