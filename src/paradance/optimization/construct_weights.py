from typing import TYPE_CHECKING, List

import optuna

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

    if ob.dirichlet:
        for i in range(ob.weights_num - 1):
            power_weights.append(
                trial.suggest_float(
                    f"w_po_{i+1}", 0, max((1 - sum(power_weights), 0.1))
                )
            )
        power_weights.append(1 - sum(power_weights))
    else:
        for i in range(ob.weights_num):
            power_weights.append(
                trial.suggest_float(
                    f"w{i+1}", ob.power_lower_bound, ob.power_upper_bound
                )
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

    if not (ob.first_order):
        weights = construct_power_weights(ob, trial)
    elif ob.calculator.equation_type == "product" and ob.power:
        weights = construct_power_weights(ob, trial) + construct_first_order_weights(
            ob, trial
        )
    elif ob.calculator.equation_type == "sum":
        weights = construct_first_order_weights(ob, trial)
    elif ob.calculator.equation_type == "free_style":
        weights = construct_first_order_weights(ob, trial)
    ob.calculator.get_overall_score(
        weights_for_equation=weights,
    )

    return weights
