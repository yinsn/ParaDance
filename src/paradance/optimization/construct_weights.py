from typing import TYPE_CHECKING, List

import optuna

if TYPE_CHECKING:
    from .multiple_objective import MultipleObjective


def construct_weights(ob: "MultipleObjective", trial: optuna.Trial) -> List[float]:
    """
    Construct a list of weights based on the attributes of the given MultipleObjective instance
    and the current optuna trial.

    Args:
        ob (MultipleObjective): An instance of the MultipleObjective class, encapsulating various
                                attributes like 'power', 'first_order', 'weights_num', and so forth,
                                which guide the construction of the weight values.
        trial (optuna.Trial): The current optuna trial from which to suggest values for the weights
                              based on the properties of the given MultipleObjective instance.

    Returns:
        List[float]: A list of weight values constructed based on the given MultipleObjective
                     instance and the current optuna trial.
    """
    power_weights: List[float] = []
    first_order_weights: List[float] = []

    if ob.power:
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

    if ob.first_order:
        if ob.first_order_lower_bound < 0:
            log = False
        else:
            log = True
        for i in range(ob.weights_num):
            first_order_weights.append(
                trial.suggest_float(
                    f"w_fo_{i+1}",
                    ob.first_order_lower_bound,
                    ob.first_order_upper_bound,
                    log=log,
                )
            )

    weights: List[float] = []
    if not (ob.first_order):
        weights = power_weights
    elif ob.calculator.equation_type == "product" and ob.power:
        weights = power_weights + first_order_weights
    elif ob.calculator.equation_type == "sum":
        weights = first_order_weights
    elif ob.calculator.equation_type == "free_style":
        weights = first_order_weights
    ob.calculator.get_overall_score(
        weights_for_equation=weights,
    )
    return weights
