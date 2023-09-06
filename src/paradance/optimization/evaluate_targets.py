from typing import List, Optional

from ..evaluation.calculator import Calculator


def evaluate_targets(
    calculator: Calculator,
    evaluator_flags: List[str],
    hyperparameters: List[Optional[float]],
    groupbys: List[Optional[str]],
    target_columns: List[str],
    weights: List[float],
) -> List[float]:
    targets = []
    for flag, hyperparameter, groupby, target_column in zip(
        evaluator_flags, hyperparameters, groupbys, target_columns
    ):
        if flag == "portfolio":
            _, concentration = calculator.calculate_portfolio_concentration(
                target_column=target_column,
                expected_return=hyperparameter,
            )
            targets.append(concentration)
        elif flag == "wuauc":
            wuauc = calculator.calculate_wuauc(
                groupby=groupby,
                label_column=target_column,
                weights_for_equation=weights,
            )
            targets.append(wuauc)
        elif flag == "auc":
            auc = calculator.calculate_wuauc(
                groupby=groupby,
                label_column=target_column,
                weights_for_equation=weights,
                auc=True,
            )
            targets.append(auc)
        elif flag == "woauc":
            woauc = calculator.calculate_woauc(
                target_column=target_column,
                weights_for_equation=weights,
            )
            targets.append(sum(woauc))
        elif flag == "logmse":
            mse = calculator.calculate_log_mse(
                target_column=target_column,
            )
            targets.append(mse)
        elif flag == "neg_rank_ratio":
            neg_rank_ratio = calculator.calculate_neg_rank_ratio(
                weights_for_equation=weights, label_column=target_column
            )
            targets.append(neg_rank_ratio)
    return targets