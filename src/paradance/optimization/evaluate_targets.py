from typing import List, Optional

from ..evaluation.calculator import Calculator


def evaluate_targets(
    calculator: Calculator,
    evaluator_flags: List[str],
    hyperparameters: List[Optional[float]],
    evaluator_propertys: List[Optional[str]],
    groupbys: List[Optional[str]],
    target_columns: List[str],
    weights: List[float],
) -> List[float]:
    targets = []
    for flag, hyperparameter, evaluator_property, groupby, target_column in zip(
        evaluator_flags, hyperparameters, evaluator_propertys, groupbys, target_columns
    ):
        if flag == "portfolio":
            _, concentration = calculator.calculate_portfolio_concentration(
                target_column=target_column,
                expected_return=hyperparameter,
            )
            targets.append(concentration)

        elif flag == "distinct_count_portfolio":
            (
                _,
                concentration,
            ) = calculator.calculate_distinct_count_portfolio_concentration(
                target_column=target_column,
                expected_coverage=hyperparameter,
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
                groupby=groupby,
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
        elif flag == "inverse_pairs":
            inverse_score = calculator.calculate_inverse_pair(
                calculator=calculator,
                weights_for_equation=weights,
                weights_type=evaluator_property,
            )
            targets.append(inverse_score)
        elif flag == "tau":
            tau = calculator.calculate_tau(
                groupby=groupby,
                target_column=target_column,
                num_bins=hyperparameter,
            )
            targets.append(tau)
    return targets
