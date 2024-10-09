from typing import List, Optional, Union

import pandas as pd

from ..evaluation import Calculator, LogarithmPCACalculator


def evaluate_targets(
    calculator: Union[Calculator, LogarithmPCACalculator],
    evaluator_flags: List[str],
    target_columns: List[str],
    mask_columns: List[Optional[str]],
    hyperparameters: List[Optional[float]],
    evaluator_propertys: List[Optional[str]],
    groupbys: List[Optional[str]],
    group_weights: List[Optional[pd.Series]],
) -> List[float]:
    targets = []
    for (
        flag,
        mask_column,
        hyperparameter,
        evaluator_property,
        groupby,
        target_column,
        weights_for_groups,
    ) in zip(
        evaluator_flags,
        mask_columns,
        hyperparameters,
        evaluator_propertys,
        groupbys,
        target_columns,
        group_weights,
    ):
        if flag == "pearson":
            corrcoef = calculator.calculate_corrcoef(
                target_column=target_column,
                mask_column=mask_column,
            )
            targets.append(corrcoef)

        elif flag == "portfolio":
            _, concentration = calculator.calculate_portfolio_concentration(
                target_column=target_column,
                mask_column=mask_column,
                expected_return=hyperparameter,
            )
            targets.append(concentration)

        elif flag == "distinct_count_portfolio":
            (
                _,
                concentration,
            ) = calculator.calculate_distinct_count_portfolio_concentration(
                target_column=target_column,
                mask_column=mask_column,
                expected_coverage=hyperparameter,
            )
            targets.append(concentration)

        elif flag == "top_coverage":
            top_coverage = calculator.calculate_top_coverage(
                target_column=target_column,
                mask_column=mask_column,
                head_percentage=hyperparameter,
            )
            targets.append(top_coverage)

        elif flag == "distinct_top_coverage":
            distinct_top_coverage = calculator.calculate_distinct_top_coverage(
                target_column=target_column,
                mask_column=mask_column,
                head_percentage=hyperparameter,
            )
            targets.append(distinct_top_coverage)

        elif flag == "wuauc":
            wuauc = calculator.calculate_wuauc(
                target_column=target_column,
                mask_column=mask_column,
                groupby=groupby,
                weights_for_groups=weights_for_groups,
            )
            targets.append(wuauc)

        elif flag == "auc":
            auc = calculator.calculate_wuauc(
                target_column=target_column,
                mask_column=mask_column,
                groupby=groupby,
                weights_for_groups=weights_for_groups,
                auc=True,
            )
            targets.append(auc)

        elif flag == "woauc":
            woauc = calculator.calculate_woauc(
                target_column=target_column,
                groupby=groupby,
                weights_for_groups=weights_for_groups,
            )
            targets.append(sum(woauc))

        elif flag == "logmse":
            mse = calculator.calculate_log_mse(
                target_column=target_column,
            )
            targets.append(mse)
        elif flag == "neg_rank_ratio":
            neg_rank_ratio = calculator.calculate_neg_rank_ratio(
                label_column=target_column
            )
            targets.append(neg_rank_ratio)

        elif flag == "inverse_pairs":
            inverse_score = calculator.calculate_inverse_pair(
                calculator=calculator,
                weights_type=evaluator_property,
            )
            targets.append(inverse_score)

        elif flag == "tau":
            tau = calculator.calculate_tau(
                groupby=groupby,
                target_column=target_column,
                weights_for_groups=weights_for_groups,
                num_bins=hyperparameter,
            )
            targets.append(tau)
    return targets
